"""Build a label-free PCA/LSI scaffold for geometry initialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from motrust.data.integration import TaskData, top_prevalent_features
from motrust.data.output import write_common_output



def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--domain", choices=("benchmark", "cortex"), default="benchmark")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--rna-features", type=int, default=4000)
    parser.add_argument("--atac-features", type=int, default=10000)
    parser.add_argument("--adt-features", type=int, default=256)
    return parser.parse_args(argv)


def _log_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    output = matrix.astype(np.float32, copy=True).tocsr()
    library = np.asarray(output.sum(axis=1)).ravel()
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    output = sparse.diags(scale).dot(output).tocsr()
    output.data = np.log1p(output.data)
    return output


def _tf_idf(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    output = matrix.astype(np.float32, copy=True).tocsr()
    output.data[:] = 1.0
    row_sum = np.asarray(output.sum(axis=1)).ravel()
    document_frequency = np.asarray(output.getnnz(axis=0)).ravel()
    tf = sparse.diags(
        np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    ).dot(output)
    idf = np.log1p(output.shape[0] / np.maximum(document_frequency, 1))
    return tf.multiply(idf).tocsr()


def _reduce(matrix: sparse.csr_matrix, modality: str, dim: int, seed: int) -> np.ndarray:
    work = _tf_idf(matrix) if modality == "atac" else _log_normalize(matrix)
    use_dim = min(dim, work.shape[0] - 1, work.shape[1] - 1)
    embedding = TruncatedSVD(n_components=use_dim, random_state=seed).fit_transform(work)
    if use_dim < dim:
        embedding = np.pad(embedding, ((0, 0), (0, dim - use_dim)))
    return StandardScaler().fit_transform(embedding).astype(np.float32)


def main(argv=None) -> None:
    args = parse_args(argv)
    data = TaskData(args.domain)
    blocks, _ = data.load_observation_blocks(args.task_id)
    limits = {"rna": args.rna_features, "atac": args.atac_features, "adt": args.adt_features}
    modality_embeddings: dict[str, np.ndarray] = {}
    modality_ids: dict[str, np.ndarray] = {}
    selected_counts: dict[str, int] = {}
    for modality, limit in limits.items():
        available = [block for block in blocks if modality in block["matrices"]]
        if not available:
            continue
        matrices = [block["matrices"][modality] for block in available]
        pooled = sparse.vstack(matrices, format="csr")
        selected = np.sort(top_prevalent_features(pooled, limit))
        modality_embeddings[modality] = _reduce(pooled[:, selected], modality, args.dim, args.seed)
        modality_ids[modality] = np.concatenate([block["cell_ids"] for block in available]).astype(str)
        selected_counts[modality] = int(len(selected))

    anchor = "rna" if "rna" in modality_embeddings else next(iter(modality_embeddings))
    aligned = {anchor}
    bridge_counts: dict[str, int] = {}
    pending = [name for name in modality_embeddings if name != anchor]
    while pending:
        progressed = False
        for modality in pending.copy():
            best_reference, best_common = None, np.array([], dtype=str)
            for reference in aligned:
                common = np.intersect1d(modality_ids[modality], modality_ids[reference])
                if len(common) > len(best_common):
                    best_reference, best_common = reference, common
            if best_reference is None or len(best_common) < 2:
                continue
            source_lookup = {cell: index for index, cell in enumerate(modality_ids[modality])}
            target_lookup = {cell: index for index, cell in enumerate(modality_ids[best_reference])}
            source_rows = np.fromiter((source_lookup[cell] for cell in best_common), dtype=np.int64)
            target_rows = np.fromiter((target_lookup[cell] for cell in best_common), dtype=np.int64)
            rotation, _ = orthogonal_procrustes(
                modality_embeddings[modality][source_rows],
                modality_embeddings[best_reference][target_rows],
            )
            modality_embeddings[modality] = modality_embeddings[modality] @ rotation
            bridge_counts[f"{modality}_to_{best_reference}"] = int(len(best_common))
            aligned.add(modality)
            pending.remove(modality)
            progressed = True
        if not progressed:
            break

    metadata = data.load_task_metadata(args.task_id, evaluation=False)
    global_ids = metadata["cell_id"].astype(str).to_numpy()
    lookup = {cell: index for index, cell in enumerate(global_ids)}
    embedding = np.zeros((len(global_ids), args.dim), dtype=np.float32)
    counts = np.zeros(len(global_ids), dtype=np.float32)
    for modality, values in modality_embeddings.items():
        rows = np.fromiter((lookup[cell] for cell in modality_ids[modality]), dtype=np.int64)
        np.add.at(embedding, rows, values)
        np.add.at(counts, rows, 1.0)
    if np.any(counts == 0):
        raise RuntimeError(f"No scaffold view for {(counts == 0).sum()} cells")
    embedding /= counts[:, None]

    manifest = {
        "schema_version": "1.0",
        "method": "MoTRUST-spectral-scaffold",
        "task_id": args.task_id,
        "seed": args.seed,
        "labels_used_for_training": False,
        "label_selected_checkpoint": False,
        "selected_feature_counts": selected_counts,
        "bridge_counts": bridge_counts,
        "preprocessing": {"rna_adt": "log-normalized TruncatedSVD", "atac": "binary TF-IDF LSI"},
        "purpose": "label-free spectral candidate",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_common_output(
        args.output_dir,
        args.task_id,
        "MoTRUST-spectral-scaffold",
        embedding,
        manifest,
        data=data,
    )
    (args.output_dir / "scaffold_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
