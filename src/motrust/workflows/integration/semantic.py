"""Build the MoTRUST shared semantic anchor for one mosaic task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from motrust.preprocessing import build_shared_semantic_anchor
from motrust.data.integration import TaskData
from motrust.data.output import write_common_output


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--domain", choices=("benchmark", "cortex"), default="benchmark")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gene-annotation", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--genes", type=int, default=4000)
    parser.add_argument("--max-distance", type=int, default=100000)
    parser.add_argument("--distance-scale", type=float, default=50000.0)
    parser.add_argument("--no-modality-centering", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    data = TaskData(args.domain)
    matrices, masks, _metadata = data.align_modalities_to_task(args.task_id)
    _raw_matrices, _ids, feature_names = data.load_task_modalities(args.task_id)
    result = build_shared_semantic_anchor(
        matrices,
        masks,
        feature_names,
        annotation_path=args.gene_annotation,
        dim=args.dim,
        n_genes=args.genes,
        seed=args.seed,
        max_distance=args.max_distance,
        distance_scale=args.distance_scale,
        center_modalities=not args.no_modality_centering,
    )
    manifest = {
        "configuration_id": "shared_semantic_anchor_v1",
        "seed": args.seed,
        "gene_annotation": str(args.gene_annotation.resolve()),
        **result.manifest,
    }
    write_common_output(
        args.output_dir,
        args.task_id,
        "MoTRUST-semantic-anchor",
        result.embedding,
        manifest,
        data=data,
    )
    np.save(args.output_dir / "semantic_reliability.npy", result.reliability)
    for modality, values in result.modality_reliability.items():
        np.save(args.output_dir / f"{modality}_reliability.npy", values)
        rows = np.flatnonzero(values > 0)
        np.save(args.output_dir / f"{modality}_observed_rows.npy", rows)
        np.save(args.output_dir / f"{modality}_embedding.npy", result.modality_embeddings[modality])
    (args.output_dir / "semantic_anchor_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"task_id": args.task_id, **result.manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
