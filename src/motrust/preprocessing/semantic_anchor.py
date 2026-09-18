"""Build a shared biological anchor without requiring paired cells.

RNA genes, ATAC-derived gene activity, and ADT proteins are projected onto a
common gene basis before a joint low-rank decomposition. This gives diagonal
mosaic data a biologically meaningful orientation even when no cells are
co-observed across modalities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


_PEAK_PATTERN = re.compile(r"^(chr[^:\-_]+)[:-](\d+)[:-](\d+)$", re.IGNORECASE)
_ADT_SUFFIXES = re.compile(
    r"(?:[-_](?:TOTAL)?SEQ|[-_]PROTEIN|[-_]ADT|[-_]AB)$", re.IGNORECASE
)


@dataclass(frozen=True)
class SemanticAnchorResult:
    """Shared anchor and its auditable modality-level evidence."""

    embedding: np.ndarray
    reliability: np.ndarray
    modality_embeddings: dict[str, np.ndarray]
    modality_reliability: dict[str, np.ndarray]
    manifest: dict[str, object]


@dataclass(frozen=True)
class UnpairedSemanticAnchorResult:
    """Shared embedding for modalities with unequal, unpaired cell rows."""

    embedding: np.ndarray
    reliability: np.ndarray
    modality: np.ndarray
    manifest: dict[str, object]


def normalize_gene_symbol(value: str) -> str:
    """Normalize RNA or antibody labels to a conservative gene-symbol key."""

    symbol = str(value).strip().upper()
    symbol = _ADT_SUFFIXES.sub("", symbol)
    symbol = symbol.replace(" ", "")
    aliases = {
        "HLA-DR": "HLA-DRA",
        "CD45RA": "PTPRC",
        "CD45RO": "PTPRC",
        "CD45": "PTPRC",
    }
    return aliases.get(symbol, symbol)


def parse_peak_names(names: np.ndarray) -> pd.DataFrame:
    """Parse ``chr-start-end`` or ``chr:start-end`` feature names."""

    rows: list[tuple[str, int, int, int]] = []
    for index, raw in enumerate(np.asarray(names, dtype=str)):
        match = _PEAK_PATTERN.match(raw.strip())
        if match is None:
            continue
        start, end = int(match.group(2)), int(match.group(3))
        if end < start:
            start, end = end, start
        rows.append((match.group(1), start, end, index))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "feature_index"])


def read_gene_annotation_bed(path: str | Path) -> pd.DataFrame:
    """Read a BED-like chromosome/start/end/gene/strand annotation."""

    frame = pd.read_csv(path, sep="\t", header=None, comment="#", dtype={0: str})
    if frame.shape[1] < 4:
        raise ValueError("Gene annotation must contain chromosome, start, end, and symbol")
    output = pd.DataFrame(
        {
            "chrom": frame.iloc[:, 0].astype(str),
            "start": pd.to_numeric(frame.iloc[:, 1], errors="coerce"),
            "end": pd.to_numeric(frame.iloc[:, 2], errors="coerce"),
            "symbol": frame.iloc[:, 3].astype(str).map(normalize_gene_symbol),
            "strand": frame.iloc[:, 4].astype(str) if frame.shape[1] >= 5 else "+",
        }
    ).dropna(subset=["start", "end"])
    output["start"] = output["start"].astype(np.int64)
    output["end"] = output["end"].astype(np.int64)
    output = output[output["symbol"] != ""].drop_duplicates("symbol", keep="first")
    output["tss"] = np.where(output["strand"].to_numpy() == "-", output["end"], output["start"])
    return output.reset_index(drop=True)


def nearest_tss_mapping(
    peak_names: np.ndarray,
    gene_symbols: np.ndarray,
    annotation: pd.DataFrame,
    *,
    max_distance: int = 100_000,
    distance_scale: float = 50_000.0,
) -> sparse.csr_matrix:
    """Map peaks to the nearest requested gene TSS with distance decay."""

    peaks = parse_peak_names(peak_names)
    symbol_to_column = {
        normalize_gene_symbol(symbol): index for index, symbol in enumerate(gene_symbols)
    }
    genes = annotation[annotation["symbol"].isin(symbol_to_column)].copy()
    if peaks.empty or genes.empty:
        return sparse.csr_matrix((len(peak_names), len(gene_symbols)), dtype=np.float32)

    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for chromosome in sorted(set(peaks["chrom"]) & set(genes["chrom"])):
        chromosome_peaks = peaks[peaks["chrom"] == chromosome]
        chromosome_genes = genes[genes["chrom"] == chromosome].sort_values("tss")
        tss = chromosome_genes["tss"].to_numpy(dtype=np.int64)
        centers = (
            chromosome_peaks["start"].to_numpy(dtype=np.int64)
            + chromosome_peaks["end"].to_numpy(dtype=np.int64)
        ) // 2
        right = np.searchsorted(tss, centers).clip(0, len(tss) - 1)
        left = np.maximum(right - 1, 0)
        use_right = np.abs(tss[right] - centers) < np.abs(tss[left] - centers)
        nearest = np.where(use_right, right, left)
        distance = np.abs(tss[nearest] - centers)
        keep = distance <= int(max_distance)
        if not np.any(keep):
            continue
        rows.append(chromosome_peaks["feature_index"].to_numpy(dtype=np.int64)[keep])
        selected_symbols = chromosome_genes["symbol"].to_numpy()[nearest[keep]]
        columns.append(np.asarray([symbol_to_column[value] for value in selected_symbols]))
        weights.append(np.exp(-distance[keep] / max(float(distance_scale), 1.0)).astype(np.float32))
    if not rows:
        return sparse.csr_matrix((len(peak_names), len(gene_symbols)), dtype=np.float32)
    return sparse.csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(peak_names), len(gene_symbols)),
        dtype=np.float32,
    )


def _coordinate_tss_mapping(
    genes: pd.DataFrame,
    peaks: pd.DataFrame,
    *,
    max_distance: int,
    distance_scale: float,
) -> sparse.csr_matrix:
    """Map coordinate tables while preserving the supplied gene column order."""

    gene_table = genes.copy().reset_index(drop=True)
    peak_table = peaks.copy().reset_index(drop=True)
    gene_table["tss"] = np.where(
        gene_table.get("strand", pd.Series("+", index=gene_table.index)).astype(str) == "-",
        gene_table["chromEnd"].to_numpy(),
        gene_table["chromStart"].to_numpy(),
    )
    centers = (
        peak_table["chromStart"].to_numpy(dtype=np.int64)
        + peak_table["chromEnd"].to_numpy(dtype=np.int64)
    ) // 2
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for chromosome in sorted(set(gene_table["chrom"]) & set(peak_table["chrom"])):
        gene_index = np.flatnonzero(gene_table["chrom"].to_numpy() == chromosome)
        peak_index = np.flatnonzero(peak_table["chrom"].to_numpy() == chromosome)
        order = np.argsort(gene_table.iloc[gene_index]["tss"].to_numpy())
        sorted_gene_index = gene_index[order]
        tss = gene_table.iloc[sorted_gene_index]["tss"].to_numpy(dtype=np.int64)
        current_centers = centers[peak_index]
        right = np.searchsorted(tss, current_centers).clip(0, len(tss) - 1)
        left = np.maximum(right - 1, 0)
        choose_right = np.abs(tss[right] - current_centers) < np.abs(tss[left] - current_centers)
        nearest = np.where(choose_right, right, left)
        distance = np.abs(tss[nearest] - current_centers)
        keep = distance <= int(max_distance)
        rows.append(peak_index[keep])
        columns.append(sorted_gene_index[nearest[keep]])
        weights.append(np.exp(-distance[keep] / float(distance_scale)).astype(np.float32))
    return sparse.csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(peak_table), len(gene_table)),
        dtype=np.float32,
    )


def build_unpaired_shared_gene_anchor(
    rna: sparse.spmatrix,
    atac: sparse.spmatrix,
    gene_table: pd.DataFrame,
    peak_table: pd.DataFrame,
    *,
    gene_scores: np.ndarray | None = None,
    n_genes: int = 2000,
    dim: int = 64,
    seed: int = 0,
    max_distance: int = 100_000,
    distance_scale: float = 50_000.0,
    center_modalities: bool = True,
) -> UnpairedSemanticAnchorResult:
    """Build the Muto-style shared-gene anchor through the common MoTRUST API."""

    rna = sparse.csr_matrix(rna, dtype=np.float32)
    atac = sparse.csr_matrix(atac, dtype=np.float32)
    gene_table = gene_table.copy().reset_index(drop=True)
    peak_table = peak_table.copy().reset_index(drop=True)
    if len(gene_table) != rna.shape[1] or len(peak_table) != atac.shape[1]:
        raise ValueError("Feature annotation rows must match matrix columns")
    use_genes = min(max(2, int(n_genes)), rna.shape[1])
    if gene_scores is None:
        scores = np.asarray(rna.getnnz(axis=0)).ravel().astype(np.float64)
        selection = "prevalence"
    else:
        scores = np.nan_to_num(np.asarray(gene_scores, dtype=np.float64), nan=-np.inf)
        if len(scores) != rna.shape[1]:
            raise ValueError("gene_scores must contain one value per RNA feature")
        selection = "supplied_score"
    selected = np.argpartition(scores, -use_genes)[-use_genes:]
    selected = selected[np.argsort(-scores[selected], kind="stable")]
    selected_genes = gene_table.iloc[selected].reset_index(drop=True)
    mapping = _coordinate_tss_mapping(
        selected_genes,
        peak_table,
        max_distance=max_distance,
        distance_scale=distance_scale,
    )
    rna_selected = rna[:, selected]
    activity = atac @ mapping
    rna_normalized = StandardScaler(with_mean=False).fit_transform(
        _log_library_normalize(rna_selected)
    ).tocsr()
    activity_normalized = StandardScaler(with_mean=False).fit_transform(
        _log_library_normalize(activity)
    ).tocsr()
    joined = sparse.vstack((rna_normalized, activity_normalized), format="csr")
    use_dim = min(int(dim), joined.shape[0] - 1, joined.shape[1] - 1)
    embedding = TruncatedSVD(n_components=use_dim, random_state=seed).fit_transform(joined)
    if center_modalities:
        rna_count = rna.shape[0]
        embedding[:rna_count] -= embedding[:rna_count].mean(axis=0, keepdims=True)
        embedding[rna_count:] -= embedding[rna_count:].mean(axis=0, keepdims=True)
    modality = np.asarray(["RNA"] * rna.shape[0] + ["ATAC"] * atac.shape[0])
    reliability = np.concatenate((_quality_reliability(rna), _quality_reliability(atac)))
    return UnpairedSemanticAnchorResult(
        embedding=embedding.astype(np.float32),
        reliability=reliability.astype(np.float32),
        modality=modality,
        manifest={
            "labels_used": False,
            "cell_correspondence_required": False,
            "gene_selection": selection,
            "selected_rna_genes": int(use_genes),
            "mapped_peaks": int(mapping.nnz),
            "components": int(use_dim),
            "center_modalities": bool(center_modalities),
            "max_distance": int(max_distance),
            "distance_scale": float(distance_scale),
        },
    )


def _log_library_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    output = matrix.astype(np.float32, copy=True).tocsr()
    library = np.asarray(output.sum(axis=1)).ravel()
    scale = np.divide(
        1e4,
        library,
        out=np.zeros_like(library, dtype=np.float32),
        where=library > 0,
    )
    output = sparse.diags(scale).dot(output).tocsr()
    output.data = np.log1p(output.data)
    return output


def _quality_reliability(matrix: sparse.csr_matrix) -> np.ndarray:
    library = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    detected = np.asarray(matrix.getnnz(axis=1)).ravel().astype(np.float64)

    def robust_scale(values: np.ndarray) -> np.ndarray:
        positive = values[values > 0]
        if len(positive) == 0:
            return np.zeros_like(values, dtype=np.float32)
        low, high = np.quantile(np.log1p(positive), [0.1, 0.9])
        if high <= low:
            return (values > 0).astype(np.float32)
        return np.clip((np.log1p(values) - low) / (high - low), 0.0, 1.0).astype(np.float32)

    return np.sqrt(robust_scale(library) * robust_scale(detected)).astype(np.float32)


def _protein_projection(proteins: np.ndarray, genes: np.ndarray) -> sparse.csr_matrix:
    gene_lookup = {normalize_gene_symbol(value): index for index, value in enumerate(genes)}
    rows, columns = [], []
    for row, protein in enumerate(proteins):
        symbol = normalize_gene_symbol(protein)
        if symbol in gene_lookup:
            rows.append(row)
            columns.append(gene_lookup[symbol])
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(proteins), len(genes)),
    )


def build_shared_semantic_anchor(
    matrices: Mapping[str, sparse.spmatrix],
    masks: Mapping[str, np.ndarray],
    feature_names: Mapping[str, np.ndarray],
    *,
    annotation_path: str | Path | None = None,
    dim: int = 32,
    n_genes: int = 4000,
    seed: int = 0,
    max_distance: int = 100_000,
    distance_scale: float = 50_000.0,
    center_modalities: bool = True,
    sparse_shared_feature_threshold: int = 64,
    reference_modality: str | None = "rna",
) -> SemanticAnchorResult:
    """Construct one shared, label-free anchor for aligned mosaic matrices."""

    if "rna" not in matrices or "rna" not in feature_names:
        raise ValueError("A shared semantic anchor requires an RNA feature universe")
    n_cells = next(iter(matrices.values())).shape[0]
    if any(matrix.shape[0] != n_cells for matrix in matrices.values()):
        raise ValueError("All modality matrices must be aligned to the same cell rows")

    rna = sparse.csr_matrix(matrices["rna"], dtype=np.float32)
    rna_observed = np.asarray(masks["rna"]) > 0
    prevalence = np.asarray(rna[rna_observed].getnnz(axis=0)).ravel()
    rna_names = np.asarray(feature_names["rna"], dtype=str)
    rna_lookup = {normalize_gene_symbol(value): index for index, value in enumerate(rna_names)}
    forced: set[int] = set()
    if "adt" in feature_names:
        for protein in np.asarray(feature_names["adt"], dtype=str):
            index = rna_lookup.get(normalize_gene_symbol(protein))
            if index is not None:
                forced.add(index)
    use_genes = min(max(2, int(n_genes)), rna.shape[1])
    selected = set(np.argpartition(prevalence, -use_genes)[-use_genes:].tolist())
    selected.update(forced)
    selected = np.asarray(sorted(selected, key=lambda index: (-prevalence[index], index)))
    genes = rna_names[selected]

    shared_views: dict[str, sparse.csr_matrix] = {"rna": rna[:, selected].tocsr()}
    mapping_counts: dict[str, int] = {"rna": int(len(selected))}
    if "atac" in matrices and "atac" in feature_names:
        if annotation_path is None:
            raise ValueError("annotation_path is required when ATAC is present")
        annotation = read_gene_annotation_bed(annotation_path)
        mapping = nearest_tss_mapping(
            np.asarray(feature_names["atac"], dtype=str),
            genes,
            annotation,
            max_distance=max_distance,
            distance_scale=distance_scale,
        )
        shared_views["atac"] = sparse.csr_matrix(matrices["atac"], dtype=np.float32) @ mapping
        mapping_counts["atac"] = int(mapping.nnz)
    if "adt" in matrices and "adt" in feature_names:
        projection = _protein_projection(np.asarray(feature_names["adt"], dtype=str), genes)
        if projection.nnz >= 2:
            shared_views["adt"] = sparse.csr_matrix(matrices["adt"], dtype=np.float32) @ projection
            mapping_counts["adt"] = int(projection.nnz)

    normalized: dict[str, sparse.csr_matrix] = {}
    observed_rows: dict[str, np.ndarray] = {}
    reliability: dict[str, np.ndarray] = {}
    stacked: list[sparse.csr_matrix] = []
    stack_modalities: list[str] = []
    for modality, matrix in shared_views.items():
        observed = np.asarray(masks.get(modality, np.zeros(n_cells))) > 0
        rows = np.flatnonzero(observed)
        if len(rows) < 2 or matrix[:, matrix.getnnz(axis=0) > 0].shape[1] < 2:
            continue
        values = _log_library_normalize(matrix[rows].tocsr())
        values = StandardScaler(with_mean=False).fit_transform(values).tocsr()
        normalized[modality] = values
        observed_rows[modality] = rows
        reliability[modality] = _quality_reliability(matrix[rows].tocsr())
        stacked.append(values)
        stack_modalities.append(modality)
    if len(stacked) < 2:
        raise ValueError("At least two modalities need two or more shared semantic features")

    joined = sparse.vstack(stacked, format="csr")
    use_dim = min(int(dim), joined.shape[0] - 1, joined.shape[1] - 1)
    if use_dim < 1:
        raise ValueError("Shared semantic matrix has insufficient rank")
    reduced = TruncatedSVD(n_components=use_dim, random_state=seed).fit_transform(joined)
    if use_dim < dim:
        reduced = np.pad(reduced, ((0, 0), (0, dim - use_dim)))

    modality_embeddings: dict[str, np.ndarray] = {}
    offset = 0
    for modality, values in zip(stack_modalities, stacked):
        n_rows = values.shape[0]
        current = StandardScaler().fit_transform(reduced[offset : offset + n_rows]).astype(np.float32)
        if center_modalities:
            current -= current.mean(axis=0, keepdims=True)
        modality_embeddings[modality] = current
        offset += n_rows

    # Sparse antibody-to-gene overlap is useful for orientation but too narrow
    # to represent an ADT-only cell. When bridge cells exist, align a full ADT
    # spectral view to the RNA/ATAC semantic coordinate system and then use it
    # for every ADT cell. The same rule is label-free and does not affect the
    # RNA-ATAC no-bridge path.
    bridge_refinements: dict[str, int] = {}
    robust_modalities = [
        name
        for name in modality_embeddings
        if mapping_counts.get(name, 0) >= int(sparse_shared_feature_threshold)
    ]
    provisional = np.zeros((n_cells, dim), dtype=np.float32)
    provisional_count = np.zeros(n_cells, dtype=np.float32)
    for modality in robust_modalities:
        rows = observed_rows[modality]
        provisional[rows] += modality_embeddings[modality]
        provisional_count[rows] += 1.0
    available = provisional_count > 0
    provisional[available] /= provisional_count[available, None]
    for modality in list(modality_embeddings):
        if mapping_counts.get(modality, 0) >= int(sparse_shared_feature_threshold):
            continue
        rows = observed_rows[modality]
        bridge = available[rows]
        if int(bridge.sum()) < 2:
            continue
        raw = sparse.csr_matrix(matrices[modality], dtype=np.float32)[rows]
        raw = _log_library_normalize(raw)
        spectral_dim = min(dim, raw.shape[0] - 1, raw.shape[1] - 1)
        if spectral_dim < 1:
            continue
        spectral = TruncatedSVD(n_components=spectral_dim, random_state=seed).fit_transform(raw)
        if spectral_dim < dim:
            spectral = np.pad(spectral, ((0, 0), (0, dim - spectral_dim)))
        spectral = StandardScaler().fit_transform(spectral).astype(np.float32)
        target = provisional[rows[bridge]]
        rotation, _ = orthogonal_procrustes(spectral[bridge], target)
        modality_embeddings[modality] = (spectral @ rotation).astype(np.float32)
        bridge_refinements[modality] = int(bridge.sum())

    embedding = np.zeros((n_cells, dim), dtype=np.float32)
    total_weight = np.zeros(n_cells, dtype=np.float32)
    modality_reliability: dict[str, np.ndarray] = {}
    modality_order = list(modality_embeddings)
    if reference_modality in modality_order:
        modality_order.remove(reference_modality)
        modality_order.insert(0, reference_modality)
    for modality in modality_order:
        current = modality_embeddings[modality]
        rows = observed_rows[modality]
        quality = np.clip(reliability[modality], 0.05, 1.0)
        full_quality = np.zeros(n_cells, dtype=np.float32)
        full_quality[rows] = quality
        modality_reliability[modality] = full_quality
        if modality == reference_modality:
            use = np.ones(len(rows), dtype=bool)
        else:
            use = total_weight[rows] == 0
        selected_rows = rows[use]
        embedding[selected_rows] += current[use] * quality[use, None]
        total_weight[selected_rows] += quality[use]
    valid = total_weight > 0
    embedding[valid] /= total_weight[valid, None]
    if not np.all(valid):
        raise ValueError(f"No shared semantic view for {int((~valid).sum())} cells")

    cell_reliability = total_weight
    manifest: dict[str, object] = {
        "labels_used": False,
        "cell_correspondence_required": False,
        "modalities": stack_modalities,
        "selected_rna_genes": int(len(genes)),
        "forced_adt_genes": int(len(forced)),
        "mapping_counts": mapping_counts,
        "bridge_refinements": bridge_refinements,
        "sparse_shared_feature_threshold": int(sparse_shared_feature_threshold),
        "components": int(dim),
        "center_modalities": bool(center_modalities),
        "reference_modality": reference_modality,
        "max_distance": int(max_distance),
        "distance_scale": float(distance_scale),
    }
    return SemanticAnchorResult(
        embedding=embedding,
        reliability=np.clip(cell_reliability, 0.0, 1.0).astype(np.float32),
        modality_embeddings=modality_embeddings,
        modality_reliability=modality_reliability,
        manifest=manifest,
    )
