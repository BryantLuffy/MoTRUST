"""Method-independent metrics following the Palette benchmark definitions.

See resources/integration/protocol.json for source attribution and the fixed
score definitions. This implementation does not require scib-metrics.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_samples,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MetricConfig:
    n_neighbors: int = 30
    silhouette_sample_size: int = 5000
    mixing_repeats: int = 10
    csas_repeats: int = 4
    csas_k_mode: str = "official"
    seed: int = 0
    standardize: bool = True


def clean_embedding(x: np.ndarray, standardize: bool = True) -> np.ndarray:
    out = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0)
    if out.ndim != 2 or out.shape[0] < 3:
        raise ValueError("Embedding must be a two-dimensional matrix with at least three rows")
    if standardize:
        out = StandardScaler().fit_transform(out).astype(np.float32)
    return out


def neighbor_indices(x: np.ndarray, k: int) -> np.ndarray:
    use_k = min(max(1, int(k)), len(x) - 1)
    # Keep process-level parallelism disabled on Windows. This avoids spawning
    # joblib worker pipes in restricted runs and makes evaluation predictable.
    model = NearestNeighbors(n_neighbors=use_k + 1, metric="euclidean", n_jobs=1).fit(x)
    return model.kneighbors(x, return_distance=False)[:, 1:]


def _sample_indices(n: int, sample_size: int, seed: int) -> np.ndarray:
    if n <= sample_size:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, sample_size, replace=False))


def cell_type_asw(x: np.ndarray, labels: np.ndarray, sample_size: int, seed: int) -> float:
    if len(np.unique(labels)) < 2:
        return math.nan
    idx = _sample_indices(len(x), sample_size, seed)
    raw = float(np.mean(silhouette_samples(x[idx], labels[idx])))
    return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))


def batch_asw(x: np.ndarray, batches: np.ndarray, sample_size: int, seed: int) -> float:
    """Equation 21: equal-weight average of 1-|batch silhouette| per batch."""
    if len(np.unique(batches)) < 2:
        return math.nan
    idx = _sample_indices(len(x), sample_size, seed)
    sampled_batches = batches[idx]
    values = silhouette_samples(x[idx], sampled_batches)
    per_batch = [np.mean(1.0 - np.abs(values[sampled_batches == b])) for b in np.unique(sampled_batches)]
    return float(np.clip(np.mean(per_batch), 0.0, 1.0))


def lisi_values(labels: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    categories = pd.Categorical(labels.astype(str))
    counts = np.zeros((len(neighbors), len(categories.categories)), dtype=np.float32)
    rows = np.repeat(np.arange(len(neighbors)), neighbors.shape[1])
    np.add.at(counts, (rows, categories.codes[neighbors].reshape(-1)), 1.0)
    probabilities = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    return 1.0 / np.maximum(np.square(probabilities).sum(axis=1), 1e-12)


def scaled_lisi(labels: np.ndarray, neighbors: np.ndarray, conservation: bool) -> float:
    n_groups = len(np.unique(labels))
    if n_groups < 2:
        return math.nan
    raw = lisi_values(labels, neighbors)
    mixing = np.clip((raw - 1.0) / (n_groups - 1.0), 0.0, 1.0)
    score = 1.0 - mixing if conservation else mixing
    return float(np.median(score))


def graph_connectivity(labels: np.ndarray, neighbors: np.ndarray) -> float:
    rows = np.repeat(np.arange(len(neighbors)), neighbors.shape[1])
    cols = neighbors.reshape(-1)
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(labels), len(labels)))
    graph = graph.maximum(graph.T)
    scores: list[float] = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) <= 1:
            continue
        _, components = connected_components(graph[idx][:, idx], directed=False)
        scores.append(float(np.bincount(components).max() / len(idx)))
    return float(np.mean(scores)) if scores else math.nan


def kbet_score(group_labels: np.ndarray, neighbors: np.ndarray, alpha: float = 0.05) -> float:
    """Local Pearson chi-square kBET acceptance score in [0, 1]."""
    categories = pd.Categorical(group_labels.astype(str))
    n_groups = len(categories.categories)
    if n_groups < 2:
        return math.nan
    global_prob = np.bincount(categories.codes, minlength=n_groups).astype(float)
    global_prob /= global_prob.sum()
    local_codes = categories.codes[neighbors]
    observed = np.zeros((len(neighbors), n_groups), dtype=float)
    rows = np.repeat(np.arange(len(neighbors)), neighbors.shape[1])
    np.add.at(observed, (rows, local_codes.reshape(-1)), 1.0)
    expected = neighbors.shape[1] * global_prob[None, :]
    valid = expected[0] > 0
    statistic = np.sum(np.square(observed[:, valid] - expected[:, valid]) / expected[:, valid], axis=1)
    pvalues = chi2.sf(statistic, df=max(1, valid.sum() - 1))
    return float(np.mean(pvalues >= alpha))


def _balanced_subsample(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    size = min(len(group) for group in groups)
    return np.concatenate([rng.choice(group, size=size, replace=False) for group in groups])


def seurat_alignment_score(
    x: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
    k_mode: str = "official",
) -> float:
    if len(np.unique(groups)) < 2:
        return math.nan
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        idx = _balanced_subsample(groups, rng)
        local_x, local_groups = x[idx], groups[idx]
        n_group = len(np.unique(local_groups))
        if k_mode == "official":
            k = min(30, len(idx) - 1)
        elif k_mode == "paper":
            k = min(max(1, round(len(idx) * 0.01)), len(idx) - 1)
        else:
            raise ValueError("csas_k_mode must be 'official' or 'paper'")
        nn = neighbor_indices(local_x, k)
        same_hits = float(np.mean(np.sum(local_groups[nn] == local_groups[:, None], axis=1)))
        score = (k - same_hits) * n_group / (k * (n_group - 1))
        scores.append(min(float(score), 1.0))
    return float(np.mean(scores))


def stratified_mixing_metrics(
    x: np.ndarray,
    cell_types: np.ndarray,
    groups: np.ndarray,
    config: MetricConfig,
) -> dict[str, float]:
    """Cell-type-stratified CSAS, kBET, and CiLISI with balanced groups."""
    rng = np.random.default_rng(config.seed)
    csas_by_type: list[float] = []
    kbet_by_type: list[float] = []
    cilisi_by_type: list[float] = []
    for cell_type in np.unique(cell_types):
        idx = np.flatnonzero(cell_types == cell_type)
        local_groups = groups[idx]
        if len(idx) < 6 or len(np.unique(local_groups)) < 2:
            continue
        counts = pd.Series(local_groups).value_counts()
        if counts.min() < 3:
            continue
        local_x = x[idx]
        csas_by_type.append(
            seurat_alignment_score(
                local_x,
                local_groups,
                repeats=config.csas_repeats,
                seed=config.seed,
                k_mode=config.csas_k_mode,
            )
        )
        repeat_kbet, repeat_lisi = [], []
        for _ in range(config.mixing_repeats):
            keep = _balanced_subsample(local_groups, rng)
            sub_x, sub_groups = local_x[keep], local_groups[keep]
            nn = neighbor_indices(sub_x, min(config.n_neighbors, len(sub_x) - 1))
            repeat_kbet.append(kbet_score(sub_groups, nn))
            repeat_lisi.append(scaled_lisi(sub_groups, nn, conservation=False))
        kbet_by_type.append(float(np.nanmedian(repeat_kbet)))
        cilisi_by_type.append(float(np.nanmedian(repeat_lisi)))
    return {
        "CSAS": float(np.nanmedian(csas_by_type)) if csas_by_type else math.nan,
        "kBET": float(np.nanmedian(kbet_by_type)) if kbet_by_type else math.nan,
        "CiLISI": float(np.nanmedian(cilisi_by_type)) if cilisi_by_type else math.nan,
        "n_evaluable_cell_types": len(csas_by_type),
    }


def _mean_available(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.nanmean(array)) if np.any(np.isfinite(array)) else math.nan


def _mean_complete(values: Iterable[float]) -> float:
    """Paper aggregate: undefined unless every named component is available."""
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array)) if np.all(np.isfinite(array)) else math.nan


def integration_metrics(
    embedding: np.ndarray,
    cell_types: np.ndarray,
    batches: np.ndarray,
    modalities: np.ndarray,
    config: MetricConfig,
    reference_query: np.ndarray | None = None,
    species: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    x = clean_embedding(embedding, config.standardize)
    cell_types = np.asarray(cell_types).astype(str)
    batches = np.asarray(batches).astype(str)
    modalities = np.asarray(modalities).astype(str)
    if not (len(x) == len(cell_types) == len(batches) == len(modalities)):
        raise ValueError("Embedding and metadata lengths differ")
    nn = neighbor_indices(x, config.n_neighbors)
    n_clusters = len(np.unique(cell_types))
    predicted = KMeans(n_clusters=n_clusters, n_init=20, random_state=config.seed).fit_predict(x)

    result: dict[str, float | int | str] = {
        "n_cells": len(x),
        "n_dimensions": x.shape[1],
        "n_cell_types": n_clusters,
        "NMI": float(normalized_mutual_info_score(cell_types, predicted)),
        "ARI": float(adjusted_rand_score(cell_types, predicted)),
        "cASW": cell_type_asw(x, cell_types, config.silhouette_sample_size, config.seed),
        "cLISI": scaled_lisi(cell_types, nn, conservation=True),
        "bASW": batch_asw(x, batches, config.silhouette_sample_size, config.seed),
        "iLISI": scaled_lisi(batches, nn, conservation=False),
        "batch_kBET": kbet_score(batches, nn),
        "graph_connectivity": graph_connectivity(cell_types, nn),
        "csas_k_mode": config.csas_k_mode,
    }
    result["biological_conservation_score"] = _mean_complete(
        result[key] for key in ("NMI", "ARI", "cASW", "cLISI")
    )
    result["batch_correction_score"] = _mean_complete(
        result[key] for key in ("bASW", "iLISI", "batch_kBET", "graph_connectivity")
    )
    bio_score = float(result["biological_conservation_score"])
    batch_score = float(result["batch_correction_score"])
    result["overall_integration_score"] = (
        0.6 * bio_score + 0.4 * batch_score
        if np.isfinite(bio_score) and np.isfinite(batch_score)
        else math.nan
    )

    mixing_sets = {"modality": modalities}
    if reference_query is not None:
        mixing_sets["reference_query"] = np.asarray(reference_query).astype(str)
    if species is not None:
        mixing_sets["species"] = np.asarray(species).astype(str)
    for prefix, groups in mixing_sets.items():
        values = stratified_mixing_metrics(x, cell_types, groups, config)
        result[f"{prefix}_CSAS"] = values["CSAS"]
        result[f"{prefix}_kBET"] = values["kBET"]
        result[f"{prefix}_CiLISI"] = values["CiLISI"]
        result[f"{prefix}_evaluable_cell_types"] = values["n_evaluable_cell_types"]
        result[f"{prefix}_mixing_score"] = _mean_complete(
            values[key] for key in ("CSAS", "kBET", "CiLISI")
        )
    return result


def label_transfer(
    embedding: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    reference_value: str,
    query_value: str,
    k: int = 5,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    x = np.asarray(embedding, dtype=np.float32)
    labels = np.asarray(labels).astype(str)
    split = np.asarray(split).astype(str)
    ref_idx = np.flatnonzero(split == reference_value)
    query_idx = np.flatnonzero(split == query_value)
    if len(ref_idx) < k or len(query_idx) == 0:
        raise ValueError(f"Insufficient rows for {reference_value} -> {query_value} transfer")
    model = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean")
    model.fit(x[ref_idx], labels[ref_idx])
    predicted = model.predict(x[query_idx])
    truth = labels[query_idx]
    metrics: dict[str, float | int | str] = {
        "direction": f"{reference_value}->{query_value}",
        "k": k,
        "n_reference": len(ref_idx),
        "n_query": len(query_idx),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", labels=np.unique(truth), zero_division=0)),
    }
    predictions = pd.DataFrame(
        {"row_index": query_idx, "true_label": truth, "predicted_label": predicted, "direction": metrics["direction"]}
    )
    return metrics, predictions
