"""Formatting helpers shared by method-specific recovery evaluators."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .recovery_metrics import evaluate_all


def formal_metric_rows(
    *,
    method: str,
    scenario: str,
    replicate: int,
    seed: int,
    modality: str,
    recovery_direction: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    labels: np.ndarray,
    reference_truth: np.ndarray,
    reference_labels: np.ndarray,
    source: str | Path,
) -> list[dict[str, object]]:
    task_id = f"{scenario.lower()}__{recovery_direction}"
    values = evaluate_all(
        prediction,
        truth,
        labels,
        modality=modality,
        seed=seed,
        reference_truth=reference_truth,
        reference_labels=reference_labels,
    )
    return [
        {
            "task_id": task_id,
            "scenario": scenario,
            "replicate": replicate,
            "recovery_direction": recovery_direction,
            "modality": modality,
            "method": method,
            "seed": seed,
            "category": value.category,
            "metric": value.metric,
            "value": value.value,
            "higher_is_better": value.higher_is_better,
            "status": "completed",
            "source": str(source),
        }
        for value in values
    ]


def observed_reference(truth_adata, modality: str) -> tuple[np.ndarray, np.ndarray]:
    column = f"{modality.lower()}_observed"
    if column not in truth_adata.obs:
        raise KeyError(f"Frozen truth data are missing {column}")
    mask = truth_adata.obs[column].astype(bool).to_numpy()
    matrix = truth_adata.layers["counts"] if "counts" in truth_adata.layers else truth_adata.X
    matrix = matrix[mask]
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    labels = truth_adata.obs.loc[mask, "cell_type"].astype(str).to_numpy()
    return np.asarray(matrix, dtype=np.float32), labels
