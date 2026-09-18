"""Row-aligned, label-safe integration output contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from motrust.data.integration import TaskData


REQUIRED_FILES = ("embedding.npy", "metadata.csv", "run_manifest.json")


def write_common_output(
    output_dir: str | Path,
    task_id: str,
    method: str,
    embedding: np.ndarray,
    manifest: dict[str, Any],
    *, data: TaskData | None = None,
) -> Path:
    """Write row-aligned, label-free method output and validate it immediately."""

    data = data or TaskData()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = data.load_task_metadata(task_id, evaluation=False)
    values = np.nan_to_num(
        np.asarray(embedding, dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0
    )
    if values.ndim != 2 or values.shape[0] != len(metadata):
        raise ValueError(
            f"{method}/{task_id}: embedding shape {values.shape} does not match "
            f"{len(metadata)} task rows"
        )
    if values.shape[1] < 2:
        raise ValueError(f"{method}/{task_id}: embedding must have at least two columns")

    np.save(output / "embedding.npy", values, allow_pickle=False)
    metadata.to_csv(output / "metadata.csv", index=False)
    payload = {
        "schema_version": "1.0",
        "method": method,
        "task_id": task_id,
        "status": "completed",
        "n_cells": int(values.shape[0]),
        "n_dimensions": int(values.shape[1]),
        "labels_used_for_training": False,
        "label_selected_checkpoint": False,
        **manifest,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    validate_common_output(output, task_id=task_id, method=method, data=data)
    return output


def validate_common_output(
    output_dir: str | Path,
    task_id: str | None = None,
    method: str | None = None,
    *, data: TaskData | None = None,
) -> dict[str, Any]:
    data = data or TaskData()
    output = Path(output_dir)
    missing = [name for name in REQUIRED_FILES if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing common outputs in {output}: {missing}")
    embedding = np.load(output / "embedding.npy", mmap_mode="r", allow_pickle=False)
    metadata = pd.read_csv(output / "metadata.csv", dtype=str)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    if embedding.ndim != 2 or embedding.shape[0] != len(metadata):
        raise ValueError(f"Embedding/metadata row mismatch in {output}")
    if metadata["cell_id"].duplicated().any():
        raise ValueError(f"Duplicate cell IDs in {output / 'metadata.csv'}")
    forbidden = {"cell_type", "source_cell_id", "broad_class", "fine_cluster"}.intersection(metadata.columns)
    if forbidden:
        raise ValueError(f"Evaluation-only columns leaked into method output: {sorted(forbidden)}")
    if task_id is not None:
        expected = data.load_task_metadata(task_id, evaluation=False)
        if not np.array_equal(metadata["cell_id"].to_numpy(), expected["cell_id"].to_numpy()):
            raise ValueError(f"Cell order differs from frozen task {task_id}")
        if manifest.get("task_id") != task_id:
            raise ValueError(f"Manifest task ID differs from {task_id}")
    if method is not None and manifest.get("method") != method:
        raise ValueError(f"Manifest method differs from {method}")
    if manifest.get("labels_used_for_training") is not False:
        raise ValueError("Benchmark adapters must not use labels for training")
    return manifest

