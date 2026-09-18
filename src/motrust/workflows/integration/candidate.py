"""Finalize a candidate representation with the fixed geometry-drift rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from motrust.data.integration import TaskData
from motrust.data.output import write_common_output



def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--domain", choices=("benchmark", "cortex"), default="benchmark")
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--spectral-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drift-threshold", type=float, default=3.0)
    parser.add_argument("--clusters", type=int, default=80)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--shrinkage", type=float, default=20.0)
    return parser.parse_args(argv)


def cluster_conditional_align(
    embedding: np.ndarray,
    batches: np.ndarray,
    clusters: int,
    iterations: int,
    strength: float,
    shrinkage: float,
) -> np.ndarray:
    corrected = StandardScaler().fit_transform(embedding).astype(np.float32)
    for iteration in range(iterations):
        assignments = MiniBatchKMeans(
            n_clusters=min(clusters, len(corrected)),
            batch_size=2048,
            n_init=5,
            random_state=iteration,
        ).fit_predict(corrected)
        updated = corrected.copy()
        for cluster in np.unique(assignments):
            cluster_mask = assignments == cluster
            cluster_center = corrected[cluster_mask].mean(axis=0)
            for batch in np.unique(batches[cluster_mask]):
                selected = cluster_mask & (batches == batch)
                count = int(selected.sum())
                if count < 2:
                    continue
                batch_center = corrected[selected].mean(axis=0)
                reliability = count / (count + shrinkage)
                updated[selected] -= strength * reliability * (batch_center - cluster_center)
        corrected = updated
    return corrected


def drift_ratio(history: pd.DataFrame) -> tuple[float, float, float]:
    geometry = history["geometry"].astype(float).to_numpy()
    if len(geometry) < 30:
        stable = geometry[max(0, len(geometry) // 2 - 5) : max(1, len(geometry) // 2 + 5)]
    else:
        stable = geometry[19:30]
    stable_mean = float(np.mean(stable))
    final = float(geometry[-1])
    return final / max(stable_mean, 1e-8), stable_mean, final


def main(argv=None) -> None:
    args = parse_args(argv)
    data = TaskData(args.domain)
    primary_dir = args.primary_root.resolve() / args.task_id
    spectral_dir = args.spectral_root.resolve() / args.task_id
    primary_metadata = pd.read_csv(primary_dir / "metadata.csv", dtype=str)
    spectral_metadata = pd.read_csv(spectral_dir / "metadata.csv", dtype=str)
    primary_manifest = json.loads((primary_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if not primary_metadata["cell_id"].equals(spectral_metadata["cell_id"]):
        raise ValueError(f"Primary and spectral cell order differs for {args.task_id}")

    ratio, stable_geometry, final_geometry = drift_ratio(
        pd.read_csv(primary_dir / "training_history.csv")
    )
    if ratio > args.drift_threshold:
        embedding = np.load(spectral_dir / "embedding.npy", allow_pickle=False)
        selected_path = "spectral_drift_fallback"
        refinement = "none"
    else:
        primary = np.load(primary_dir / "embedding.npy", allow_pickle=False)
        batches = primary_metadata["instance_batch"].astype(str).to_numpy()
        embedding = cluster_conditional_align(
            primary,
            batches,
            args.clusters,
            args.iterations,
            args.strength,
            args.shrinkage,
        )
        selected_path = "probabilistic"
        refinement = "cluster_conditional_batch_alignment"

    manifest = {
        "labels_used_for_training": False,
        "label_selected_checkpoint": False,
        "seed": primary_manifest.get("seed"),
        "configuration_id": "drift_rejection_cluster80_i8",
        "primary_run": str(primary_dir),
        "spectral_run": str(spectral_dir),
        "selection_rule": "spectral iff final_geometry/stable_geometry > drift_threshold",
        "selected_path": selected_path,
        "drift_ratio": ratio,
        "stable_geometry": stable_geometry,
        "final_geometry": final_geometry,
        "drift_threshold": args.drift_threshold,
        "refinement": refinement,
        "refinement_parameters": {
            "clusters": args.clusters,
            "iterations": args.iterations,
            "strength": args.strength,
            "shrinkage": args.shrinkage,
        },
    }
    write_common_output(
        args.output_dir,
        args.task_id,
        "MoTRUST-candidate",
        embedding,
        manifest,
        data=data,
    )
    print(json.dumps({"task_id": args.task_id, **manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
