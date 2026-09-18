"""Evaluate an integration output with a fixed, method-independent metric suite."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from motrust.benchmark.metrics import MetricConfig, integration_metrics
from motrust.data.integration import TaskData
from motrust.data.output import validate_common_output


def evaluate_output(run_dir, task_id, method=None, *, domain="benchmark", seed=2024,
                    neighbors=30, silhouette_sample_size=5000, data=None):
    data = data or TaskData(domain)
    run_dir = Path(run_dir).resolve()
    manifest = validate_common_output(run_dir, task_id=task_id, method=method, data=data)
    method = method or manifest["method"]
    task = data.load_task_definition(task_id)
    embedding = np.load(run_dir / "embedding.npy", allow_pickle=False)
    method_metadata = pd.read_csv(run_dir / "metadata.csv", dtype=str)
    evaluation = data.load_task_metadata(task_id, evaluation=True).set_index("cell_id")
    try:
        evaluation = evaluation.loc[method_metadata["cell_id"].astype(str)].reset_index()
    except KeyError as exc:
        raise ValueError("Method output contains cells outside the frozen task") from exc
    if len(evaluation) != len(embedding):
        raise ValueError("Evaluation metadata and embedding rows differ")

    config = MetricConfig(
        n_neighbors=neighbors,
        silhouette_sample_size=silhouette_sample_size,
        seed=seed,
        standardize=True,
    )
    metrics = integration_metrics(
        embedding,
        evaluation["cell_type"].astype(str).to_numpy(),
        evaluation["instance_batch"].astype(str).to_numpy(),
        evaluation["modality"].astype(str).to_numpy(),
        config,
    )
    payload = {
        "schema_version": "1.0",
        "method": method,
        "task_id": task_id,
        "scenario": task["scenario"],
        "replicate": task["replicate"],
        "evaluation_seed": seed,
        **metrics,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=True), encoding="utf-8"
    )
    pd.DataFrame([payload]).to_csv(run_dir / "metrics.csv", index=False)
    palette_row = {
        "Embedding": method,
        "KMeans NMI": metrics["NMI"],
        "KMeans ARI": metrics["ARI"],
        "Silhouette label": metrics["cASW"],
        "cLISI": metrics["cLISI"],
        "Silhouette batch": metrics["bASW"],
        "iLISI": metrics["iLISI"],
        "KBET": metrics["batch_kBET"],
        "Graph connectivity": metrics["graph_connectivity"],
        "CkBET": metrics["modality_kBET"],
        "CSAS": metrics["modality_CSAS"],
        "CiLISI": metrics["modality_CiLISI"],
        "Bio conservation": metrics["biological_conservation_score"],
        "Batch correction": metrics["batch_correction_score"],
        "Total": metrics["overall_integration_score"],
        "Modality mixing": metrics["modality_mixing_score"],
        "scenario": task["scenario"],
        "replicate": task["replicate"],
        "task_id": task_id,
    }
    pd.DataFrame([palette_row]).to_csv(run_dir / "metrics_summary.csv", index=False)
    return payload

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--method")
    parser.add_argument("--domain", choices=("benchmark", "cortex"), default="benchmark")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--silhouette-sample-size", type=int, default=5000)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_output(**vars(args)), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
