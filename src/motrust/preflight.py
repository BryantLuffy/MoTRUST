"""Input checks that do not fit models or generate experimental outputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

from .paths import resource_path, workdir


def _require(path: Path, issues: list[str]) -> bool:
    if path.is_file():
        return True
    try:
        label = path.relative_to(workdir()).as_posix()
    except ValueError:
        label = str(path)
    issues.append("Missing: " + label)
    return False


def _json(path: Path, issues: list[str]) -> dict | None:
    if not _require(path, issues):
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except (OSError, ValueError) as error:
        issues.append(f"Invalid JSON in {path.name}: {error}")
        return None


def integration_check(task_id: str, domain: str = "benchmark", *, evaluate: bool = True) -> list[str]:
    issues: list[str] = []
    root = workdir()
    if domain not in {"benchmark", "cortex"}:
        return ["Unknown integration domain."]
    if not task_id or Path(task_id).name != task_id or task_id in {".", ".."} or "/" in task_id or "\\" in task_id:
        return ["Task ID must be a single valid identifier."]
    data = root / "data" / ("cortex" if domain == "cortex" else "")
    run = root / "runs/integration" / ("cortex" if domain == "cortex" else "")
    definitions = data / "tasks/tasks.json"
    if not definitions.is_file():
        definitions = resource_path("integration", "cortex_tasks.json" if domain == "cortex" else "tasks.json")
    tasks = _json(definitions, issues)
    if tasks is not None and task_id not in {entry.get("task_id") for entry in tasks.get("tasks", [])}:
        return [f"Unknown task: {task_id}. Provide its definition in data/tasks/tasks.json for custom tasks."]
    _require(data / "tasks" / task_id / "metadata_training.csv", issues)
    if evaluate:
        _require(data / "tasks" / task_id / "metadata_evaluation.csv", issues)
    for representation in ("semantic", "candidate"):
        _require(run / representation / task_id / "metadata.csv", issues)
    manifest = _json(run / "composition" / task_id / "composition_manifest.json", issues)
    if manifest is not None:
        try:
            settings = manifest["adaptive_mixing"]
            intervention = float(settings["semantic_intervention_fraction"])
            strength = float(settings["effective_strength"])
            if not 0 <= intervention <= 1 or not 0 <= strength:
                raise ValueError("intervention or strength is out of range")
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"Invalid composition settings: {error}")
        else:
            if intervention == 0:
                _require(run / "reference" / task_id / "embedding.npy", issues)
                _require(run / "reference" / task_id / "metadata.csv", issues)
            else:
                for rel in ("semantic/{}/embedding.npy", "semantic/{}/semantic_reliability.npy", "candidate/{}/embedding.npy"):
                    _require(run / rel.format(task_id), issues)
    return issues


def _rscript_check(issues: list[str]) -> None:
    configured = os.environ.get("RSCRIPT", "Rscript")
    executable = shutil.which(configured)
    if executable is None and Path(configured).is_file():
        executable = configured
    if executable is None:
        issues.append("Rscript is unavailable; set RSCRIPT or add Rscript to PATH (jsonlite required).")
        return
    try:
        completed = subprocess.run([executable, "--vanilla", "-e", 'quit(status=if(requireNamespace("jsonlite",quietly=TRUE)) 0L else 2L)'],
                                   capture_output=True, timeout=30, check=False)
        if completed.returncode:
            issues.append("The selected R installation needs jsonlite.")
    except (OSError, subprocess.TimeoutExpired):
        issues.append("Rscript could not complete its dependency check.")


def rna_check(stage: str) -> list[str]:
    if stage not in {"prepare", "points", "evaluate"}:
        return ["Unknown RNA workflow step."]
    issues: list[str] = []
    root = workdir()
    output = root / "results/rna_diffusion"
    index = _json(root / "data/cache/cache_index.json", issues)
    if stage == "prepare":
        if index is not None:
            path_base = index.get("path_base", "workdir")
            if path_base not in {"workdir", "index"}:
                issues.append("Cache path_base must be 'workdir' or 'index'.")
            reference_root = root / "data/cache" if path_base == "index" else root
            for scenario in ("TEA_s1", "BMMC_s1", "Retina"):
                records = index.get("scenarios", {}).get(scenario)
                if not isinstance(records, dict):
                    issues.append(f"Cache index lacks scenario {scenario}.")
                    continue
                paired = [record for record in records.values() if {"rna", "atac"} <= set(record.get("modalities", {}))]
                if not paired:
                    issues.append(f"No paired RNA/ATAC library in {scenario}.")
                for record in paired:
                    for modality in ("rna", "atac"):
                        for key in ("matrix", "features", "barcodes"):
                            reference = record["modalities"][modality].get(key)
                            if isinstance(reference, str) and reference:
                                _require(reference_root / reference, issues)
                            else:
                                issues.append(f"Missing {scenario}/{modality}/{key} in cache index.")
                    metadata = record.get("metadata")
                    if isinstance(metadata, str) and metadata:
                        _require(reference_root / metadata, issues)
                    else:
                        issues.append(f"Missing library metadata in {scenario}.")
        if not (output / "metadata_export_complete.json").is_file():
            _rscript_check(issues)
    else:
        for dataset in ("TEA", "BMMC-Multiome", "Retina"):
            for name in ("manifest.json", "metadata.csv", "split.npz", "rna_counts.npz", "atac_counts.npz"):
                _require(output / "inputs" / dataset / name, issues)
            if stage == "evaluate":
                _require(output / "anchors" / dataset / "prepared.npz", issues)
                for seed in (42, 0, 1):
                    _require(output / "point/runs" / dataset / f"seed_{seed}/balanced_atac/model.pt", issues)
        import torch
        if not torch.cuda.is_available():
            issues.append("This training/sampling workflow requires a CUDA-enabled PyTorch installation and an available GPU.")
    return list(dict.fromkeys(issues))
