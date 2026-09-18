"""Apply adaptive semantic protection using fixed, label-free task settings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.preprocessing import StandardScaler

from motrust.data.integration import TaskData
from motrust.data.output import write_common_output, validate_common_output
from motrust.integration import reliability_mnn_align, conditional_location_scale_align
from .protection import guarded


def load_settings(path):
    """Read the prepared gate; final integration never refits this diagnostic."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    settings = manifest["adaptive_mixing"]
    intervention = float(settings["semantic_intervention_fraction"])
    strength = float(settings["effective_strength"])
    if not np.isfinite([intervention, strength]).all() or not 0 <= intervention <= 1 or strength < 0:
        raise ValueError("Invalid composition intervention or strength")
    return manifest, intervention, strength


def validate_representation_metadata(folder, expected):
    metadata = pd.read_csv(Path(folder) / "metadata.csv", dtype=str)
    if not metadata.cell_id.equals(expected.cell_id):
        raise ValueError(f"Representation cell order differs in {folder}")
    if {"cell_type", "source_cell_id", "broad_class", "fine_cluster"}.intersection(metadata.columns):
        raise ValueError("Evaluation-only columns in representation metadata")
    for column in ("modality", "instance_batch"):
        if not metadata[column].equals(expected[column]):
            raise ValueError(f"Representation {column} differs from task metadata")
    return metadata


def alignment_tail(embedding, modality, batches, reliability, strength, *, reference=False):
    """Fixed three-pass modality and two-pass batch calibration protocol."""
    z = embedding
    amount = min(strength, 1.0) if reference else 1
    for _ in range(3):
        z = reliability_mnn_align(z, modality, reliability, neighbors=20,
                                  smoothing_neighbors=40, bandwidth=1, strength=amount)
    z = conditional_location_scale_align(z, modality, clusters=40, iterations=1,
                                         strength=strength, shrinkage=20,
                                         scale_strength=1, random_state=2024)
    z = reliability_mnn_align(z, batches, neighbors=20, smoothing_neighbors=40,
                              bandwidth=1, strength=1)
    z = conditional_location_scale_align(z, batches, clusters=40, iterations=2,
                                         strength=1, shrinkage=20,
                                         scale_strength=1, random_state=2024)
    if not np.isfinite(z).all():
        raise FloatingPointError("Nonfinite aligned embedding")
    return z


def protected_fusion(semantic, neural, groups, intervention):
    """Protect in the semantic frame and rotate the mixture back to neural coordinates."""
    if np.shape(semantic) != np.shape(neural) or np.ndim(semantic) != 2:
        raise ValueError("Paired representation shapes differ")
    if len(groups) != len(semantic) or not np.isfinite(semantic).all() or not np.isfinite(neural).all():
        raise ValueError("Invalid representation or observation groups")
    if not np.isfinite(intervention) or not 0 <= intervention <= 1:
        raise ValueError("Intervention must be within [0, 1]")
    s = StandardScaler().fit_transform(semantic)
    n = StandardScaler().fit_transform(neural)
    rotation, _ = orthogonal_procrustes(n, s)
    n = n @ rotation
    protected, diagnostics = guarded(s, n, np.asarray(groups))
    z = (((1 - intervention) * n + intervention * protected) @ rotation.T).astype(np.float32)
    return z, diagnostics


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_integration(task_id, domain="benchmark", evaluate=True, force=False, *, root=None):
    """Integrate prepared representations; no model training is performed here.

    At zero intervention the saved reference array is reused exactly. At nonzero
    intervention the gate and calibration strength come from the task manifest.
    """
    data = TaskData(domain, root=root)
    metadata = data.load_task_metadata(task_id)
    base = data.integration_root
    semantic = base / "semantic" / task_id
    candidate = base / "candidate" / task_id
    setting_path = base / "composition" / task_id / "composition_manifest.json"
    manifest, intervention, strength = load_settings(setting_path)
    validate_representation_metadata(semantic, metadata)
    validate_representation_metadata(candidate, metadata)
    if intervention == 0:
        reference = base / "reference" / task_id
        validate_representation_metadata(reference, metadata)
        sources = {"reference": _sha256(reference / "embedding.npy")}
    else:
        sources = {"semantic": _sha256(semantic / "embedding.npy"),
                   "candidate": _sha256(candidate / "embedding.npy"),
                   "reliability": _sha256(semantic / "semantic_reliability.npy")}
    output = data.root / "results/integration" / domain / task_id
    if (output / "embedding.npy").exists() and not force:
        existing = validate_common_output(output, task_id, "MoTRUST", data=data)
        if existing.get("composition_sha256") != _sha256(setting_path):
            raise ValueError("Composition settings changed; choose --force to replace this result")
        if existing.get("input_sha256") != sources:
            raise ValueError("Input representations changed; choose --force to replace this result")
    else:
        if intervention == 0:
            z = np.load(reference / "embedding.npy", allow_pickle=False)
            if z.dtype != np.float32:
                raise ValueError("Reference must use the common float32 output format for exact bypass")
            if z.ndim != 2 or z.shape[0] != len(metadata) or z.shape[1] < 2:
                raise ValueError("Reference shape differs from the task output contract")
            diagnostics = {"exact_reference_bypass": True}
        else:
            s = np.load(semantic / "embedding.npy", allow_pickle=False)
            n = np.load(candidate / "embedding.npy", allow_pickle=False)
            rho = np.load(semantic / "semantic_reliability.npy", allow_pickle=False)
            if np.asarray(rho).shape != (len(metadata),) or not np.isfinite(rho).all():
                raise ValueError("Semantic reliability must be a finite vector in task cell order")
            z, diagnostics = protected_fusion(s, n, metadata.modality.to_numpy(), intervention)
            z = alignment_tail(z, metadata.modality.to_numpy(), metadata.instance_batch.to_numpy(), rho, strength)
        if not np.isfinite(z).all():
            raise FloatingPointError("Nonfinite integrated embedding")
        write_common_output(output, task_id, "MoTRUST", z,
                            dict(seed=2024, domain=domain, intervention=intervention,
                                 strength=strength, diagnostics=diagnostics,
                                 composition_sha256=_sha256(setting_path), input_sha256=sources), data=data)
        if intervention == 0 and not np.array_equal(z, np.load(output / "embedding.npy", allow_pickle=False)):
            raise ValueError("Reference must use the common float32 output format for exact bypass")
    if evaluate and (force or not (output / "metrics.json").exists()):
        from motrust.benchmark.evaluate import evaluate_output
        evaluate_output(output, task_id, "MoTRUST", domain=domain, seed=2024, data=data)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--domain", choices=("benchmark", "cortex"), default="benchmark")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    print(run_integration(args.task_id, args.domain, evaluate=not args.no_evaluate, force=args.force))


if __name__ == "__main__":
    main()
