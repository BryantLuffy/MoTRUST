"""Prepare the fixed modality-separability gate and coordinate-consistent reference."""
from __future__ import annotations
import argparse
import json
import numpy as np
from motrust.data.integration import TaskData
from motrust.data.output import write_common_output
from motrust.integration import local_group_separability
from .protection import frame_consistent_fusion
from .integrate import alignment_tail, load_settings, validate_representation_metadata


def compose_task(task_id, domain="benchmark", *, neural_fraction=0.9, force=False, root=None):
    """Create absent settings; existing task settings are authoritative unless forced."""
    data = TaskData(domain, root=root)
    metadata = data.load_task_metadata(task_id)
    base = data.integration_root
    sr, nr = base / "semantic" / task_id, base / "candidate" / task_id
    validate_representation_metadata(sr, metadata)
    validate_representation_metadata(nr, metadata)
    semantic = np.load(sr / "embedding.npy", allow_pickle=False)
    neural = np.load(nr / "embedding.npy", allow_pickle=False)
    rho = np.load(sr / "semantic_reliability.npy", allow_pickle=False)
    path = base / "composition" / task_id / "composition_manifest.json"
    if path.is_file() and not force:
        manifest, intervention, strength = load_settings(path)
        effective_fraction = float(manifest["effective_neural_fraction"])
    else:
        if not np.isfinite(neural_fraction) or not 0 <= neural_fraction <= 1:
            raise ValueError("Neural fraction must be within [0, 1]")
        separability = local_group_separability(neural, metadata.modality.to_numpy())
        denominator = 0.4 if domain == "cortex" else max(0.55 - 0.15, 1e-8)
        position = float(np.clip((separability - 0.15) / denominator, 0.0, 1.0))
        intervention = position * position * (3.0 - 2.0 * position)
        strength = 1.0 + intervention
        effective_fraction = float(neural_fraction)
        manifest = dict(schema_version="1.0", task_id=task_id, seed=2024,
                        effective_neural_fraction=effective_fraction,
                        adaptive_mixing=dict(enabled=True, observed_group_separability=float(separability),
                                             low=0.15, high=0.55, minimum_strength=1.0, maximum_strength=2.0,
                                             semantic_intervention_fraction=intervention,
                                             effective_strength=strength, diagnostic_source="candidate_pre_correction"))
    output = base / "reference" / task_id
    if force or not (output / "embedding.npy").is_file():
        z = frame_consistent_fusion(semantic, neural, rho, intervention, effective_fraction)
        z = alignment_tail(z, metadata.modality.to_numpy(), metadata.instance_batch.to_numpy(),
                           rho, strength, reference=True)
        write_common_output(output, task_id, "MoTRUST-reference", z,
                            dict(seed=2024, intervention=intervention, strength=strength,
                                 effective_neural_fraction=effective_fraction), data=data)
    if force or not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--domain", choices=("benchmark", "cortex"), default="benchmark")
    parser.add_argument("--neural-fraction", type=float, default=0.9)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    print(compose_task(**vars(args)))


if __name__ == "__main__":
    main()
