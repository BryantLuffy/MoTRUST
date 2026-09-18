"""Exercise semantic protection and sample-mean projection on small CPU arrays.

Install MoTRUST before running: python -m pip install -e .
This example demonstrates numerical interfaces, not a biological experiment.
"""
import json
import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(variable, "1")

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.preprocessing import StandardScaler
import torch

from motrust.workflows.integration.protection import guarded
from motrust.workflows.rna_diffusion.model import project_mean


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    groups = np.repeat(np.asarray(["rna", "atac"]), 40)
    semantic = StandardScaler().fit_transform(rng.normal(size=(80, 8)))
    candidate = StandardScaler().fit_transform(semantic + rng.normal(scale=0.3, size=(80, 8)))
    rotation, _ = orthogonal_procrustes(candidate, semantic)
    protected, diagnostics = guarded(semantic, candidate @ rotation, groups)
    assert protected.shape == semantic.shape and np.isfinite(protected).all()

    generator = torch.Generator().manual_seed(42)
    anchor = torch.rand((6, 10), generator=generator) * 3.0
    raw_samples = anchor.unsqueeze(0) + torch.randn((32, 6, 10), generator=generator)
    samples = project_mean(raw_samples, anchor)
    mean_error = float((samples.mean(0) - anchor).abs().max())
    assert mean_error <= 2e-5
    print(json.dumps({
        "example": "synthetic CPU inputs",
        "protected_embedding_shape": list(protected.shape),
        "semantic_protection": diagnostics,
        "rna_sample_shape": list(samples.shape),
        "maximum_sample_mean_error": mean_error,
    }, indent=2))


if __name__ == "__main__":
    main()
