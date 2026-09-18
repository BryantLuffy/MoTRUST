"""Focused tests for MoTRUST reliability-aware recovery."""

from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from motrust.recovery.diffusion import (
    ReliabilityAdaptiveDiffusion,
    geometric_reliability,
)
from motrust.recovery.reliability import (
    fit_calibration,
    prediction_interval,
    robust_unit_scale,
)
from motrust.training.recovery import target_reconstruction_loss


class DummyDecoder:
    rna_distribution = "poisson"

    def decode(self, latent: torch.Tensor):
        rna = torch.nn.functional.softplus(latent @ torch.ones(latent.shape[1], 12))
        atac = latent @ torch.linspace(-1.0, 1.0, 12).repeat(latent.shape[1], 1)
        return rna, atac, None


def recovery_args() -> SimpleNamespace:
    return SimpleNamespace(
        rna_smse_weight=0.25,
        rna_moment_weight=0.1,
        rna_correlation_weight=0.25,
        rna_correlation_features=8,
        atac_focal_gamma=2.0,
        atac_positive_margin=0.2,
        atac_rank_weight=0.1,
        atac_hard_negative_weight=0.25,
        atac_focal_weight=0.1,
        atac_density_weight=0.5,
        atac_frequency_weight=0.1,
    )


def make_model(latent_dim: int = 4) -> ReliabilityAdaptiveDiffusion:
    condition_dim = latent_dim * 2
    return ReliabilityAdaptiveDiffusion(
        latent_dim=latent_dim,
        condition_mean=torch.zeros(condition_dim),
        condition_std=torch.ones(condition_dim),
        residual_mean=torch.zeros(latent_dim),
        residual_std=torch.ones(latent_dim),
        timesteps=20,
        hidden_dim=16,
        time_dim=8,
        dropout=0.0,
    )


def test_reliability_is_monotone_and_handles_constant_scores() -> None:
    components = torch.tensor([
        [0.2, 0.2, 0.2, 0.2],
        [0.5, 0.5, 0.5, 0.5],
        [0.9, 0.9, 0.9, 0.9],
    ])
    rho = geometric_reliability(components)
    assert torch.all(rho[1:] > rho[:-1])
    assert torch.allclose(robust_unit_scale(torch.ones(7)), torch.ones(7))


def test_reliability_controls_cellwise_diffusion() -> None:
    model = make_model().eval()
    source_mu = torch.zeros(3, 4)
    source_logvar = torch.zeros(3, 4)
    components = torch.tensor([
        [0.1, 0.1, 0.1, 0.1],
        [0.5, 0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0, 1.0],
    ])
    output = model.predict_distribution(
        source_mu, source_logvar, components,
        max_timestep=9, max_blend=0.5, ensemble_size=3, seed=11,
    )
    assert output["samples"].shape == (3, 3, 4)
    assert torch.all(output["timestep"][1:] >= output["timestep"][:-1])
    assert torch.all(output["blend"][1:] > output["blend"][:-1])


def test_conformal_interval_reaches_in_sample_target_coverage() -> None:
    generator = torch.Generator().manual_seed(19)
    target = torch.randn(200, 5, generator=generator)
    sample_mean = target + 0.2 * torch.randn(200, 5, generator=generator)
    sample_std = 0.05 + 0.1 * torch.rand(200, 5, generator=generator)
    uncertainty = sample_std.mean(dim=1)
    base_reliability = torch.linspace(0.2, 1.0, 200)
    calibration = fit_calibration(
        uncertainty, sample_mean, sample_std, target, base_reliability,
        target_coverage=0.9,
    )
    generation_confidence = calibration.generation_confidence(uncertainty)
    components = torch.stack((
        base_reliability, base_reliability, base_reliability,
        generation_confidence,
    ), dim=1)
    rho = calibration.combined_reliability(components)
    lower, upper = prediction_interval(
        sample_mean, sample_std, calibration, rho
    )
    coverage = ((target >= lower) & (target <= upper)).float().mean().item()
    assert coverage >= 0.895
    confidence = calibration.generation_confidence(
        torch.tensor([float(uncertainty.min()), float(uncertainty.max())])
    )
    assert np.isfinite(confidence.numpy()).all()
    assert confidence[0] >= confidence[1]
    weighted = fit_calibration(
        uncertainty, sample_mean, sample_std, target, base_reliability,
        target_coverage=0.9,
        reliability_components=torch.rand(200, 4, generator=generator),
    )
    assert np.isclose(weighted.reliability_weights.sum(), 1.0)


def test_metric_aligned_decoder_losses_are_finite() -> None:
    generator = torch.Generator().manual_seed(31)
    latent = torch.randn(16, 4, generator=generator, requires_grad=True)
    weights = torch.ones(16)
    rna_truth = torch.poisson(torch.full((16, 12), 2.0), generator=generator)
    atac_truth = (torch.rand(16, 12, generator=generator) < 0.15).float()
    for modality, truth in (("rna", rna_truth), ("atac", atac_truth)):
        loss, values = target_reconstruction_loss(
            DummyDecoder(), latent, truth, modality, weights, recovery_args()
        )
        assert torch.isfinite(loss)
        assert all(np.isfinite(value) for value in values.values())


if __name__ == "__main__":
    test_reliability_is_monotone_and_handles_constant_scores()
    test_reliability_controls_cellwise_diffusion()
    test_conformal_interval_reaches_in_sample_target_coverage()
    print("MoTRUST focused tests passed")
