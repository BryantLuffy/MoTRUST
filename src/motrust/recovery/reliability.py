"""Reliability, transport pseudo-bridges, and conformal calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from motrust.models.mosaic_regularizers import transport_correspondence_confidence
from motrust.models.unbalanced_ot import UncertaintyAwareUnbalancedOT


def robust_unit_scale(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float().reshape(-1)
    low = torch.quantile(values, 0.05)
    high = torch.quantile(values, 0.95)
    if bool((high - low).abs() < 1e-8):
        return torch.ones_like(values)
    return ((values - low) / (high - low).clamp_min(1e-8)).clamp(0.0, 1.0)


@torch.inference_mode()
def modality_quality(
    vae,
    means: torch.Tensor,
    logvars: torch.Tensor,
    observed: np.ndarray,
    modality: str,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    result = torch.zeros(len(means))
    indices = np.flatnonzero(observed)
    gates = []
    precisions = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        mu = means[selected].to(device)
        logvar = logvars[selected].to(device)
        gate = vae._gate_weight(modality, mu, logvar).reshape(-1).clamp(0.0, 2.0) / 2.0
        precision = torch.exp(-logvar.clamp(-12.0, 12.0)).mean(dim=1)
        gates.append(gate.cpu())
        precisions.append(precision.cpu())
    gate_score = torch.cat(gates)
    precision_score = robust_unit_scale(torch.cat(precisions))
    result[indices] = (gate_score.clamp_min(1e-6) * precision_score.clamp_min(1e-6)).sqrt()
    return result.clamp(0.0, 1.0)


@torch.inference_mode()
def bridge_proximity(
    means: torch.Tensor,
    query_indices: np.ndarray,
    bridge_indices: np.ndarray,
    device: torch.device,
    max_anchors: int = 1024,
    chunk_size: int = 2048,
) -> torch.Tensor:
    anchors_index = np.asarray(bridge_indices)
    if len(anchors_index) > max_anchors:
        positions = np.linspace(0, len(anchors_index) - 1, max_anchors).round().astype(int)
        anchors_index = anchors_index[positions]
    anchors = means[anchors_index].to(device)
    distances = []
    for start in range(0, len(query_indices), chunk_size):
        query = means[query_indices[start : start + chunk_size]].to(device)
        distance = torch.cdist(query, anchors).min(dim=1).values / means.shape[1] ** 0.5
        distances.append(distance.cpu())
    distance = torch.cat(distances)
    scale = torch.quantile(distance, 0.5).clamp_min(1e-4)
    return torch.exp(-distance / scale).clamp(0.0, 1.0)


@dataclass
class TransportTargets:
    source_indices: np.ndarray
    target_indices: np.ndarray
    barycenter: torch.Tensor
    confidence: torch.Tensor
    components: dict[str, torch.Tensor]


@torch.inference_mode()
def transport_targets(
    source_mu: torch.Tensor,
    source_logvar: torch.Tensor,
    target_mu: torch.Tensor,
    target_logvar: torch.Tensor,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    device: torch.device,
    max_source: int | None = None,
    max_target: int = 2048,
    seed: int = 0,
) -> TransportTargets:
    rng = np.random.default_rng(seed)
    source_indices = np.asarray(source_indices)
    target_indices = np.asarray(target_indices)
    if max_source is not None and len(source_indices) > max_source:
        source_indices = np.sort(rng.choice(source_indices, max_source, replace=False))
    if len(target_indices) > max_target:
        target_indices = np.sort(rng.choice(target_indices, max_target, replace=False))
    solver = UncertaintyAwareUnbalancedOT(
        epsilon=0.1, mass_regularization=1.0,
        uncertainty_weight=0.1, variance_weight=0.1, num_iters=30,
    ).to(device)
    output = solver(
        source_mu[source_indices].to(device), source_logvar[source_indices].to(device),
        target_mu[target_indices].to(device), target_logvar[target_indices].to(device),
    )
    plan = output["transport"]
    confidence, components = transport_correspondence_confidence(plan, mode="composite")
    probabilities = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-8)
    barycenter = probabilities @ target_mu[target_indices].to(device)
    hard_target = target_indices[probabilities.argmax(dim=1).cpu().numpy()]
    return TransportTargets(
        source_indices=source_indices,
        target_indices=hard_target,
        barycenter=barycenter.cpu(),
        confidence=confidence.cpu(),
        components={key: value.cpu() for key, value in components.items()},
    )


def generation_uncertainty(
    distribution: dict[str, torch.Tensor],
    residual_scale: torch.Tensor,
) -> torch.Tensor:
    scale = residual_scale.to(distribution["samples"].device).clamp_min(1e-4)
    stochastic = (distribution["sample_std"] / scale).mean(dim=1)
    displacement = (
        (distribution["sample_mean"] - distribution["conditional_mean"]).abs() / scale
    ).mean(dim=1)
    return 0.5 * stochastic + 0.5 * displacement


@dataclass
class Calibration:
    uncertainty_x: np.ndarray
    predicted_error_y: np.ndarray
    error_scale: float
    conformal_quantile: float
    sigma_floor: np.ndarray
    interval_scale: np.ndarray
    rejection_threshold: float
    target_coverage: float
    reliability_low: float = 0.0
    reliability_high: float = 1.0
    selected_max_timestep: int = 0
    selected_max_blend: float = 0.0
    selected_transport_blend: float = 0.0
    reliability_weights: np.ndarray | None = None

    def generation_confidence(self, uncertainty: torch.Tensor) -> torch.Tensor:
        predicted = np.interp(
            uncertainty.detach().cpu().numpy(), self.uncertainty_x,
            self.predicted_error_y,
        )
        confidence = np.exp(-predicted / max(self.error_scale, 1e-8))
        return torch.from_numpy(confidence.astype(np.float32)).to(uncertainty.device)

    def policy_reliability(self, reliability: torch.Tensor) -> torch.Tensor:
        denominator = max(self.reliability_high - self.reliability_low, 1e-8)
        return ((reliability - self.reliability_low) / denominator).clamp(0.0, 1.0)

    def combined_reliability(self, components: torch.Tensor) -> torch.Tensor:
        clipped = components.float().clamp(1e-6, 1.0)
        if self.reliability_weights is None:
            weights = torch.full(
                (clipped.shape[1],), 1.0 / clipped.shape[1],
                device=clipped.device,
            )
        else:
            weights = torch.from_numpy(self.reliability_weights).to(clipped.device)
            weights = weights / weights.sum().clamp_min(1e-8)
        return torch.exp((clipped.log() * weights[None, :]).sum(dim=1))


def fit_calibration(
    uncertainty: torch.Tensor,
    sample_mean: torch.Tensor,
    sample_std: torch.Tensor,
    target: torch.Tensor,
    base_reliability: torch.Tensor,
    target_coverage: float = 0.9,
    reliability_components: torch.Tensor | None = None,
) -> Calibration:
    from sklearn.isotonic import IsotonicRegression

    uncertainty_np = uncertainty.detach().cpu().numpy()
    error = ((sample_mean - target).square().mean(dim=1).sqrt()).detach().cpu().numpy()
    isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip")
    isotonic.fit(uncertainty_np, error)
    error_scale = float(np.quantile(error, 0.5))
    predicted_error = isotonic.predict(uncertainty_np)
    generation_confidence = np.exp(-predicted_error / max(error_scale, 1e-8))
    reliability_weights = np.full(4, 0.25, dtype=np.float32)
    if reliability_components is not None:
        from scipy.optimize import nnls

        components_np = reliability_components.detach().cpu().numpy().copy()
        components_np[:, 3] = generation_confidence
        risks = -np.log(np.clip(components_np, 1e-6, 1.0))
        learned, _ = nnls(risks, error)
        if learned.sum() > 1e-8:
            reliability_weights = (learned / learned.sum()).astype(np.float32)
        rho = np.exp(
            np.log(np.clip(components_np, 1e-6, 1.0)) @ reliability_weights
        )
    else:
        rho = (
            base_reliability.detach().cpu().numpy().clip(1e-8, 1.0) ** 3
            * generation_confidence.clip(1e-8, 1.0)
        ) ** 0.25
    rejection_threshold = float(np.quantile(rho, 1.0 - target_coverage))
    reliability_low = float(np.quantile(rho, 0.1))
    reliability_high = float(np.quantile(rho, 0.9))

    absolute_error = (sample_mean - target).abs().detach().cpu().numpy()
    sigma = sample_std.detach().cpu().numpy()
    sigma_floor = np.quantile(absolute_error, 0.1, axis=0).astype(np.float32)
    interval_scale = (
        np.median(sigma, axis=0) + sigma_floor
    ).astype(np.float32)
    effective_sigma = interval_scale[None, :] / np.sqrt(
        np.clip(rho[:, None], 1e-4, 1.0)
    )
    nonconformity = absolute_error / np.maximum(effective_sigma, 1e-6)
    conformal_quantile = float(np.quantile(nonconformity, target_coverage))
    return Calibration(
        uncertainty_x=np.asarray(isotonic.X_thresholds_, dtype=np.float32),
        predicted_error_y=np.asarray(isotonic.y_thresholds_, dtype=np.float32),
        error_scale=error_scale,
        conformal_quantile=conformal_quantile,
        sigma_floor=sigma_floor,
        interval_scale=interval_scale,
        rejection_threshold=rejection_threshold,
        target_coverage=float(target_coverage),
        reliability_low=reliability_low,
        reliability_high=reliability_high,
        reliability_weights=reliability_weights,
    )


def prediction_interval(
    sample_mean: torch.Tensor,
    sample_std: torch.Tensor,
    calibration: Calibration,
    reliability: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.from_numpy(calibration.interval_scale).to(sample_mean.device)
    if reliability is None:
        reliability = torch.ones(len(sample_mean), device=sample_mean.device)
    inflation = reliability.reshape(-1, 1).clamp(1e-4, 1.0).rsqrt()
    radius = calibration.conformal_quantile * scale * inflation
    return sample_mean - radius, sample_mean + radius
