"""Label-free regularizers for mosaic single-cell integration."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _evenly_spaced_indices(length: int, limit: int, device: torch.device) -> torch.Tensor:
    if limit <= 0 or length <= limit:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, steps=limit, device=device).round().long().unique()


def soft_neighborhood_preservation_loss(
    latent: torch.Tensor,
    features: torch.Tensor,
    *,
    max_cells: int = 96,
    max_features: int = 256,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Match input-space and latent-space soft neighborhoods within a modality.

    The input neighborhood distribution is detached and acts as a label-free
    geometric target. Deterministic subsampling bounds the quadratic cost.
    """

    if latent.ndim != 2 or features.ndim != 2 or len(latent) != len(features):
        raise ValueError("latent and features must be aligned two-dimensional tensors")
    if len(latent) < 3:
        return latent.sum() * 0.0

    cell_index = _evenly_spaced_indices(len(latent), max_cells, latent.device)
    feature_index = _evenly_spaced_indices(features.shape[1], max_features, features.device)
    c = F.normalize(latent[cell_index].float(), dim=1, eps=1e-8)
    x = F.normalize(features[cell_index][:, feature_index].float(), dim=1, eps=1e-8)
    tau = max(float(temperature), 1e-4)

    input_logits = (x @ x.T) / tau
    latent_logits = (c @ c.T) / tau
    diagonal = torch.eye(len(cell_index), dtype=torch.bool, device=latent.device)
    input_logits = input_logits.masked_fill(diagonal, -1e4)
    latent_logits = latent_logits.masked_fill(diagonal, -1e4)
    target = torch.softmax(input_logits.detach(), dim=1)
    return F.kl_div(torch.log_softmax(latent_logits, dim=1), target, reduction="batchmean")


def transport_correspondence_confidence(
    transport: torch.Tensor,
    *,
    mode: str = "composite",
    mass_weight: float = 1.0,
    entropy_weight: float = 1.0,
    margin_weight: float = 1.0,
    reciprocity_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Estimate per-source correspondence reliability from a transport plan.

    The composite score is a weighted geometric mean of relative transported
    mass, inverse normalized row entropy, the top-1/top-2 probability margin,
    and reverse-direction agreement. All outputs are detached because they are
    evidence used to choose fusion strength, not a shortcut through the solver.
    """

    if transport.ndim != 2:
        raise ValueError("transport must be a two-dimensional matrix")
    if mode not in {"legacy_mass", "legacy_mass_entropy", "mass", "mass_entropy", "composite"}:
        raise ValueError(f"Unknown correspondence confidence mode: {mode}")
    plan = transport.detach().float().clamp_min(0.0)
    eps = torch.finfo(plan.dtype).eps
    row_mass = plan.sum(dim=1)
    valid = row_mass > eps
    probabilities = plan / row_mass.clamp_min(eps).unsqueeze(1)

    positive_mass = row_mass[valid]
    mass_scale = positive_mass.mean() if bool(valid.any()) else row_mass.new_tensor(1.0)
    relative_mass = (row_mass / mass_scale.clamp_min(eps)).clamp(0.0, 1.0)
    legacy_mass = (row_mass * float(plan.shape[0])).clamp(0.0, 1.0)

    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=1)
    entropy_max = math.log(max(2, transport.shape[1]))
    inverse_entropy = (1.0 - entropy / entropy_max).clamp(0.0, 1.0)

    top_k = min(2, transport.shape[1])
    top = probabilities.topk(top_k, dim=1)
    top1 = top.values[:, 0]
    top2 = top.values[:, 1] if top_k == 2 else torch.zeros_like(top1)
    margin = ((top1 - top2) / top1.clamp_min(eps)).clamp(0.0, 1.0)

    best_target = top.indices[:, 0]
    column_mass = plan.sum(dim=0)
    reverse = plan / column_mass.clamp_min(eps).unsqueeze(0)
    reverse_best = reverse.max(dim=0).values
    row_index = torch.arange(plan.shape[0], device=plan.device)
    reciprocity = (
        reverse[row_index, best_target] / reverse_best[best_target].clamp_min(eps)
    ).clamp(0.0, 1.0)

    components = {
        "mass": relative_mass,
        "inverse_entropy": inverse_entropy,
        "margin": margin,
        "reciprocity": reciprocity,
    }
    if mode == "legacy_mass":
        confidence = legacy_mass
    elif mode == "legacy_mass_entropy":
        confidence = legacy_mass * inverse_entropy
    elif mode == "mass":
        confidence = relative_mass
    elif mode == "mass_entropy":
        confidence = relative_mass * inverse_entropy
    else:
        weights = {
            "mass": max(0.0, float(mass_weight)),
            "inverse_entropy": max(0.0, float(entropy_weight)),
            "margin": max(0.0, float(margin_weight)),
            "reciprocity": max(0.0, float(reciprocity_weight)),
        }
        weight_sum = sum(weights.values())
        if weight_sum <= 0:
            raise ValueError("At least one correspondence confidence weight must be positive")
        log_confidence = sum(
            weight * components[name].clamp_min(eps).log()
            for name, weight in weights.items()
        ) / weight_sum
        confidence = log_confidence.exp()
    confidence = torch.where(valid, confidence, torch.zeros_like(confidence)).clamp(0.0, 1.0)
    return confidence.detach(), {name: value.detach() for name, value in components.items()}


def confidence_weighted_transport_pull(
    source: torch.Tensor,
    target: torch.Tensor,
    transport: torch.Tensor,
    *,
    confidence_min: float = 0.15,
    entropy_power: float = 1.0,
    confidence_mode: str = "legacy_mass_entropy",
    confidence_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull source cells toward high-confidence transport barycenters.

    Confidence combines transported row mass with normalized inverse entropy.
    The transport plan and barycentric targets are detached so the term trains
    the encoders rather than exploiting gradients through the OT solver.
    """

    if transport.shape != (len(source), len(target)):
        raise ValueError("transport shape must match source and target cell counts")
    eps = torch.finfo(source.dtype).eps
    plan = transport.detach().clamp_min(0.0)
    row_mass = plan.sum(dim=1)
    valid = row_mass > eps
    if not bool(valid.any()):
        zero = source.sum() * 0.0
        return zero, zero.detach(), zero.detach()

    probabilities = plan / row_mass.clamp_min(eps).unsqueeze(1)
    confidence, _ = transport_correspondence_confidence(
        plan,
        mode=confidence_mode,
        mass_weight=confidence_weights[0],
        entropy_weight=confidence_weights[1],
        margin_weight=confidence_weights[2],
        reciprocity_weight=confidence_weights[3],
    )
    if entropy_power != 1.0:
        confidence = confidence.pow(max(float(entropy_power), 1e-4))
    accepted = valid & (confidence >= float(confidence_min))
    if not bool(accepted.any()):
        zero = source.sum() * 0.0
        return zero, confidence.mean(), accepted.float().mean()

    barycenter = probabilities @ target.detach()
    distance = (source - barycenter).square().mean(dim=1)
    weights = confidence * accepted.float()
    loss = (weights * distance).sum() / weights.sum().clamp_min(eps)
    return loss, confidence.mean(), accepted.float().mean()


def symmetric_confidence_transport_loss(
    rna_c: torch.Tensor,
    atac_c: torch.Tensor,
    transport: torch.Tensor,
    *,
    confidence_min: float = 0.15,
    entropy_power: float = 1.0,
    confidence_mode: str = "legacy_mass_entropy",
    confidence_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rna_loss, rna_confidence, rna_accepted = confidence_weighted_transport_pull(
        rna_c,
        atac_c,
        transport,
        confidence_min=confidence_min,
        entropy_power=entropy_power,
        confidence_mode=confidence_mode,
        confidence_weights=confidence_weights,
    )
    atac_loss, atac_confidence, atac_accepted = confidence_weighted_transport_pull(
        atac_c,
        rna_c,
        transport.T,
        confidence_min=confidence_min,
        entropy_power=entropy_power,
        confidence_mode=confidence_mode,
        confidence_weights=confidence_weights,
    )
    return (
        0.5 * (rna_loss + atac_loss),
        0.5 * (rna_confidence + atac_confidence),
        0.5 * (rna_accepted + atac_accepted),
    )
