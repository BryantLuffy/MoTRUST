"""Train MoTRUST reliability-adaptive diffusion on mosaic RNA/ATAC data."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from motrust.recovery.bridge import (
    DIRECTIONS,
    build_vae,
    decoder_reconstruction_loss,
    decode_target,
    encode_observed,
    evaluate_variant,
    matrix_rows,
    set_seed,
)
from motrust.recovery.diffusion import (
    ReliabilityAdaptiveDiffusion,
    geometric_reliability,
)
from motrust.recovery.reliability import (
    Calibration,
    bridge_proximity,
    fit_calibration,
    generation_uncertainty,
    modality_quality,
    prediction_interval,
    transport_targets,
)
from motrust.evaluation.recovery_metrics import (
    atac_ranking_metrics,
    normalize_for_evaluation,
    standardized_mse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--max-inference-timestep", type=int, default=9)
    parser.add_argument("--max-diffusion-blend", type=float, default=0.5)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--decoder-weight", type=float, default=0.15)
    parser.add_argument("--pseudo-weight", type=float, default=0.0)
    parser.add_argument("--max-pseudo-cells", type=int, default=2048)
    parser.add_argument("--transport-target-anchors", type=int, default=2048)
    parser.add_argument("--transport-confidence-min", type=float, default=0.15)
    parser.add_argument("--target-coverage", type=float, default=0.9)
    parser.add_argument("--atac-rank-weight", type=float, default=0.1)
    parser.add_argument("--atac-density-weight", type=float, default=0.5)
    parser.add_argument("--atac-frequency-weight", type=float, default=0.1)
    parser.add_argument("--atac-hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--atac-focal-weight", type=float, default=0.0)
    parser.add_argument("--atac-focal-gamma", type=float, default=2.0)
    parser.add_argument("--atac-positive-margin", type=float, default=0.2)
    parser.add_argument("--rna-smse-weight", type=float, default=0.0)
    parser.add_argument("--rna-moment-weight", type=float, default=0.0)
    parser.add_argument("--rna-correlation-weight", type=float, default=0.0)
    parser.add_argument("--rna-correlation-features", type=int, default=128)
    parser.add_argument(
        "--selection-objective", choices=("decoder", "metric_aligned"), default="decoder"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    cell_values = values.reshape(len(values), -1).mean(dim=1)
    weights = weights.reshape(-1).clamp_min(0.0)
    return (cell_values * weights).sum() / weights.sum().clamp_min(1e-8)


def target_reconstruction_loss(
    vae,
    latent: torch.Tensor,
    truth: torch.Tensor,
    modality: str,
    weights: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    rna_output, atac_logits, _ = vae.decode(latent)
    if modality == "rna":
        prediction = rna_output if vae.rna_distribution == "poisson" else rna_output[0]
        prediction = prediction.clamp_min(0.0)
        truth = truth.clamp_min(0.0)
        prediction = torch.log1p(
            prediction * (1e4 / prediction.sum(dim=1, keepdim=True).clamp_min(1e-8))
        )
        normalized_truth = torch.log1p(
            truth * (1e4 / truth.sum(dim=1, keepdim=True).clamp_min(1e-8))
        )
        normalized_loss = weighted_mean(
            F.smooth_l1_loss(prediction, normalized_truth, reduction="none"), weights
        )
        feature_variance = normalized_truth.var(dim=0, unbiased=False).clamp_min(0.05)
        standardized_error = (
            (prediction - normalized_truth).square() / feature_variance[None, :]
        ).clamp_max(25.0)
        smse_loss = weighted_mean(standardized_error, weights)
        normalized_weights = weights / weights.sum().clamp_min(1e-8)
        predicted_mean = (prediction * normalized_weights[:, None]).sum(dim=0)
        truth_mean = (normalized_truth * normalized_weights[:, None]).sum(dim=0)
        moment_loss = F.smooth_l1_loss(predicted_mean, truth_mean)
        correlation_features = min(
            max(2, int(args.rna_correlation_features)), normalized_truth.shape[1]
        )
        selected_features = torch.topk(
            normalized_truth.var(dim=0, unbiased=False), correlation_features
        ).indices
        prediction_selected = prediction[:, selected_features]
        truth_selected = normalized_truth[:, selected_features]

        def weighted_correlation(values: torch.Tensor) -> torch.Tensor:
            mean = (values * normalized_weights[:, None]).sum(dim=0)
            centered = values - mean
            variance = (
                centered.square() * normalized_weights[:, None]
            ).sum(dim=0).clamp_min(1e-4)
            standardized = centered / variance.sqrt()[None, :]
            return (standardized * normalized_weights[:, None]).T @ standardized

        correlation_loss = F.smooth_l1_loss(
            weighted_correlation(prediction_selected),
            weighted_correlation(truth_selected),
        )
        loss = (
            normalized_loss
            + args.rna_smse_weight * smse_loss
            + args.rna_moment_weight * moment_loss
            + args.rna_correlation_weight * correlation_loss
        )
        return loss, {
            "decoder": float(normalized_loss.detach()),
            "rna_smse": float(smse_loss.detach()),
            "moment": float(moment_loss.detach()),
            "correlation": float(correlation_loss.detach()),
            "rank": 0.0,
            "density": 0.0,
        }

    binary = (truth > 0).float()
    element = F.binary_cross_entropy_with_logits(atac_logits, binary, reduction="none")
    positive = (element * binary).sum(dim=1) / binary.sum(dim=1).clamp_min(1.0)
    negative_mask = 1.0 - binary
    negative = (element * negative_mask).sum(dim=1) / negative_mask.sum(dim=1).clamp_min(1.0)
    balanced_bce = weighted_mean(0.5 * (positive + negative), weights)

    probability = torch.sigmoid(atac_logits)
    probability_true = probability * binary + (1.0 - probability) * negative_mask
    focal_element = (1.0 - probability_true).pow(args.atac_focal_gamma) * element
    focal_positive = (focal_element * binary).sum(dim=1) / binary.sum(dim=1).clamp_min(1.0)
    focal_negative = (focal_element * negative_mask).sum(dim=1) / negative_mask.sum(dim=1).clamp_min(1.0)
    focal_loss = weighted_mean(0.5 * (focal_positive + focal_negative), weights)

    positive_index = torch.multinomial(binary.clamp_min(1e-8), 32, replacement=True)
    negative_index = torch.multinomial(negative_mask.clamp_min(1e-8), 32, replacement=True)
    positive_logits = atac_logits.gather(1, positive_index)
    negative_logits = atac_logits.gather(1, negative_index)
    rank_cell = F.softplus(negative_logits - positive_logits + 0.1).mean(dim=1)
    rank_loss = weighted_mean(rank_cell, weights)
    hard_count = min(32, atac_logits.shape[1])
    hard_negative_logits = atac_logits.masked_fill(binary.bool(), float("-inf")).topk(
        hard_count, dim=1
    ).values
    hard_rank_cell = F.softplus(
        hard_negative_logits - positive_logits[:, :hard_count] + args.atac_positive_margin
    ).mean(dim=1)
    hard_negative_loss = weighted_mean(hard_rank_cell, weights)
    density_loss = weighted_mean(
        (probability.mean(dim=1) - binary.mean(dim=1)).abs(), weights
    )
    normalized_weights = weights / weights.sum().clamp_min(1e-8)
    predicted_frequency = (probability * normalized_weights[:, None]).sum(dim=0)
    truth_frequency = (binary * normalized_weights[:, None]).sum(dim=0)
    frequency_loss = F.smooth_l1_loss(predicted_frequency, truth_frequency)
    loss = (
        balanced_bce
        + args.atac_rank_weight * rank_loss
        + args.atac_hard_negative_weight * hard_negative_loss
        + args.atac_focal_weight * focal_loss
        + args.atac_density_weight * density_loss
        + args.atac_frequency_weight * frequency_loss
    )
    return loss, {
        "decoder": float(balanced_bce.detach()),
        "rank": float(rank_loss.detach()),
        "hard_negative": float(hard_negative_loss.detach()),
        "focal": float(focal_loss.detach()),
        "density": float(density_loss.detach()),
        "frequency": float(frequency_loss.detach()),
    }


@torch.inference_mode()
def validation_recovery_metrics(
    vae,
    latent: torch.Tensor,
    truth: torch.Tensor,
    modality: str,
) -> dict[str, float]:
    """Compute label-free target metrics on held-out observed bridge cells."""

    rna_output, atac_logits, _ = vae.decode(latent)
    truth_array = truth.detach().cpu().numpy()
    if modality == "rna":
        prediction = rna_output if vae.rna_distribution == "poisson" else rna_output[0]
        prediction_array = prediction.clamp_min(0.0).detach().cpu().numpy()
        prediction_array = normalize_for_evaluation(prediction_array, "rna", truth=False)
        truth_normalized = normalize_for_evaluation(truth_array, "rna", truth=True)
        variable = np.argsort(np.var(truth_normalized, axis=0))[
            -min(100, truth_normalized.shape[1]) :
        ]
        prediction_correlation = np.corrcoef(prediction_array[:, variable], rowvar=False)
        truth_correlation = np.corrcoef(truth_normalized[:, variable], rowvar=False)
        correlation_mse = float(
            np.nanmean(np.square(prediction_correlation - truth_correlation))
        )
        return {
            "rna_smse": standardized_mse(prediction_array, truth_normalized),
            "rna_correlation_mse": correlation_mse,
        }
    probability = torch.sigmoid(atac_logits).detach().cpu().numpy()
    auprc, recall = atac_ranking_metrics(probability, (truth_array > 0).astype(np.float32))
    return {"atac_auprc": auprc, "atac_positive_recall": recall}


def reliability_components(
    quality: torch.Tensor,
    indices: np.ndarray,
    correspondence: torch.Tensor,
    proximity: torch.Tensor,
    generation_confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    if generation_confidence is None:
        generation_confidence = torch.ones(len(indices))
    return torch.stack((
        quality[indices], correspondence, proximity, generation_confidence
    ), dim=1).float().clamp(0.0, 1.0)


def build_training_pairs(
    direction: str,
    latent: dict[str, tuple[torch.Tensor, torch.Tensor]],
    quality: dict[str, torch.Tensor],
    observed: dict[str, np.ndarray],
    bridge_train: np.ndarray,
    proximity_all: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    source, target = DIRECTIONS[direction]
    source_mu, source_logvar = latent[source]
    target_mu, target_logvar = latent[target]
    source_only = np.flatnonzero(observed[source] & ~observed[target])
    target_only = np.flatnonzero(observed[target] & ~observed[source])
    pseudo = transport_targets(
        source_mu, source_logvar, target_mu, target_logvar,
        source_only, target_only, device,
        max_source=args.max_pseudo_cells,
        max_target=args.transport_target_anchors,
        seed=args.seed + (1103 if direction == "rna_to_atac" else 2203),
    )
    qualified = pseudo.confidence >= args.transport_confidence_min
    accepted = qualified & (args.pseudo_weight > 0)
    pseudo_source = pseudo.source_indices[accepted.numpy()]
    pseudo_target_rows = pseudo.target_indices[accepted.numpy()]
    pseudo_target_mu = pseudo.barycenter[accepted]
    pseudo_confidence = pseudo.confidence[accepted]

    real_components = reliability_components(
        quality[source], bridge_train, torch.ones(len(bridge_train)),
        torch.ones(len(bridge_train)),
    )
    pseudo_components = reliability_components(
        quality[source], pseudo_source, pseudo_confidence,
        proximity_all[source][pseudo_source],
    )
    return {
        "source_indices": np.concatenate((bridge_train, pseudo_source)),
        "target_rows": np.concatenate((bridge_train, pseudo_target_rows)),
        "target_mu": torch.cat((target_mu[bridge_train], pseudo_target_mu), dim=0),
        "components": torch.cat((real_components, pseudo_components), dim=0),
        "weights": torch.cat((
            torch.ones(len(bridge_train)),
            args.pseudo_weight * pseudo_confidence,
        )),
        "pair_type": np.concatenate((
            np.repeat("bridge", len(bridge_train)),
            np.repeat("transport", len(pseudo_source)),
        )),
        "transport_summary": {
            "candidate_cells": int(len(pseudo.source_indices)),
            "qualified_cells": int(qualified.sum()),
            "qualification_rate": float(qualified.float().mean()),
            "training_cells": int(accepted.sum()),
            "mean_confidence": float(pseudo.confidence.mean()),
            "qualified_mean_confidence": float(pseudo.confidence[qualified].mean()) if bool(qualified.any()) else 0.0,
            "component_means": {
                key: float(value.mean()) for key, value in pseudo.components.items()
            },
        },
    }


def make_model(
    source_mu: torch.Tensor,
    source_logvar: torch.Tensor,
    pairs: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> ReliabilityAdaptiveDiffusion:
    indices = pairs["source_indices"]
    real = pairs["pair_type"] == "bridge"
    real_indices = indices[real]
    raw_condition = torch.cat((
        source_mu[real_indices], source_logvar[real_indices]
    ), dim=1)
    residual = pairs["target_mu"][real] - source_mu[real_indices]
    return ReliabilityAdaptiveDiffusion(
        latent_dim=source_mu.shape[1],
        condition_mean=raw_condition.mean(dim=0),
        condition_std=raw_condition.std(dim=0).clamp_min(1e-4),
        residual_mean=residual.mean(dim=0),
        residual_std=residual.std(dim=0).clamp_min(1e-4),
        timesteps=args.timesteps,
        hidden_dim=args.hidden_dim,
    ).to(device)


def train_direction(
    direction: str,
    model: ReliabilityAdaptiveDiffusion,
    vae,
    pairs: dict,
    latent: dict[str, tuple[torch.Tensor, torch.Tensor]],
    target_data: ad.AnnData,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    source, target = DIRECTIONS[direction]
    source_mu, source_logvar = latent[source]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(
        args.seed + (0 if direction == "rna_to_atac" else 10000)
    )
    history = []
    n_pairs = len(pairs["source_indices"])
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(n_pairs, generator=generator)
        totals: dict[str, float] = {}
        batches = 0
        for start in range(0, n_pairs, args.batch_size):
            selected = permutation[start : start + args.batch_size]
            source_indices = pairs["source_indices"][selected.numpy()]
            weights = pairs["weights"][selected].to(device)
            loss, values, diffusion_target, mean_target = model.training_loss(
                source_mu[source_indices].to(device),
                source_logvar[source_indices].to(device),
                pairs["target_mu"][selected].to(device),
                pairs["components"][selected].to(device),
                sample_weight=weights,
                return_predictions=True,
            )
            truth = torch.from_numpy(matrix_rows(
                target_data, pairs["target_rows"][selected.numpy()]
            ).astype(np.float32, copy=False)).to(device)
            diffusion_decoder, decoder_values = target_reconstruction_loss(
                vae=vae,
                latent=diffusion_target,
                truth=truth,
                modality=target,
                weights=weights,
                args=args,
            )
            mean_decoder, _ = target_reconstruction_loss(
                vae, mean_target, truth, target, weights, args
            )
            decoder_loss = 0.5 * (diffusion_decoder + mean_decoder)
            loss = loss + args.decoder_weight * decoder_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            values.update(decoder_values)
            values["loss"] = float(loss.detach())
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            batches += 1
        row = {"direction": direction, "epoch": epoch}
        row.update({key: value / max(1, batches) for key, value in totals.items()})
        history.append(row)
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(
                f"{direction} epoch={epoch} loss={row['loss']:.5f} "
                f"x0={row['diffusion']:.5f} mean={row['mean']:.5f} "
                f"decoder={row['decoder']:.5f} rho={row['rho']:.3f}",
                flush=True,
            )
    return pd.DataFrame(history)


@torch.inference_mode()
def predict_in_batches(
    model: ReliabilityAdaptiveDiffusion,
    source_mu: torch.Tensor,
    source_logvar: torch.Tensor,
    indices: np.ndarray,
    components: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    calibration: Calibration | None = None,
    max_timestep: int | None = None,
    max_blend: float | None = None,
) -> dict[str, torch.Tensor]:
    collected: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(indices), args.batch_size):
        selected = indices[start : start + args.batch_size]
        component_batch = components[start : start + len(selected)].to(device)
        policy = None
        if calibration is not None:
            policy = calibration.policy_reliability(
                calibration.combined_reliability(component_batch)
            )
        distribution = model.predict_distribution(
            source_mu[selected].to(device), source_logvar[selected].to(device),
            component_batch,
            max_timestep=(
                calibration.selected_max_timestep if calibration is not None
                else args.max_inference_timestep
            ) if max_timestep is None else max_timestep,
            max_blend=(
                calibration.selected_max_blend if calibration is not None
                else args.max_diffusion_blend
            ) if max_blend is None else max_blend,
            ensemble_size=args.ensemble_size,
            seed=args.seed + start + 1701,
            policy_reliability=policy,
        )
        for key, value in distribution.items():
            if key == "samples":
                continue
            collected.setdefault(key, []).append(value.cpu())
        collected.setdefault("samples", []).append(distribution["samples"].cpu())
    result = {}
    for key, values in collected.items():
        result[key] = torch.cat(values, dim=1 if key == "samples" else 0)
    return result


def base_reliability(components: torch.Tensor) -> torch.Tensor:
    return components[:, :3].clamp_min(1e-8).log().mean(dim=1).exp()


def apply_transport_support(
    distribution: dict[str, torch.Tensor],
    barycenter: torch.Tensor,
    confidence: torch.Tensor,
    max_blend: float,
) -> dict[str, torch.Tensor]:
    """Shift a generated latent distribution toward confidence-weighted OT support."""
    supported = {key: value for key, value in distribution.items()}
    supported["base_conditional_mean"] = distribution["conditional_mean"].clone()
    weight = (float(max_blend) * confidence).clamp(0.0, 1.0)
    shift = weight[:, None] * (barycenter - distribution["conditional_mean"])
    supported["samples"] = distribution["samples"] + shift[None, :, :]
    supported["sample_mean"] = distribution["sample_mean"] + shift
    supported["conditional_mean"] = distribution["conditional_mean"] + shift
    supported["transport_blend"] = weight
    return supported


def calibrate_direction(
    direction: str,
    model: ReliabilityAdaptiveDiffusion,
    latent: dict[str, tuple[torch.Tensor, torch.Tensor]],
    quality: dict[str, torch.Tensor],
    vae,
    target_data: ad.AnnData,
    bridge_train: np.ndarray,
    bridge_validation: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Calibration, dict]:
    source, target = DIRECTIONS[direction]
    source_mu, source_logvar = latent[source]
    target_mu, target_logvar = latent[target]
    transported = transport_targets(
        source_mu, source_logvar, target_mu, target_logvar,
        bridge_validation, bridge_train, device,
        max_target=args.transport_target_anchors,
        seed=args.seed + (3109 if direction == "rna_to_atac" else 3209),
    )
    proximity = bridge_proximity(
        source_mu, bridge_validation, bridge_train, device
    )
    initial_components = reliability_components(
        quality[source], bridge_validation, transported.confidence, proximity
    )
    initial = predict_in_batches(
        model, source_mu, source_logvar, bridge_validation,
        initial_components, args, device,
    )
    initial_uncertainty = generation_uncertainty(initial, model.residual_std.cpu())
    preliminary = fit_calibration(
        initial_uncertainty[: len(bridge_validation) // 2],
        initial["sample_mean"][: len(bridge_validation) // 2],
        initial["sample_std"][: len(bridge_validation) // 2],
        target_mu[bridge_validation[: len(bridge_validation) // 2]],
        base_reliability(initial_components[: len(bridge_validation) // 2]),
        target_coverage=args.target_coverage,
        reliability_components=initial_components[: len(bridge_validation) // 2],
    )
    generation_confidence = preliminary.generation_confidence(initial_uncertainty)
    final_components = initial_components.clone()
    final_components[:, 3] = generation_confidence
    calibration_count = len(bridge_validation) // 2
    calibration_indices = bridge_validation[:calibration_count]
    candidate_rows = []
    target_scale = target_mu[calibration_indices].std(dim=0).clamp_min(1e-4)
    truth = torch.from_numpy(matrix_rows(
        target_data, calibration_indices
    ).astype(np.float32, copy=False)).to(device)
    diffusion_settings = [(0, 0.0)]
    diffusion_settings.extend(
        (timestep, blend)
        for timestep in sorted({4, args.max_inference_timestep})
        for blend in sorted({0.25, args.max_diffusion_blend})
    )
    candidate_outputs = {}
    for max_timestep, max_blend in diffusion_settings:
        base_candidate = predict_in_batches(
            model, source_mu, source_logvar, bridge_validation,
            final_components, args, device, calibration=preliminary,
            max_timestep=max_timestep, max_blend=max_blend,
        )
        for transport_blend in (0.0, 0.5, 1.0):
            candidate = apply_transport_support(
                base_candidate, transported.barycenter,
                transported.confidence, transport_blend,
            )
            key = (max_timestep, max_blend, transport_blend)
            candidate_outputs[key] = candidate
            predicted = candidate["sample_mean"][:calibration_count]
            latent_loss = torch.mean(
                ((predicted - target_mu[calibration_indices]) / target_scale) ** 2
            ).item()
            decoder_loss = decoder_reconstruction_loss(
                vae, predicted.to(device), truth, target
            ).item()
            target_metrics = validation_recovery_metrics(
                vae, predicted.to(device), truth, target
            )
            candidate_row = {
                "max_timestep": int(max_timestep),
                "max_blend": float(max_blend),
                "transport_blend": float(transport_blend),
                "latent_smse": latent_loss, "decoder_loss": decoder_loss,
                "rna_smse": np.nan,
                "rna_correlation_mse": np.nan,
                "atac_auprc": np.nan,
                "atac_positive_recall": np.nan,
            }
            candidate_row.update(target_metrics)
            candidate_rows.append(candidate_row)
    candidate_frame = pd.DataFrame(candidate_rows)
    baseline = candidate_frame[
        (candidate_frame.max_timestep == 0)
        & (candidate_frame.max_blend == 0)
        & (candidate_frame.transport_blend == 0)
    ].iloc[0]
    eligible = candidate_frame[
        (candidate_frame.latent_smse <= baseline.latent_smse)
        & (candidate_frame.decoder_loss <= baseline.decoder_loss)
    ]
    if args.selection_objective == "decoder":
        selected = eligible.sort_values(
            ["decoder_loss", "latent_smse", "transport_blend", "max_blend", "max_timestep"]
        ).iloc[0]
    elif target == "atac":
        selected = eligible.sort_values(
            ["atac_auprc", "atac_positive_recall", "decoder_loss", "latent_smse"],
            ascending=[False, False, True, True],
        ).iloc[0]
    else:
        baseline_smse = max(float(baseline.rna_smse), 1e-8)
        baseline_correlation = max(float(baseline.rna_correlation_mse), 1e-8)
        eligible = eligible.assign(
            rna_joint_objective=(
                eligible.rna_smse / baseline_smse
                + eligible.rna_correlation_mse / baseline_correlation
            )
        )
        selected = eligible.sort_values(
            ["rna_joint_objective", "rna_smse", "decoder_loss", "latent_smse"],
            ascending=[True, True, True, True],
        ).iloc[0]
    selected_key = (
        int(selected.max_timestep), float(selected.max_blend),
        float(selected.transport_blend),
    )
    final = candidate_outputs[selected_key]
    final_uncertainty = generation_uncertainty(final, model.residual_std.cpu())
    calibration = fit_calibration(
        final_uncertainty[:calibration_count],
        final["sample_mean"][:calibration_count],
        final["sample_std"][:calibration_count],
        target_mu[calibration_indices],
        base_reliability(final_components[:calibration_count]),
        target_coverage=args.target_coverage,
        reliability_components=final_components[:calibration_count],
    )
    calibration.selected_max_timestep = selected_key[0]
    calibration.selected_max_blend = selected_key[1]
    calibration.selected_transport_blend = selected_key[2]
    final_components[:, 3] = calibration.generation_confidence(final_uncertainty)
    rho = calibration.combined_reliability(final_components)
    lower, upper = prediction_interval(
        final["sample_mean"], final["sample_std"], calibration, rho
    )
    audit_slice = slice(calibration_count, None)
    target_values = target_mu[bridge_validation][audit_slice]
    coverage = (
        (target_values >= lower[audit_slice]) & (target_values <= upper[audit_slice])
    ).float().mean()
    accepted = rho >= calibration.rejection_threshold
    error = (
        final["sample_mean"][audit_slice] - target_values
    ).square().mean(dim=1).sqrt()
    audit_accepted = accepted[audit_slice]
    summary = {
        "latent_interval_coverage": float(coverage),
        "target_coverage": args.target_coverage,
        "rejection_threshold": calibration.rejection_threshold,
        "validation_acceptance_rate": float(audit_accepted.float().mean()),
        "validation_rmse_all": float(error.mean()),
        "validation_rmse_accepted": float(error[audit_accepted].mean()) if bool(audit_accepted.any()) else None,
        "mean_transport_confidence": float(transported.confidence.mean()),
        "selected_max_timestep": selected_key[0],
        "selected_max_blend": selected_key[1],
        "selected_transport_blend": selected_key[2],
        "candidate_scores": candidate_rows,
        "reliability_weights": calibration.reliability_weights.tolist(),
    }
    return calibration, summary


def recover_direction(
    direction: str,
    model: ReliabilityAdaptiveDiffusion,
    calibration: Calibration,
    latent: dict[str, tuple[torch.Tensor, torch.Tensor]],
    quality: dict[str, torch.Tensor],
    observed: dict[str, np.ndarray],
    bridge_train: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    source, target = DIRECTIONS[direction]
    source_mu, source_logvar = latent[source]
    target_mu, target_logvar = latent[target]
    hidden = np.flatnonzero(observed[source] & ~observed[target])
    target_pool = np.flatnonzero(observed[target])
    transported = transport_targets(
        source_mu, source_logvar, target_mu, target_logvar,
        hidden, target_pool, device,
        max_target=args.transport_target_anchors,
        seed=args.seed + (4109 if direction == "rna_to_atac" else 4209),
    )
    proximity = bridge_proximity(source_mu, hidden, bridge_train, device)
    initial_components = reliability_components(
        quality[source], hidden, transported.confidence, proximity
    )
    initial = predict_in_batches(
        model, source_mu, source_logvar, hidden, initial_components, args, device
    )
    uncertainty = generation_uncertainty(initial, model.residual_std.cpu())
    generation_confidence = calibration.generation_confidence(uncertainty)
    final_components = initial_components.clone()
    final_components[:, 3] = generation_confidence
    final = predict_in_batches(
        model, source_mu, source_logvar, hidden, final_components, args, device,
        calibration=calibration,
    )
    final = apply_transport_support(
        final, transported.barycenter, transported.confidence,
        calibration.selected_transport_blend,
    )
    final_uncertainty = generation_uncertainty(final, model.residual_std.cpu())
    final_components[:, 3] = calibration.generation_confidence(final_uncertainty)
    final = predict_in_batches(
        model, source_mu, source_logvar, hidden, final_components, args, device,
        calibration=calibration,
    )
    final = apply_transport_support(
        final, transported.barycenter, transported.confidence,
        calibration.selected_transport_blend,
    )
    final_uncertainty = generation_uncertainty(final, model.residual_std.cpu())
    rho = calibration.combined_reliability(final_components)
    accepted = rho >= calibration.rejection_threshold
    point_latent = final["sample_mean"].clone()
    point_latent[~accepted] = final["base_conditional_mean"][~accepted]
    lower, upper = prediction_interval(
        final["sample_mean"], final["sample_std"], calibration, rho
    )
    output = {
        "cell_index": hidden.astype(np.int64),
        "latent_samples": final["samples"].numpy().astype(np.float16),
        "latent_mean": final["sample_mean"].numpy().astype(np.float32),
        "latent_std": final["sample_std"].numpy().astype(np.float32),
        "latent_lower": lower.numpy().astype(np.float32),
        "latent_upper": upper.numpy().astype(np.float32),
        "conditional_mean": final["conditional_mean"].numpy().astype(np.float32),
        "base_conditional_mean": final["base_conditional_mean"].numpy().astype(np.float32),
        "quality": final_components[:, 0].numpy(),
        "correspondence_confidence": final_components[:, 1].numpy(),
        "bridge_proximity": final_components[:, 2].numpy(),
        "generation_confidence": final_components[:, 3].numpy(),
        "rho": rho.numpy(),
        "accepted": accepted.numpy(),
        "adaptive_blend": final["blend"].numpy(),
        "adaptive_timestep": final["timestep"].numpy(),
        "policy_reliability": final["policy_reliability"].numpy(),
        "transport_blend": final["transport_blend"].numpy(),
        "uncertainty": final_uncertainty.numpy(),
    }
    summary = {
        "n_cells": int(len(hidden)),
        "acceptance_rate": float(accepted.float().mean()),
        "mean_rho": float(rho.mean()),
        "mean_quality": float(final_components[:, 0].mean()),
        "mean_correspondence_confidence": float(final_components[:, 1].mean()),
        "mean_generation_confidence": float(final_components[:, 3].mean()),
        "mean_adaptive_blend": float(final["blend"].mean()),
        "mean_adaptive_timestep": float(final["timestep"].float().mean()),
        "mean_transport_blend": float(final["transport_blend"].mean()),
    }
    return hidden, point_latent.numpy(), {"arrays": output, "summary": summary}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    rna = ad.read_h5ad(args.data_dir / "truth_rna_hvg.h5ad")
    atac = ad.read_h5ad(args.data_dir / "truth_atac_hvg.h5ad")
    truth_data = {"rna": rna, "atac": atac}
    observed = {
        "rna": rna.obs["rna_observed"].astype(bool).to_numpy(),
        "atac": rna.obs["atac_observed"].astype(bool).to_numpy(),
    }
    bridge = np.flatnonzero(observed["rna"] & observed["atac"])
    rng = np.random.default_rng(args.seed + 313)
    bridge = rng.permutation(bridge)
    n_validation = max(1, int(round(len(bridge) * args.validation_fraction)))
    bridge_validation = bridge[:n_validation].copy()
    bridge_train = bridge[n_validation:].copy()

    checkpoint = torch.load(args.vae_checkpoint, map_location="cpu", weights_only=False)
    vae = build_vae(checkpoint, rna.n_vars, atac.n_vars, device)
    latent = {
        "rna": encode_observed(vae, rna, observed["rna"], "rna", args.batch_size, device),
        "atac": encode_observed(vae, atac, observed["atac"], "atac", args.batch_size, device),
    }
    quality = {
        modality: modality_quality(
            vae, latent[modality][0], latent[modality][1], observed[modality],
            modality, device, args.batch_size,
        )
        for modality in ("rna", "atac")
    }
    proximity_all = {}
    for modality in ("rna", "atac"):
        proximity_all[modality] = torch.zeros(rna.n_obs)
        indices = np.flatnonzero(observed[modality])
        proximity_all[modality][indices] = bridge_proximity(
            latent[modality][0], indices, bridge_train, device
        )

    models = {}
    calibrations = {}
    histories = []
    transport_summaries = {}
    calibration_summaries = {}
    for direction, (source, target) in DIRECTIONS.items():
        pairs = build_training_pairs(
            direction, latent, quality, observed, bridge_train,
            proximity_all, args, device,
        )
        transport_summaries[direction] = pairs["transport_summary"]
        model = make_model(
            latent[source][0], latent[source][1], pairs, args, device
        )
        model_path = args.output_dir / f"{direction}_reliability_diffusion.pt"
        if model_path.exists() and not args.force:
            saved = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(saved["model"])
            history = pd.DataFrame(saved.get("history", []))
        else:
            history = train_direction(
                direction, model, vae, pairs, latent, truth_data[target], args, device
            )
        calibration, calibration_summary = calibrate_direction(
            direction, model.eval(), latent, quality,
            vae, truth_data[target],
            bridge_train, bridge_validation, args, device,
        )
        torch.save({
            "model": model.state_dict(), "history": history.to_dict("records"),
            "calibration": asdict(calibration), "args": vars(args),
            "transport_summary": pairs["transport_summary"],
        }, model_path)
        models[direction] = model.eval()
        calibrations[direction] = calibration
        histories.append(history)
        calibration_summaries[direction] = calibration_summary
        print(f"{direction} calibration {calibration_summary}", flush=True)

    pd.concat(histories, ignore_index=True).to_csv(
        args.output_dir / "training_history.csv", index=False
    )
    predictions = {}
    mean_predictions = {}
    hidden_indices = {}
    recovery_summaries = {}
    for direction, (source, target) in DIRECTIONS.items():
        hidden, point_latent, recovery = recover_direction(
            direction, models[direction], calibrations[direction], latent,
            quality, observed, bridge_train, args, device,
        )
        hidden_indices[direction] = hidden
        predictions[direction] = decode_target(
            vae, torch.from_numpy(point_latent), target, args.batch_size, device
        )
        mean_predictions[direction] = decode_target(
            vae, torch.from_numpy(recovery["arrays"]["conditional_mean"]),
            target, args.batch_size, device,
        )
        np.savez_compressed(
            args.output_dir / f"{direction}_recovery_distribution.npz",
            **recovery["arrays"],
        )
        recovery_summaries[direction] = recovery["summary"]
        print(f"{direction} recovery {recovery['summary']}", flush=True)

    metrics = evaluate_variant(
        "MoTRUST", predictions, truth_data,
        hidden_indices, args.seed, args.output_dir,
    )
    mean_metrics = evaluate_variant(
        "MoTRUST-ConditionalMean", mean_predictions, truth_data,
        hidden_indices, args.seed, args.output_dir,
    )
    metrics.to_csv(args.output_dir / "formal_recovery_metrics_long.csv", index=False)
    mean_metrics.to_csv(
        args.output_dir / "formal_reliability_mean_metrics_long.csv", index=False
    )
    manifest = {
        "method": "MoTRUST",
        "seed": args.seed,
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "bridge_training_cells": int(len(bridge_train)),
        "bridge_validation_cells": int(len(bridge_validation)),
        "transport_training": transport_summaries,
        "calibration": calibration_summaries,
        "recovery": recovery_summaries,
        "hidden_truth_used_for_training_or_selection": False,
        "cell_type_labels_used_for_training_or_selection": False,
        "hidden_truth_used_for_posthoc_evaluation": True,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Completed in {manifest['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
