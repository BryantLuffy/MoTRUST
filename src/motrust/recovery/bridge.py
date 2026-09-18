"""Train and evaluate bridge-conditioned latent diffusion on frozen Ma-2020."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse


from motrust.evaluation.formal_rows import (
    formal_metric_rows,
    observed_reference,
)
from motrust.models.multimodal_vae import ImprovedMultiModalVAE
from motrust.models.poe import ProductOfExperts
from motrust.recovery.base_diffusion import (
    ConditionalResidualLatentDiffusion,
)


DIRECTIONS = {
    "rna_to_atac": ("rna", "atac"),
    "atac_to_rna": ("atac", "rna"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--decoder-weight", type=float, default=0.15)
    parser.add_argument("--max-diffusion-timestep", type=int, default=9)
    parser.add_argument("--diffusion-blend", type=float, default=0.25)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def matrix_rows(data: ad.AnnData, indices: np.ndarray) -> np.ndarray:
    values = data.layers["counts"][indices]
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


def build_vae(checkpoint: dict, rna_dim: int, atac_dim: int, device: torch.device):
    saved = checkpoint.get("args", {})
    dim_c = int(saved.get("dim_c", 32))
    dim_u = int(saved.get("dim_u", 2))
    model = ImprovedMultiModalVAE(
        rna_dim=rna_dim,
        atac_dim=atac_dim,
        latent_dim=dim_c + dim_u,
        dim_c=dim_c,
        dim_u=dim_u,
        hidden_dims=[512, 256],
        dropout=0.1,
        beta_c=float(saved.get("lam_kld_c", 1.0)),
        beta_u=float(saved.get("lam_kld_u", 0.5)),
        use_poe=True,
        batch_correction=bool(saved.get("use_batch_correction", True)),
        rna_distribution=str(saved.get("rna_distribution", "zinb")),
        use_gated_poe=bool(saved.get("use_gated_poe", True)),
        gate_hidden_dim=int(saved.get("gate_hidden_dim", 64)),
        use_shared_backbone=bool(saved.get("use_shared_backbone", False)),
    ).to(device)
    model.load_state_dict(checkpoint["vae"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_observed(
    vae: ImprovedMultiModalVAE,
    data: ad.AnnData,
    observed: np.ndarray,
    modality: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_dim = vae.latent_dim
    means = torch.full((data.n_obs, latent_dim), torch.nan)
    logvars = torch.full((data.n_obs, latent_dim), torch.nan)
    indices = np.flatnonzero(observed)
    encoder = vae.rna_encoder if modality == "rna" else vae.atac_encoder
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        values = torch.from_numpy(
            matrix_rows(data, selected).astype(np.float32, copy=False)
        ).to(device)
        mu, logvar = encoder(values)
        weights = None
        if vae.use_gated_poe:
            weights = [vae._gate_weight(modality, mu, logvar)]
        posterior_mu, posterior_logvar = ProductOfExperts.poe_with_prior(
            [mu], [logvar], weights=weights
        )
        means[selected] = posterior_mu.cpu()
        logvars[selected] = posterior_logvar.cpu()
    return means, logvars


def make_diffusion(
    source_mu: torch.Tensor,
    source_logvar: torch.Tensor,
    target_mu: torch.Tensor,
    train_indices: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> ConditionalResidualLatentDiffusion:
    raw_condition = torch.cat((source_mu[train_indices], source_logvar[train_indices]), dim=1)
    residual = target_mu[train_indices] - source_mu[train_indices]
    return ConditionalResidualLatentDiffusion(
        latent_dim=source_mu.shape[1],
        condition_mean=raw_condition.mean(dim=0),
        condition_std=raw_condition.std(dim=0).clamp_min(1e-4),
        residual_mean=residual.mean(dim=0),
        residual_std=residual.std(dim=0).clamp_min(1e-4),
        timesteps=args.timesteps,
        hidden_dim=args.hidden_dim,
    ).to(device)


def decoder_reconstruction_loss(
    vae: ImprovedMultiModalVAE,
    latent: torch.Tensor,
    truth: torch.Tensor,
    modality: str,
) -> torch.Tensor:
    """Measure paired-target reconstruction without labels or hidden test cells."""
    rna_output, atac_logits, _ = vae.decode(latent)
    if modality == "rna":
        prediction = rna_output if vae.rna_distribution == "poisson" else rna_output[0]
        prediction = prediction.clamp_min(0.0)
        truth = truth.clamp_min(0.0)
        prediction = torch.log1p(
            prediction * (1e4 / prediction.sum(dim=1, keepdim=True).clamp_min(1e-8))
        )
        truth = torch.log1p(
            truth * (1e4 / truth.sum(dim=1, keepdim=True).clamp_min(1e-8))
        )
        return F.smooth_l1_loss(prediction, truth)

    if modality != "atac":
        raise ValueError(f"Unsupported target modality: {modality}")
    binary_truth = (truth > 0).float()
    element_loss = F.binary_cross_entropy_with_logits(
        atac_logits, binary_truth, reduction="none"
    )
    positive_loss = (element_loss * binary_truth).sum(dim=1) / binary_truth.sum(
        dim=1
    ).clamp_min(1.0)
    negative = 1.0 - binary_truth
    negative_loss = (element_loss * negative).sum(dim=1) / negative.sum(dim=1).clamp_min(
        1.0
    )
    return 0.5 * (positive_loss + negative_loss).mean()


def train_direction(
    name: str,
    model: ConditionalResidualLatentDiffusion,
    source_mu: torch.Tensor,
    source_logvar: torch.Tensor,
    target_mu: torch.Tensor,
    train_indices: torch.Tensor,
    vae: ImprovedMultiModalVAE,
    target_data: ad.AnnData,
    target_modality: str,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    generator = torch.Generator().manual_seed(args.seed + (0 if name == "rna_to_atac" else 10000))
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
        totals = {"loss": 0.0, "diffusion": 0.0, "mean": 0.0, "decoder": 0.0}
        batches = 0
        for start in range(0, len(permutation), args.batch_size):
            selected = permutation[start : start + args.batch_size]
            loss, values, diffusion_target, mean_target = model.training_loss(
                source_mu[selected].to(device),
                source_logvar[selected].to(device),
                target_mu[selected].to(device),
                return_predictions=True,
            )
            truth = torch.from_numpy(matrix_rows(
                target_data, selected.numpy()
            ).astype(np.float32, copy=False)).to(device)
            decoder_loss = 0.5 * (
                decoder_reconstruction_loss(vae, diffusion_target, truth, target_modality)
                + decoder_reconstruction_loss(vae, mean_target, truth, target_modality)
            )
            loss = loss + args.decoder_weight * decoder_loss
            values["loss"] = float(loss.detach())
            values["decoder"] = float(decoder_loss.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            for key in totals:
                totals[key] += values[key]
            batches += 1
        row = {"direction": name, "epoch": epoch}
        row.update({key: value / max(batches, 1) for key, value in totals.items()})
        history.append(row)
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(
                f"{name} epoch={epoch} loss={row['loss']:.5f} "
                f"noise={row['diffusion']:.5f} mean={row['mean']:.5f} "
                f"decoder={row['decoder']:.5f}",
                flush=True,
            )
    return pd.DataFrame(history)


@torch.inference_mode()
def candidate_selection(
    name: str,
    model: ConditionalResidualLatentDiffusion,
    source_mu: torch.Tensor,
    source_logvar: torch.Tensor,
    target_mu: torch.Tensor,
    validation_indices: torch.Tensor,
    vae: ImprovedMultiModalVAE,
    target_data: ad.AnnData,
    target_modality: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, dict, dict]:
    source = source_mu[validation_indices].to(device)
    logvar = source_logvar[validation_indices].to(device)
    target = target_mu[validation_indices].to(device)
    target_scale = target_mu[validation_indices].std(dim=0).to(device).clamp_min(1e-4)
    truth = torch.from_numpy(matrix_rows(
        target_data, validation_indices.numpy()
    ).astype(np.float32, copy=False)).to(device)
    rows = []
    mean_target, _ = model.predict(
        source, logvar, diffusion_strength=0.0,
        ddim_steps=args.ddim_steps, ensemble_size=1, seed=args.seed + 701,
    )
    rows.append({
        "direction": name, "sampler": "mean", "noise_timestep": -1,
        "diffusion_strength": 0.0, "target_blend": 0.0,
        "validation_latent_smse": torch.mean(
            ((mean_target - target) / target_scale) ** 2
        ).item(),
        "validation_decoder_loss": decoder_reconstruction_loss(
            vae, mean_target, truth, target_modality
        ).item(),
    })
    candidate_timesteps = sorted({
        timestep for timestep in (0, 4, 9, args.max_diffusion_timestep)
        if 0 <= timestep <= args.max_diffusion_timestep
    })
    for noise_timestep in candidate_timesteps:
        predicted, mean_target = model.predict_tweedie(
            source, logvar, noise_timestep=noise_timestep,
            ensemble_size=max(2, args.ensemble_size), seed=args.seed + 1709,
        )
        blend = float(args.diffusion_blend)
        blended = (1.0 - blend) * mean_target + blend * predicted
        latent_loss = torch.mean(((blended - target) / target_scale) ** 2).item()
        decoder_loss = decoder_reconstruction_loss(
            vae, blended, truth, target_modality
        ).item()
        rows.append({
            "direction": name, "sampler": "tweedie",
            "noise_timestep": int(noise_timestep), "diffusion_strength": 1.0,
            "target_blend": blend, "validation_latent_smse": latent_loss,
            "validation_decoder_loss": decoder_loss,
        })
    frame = pd.DataFrame(rows)
    mean_best = frame[frame.sampler == "mean"].iloc[0].to_dict()
    eligible = frame[
        (frame.validation_latent_smse <= mean_best["validation_latent_smse"])
        & (frame.validation_decoder_loss <= mean_best["validation_decoder_loss"])
    ]
    best = eligible.sort_values(
        ["validation_decoder_loss", "sampler", "noise_timestep", "target_blend"]
    ).iloc[0].to_dict()
    return frame, best, mean_best


@torch.inference_mode()
def predict_hidden(
    model: ConditionalResidualLatentDiffusion,
    source_mu: torch.Tensor,
    source_logvar: torch.Tensor,
    hidden_indices: np.ndarray,
    setting: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(hidden_indices), args.batch_size):
        selected = hidden_indices[start : start + args.batch_size]
        source = source_mu[selected].to(device)
        logvar = source_logvar[selected].to(device)
        if setting.get("sampler", "ddim") == "tweedie":
            predicted, mean_target = model.predict_tweedie(
                source, logvar,
                noise_timestep=int(setting["noise_timestep"]),
                ensemble_size=args.ensemble_size,
                seed=args.seed + start + 1701,
            )
        else:
            predicted, mean_target = model.predict(
                source, logvar,
                diffusion_strength=float(setting["diffusion_strength"]),
                ddim_steps=args.ddim_steps, ensemble_size=args.ensemble_size,
                seed=args.seed + start + 1701,
            )
        blend = float(setting["target_blend"])
        outputs.append(((1.0 - blend) * mean_target + blend * predicted).cpu())
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def decode_target(
    vae: ImprovedMultiModalVAE,
    latent: torch.Tensor,
    modality: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    rows = []
    for start in range(0, len(latent), batch_size):
        values = latent[start : start + batch_size].to(device)
        rna_output, atac_logits, _ = vae.decode(values)
        if modality == "rna":
            prediction = rna_output if vae.rna_distribution == "poisson" else rna_output[0]
        else:
            prediction = torch.sigmoid(atac_logits)
        rows.append(prediction.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(rows)


def evaluate_variant(
    method: str,
    predictions: dict[str, np.ndarray],
    truth_data: dict[str, ad.AnnData],
    hidden_indices: dict[str, np.ndarray],
    seed: int,
    source: Path,
) -> pd.DataFrame:
    rows = []
    for direction, (_source, target) in DIRECTIONS.items():
        truth = truth_data[target]
        selected = hidden_indices[direction]
        truth_x = matrix_rows(truth, selected).astype(np.float32, copy=False)
        labels = truth.obs.iloc[selected]["cell_type"].astype(str).to_numpy()
        reference_truth, reference_labels = observed_reference(truth, target)
        rows.extend(formal_metric_rows(
            method=method, scenario="MA2020", replicate=1, seed=seed,
            modality=target, recovery_direction=direction,
            prediction=predictions[direction], truth=truth_x, labels=labels,
            reference_truth=reference_truth, reference_labels=reference_labels,
            source=source,
        ))
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    rna = ad.read_h5ad(args.data_dir / "truth_rna_hvg.h5ad")
    atac = ad.read_h5ad(args.data_dir / "truth_atac_hvg.h5ad")
    truth_data = {"rna": rna, "atac": atac}
    rna_observed = rna.obs["rna_observed"].astype(bool).to_numpy()
    atac_observed = rna.obs["atac_observed"].astype(bool).to_numpy()
    bridge = np.flatnonzero(rna_observed & atac_observed)
    rng = np.random.default_rng(args.seed + 313)
    bridge = rng.permutation(bridge)
    n_validation = max(1, int(round(len(bridge) * args.validation_fraction)))
    validation_indices = torch.from_numpy(bridge[:n_validation].copy()).long()
    train_indices = torch.from_numpy(bridge[n_validation:].copy()).long()

    checkpoint = torch.load(args.vae_checkpoint, map_location="cpu", weights_only=False)
    vae = build_vae(checkpoint, rna.n_vars, atac.n_vars, device)
    latent = {}
    latent["rna"] = encode_observed(
        vae, rna, rna_observed, "rna", args.batch_size, device
    )
    latent["atac"] = encode_observed(
        vae, atac, atac_observed, "atac", args.batch_size, device
    )

    histories = []
    selections = []
    models = {}
    settings = {}
    mean_settings = {}
    for direction, (source, target) in DIRECTIONS.items():
        model_path = args.output_dir / f"{direction}_diffusion.pt"
        source_mu, source_logvar = latent[source]
        target_mu, _target_logvar = latent[target]
        model = make_diffusion(
            source_mu, source_logvar, target_mu, train_indices, args, device
        )
        if model_path.exists() and not args.force:
            saved = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(saved["model"])
            history = pd.DataFrame(saved.get("history", []))
        else:
            history = train_direction(
                direction, model, source_mu, source_logvar, target_mu,
                train_indices, vae, truth_data[target], target, args, device,
            )
        selection, best, mean_best = candidate_selection(
            direction, model, source_mu, source_logvar, target_mu,
            validation_indices, vae, truth_data[target], target, args, device,
        )
        torch.save({
            "model": model.state_dict(), "history": history.to_dict("records"),
            "selection": selection.to_dict("records"), "best": best,
            "mean_best": mean_best, "args": vars(args),
        }, model_path)
        models[direction] = model.eval()
        settings[direction] = best
        mean_settings[direction] = mean_best
        histories.append(history)
        selections.append(selection)
        print(f"{direction} selected {best}", flush=True)

    pd.concat(histories, ignore_index=True).to_csv(args.output_dir / "training_history.csv", index=False)
    pd.concat(selections, ignore_index=True).to_csv(args.output_dir / "validation_selection.csv", index=False)
    hidden_indices = {
        "rna_to_atac": np.flatnonzero(rna_observed & ~atac_observed),
        "atac_to_rna": np.flatnonzero(atac_observed & ~rna_observed),
    }
    diffusion_predictions = {}
    mean_predictions = {}
    for direction, (source, target) in DIRECTIONS.items():
        model = models[direction]
        source_mu, source_logvar = latent[source]
        diffusion_latent = predict_hidden(
            model, source_mu, source_logvar, hidden_indices[direction],
            settings[direction], args, device,
        )
        mean_latent = predict_hidden(
            model, source_mu, source_logvar, hidden_indices[direction],
            mean_settings[direction], args, device,
        )
        diffusion_predictions[direction] = decode_target(
            vae, diffusion_latent, target, args.batch_size, device
        )
        mean_predictions[direction] = decode_target(
            vae, mean_latent, target, args.batch_size, device
        )

    diffusion_metrics = evaluate_variant(
        "MoTRUST-BridgeDiffusion", diffusion_predictions, truth_data,
        hidden_indices, args.seed, args.output_dir,
    )
    mean_metrics = evaluate_variant(
        "MoTRUST-BridgeMean", mean_predictions, truth_data,
        hidden_indices, args.seed, args.output_dir,
    )
    diffusion_metrics.to_csv(
        args.output_dir / "formal_recovery_metrics_long.csv", index=False
    )
    mean_metrics.to_csv(
        args.output_dir / "formal_bridge_mean_metrics_long.csv", index=False
    )
    manifest = {
        "method": "MoTRUST-BridgeDiffusion",
        "seed": args.seed,
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "bridge_training_cells": int(len(train_indices)),
        "bridge_validation_cells": int(len(validation_indices)),
        "hidden_truth_used_for_training_or_selection": False,
        "labels_used_for_training_or_selection": False,
        "selected_settings": settings,
        "mean_only_settings": mean_settings,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"Completed in {manifest['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
