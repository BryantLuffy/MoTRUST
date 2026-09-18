"""Label-free MoTRUST integration trainer for RNA/ATAC/ADT mosaic tasks.

The frozen task matrices stay sparse on CPU. Only selected-feature mini-batches
are materialized as dense tensors, which makes the Retina task tractable.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
from scipy import sparse

from motrust.models import BatchDiscriminator, ImprovedMultiModalVAE
from motrust.data.integration import TaskData, top_prevalent_features
from motrust.data.output import write_common_output



MODALITIES = ("rna", "atac", "adt")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--domain", choices=("benchmark", "cortex"), default="benchmark")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dim-c", type=int, default=32)
    parser.add_argument("--dim-u", type=int, default=4)
    parser.add_argument("--hidden-dims", default="256,128")
    parser.add_argument("--rna-features", type=int, default=4000)
    parser.add_argument("--atac-features", type=int, default=10000)
    parser.add_argument("--adt-features", type=int, default=256)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--lambda-kl", type=float, default=0.001)
    parser.add_argument("--lambda-batch-adv", type=float, default=0.05)
    parser.add_argument("--lambda-modality-adv", type=float, default=0.10)
    parser.add_argument("--adversarial-warmup-epochs", type=int, default=5)
    parser.add_argument("--lambda-geometry", type=float, default=0.10)
    parser.add_argument("--geometry-warmup-epochs", type=int, default=5)
    parser.add_argument("--geometry-max-cells", type=int, default=64)
    parser.add_argument("--geometry-max-features", type=int, default=256)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--lambda-ema", type=float, default=0.02)
    parser.add_argument("--freeze-teacher", action="store_true")
    parser.add_argument("--lambda-bridge-align", type=float, default=1.0)
    parser.add_argument("--lambda-reference-align", type=float, default=0.0)
    parser.add_argument("--reference-align-warmup-epochs", type=int, default=5)
    parser.add_argument("--reference-align-max-cells", type=int, default=64)
    parser.add_argument(
        "--reference-align-keep-fraction",
        type=float,
        default=1.0,
        help="Fraction of lowest-distance cells retained in each Chamfer direction",
    )
    parser.add_argument(
        "--reference-align-direction",
        choices=("symmetric", "target_only"),
        default="symmetric",
    )
    parser.add_argument("--spectral-geometry-root", type=Path)
    parser.add_argument("--lambda-spectral-geometry", type=float, default=0.0)
    parser.add_argument("--spectral-geometry-warmup-epochs", type=int, default=5)
    parser.add_argument("--spectral-geometry-max-cells", type=int, default=128)
    parser.add_argument("--semantic-anchor-root", type=Path)
    parser.add_argument("--lambda-semantic-anchor", type=float, default=0.0)
    parser.add_argument("--semantic-anchor-warmup-epochs", type=int, default=5)
    parser.add_argument("--lambda-variance", type=float, default=0.2)
    parser.add_argument("--variance-target", type=float, default=0.5)
    parser.add_argument("--lambda-gate-prior", type=float, default=0.02)
    parser.add_argument("--gate-prior-mode", choices=("point", "mean"), default="point")
    parser.add_argument("--lambda-gate-monotonic", type=float, default=0.0)
    parser.add_argument("--gate-corruption-rate", type=float, default=0.35)
    parser.add_argument("--gate-monotonic-margin", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-cells", type=int, default=0, help="Smoke-test only")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")
    parser.add_argument("--embedding-source", choices=("student", "ema_teacher"), default="student")
    parser.add_argument("--deterministic-alignment", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-gated-poe", action="store_true")
    parser.add_argument("--no-shared-backbone", action="store_true")
    parser.add_argument("--no-geometry", action="store_true")
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _select_and_prepare(
    matrices: dict[str, sparse.csr_matrix],
    masks: dict[str, np.ndarray],
    limits: dict[str, int],
) -> tuple[dict[str, sparse.csr_matrix], dict[str, np.ndarray]]:
    prepared: dict[str, sparse.csr_matrix] = {}
    selected: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        if modality not in matrices:
            continue
        observed = masks[modality] > 0
        source = matrices[modality]
        feature_index = top_prevalent_features(source[observed], limits[modality])
        value = source[:, feature_index].astype(np.float32).tocsr()
        value.sort_indices()
        if modality == "atac":
            value.data[:] = 1.0
        else:
            library = np.asarray(value.sum(axis=1)).ravel().astype(np.float32)
            positive = library[observed & (library > 0)]
            target = float(np.median(positive)) if len(positive) else 1.0
            scale = np.divide(
                target,
                library,
                out=np.zeros_like(library, dtype=np.float32),
                where=library > 0,
            )
            value = sparse.diags(scale).dot(value).tocsr()
            value.data = np.log1p(value.data)
        prepared[modality] = value
        selected[modality] = feature_index
    return prepared, selected


def _dense_rows(matrix: sparse.csr_matrix, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(matrix[rows].toarray()).to(device=device, dtype=torch.float32)


def _observed_view(outputs: dict, modality: str, mask: torch.Tensor, dim_c: int) -> torch.Tensor:
    key = {"rna": "rna_mu_enc", "atac": "atac_mu", "adt": "adt_mu_enc"}[modality]
    return outputs[key][mask.bool(), :dim_c]


def _input_geometry_loss(
    x: torch.Tensor,
    latent: torch.Tensor,
    max_cells: int,
    max_features: int,
) -> torch.Tensor:
    if len(x) < 3:
        return latent.sum() * 0.0
    if len(x) > max_cells:
        choose = torch.linspace(0, len(x) - 1, max_cells, device=x.device).round().long().unique()
        x, latent = x[choose], latent[choose]
    if x.shape[1] > max_features:
        variance = x.var(dim=0, unbiased=False)
        columns = torch.topk(variance, k=max_features, largest=True).indices
        x = x[:, columns]
    x = F.normalize(x.float(), dim=1)
    latent = F.normalize(latent.float(), dim=1)
    source = torch.cdist(x, x)
    target = torch.cdist(latent, latent)
    diagonal = torch.eye(len(x), dtype=torch.bool, device=x.device)
    source_scale = source[~diagonal].median().detach().clamp_min(1e-4)
    target_scale = target[~diagonal].median().detach().clamp_min(1e-4)
    return F.smooth_l1_loss(target / target_scale, source / source_scale)


def _bridge_alignment_loss(
    outputs: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    dim_c: int,
) -> torch.Tensor:
    views = {
        "rna": outputs["rna_mu_enc"][:, :dim_c],
        "atac": outputs["atac_mu"][:, :dim_c],
    }
    if outputs.get("adt_mu_enc") is not None:
        views["adt"] = outputs["adt_mu_enc"][:, :dim_c]
    terms = []
    names = list(views)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            paired = masks[left_name].bool() & masks[right_name].bool()
            if paired.sum() < 2:
                continue
            left = views[left_name][paired]
            right = views[right_name][paired]
            cosine = 1.0 - F.cosine_similarity(left, right, dim=1).mean()
            direction = F.smooth_l1_loss(F.normalize(left, dim=1), F.normalize(right, dim=1))
            terms.append(cosine + direction)
    if terms:
        return torch.stack(terms).mean()
    return outputs["c"].sum() * 0.0


def _reference_chamfer_loss(
    latent: torch.Tensor,
    batch_codes: torch.Tensor,
    reference_codes: torch.Tensor,
    max_cells: int,
    direction: str,
    keep_fraction: float,
) -> torch.Tensor:
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("reference_align_keep_fraction must be in (0, 1]")

    def trimmed_mean(values: torch.Tensor) -> torch.Tensor:
        keep = max(1, int(math.ceil(len(values) * keep_fraction)))
        return values.topk(keep, largest=False).values.mean()

    reference_mask = torch.isin(batch_codes, reference_codes)
    if reference_mask.sum() < 2:
        return latent.sum() * 0.0
    reference = F.normalize(latent[reference_mask], dim=1)
    if len(reference) > max_cells:
        choose = torch.linspace(0, len(reference) - 1, max_cells, device=latent.device).round().long().unique()
        reference = reference[choose]
    terms = []
    for code in torch.unique(batch_codes[~reference_mask]):
        target = F.normalize(latent[batch_codes == code], dim=1)
        if len(target) < 2:
            continue
        if len(target) > max_cells:
            choose = torch.linspace(0, len(target) - 1, max_cells, device=latent.device).round().long().unique()
            target = target[choose]
        distances = torch.cdist(target, reference).square()
        target_to_reference = trimmed_mean(distances.min(dim=1).values)
        if direction == "target_only":
            terms.append(target_to_reference)
        else:
            reference_to_target = trimmed_mean(distances.min(dim=0).values)
            terms.append(0.5 * (target_to_reference + reference_to_target))
    if terms:
        return torch.stack(terms).mean()
    return latent.sum() * 0.0


def _variance_floor_loss(latents: list[torch.Tensor], target: float) -> torch.Tensor:
    terms = []
    for latent in latents:
        if len(latent) < 3:
            continue
        standard_deviation = torch.sqrt(latent.var(dim=0, unbiased=False) + 1e-4)
        terms.append(F.relu(target - standard_deviation).mean())
    if terms:
        return torch.stack(terms).mean()
    return latents[0].sum() * 0.0


def _corrupted_gate_weight(
    model: ImprovedMultiModalVAE,
    modality: str,
    values: torch.Tensor,
    corruption_rate: float,
) -> torch.Tensor:
    keep = torch.rand_like(values).ge(corruption_rate).to(values.dtype)
    corrupted = values * keep
    encoder = {
        "rna": model.rna_encoder,
        "atac": model.atac_encoder,
        "adt": model.adt_encoder,
    }[modality]
    mean, logvar = encoder(corrupted)
    return model._gate_weight(modality, mean, logvar)


@torch.no_grad()
def _ema_update(teacher: ImprovedMultiModalVAE, student: ImprovedMultiModalVAE, decay: float) -> None:
    for target, source in zip(teacher.parameters(), student.parameters()):
        target.mul_(decay).add_(source, alpha=1.0 - decay)
    for target, source in zip(teacher.buffers(), student.buffers()):
        target.copy_(source)


def _model_state(
    model: ImprovedMultiModalVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    selected: dict[str, np.ndarray],
    teacher: ImprovedMultiModalVAE | None = None,
    batch_discriminator: BatchDiscriminator | None = None,
    modality_discriminator: BatchDiscriminator | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
    history: list[dict[str, float | int]] | None = None,
) -> dict:
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "arguments": vars(args),
        "selected_features": {name: values.tolist() for name, values in selected.items()},
        "history": history or [],
    }
    if teacher is not None:
        state["teacher"] = teacher.state_dict()
    if batch_discriminator is not None:
        state["batch_discriminator"] = batch_discriminator.state_dict()
    if modality_discriminator is not None:
        state["modality_discriminator"] = modality_discriminator.state_dict()
    if disc_optimizer is not None:
        state["disc_optimizer"] = disc_optimizer.state_dict()
    return state


def main(argv=None) -> None:
    args = parse_args(argv)
    data = TaskData(args.domain)
    set_seed(args.seed)
    started = time.time()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    task = data.load_task_definition(args.task_id)
    output = Path(args.output_dir) if args.output_dir else data.integration_root / "neural" / args.task_id
    output.mkdir(parents=True, exist_ok=True)
    matrices, masks, metadata = data.align_modalities_to_task(args.task_id)
    keep = None
    if args.max_cells > 0 and args.max_cells < len(metadata):
        keep = np.sort(np.random.default_rng(args.seed).choice(len(metadata), args.max_cells, replace=False))
        metadata = metadata.iloc[keep].reset_index(drop=True)
        matrices = {name: value[keep] for name, value in matrices.items()}
        masks = {name: value[keep] for name, value in masks.items()}

    limits = {"rna": args.rna_features, "atac": args.atac_features, "adt": args.adt_features}
    matrices, selected = _select_and_prepare(matrices, masks, limits)
    n_cells = len(metadata)
    spectral_geometry = None
    if args.spectral_geometry_root is not None:
        spectral_dir = args.spectral_geometry_root.resolve() / args.task_id
        spectral_metadata = pd.read_csv(spectral_dir / "metadata.csv", dtype=str)
        full_metadata = data.load_task_metadata(args.task_id, evaluation=False)
        if not spectral_metadata["cell_id"].equals(full_metadata["cell_id"]):
            raise ValueError(f"Spectral geometry cell order differs for {args.task_id}")
        spectral_geometry = np.load(spectral_dir / "embedding.npy", allow_pickle=False).astype(np.float32)
        if keep is not None:
            spectral_geometry = spectral_geometry[keep]
        if len(spectral_geometry) != n_cells:
            raise ValueError(f"Spectral geometry row count differs for {args.task_id}")
    semantic_anchor = None
    semantic_reliability = None
    if args.semantic_anchor_root is not None:
        semantic_dir = args.semantic_anchor_root.resolve() / args.task_id
        semantic_metadata = pd.read_csv(semantic_dir / "metadata.csv", dtype=str)
        full_metadata = data.load_task_metadata(args.task_id, evaluation=False)
        if not semantic_metadata["cell_id"].equals(full_metadata["cell_id"]):
            raise ValueError(f"Semantic anchor cell order differs for {args.task_id}")
        semantic_anchor = np.load(semantic_dir / "embedding.npy", allow_pickle=False).astype(np.float32)
        semantic_reliability = np.load(
            semantic_dir / "semantic_reliability.npy", allow_pickle=False
        ).astype(np.float32)
        if keep is not None:
            semantic_anchor = semantic_anchor[keep]
            semantic_reliability = semantic_reliability[keep]
        if len(semantic_anchor) != n_cells or len(semantic_reliability) != n_cells:
            raise ValueError(f"Semantic anchor row count differs for {args.task_id}")
    use_adt = "adt" in matrices
    rna_dim = matrices["rna"].shape[1] if "rna" in matrices else 1
    atac_dim = matrices["atac"].shape[1] if "atac" in matrices else 1
    adt_dim = matrices["adt"].shape[1] if use_adt else None
    if "rna" not in matrices:
        matrices["rna"] = sparse.csr_matrix((n_cells, 1), dtype=np.float32)
        masks["rna"] = np.zeros(n_cells, dtype=np.float32)
    if "atac" not in matrices:
        matrices["atac"] = sparse.csr_matrix((n_cells, 1), dtype=np.float32)
        masks["atac"] = np.zeros(n_cells, dtype=np.float32)

    batch_codes, batch_names = pd.factorize(metadata["instance_batch"].astype(str), sort=True)
    observed_counts = metadata["modality"].astype(str).str.count(r"\+") + 1
    reference_batches = metadata.loc[
        observed_counts == observed_counts.max(), "instance_batch"
    ].astype(str).unique()
    reference_code_values = np.flatnonzero(np.isin(batch_names.astype(str), reference_batches)).astype(np.int64)
    reference_codes = torch.from_numpy(reference_code_values).long().to(device)
    hidden_dims = [int(value) for value in args.hidden_dims.split(",") if value.strip()]
    model = ImprovedMultiModalVAE(
        rna_dim=rna_dim,
        atac_dim=atac_dim,
        adt_dim=adt_dim,
        latent_dim=args.dim_c + args.dim_u,
        dim_c=args.dim_c,
        dim_u=args.dim_u,
        hidden_dims=hidden_dims,
        dropout=0.1,
        beta_c=1.0,
        beta_u=0.5,
        use_poe=True,
        rna_distribution="poisson",
        use_adt=use_adt,
        adt_distribution="poisson",
        use_gated_poe=not args.no_gated_poe,
        use_shared_backbone=not args.no_shared_backbone,
    ).to(device)
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    encoder_parameters = []
    decoder_parameters = []
    for name, parameter in model.named_parameters():
        if "encoder" in name or "modality_gates" in name:
            encoder_parameters.append(parameter)
        else:
            decoder_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.learning_rate * args.encoder_lr_scale},
            {"params": decoder_parameters, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    batch_discriminator = BatchDiscriminator(args.dim_c, len(batch_names), [64, 32], 0.1).to(device)
    modality_count = sum(name in matrices and masks[name].sum() > 0 for name in MODALITIES)
    modality_discriminator = BatchDiscriminator(args.dim_c, modality_count, [64, 32], 0.1).to(device)
    disc_optimizer = torch.optim.AdamW(
        list(batch_discriminator.parameters()) + list(modality_discriminator.parameters()), lr=1e-4
    )
    modality_to_code = {
        name: index for index, name in enumerate(name for name in MODALITIES if name in matrices and masks[name].sum() > 0)
    }

    history: list[dict[str, float | int]] = []
    start_epoch = 0
    if not args.no_resume:
        checkpoints = [args.resume_from] if args.resume_from else sorted(
            output.glob("checkpoint_epoch_*.pt"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        if checkpoints:
            checkpoint_path = checkpoints[-1]
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            checkpoint_arguments = checkpoint.get("arguments", {})
            current_configuration = {
                "dim_c": args.dim_c,
                "dim_u": args.dim_u,
                "hidden_dims": args.hidden_dims,
                "lambda_kl": args.lambda_kl,
                "lambda_bridge_align": args.lambda_bridge_align,
                "lambda_variance": args.lambda_variance,
                "lambda_gate_prior": args.lambda_gate_prior,
            }
            mismatched = [
                name for name, value in current_configuration.items()
                if name in checkpoint_arguments and checkpoint_arguments[name] != value
            ]
            if mismatched:
                raise RuntimeError(f"Checkpoint configuration mismatch: {mismatched}")
            model.load_state_dict(checkpoint["model"])
            if not args.reset_optimizer_on_resume:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "teacher" in checkpoint:
                teacher.load_state_dict(checkpoint["teacher"])
            else:
                teacher.load_state_dict(model.state_dict())
            if "batch_discriminator" in checkpoint:
                batch_discriminator.load_state_dict(checkpoint["batch_discriminator"])
            if "modality_discriminator" in checkpoint:
                modality_discriminator.load_state_dict(checkpoint["modality_discriminator"])
            if "disc_optimizer" in checkpoint and not args.reset_optimizer_on_resume:
                disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
            history = checkpoint.get("history", [])
            start_epoch = int(checkpoint["epoch"])
            print(f"Resuming {args.task_id} from epoch {start_epoch}: {checkpoint_path}", flush=True)
    indices = np.arange(n_cells)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        rng.shuffle(indices)
        totals: dict[str, float] = {}
        valid_steps = 0
        kl_scale = min(1.0, epoch / max(1, args.kl_warmup_epochs))
        adv_scale = min(1.0, epoch / max(1, args.adversarial_warmup_epochs))
        for start in range(0, n_cells, args.batch_size):
            rows = indices[start : start + args.batch_size]
            if len(rows) < 2:
                continue
            rna = _dense_rows(matrices["rna"], rows, device)
            atac = _dense_rows(matrices["atac"], rows, device)
            adt = _dense_rows(matrices["adt"], rows, device) if use_adt else None
            row_masks = {
                name: torch.from_numpy(masks.get(name, np.zeros(n_cells, dtype=np.float32))[rows]).to(device)
                for name in MODALITIES
            }
            batches = torch.from_numpy(batch_codes[rows].copy()).long().to(device)
            outputs = model(
                rna,
                atac,
                adt_x=adt,
                rna_mask=row_masks["rna"],
                atac_mask=row_masks["atac"],
                adt_mask=row_masks["adt"] if use_adt else None,
            )
            components = model.compute_loss(
                rna,
                atac,
                outputs,
                adt_x=adt,
                rna_loss_mask=row_masks["rna"],
                atac_loss_mask=row_masks["atac"],
                adt_loss_mask=row_masks["adt"] if use_adt else None,
            )
            reconstruction = components["rna_recon_loss"] / max(1, rna_dim)
            reconstruction = reconstruction + components["atac_recon_loss"] / max(1, atac_dim)
            if use_adt:
                reconstruction = reconstruction + components["adt_recon_loss"] / max(1, adt_dim or 1)
            loss = reconstruction + args.lambda_kl * kl_scale * components["kl_loss"]
            shared_latent = (
                outputs["z_mu"][:, : args.dim_c] if args.deterministic_alignment else outputs["c"]
            )

            views, modality_labels = [], []
            geometry_terms = []
            for modality, tensor in (("rna", rna), ("atac", atac), ("adt", adt)):
                if modality not in modality_to_code or tensor is None:
                    continue
                observed = row_masks[modality].bool()
                if observed.sum() < 2:
                    continue
                latent = _observed_view(outputs, modality, row_masks[modality], args.dim_c)
                views.append(latent)
                modality_labels.append(
                    torch.full((len(latent),), modality_to_code[modality], dtype=torch.long, device=device)
                )
                if not args.no_geometry and observed.sum() >= 3:
                    geometry_terms.append(
                        _input_geometry_loss(
                            tensor[observed], latent, args.geometry_max_cells, args.geometry_max_features
                        )
                    )

            bridge_align = _bridge_alignment_loss(outputs, row_masks, args.dim_c)
            variance_views = [shared_latent]
            variance_views.extend(views)
            variance = _variance_floor_loss(variance_views, args.variance_target)
            gate_terms = []
            gate_monotonic_terms = []
            if model.modality_gates is not None:
                gate_inputs = {
                    "rna": (outputs["rna_mu_enc"], outputs["rna_logvar_enc"]),
                    "atac": (outputs["atac_mu"], outputs["atac_logvar"]),
                    "adt": (outputs.get("adt_mu_enc"), outputs.get("adt_logvar_enc")),
                }
                for modality, (mean, logvar) in gate_inputs.items():
                    observed = row_masks[modality].bool()
                    if modality not in model.modality_gates or mean is None or observed.sum() == 0:
                        continue
                    weight = model._gate_weight(modality, mean, logvar)
                    if args.gate_prior_mode == "mean":
                        gate_terms.append((weight[observed].mean() - 1.0).square())
                    else:
                        gate_terms.append((weight[observed] - 1.0).square().mean())
                    if args.lambda_gate_monotonic > 0 and observed.sum() >= 2:
                        damaged_weight = _corrupted_gate_weight(
                            model,
                            modality,
                            {"rna": rna, "atac": atac, "adt": adt}[modality][observed],
                            args.gate_corruption_rate,
                        )
                        gate_monotonic_terms.append(
                            F.relu(args.gate_monotonic_margin - weight[observed] + damaged_weight).mean()
                        )
            gate_prior = torch.stack(gate_terms).mean() if gate_terms else loss * 0.0
            gate_monotonic = (
                torch.stack(gate_monotonic_terms).mean() if gate_monotonic_terms else loss * 0.0
            )
            loss = loss + args.lambda_bridge_align * bridge_align
            loss = loss + args.lambda_variance * variance
            loss = loss + args.lambda_gate_prior * gate_prior
            loss = loss + args.lambda_gate_monotonic * gate_monotonic

            reference_align = _reference_chamfer_loss(
                shared_latent,
                batches,
                reference_codes,
                args.reference_align_max_cells,
                args.reference_align_direction,
                args.reference_align_keep_fraction,
            )
            reference_scale = min(1.0, epoch / max(1, args.reference_align_warmup_epochs))
            loss = loss + args.lambda_reference_align * reference_scale * reference_align

            if spectral_geometry is not None and args.lambda_spectral_geometry > 0:
                spectral_rows = torch.from_numpy(spectral_geometry[rows]).to(device)
                spectral_geometry_loss = _input_geometry_loss(
                    spectral_rows,
                    outputs["z_mu"][:, : args.dim_c],
                    args.spectral_geometry_max_cells,
                    spectral_rows.shape[1],
                )
            else:
                spectral_geometry_loss = loss * 0.0
            spectral_scale = min(1.0, epoch / max(1, args.spectral_geometry_warmup_epochs))
            loss = loss + args.lambda_spectral_geometry * spectral_scale * spectral_geometry_loss

            if semantic_anchor is not None and args.lambda_semantic_anchor > 0:
                anchor_rows = torch.from_numpy(semantic_anchor[rows]).to(device)
                anchor_reliability = torch.from_numpy(semantic_reliability[rows]).to(device)
                student_direction = F.normalize(outputs["z_mu"][:, : args.dim_c], dim=1)
                anchor_direction = F.normalize(anchor_rows, dim=1)
                point_error = F.smooth_l1_loss(
                    student_direction,
                    anchor_direction,
                    reduction="none",
                ).mean(dim=1)
                semantic_anchor_loss = (
                    point_error * anchor_reliability
                ).sum() / anchor_reliability.sum().clamp_min(1e-6)
            else:
                semantic_anchor_loss = loss * 0.0
            semantic_scale = min(1.0, epoch / max(1, args.semantic_anchor_warmup_epochs))
            loss = loss + args.lambda_semantic_anchor * semantic_scale * semantic_anchor_loss

            batch_disc = batch_discriminator.compute_loss(shared_latent.detach(), batches)
            if len(views) > 1:
                joined_views = torch.cat(views, dim=0)
                joined_modality = torch.cat(modality_labels, dim=0)
                modality_disc = modality_discriminator.compute_loss(joined_views.detach(), joined_modality)
            else:
                joined_views = None
                joined_modality = None
                modality_disc = loss * 0.0
            disc_loss = batch_disc + modality_disc
            disc_optimizer.zero_grad(set_to_none=True)
            disc_loss.backward()
            disc_optimizer.step()

            batch_adv = batch_discriminator.compute_loss(shared_latent, batches)
            modality_adv = (
                modality_discriminator.compute_loss(joined_views, joined_modality)
                if joined_views is not None
                else loss * 0.0
            )
            loss = loss - adv_scale * (
                args.lambda_batch_adv * batch_adv + args.lambda_modality_adv * modality_adv
            )
            geometry = torch.stack(geometry_terms).mean() if geometry_terms else loss * 0.0
            if not args.no_geometry and epoch >= args.geometry_warmup_epochs:
                loss = loss + args.lambda_geometry * geometry

            with torch.no_grad():
                teacher_outputs = teacher(
                    rna,
                    atac,
                    adt_x=adt,
                    rna_mask=row_masks["rna"],
                    atac_mask=row_masks["atac"],
                    adt_mask=row_masks["adt"] if use_adt else None,
                )
            teacher_shared = (
                teacher_outputs["z_mu"][:, : args.dim_c]
                if args.deterministic_alignment
                else teacher_outputs["c"]
            )
            ema_loss = F.smooth_l1_loss(shared_latent, teacher_shared)
            loss = loss + args.lambda_ema * ema_loss

            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if not args.freeze_teacher:
                _ema_update(teacher, model, args.ema_decay)

            valid_steps += 1
            current = {
                "loss": loss,
                "reconstruction": reconstruction,
                "kl": components["kl_loss"],
                "batch_adv": batch_adv,
                "modality_adv": modality_adv,
                "geometry": geometry,
                "ema": ema_loss,
                "bridge_align": bridge_align,
                "variance": variance,
                "gate_prior": gate_prior,
                "gate_monotonic": gate_monotonic,
                "reference_align": reference_align,
                "spectral_geometry": spectral_geometry_loss,
                "semantic_anchor": semantic_anchor_loss,
            }
            for name, value in current.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())

        if valid_steps == 0:
            raise RuntimeError(f"No finite training steps at epoch {epoch}")
        record = {"epoch": epoch, **{name: value / valid_steps for name, value in totals.items()}}
        history.append(record)
        pd.DataFrame(history).to_csv(output / "training_history.partial.csv", index=False)
        print(json.dumps(record), flush=True)
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            torch.save(
                _model_state(
                    model, optimizer, epoch, args, selected, teacher,
                    batch_discriminator, modality_discriminator, disc_optimizer, history,
                ),
                output / f"checkpoint_epoch_{epoch}.pt",
            )

    torch.save(
        _model_state(
            model, optimizer, args.epochs, args, selected, teacher,
            batch_discriminator, modality_discriminator, disc_optimizer, history,
        ),
        output / "checkpoint_final.pt",
    )
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    (output / "selected_features.json").write_text(
        json.dumps({name: values.tolist() for name, values in selected.items()}, indent=2), encoding="utf-8"
    )

    inference_model = teacher if args.embedding_source == "ema_teacher" else model
    inference_model.eval()
    embeddings = []
    gate_rows: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for start in range(0, n_cells, args.batch_size):
            rows = np.arange(start, min(start + args.batch_size, n_cells))
            rna = _dense_rows(matrices["rna"], rows, device)
            atac = _dense_rows(matrices["atac"], rows, device)
            adt = _dense_rows(matrices["adt"], rows, device) if use_adt else None
            row_masks = {
                name: torch.from_numpy(masks.get(name, np.zeros(n_cells, dtype=np.float32))[rows]).to(device)
                for name in MODALITIES
            }
            encoded = inference_model.encode(
                rna,
                atac,
                adt_x=adt,
                rna_mask=row_masks["rna"],
                atac_mask=row_masks["atac"],
                adt_mask=row_masks["adt"] if use_adt else None,
            )
            embeddings.append(encoded[0][:, : args.dim_c].cpu().numpy())
            if inference_model.modality_gates is not None:
                means = {"rna": encoded[2], "atac": encoded[4], "adt": encoded[6]}
                logvars = {"rna": encoded[3], "atac": encoded[5], "adt": encoded[7]}
                for modality in MODALITIES:
                    if modality not in inference_model.modality_gates or means[modality] is None:
                        continue
                    weights = inference_model._gate_weight(
                        modality, means[modality], logvars[modality]
                    ).cpu().numpy()
                    for offset, weight in enumerate(weights):
                        gate_rows.append(
                            {
                                "row_index": int(rows[offset]),
                                "modality": modality,
                                "observed": int(row_masks[modality][offset].item()),
                                "gate_weight": float(weight),
                            }
                        )
    embedding = np.vstack(embeddings)
    pd.DataFrame(gate_rows).to_csv(output / "gate_weights.csv", index=False)

    manifest = {
        "scenario": task["scenario"],
        "replicate": task["replicate"],
        "seed": args.seed,
        "fixed_epoch": args.epochs,
        "checkpoint": "checkpoint_final.pt",
        "device": str(device),
        "torch_version": torch.__version__,
        "scipy_version": scipy.__version__,
        "runtime_seconds": time.time() - started,
        "resumed_from_epoch": start_epoch,
        "training_cell_count": n_cells,
        "smoke_test_subset": bool(args.max_cells > 0),
        "modalities": [name for name in MODALITIES if masks.get(name, np.zeros(1)).sum() > 0],
        "selected_feature_counts": {name: int(len(values)) for name, values in selected.items()},
        "preprocessing": {
            "rna": "top prevalence within frozen feature universe; library-size normalization to observed median; log1p",
            "atac": "top prevalence within frozen feature universe; binary",
            "adt": "all/top prevalence; library-size normalization to observed median; log1p",
        },
        "model": {
            "configuration_id": "mosaic_v2_collapse_resistant",
            "gated_poe": not args.no_gated_poe,
            "shared_backbone": not args.no_shared_backbone,
            "per_cell_modality_masks": True,
            "feature_dimension_normalized_reconstruction": True,
            "batch_adversarial": args.lambda_batch_adv,
            "modality_adversarial": args.lambda_modality_adv,
            "local_geometry": 0.0 if args.no_geometry else args.lambda_geometry,
            "ema_teacher": args.lambda_ema,
            "frozen_teacher": args.freeze_teacher,
            "kl_weight": args.lambda_kl,
            "bridge_alignment": args.lambda_bridge_align,
            "reference_distribution_alignment": args.lambda_reference_align,
            "reference_alignment_direction": args.reference_align_direction,
            "reference_alignment_keep_fraction": args.reference_align_keep_fraction,
            "spectral_geometry_teacher": (
                str(args.spectral_geometry_root.resolve()) if args.spectral_geometry_root else None
            ),
            "spectral_geometry_weight": args.lambda_spectral_geometry,
            "semantic_anchor_teacher": (
                str(args.semantic_anchor_root.resolve()) if args.semantic_anchor_root else None
            ),
            "semantic_anchor_weight": args.lambda_semantic_anchor,
            "reference_batches": reference_batches.tolist(),
            "variance_floor": args.lambda_variance,
            "variance_target": args.variance_target,
            "gate_neutral_prior": args.lambda_gate_prior,
            "gate_prior_mode": args.gate_prior_mode,
            "gate_monotonic": args.lambda_gate_monotonic,
            "gate_corruption_rate": args.gate_corruption_rate,
            "gate_monotonic_margin": args.gate_monotonic_margin,
            "embedding_source": args.embedding_source,
            "deterministic_alignment": args.deterministic_alignment,
        },
    }
    if args.max_cells > 0:
        # Smoke outputs intentionally do not satisfy the full task output contract.
        np.save(output / "embedding.npy", embedding, allow_pickle=False)
        metadata.to_csv(output / "metadata.csv", index=False)
        manifest.update(
            {
                "schema_version": "1.0",
                "method": "MoTRUST",
                "task_id": args.task_id,
                "status": "smoke_test_completed",
                "labels_used_for_training": False,
                "label_selected_checkpoint": False,
            }
        )
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        write_common_output(output, args.task_id, "MoTRUST", embedding, manifest, data=data)
    print(f"Completed MoTRUST {args.task_id}: {embedding.shape} -> {output}")


if __name__ == "__main__":
    main()
