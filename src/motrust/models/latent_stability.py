"""Label-free early-geometry and EMA trust-region stabilization."""

from __future__ import annotations

import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .poe import ProductOfExperts


def _subsample(length: int, limit: int, device: torch.device) -> torch.Tensor:
    if limit <= 0 or length <= limit:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, limit, device=device).round().long().unique()


def local_anchor_geometry_loss(
    student: torch.Tensor,
    anchor: torch.Tensor,
    *,
    n_neighbors: int = 15,
    max_cells: int = 96,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Preserve local pairwise distances defined by a frozen early encoder."""

    if student.shape != anchor.shape or student.ndim != 2:
        raise ValueError("student and anchor latents must have the same 2D shape")
    if len(student) < 3:
        return student.sum() * 0.0
    index = _subsample(len(student), max_cells, student.device)
    student = student[index].float()
    anchor = anchor[index].float().detach()
    anchor_distance = torch.cdist(anchor, anchor)
    student_distance = torch.cdist(student, student)
    diagonal = torch.eye(len(index), dtype=torch.bool, device=student.device)
    anchor_distance = anchor_distance.masked_fill(diagonal, float("inf"))
    k = min(max(1, int(n_neighbors)), len(index) - 1)
    edge_index = torch.topk(anchor_distance, k=k, dim=1, largest=False).indices
    row = torch.arange(len(index), device=student.device).unsqueeze(1).expand_as(edge_index)
    target = anchor_distance[row, edge_index]
    current = student_distance[row, edge_index]
    scale = target[target.isfinite()].median().clamp_min(1e-4)
    weights = torch.exp(-target / (scale * max(float(temperature), 1e-4))).detach()
    relative_error = (current - target) / scale
    pointwise = F.smooth_l1_loss(relative_error, torch.zeros_like(relative_error), reduction="none")
    return (weights * pointwise).sum() / weights.sum().clamp_min(1e-8)


def ema_trust_region_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Penalize abrupt coordinate changes relative to a slowly moving teacher."""

    if student.shape != teacher.shape:
        raise ValueError("student and EMA teacher latents must have the same shape")
    return F.smooth_l1_loss(student, teacher.detach())


def latent_variance_preservation_loss(student: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    student_std = student.float().std(dim=0, unbiased=False).clamp_min(1e-4)
    anchor_std = anchor.float().detach().std(dim=0, unbiased=False).clamp_min(1e-4)
    return torch.square(torch.log(student_std) - torch.log(anchor_std)).mean()


def anchor_chart_trust_region_loss(student: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Bound cumulative movement in the anchor's standardized latent chart."""

    if student.shape != anchor.shape:
        raise ValueError("student and anchor latents must have the same shape")
    anchor = anchor.float().detach()
    student = student.float()
    anchor_centered = anchor - anchor.mean(dim=0, keepdim=True)
    student_centered = student - student.mean(dim=0, keepdim=True)
    anchor_scale = anchor.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    return F.smooth_l1_loss(student_centered / anchor_scale, anchor_centered / anchor_scale)


def anchor_affinity_distillation_loss(
    student: torch.Tensor,
    anchor: torch.Tensor,
    *,
    max_cells: int = 96,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Preserve the anchor's full within-batch neighborhood ranking."""

    if student.shape != anchor.shape or student.ndim != 2:
        raise ValueError("student and anchor latents must have the same 2D shape")
    if len(student) < 3:
        return student.sum() * 0.0
    index = _subsample(len(student), max_cells, student.device)
    student = student[index].float()
    anchor = anchor[index].float().detach()
    anchor_distance = torch.cdist(anchor, anchor)
    student_distance = torch.cdist(student, student)
    diagonal = torch.eye(len(index), dtype=torch.bool, device=student.device)
    anchor_scale = anchor_distance[~diagonal].median().clamp_min(1e-4)
    student_scale = student_distance[~diagonal].median().detach().clamp_min(1e-4)
    tau = max(float(temperature), 1e-4)
    anchor_logits = (-anchor_distance / (anchor_scale * tau)).masked_fill(diagonal, -1e4)
    student_logits = (-student_distance / (student_scale * tau)).masked_fill(diagonal, -1e4)
    anchor_probability = torch.softmax(anchor_logits, dim=1).detach()
    return F.kl_div(
        torch.log_softmax(student_logits, dim=1),
        anchor_probability,
        reduction="batchmean",
    )


class LatentStabilityController:
    """Hold fixed early encoders and EMA encoders for RNA and ATAC."""

    def __init__(
        self,
        rna_encoder: nn.Module,
        atac_encoder: nn.Module,
        *,
        dim_c: int,
        ema_decay: float = 0.999,
        n_neighbors: int = 15,
        max_cells: int = 96,
        temperature: float = 0.2,
        modality_gates: nn.ModuleDict | None = None,
        use_gated_poe_views: bool = False,
    ) -> None:
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")
        self.student_rna = rna_encoder
        self.student_atac = atac_encoder
        self.anchor_rna = copy.deepcopy(rna_encoder).eval()
        self.anchor_atac = copy.deepcopy(atac_encoder).eval()
        self.ema_rna = copy.deepcopy(rna_encoder).eval()
        self.ema_atac = copy.deepcopy(atac_encoder).eval()
        self.student_gates = modality_gates
        self.anchor_gates = copy.deepcopy(modality_gates).eval() if modality_gates is not None else None
        self.ema_gates = copy.deepcopy(modality_gates).eval() if modality_gates is not None else None
        frozen_modules = [self.anchor_rna, self.anchor_atac, self.ema_rna, self.ema_atac]
        if self.anchor_gates is not None:
            frozen_modules.extend([self.anchor_gates, self.ema_gates])
        for module in frozen_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.dim_c = int(dim_c)
        self.ema_decay = float(ema_decay)
        self.n_neighbors = int(n_neighbors)
        self.max_cells = int(max_cells)
        self.temperature = float(temperature)
        self.use_gated_poe_views = bool(use_gated_poe_views and modality_gates is not None)

    @staticmethod
    def _mean(encoder: nn.Module, x: torch.Tensor, dim_c: int) -> torch.Tensor:
        mu, _logvar = encoder(x)
        return mu[:, :dim_c]

    @staticmethod
    def _gated_view_mean(
        encoder: nn.Module,
        gates: nn.ModuleDict,
        modality: str,
        x: torch.Tensor,
        dim_c: int,
    ) -> torch.Tensor:
        mu, logvar = encoder(x)
        gate_input = torch.cat([mu.detach(), logvar.detach()], dim=1)
        weight = 2.0 * torch.sigmoid(gates[modality](gate_input).view(-1))
        fused_mu, _ = ProductOfExperts.poe_with_prior(
            [mu], [logvar], weights=[weight]
        )
        return fused_mu[:, :dim_c]

    def modality_loss(
        self,
        modality: str,
        x: torch.Tensor,
        student_c: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if modality == "rna":
            anchor_encoder, ema_encoder = self.anchor_rna, self.ema_rna
        elif modality == "atac":
            anchor_encoder, ema_encoder = self.anchor_atac, self.ema_atac
        else:
            raise ValueError(f"Unknown modality: {modality}")
        with torch.no_grad():
            if self.use_gated_poe_views:
                anchor_c = self._gated_view_mean(
                    anchor_encoder, self.anchor_gates, modality, x, self.dim_c
                )
                ema_c = self._gated_view_mean(
                    ema_encoder, self.ema_gates, modality, x, self.dim_c
                )
            else:
                anchor_c = self._mean(anchor_encoder, x, self.dim_c)
                ema_c = self._mean(ema_encoder, x, self.dim_c)
        return {
            "geometry": local_anchor_geometry_loss(
                student_c,
                anchor_c,
                n_neighbors=self.n_neighbors,
                max_cells=self.max_cells,
                temperature=self.temperature,
            ),
            "ema": ema_trust_region_loss(student_c, ema_c),
            "variance": latent_variance_preservation_loss(student_c, anchor_c),
            "anchor": anchor_chart_trust_region_loss(student_c, anchor_c),
            "topology": anchor_affinity_distillation_loss(
                student_c,
                anchor_c,
                max_cells=self.max_cells,
                temperature=self.temperature,
            ),
        }

    @torch.no_grad()
    def update(self) -> None:
        for ema, student in (
            (self.ema_rna, self.student_rna),
            (self.ema_atac, self.student_atac),
        ):
            for ema_parameter, student_parameter in zip(ema.parameters(), student.parameters()):
                ema_parameter.mul_(self.ema_decay).add_(student_parameter, alpha=1.0 - self.ema_decay)
            for ema_buffer, student_buffer in zip(ema.buffers(), student.buffers()):
                if torch.is_floating_point(ema_buffer):
                    ema_buffer.mul_(self.ema_decay).add_(student_buffer, alpha=1.0 - self.ema_decay)
                else:
                    ema_buffer.copy_(student_buffer)
        if self.ema_gates is not None and self.student_gates is not None:
            for ema_parameter, student_parameter in zip(
                self.ema_gates.parameters(), self.student_gates.parameters()
            ):
                ema_parameter.mul_(self.ema_decay).add_(
                    student_parameter, alpha=1.0 - self.ema_decay
                )


class AdaptiveStabilityConstraint:
    """Epoch-level dual control for modality-specific latent constraints."""

    COMPONENTS = ("geometry", "ema", "variance", "anchor", "topology")
    MODALITIES = ("rna", "atac")

    def __init__(
        self,
        *,
        geometry_tolerance: float = 0.05,
        ema_tolerance: float = 0.02,
        variance_tolerance: float = 0.10,
        anchor_tolerance: float = 0.02,
        topology_tolerance: float = 0.02,
        task_fraction: float = 0.005,
        augmentation: float = 0.0,
        dual_lr: float = 0.5,
        dual_init: float = 1.0,
        dual_max: float = 10.0,
        rna_weight: float = 1.0,
        atac_weight: float = 2.0,
    ) -> None:
        self.tolerances = {
            "geometry": max(float(geometry_tolerance), 1e-8),
            "ema": max(float(ema_tolerance), 1e-8),
            "variance": max(float(variance_tolerance), 1e-8),
            "anchor": max(float(anchor_tolerance), 1e-8),
            "topology": max(float(topology_tolerance), 1e-8),
        }
        self.task_fraction = max(float(task_fraction), 0.0)
        self.augmentation = max(float(augmentation), 0.0)
        self.dual_lr = max(float(dual_lr), 0.0)
        self.dual_max = max(float(dual_max), 0.0)
        initial = min(max(float(dual_init), 0.0), self.dual_max)
        self.duals = {
            modality: {component: initial for component in self.COMPONENTS}
            for modality in self.MODALITIES
        }
        self.modality_weights = {
            "rna": max(float(rna_weight), 0.0),
            "atac": max(float(atac_weight), 0.0),
        }
        self._ratio_sums: defaultdict[tuple[str, str], float] = defaultdict(float)
        self._observations = 0

    def penalty(
        self,
        rna_losses: dict[str, torch.Tensor],
        atac_losses: dict[str, torch.Tensor],
        primary_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        losses = {"rna": rna_losses, "atac": atac_losses}
        weighted_penalty = primary_loss.new_tensor(0.0)
        weight_sum = 0.0
        diagnostics: dict[str, float] = {}
        for modality in self.MODALITIES:
            modality_weight = self.modality_weights[modality]
            for component in self.COMPONENTS:
                ratio = losses[modality][component] / self.tolerances[component]
                violation = torch.relu(ratio - 1.0)
                weighted_penalty = (
                    weighted_penalty
                    + modality_weight
                    * (
                        self.duals[modality][component] * violation
                        + 0.5 * self.augmentation * violation.square()
                    )
                )
                weight_sum += modality_weight
                ratio_value = float(ratio.detach().item())
                self._ratio_sums[(modality, component)] += ratio_value
                diagnostics[f"{modality}_{component}_ratio"] = ratio_value
                diagnostics[f"{modality}_{component}_dual"] = self.duals[modality][component]
        self._observations += 1
        normalized_penalty = weighted_penalty / max(weight_sum, 1e-8)
        task_scale = self.task_fraction * primary_loss.detach().abs().clamp_min(1e-8)
        contribution = task_scale * normalized_penalty
        diagnostics["constraint_penalty"] = float(normalized_penalty.detach().item())
        diagnostics["constraint_contribution"] = float(contribution.detach().item())
        return contribution, diagnostics

    def step_epoch(self) -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        count = max(self._observations, 1)
        for modality in self.MODALITIES:
            for component in self.COMPONENTS:
                ratio = self._ratio_sums[(modality, component)] / count
                updated = self.duals[modality][component] + self.dual_lr * (ratio - 1.0)
                self.duals[modality][component] = min(max(updated, 0.0), self.dual_max)
                diagnostics[f"{modality}_{component}_ratio"] = ratio
                diagnostics[f"{modality}_{component}_dual"] = self.duals[modality][component]
        self._ratio_sums.clear()
        self._observations = 0
        return diagnostics

    def state_dict(self) -> dict:
        return {
            "tolerances": dict(self.tolerances),
            "task_fraction": self.task_fraction,
            "augmentation": self.augmentation,
            "dual_lr": self.dual_lr,
            "dual_max": self.dual_max,
            "duals": {key: dict(value) for key, value in self.duals.items()},
            "modality_weights": dict(self.modality_weights),
        }
