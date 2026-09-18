"""Cross-view consistency objectives for multimodal latent inference."""

from __future__ import annotations

import torch
from torch import nn


def compute_consistency_loss(
    vae,
    rna_x: torch.Tensor,
    atac_x: torch.Tensor,
    adt_x: torch.Tensor | None = None,
    include_adt: bool = False,
) -> torch.Tensor:
    """Penalize disagreement between joint and modality-specific biological latents."""

    use_adt = bool(
        include_adt
        and adt_x is not None
        and getattr(vae, "use_adt", False)
        and getattr(vae, "adt_encoder", None) is not None
    )
    encoded = vae.encode(
        rna_x,
        atac_x,
        adt_x=adt_x if use_adt else None,
        batch_ids=None,
    )
    c_values = [
        encoded[0][:, : vae.dim_c],
        vae.rna_encoder(rna_x)[0][:, : vae.dim_c],
        vae.atac_encoder(atac_x)[0][:, : vae.dim_c],
    ]
    if use_adt:
        c_values.append(vae.adt_encoder(adt_x)[0][:, : vae.dim_c])

    stacked = torch.stack(c_values, dim=0)
    centered = stacked - stacked.mean(dim=0, keepdim=True)
    loss = centered.square().sum() / max(int(rna_x.shape[0]), 1)
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


class ConsistencyLossModule(nn.Module):
    """Weighted module wrapper for :func:`compute_consistency_loss`."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(self, vae, rna_x: torch.Tensor, atac_x: torch.Tensor) -> torch.Tensor:
        return self.weight * compute_consistency_loss(vae, rna_x, atac_x)


def compute_alignment_loss(vae, rna_x: torch.Tensor, atac_x: torch.Tensor) -> torch.Tensor:
    """Backward-compatible alias for the consistency objective."""

    return compute_consistency_loss(vae, rna_x, atac_x)
