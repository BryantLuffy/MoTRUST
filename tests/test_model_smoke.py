"""Small CPU tests for the MoTRUST representation backbone."""

from __future__ import annotations

import torch

from motrust.models import MoTRUSTVAE, UncertaintyAwareUnbalancedOT


def test_gated_poe_supports_missing_modalities() -> None:
    torch.manual_seed(3)
    model = MoTRUSTVAE(
        rna_dim=12,
        atac_dim=18,
        latent_dim=6,
        dim_c=4,
        dim_u=2,
        hidden_dims=[10],
        dropout=0.0,
        use_gated_poe=True,
    ).eval()
    rna = torch.rand(5, 12)
    atac = torch.rand(5, 18)
    rna_mask = torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32)
    atac_mask = 1.0 - rna_mask
    mu, logvar, *_ = model.encode(
        rna,
        atac,
        rna_mask=rna_mask,
        atac_mask=atac_mask,
    )
    assert mu.shape == (5, 6)
    assert logvar.shape == (5, 6)
    assert torch.isfinite(mu).all()
    assert torch.isfinite(logvar).all()


def test_unbalanced_transport_returns_finite_soft_plan() -> None:
    torch.manual_seed(5)
    source_mu = torch.randn(7, 4)
    target_mu = torch.randn(9, 4)
    source_logvar = torch.zeros_like(source_mu)
    target_logvar = torch.zeros_like(target_mu)
    result = UncertaintyAwareUnbalancedOT(num_iters=8)(
        source_mu,
        source_logvar,
        target_mu,
        target_logvar,
    )
    assert result["transport"].shape == (7, 9)
    assert torch.isfinite(result["transport"]).all()
    assert result["transport_mass"] > 0
    assert torch.isfinite(result["loss"])


if __name__ == "__main__":
    test_gated_poe_supports_missing_modalities()
    test_unbalanced_transport_returns_finite_soft_plan()
    print("MoTRUST model smoke tests passed")
