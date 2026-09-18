"""Uncertainty-aware unbalanced optimal transport for posterior alignment."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class UncertaintyAwareUnbalancedOT(nn.Module):
    """Align two Gaussian posterior sets with entropy-regularized unbalanced OT."""

    def __init__(
        self,
        epsilon: float = 0.1,
        mass_regularization: float = 1.0,
        uncertainty_weight: float = 0.1,
        variance_weight: float = 0.0,
        num_iters: int = 30,
    ) -> None:
        super().__init__()
        if epsilon <= 0 or mass_regularization <= 0:
            raise ValueError("epsilon and mass_regularization must be positive.")
        self.epsilon = float(epsilon)
        self.mass_regularization = float(mass_regularization)
        self.uncertainty_weight = float(uncertainty_weight)
        self.variance_weight = float(variance_weight)
        self.num_iters = int(num_iters)

    def forward(
        self,
        source_mu: torch.Tensor,
        source_logvar: torch.Tensor,
        target_mu: torch.Tensor,
        target_logvar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if source_mu.ndim != 2 or target_mu.ndim != 2:
            raise ValueError("UOT posterior means must be rank-2 tensors.")
        if source_mu.shape[1] != target_mu.shape[1]:
            raise ValueError("UOT posterior means must have the same latent dimension.")
        if source_mu.shape != source_logvar.shape or target_mu.shape != target_logvar.shape:
            raise ValueError("Each posterior mean and log-variance must have matching shapes.")

        dim = source_mu.shape[1]
        biological_cost = torch.cdist(source_mu, target_mu, p=2).square() / max(1, dim)
        source_std = torch.exp(0.5 * source_logvar.clamp(-20.0, 20.0))
        target_std = torch.exp(0.5 * target_logvar.clamp(-20.0, 20.0))
        variance_cost = torch.cdist(source_std, target_std, p=2).square() / max(1, dim)
        source_uncertainty = source_logvar.clamp(-20.0, 20.0).exp().mean(dim=1, keepdim=True)
        target_uncertainty = target_logvar.clamp(-20.0, 20.0).exp().mean(dim=1, keepdim=True).T
        optim_cost = biological_cost + self.variance_weight * variance_cost
        plan_cost = optim_cost + self.uncertainty_weight * (
            source_uncertainty + target_uncertainty
        )

        # The transport plan is an alignment target. Gradients optimize the
        # posterior matching cost, while the plan itself stays detached.
        with torch.no_grad():
            n_source, n_target = plan_cost.shape
            log_a = plan_cost.new_full((n_source,), -math.log(float(n_source)))
            log_b = plan_cost.new_full((n_target,), -math.log(float(n_target)))
            log_kernel = -plan_cost.detach() / self.epsilon
            tau = self.mass_regularization / (self.mass_regularization + self.epsilon)
            log_u = torch.zeros_like(log_a)
            log_v = torch.zeros_like(log_b)
            for _ in range(self.num_iters):
                log_u = tau * (log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1))
                log_v = tau * (log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0))
            transport = torch.exp(log_kernel + log_u[:, None] + log_v[None, :])

        transport_mass = transport.sum().clamp_min(1e-8)
        biological_loss = (transport * biological_cost).sum() / transport_mass
        variance_loss = (transport * variance_cost).sum() / transport_mass
        loss = biological_loss + self.variance_weight * variance_loss
        return {
            "loss": loss,
            "biological_loss": biological_loss,
            "variance_loss": variance_loss,
            "transport": transport,
            "transport_mass": transport_mass,
        }
