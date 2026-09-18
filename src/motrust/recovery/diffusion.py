"""Reliability-conditioned x0 diffusion for cross-modal latent recovery."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.projection = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.SiLU(),
            nn.Linear(dimension * 2, dimension),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequency = torch.exp(
            -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        )
        angles = timesteps.float()[:, None] * frequency[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[1]))
        return self.projection(embedding)


def mlp(input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def geometric_reliability(components: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Combine quality, correspondence, proximity, and generation confidence."""
    if components.ndim != 2 or components.shape[1] != 4:
        raise ValueError("reliability components must have shape [n_cells, 4]")
    clipped = components.float().clamp(0.0, 1.0)
    reliability = clipped.clamp_min(eps).log().mean(dim=1).exp()
    any_zero = (clipped <= 0).any(dim=1)
    return torch.where(any_zero, torch.zeros_like(reliability), reliability)


class ReliabilityAdaptiveDiffusion(nn.Module):
    """Predict target residuals while reliability controls noise and correction."""

    def __init__(
        self,
        latent_dim: int,
        condition_mean: torch.Tensor,
        condition_std: torch.Tensor,
        residual_mean: torch.Tensor,
        residual_std: torch.Tensor,
        timesteps: int = 100,
        hidden_dim: int = 256,
        time_dim: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.timesteps = int(timesteps)
        condition_dim = latent_dim * 2
        self.register_buffer("condition_mean", condition_mean.float())
        self.register_buffer("condition_std", condition_std.float().clamp_min(1e-4))
        self.register_buffer("residual_mean", residual_mean.float())
        self.register_buffer("residual_std", residual_std.float().clamp_min(1e-4))
        self.mean_network = mlp(condition_dim, latent_dim, hidden_dim, dropout)
        self.time_embedding = TimeEmbedding(time_dim)
        self.denoising_network = mlp(
            latent_dim + condition_dim + latent_dim + time_dim,
            latent_dim,
            hidden_dim,
            dropout,
        )

        steps = torch.arange(timesteps + 1, dtype=torch.float32)
        alpha_bar = torch.cos(((steps / timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-4, 0.02)
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - betas, dim=0))

    def condition(
        self,
        source_mu: torch.Tensor,
        source_logvar: torch.Tensor,
        reliability_components: torch.Tensor,
    ) -> torch.Tensor:
        # Reliability controls the stochastic inference policy. Keeping it out of
        # the biological condition prevents low-confidence test cells from
        # becoming out-of-distribution inputs to the conditional mean network.
        raw = torch.cat((source_mu, source_logvar.clamp(-12.0, 12.0)), dim=1)
        return (raw - self.condition_mean) / self.condition_std

    def normalized_residual(
        self, source_mu: torch.Tensor, target_mu: torch.Tensor
    ) -> torch.Tensor:
        return (target_mu - source_mu - self.residual_mean) / self.residual_std

    def denoise(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        mean_residual: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoising_network(torch.cat(
            (noisy, condition, mean_residual, self.time_embedding(timesteps)), dim=1
        ))

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.reshape(-1).clamp_min(0.0)
        return (values.reshape(len(values), -1).mean(dim=1) * weights).sum() / weights.sum().clamp_min(1e-8)

    def training_loss(
        self,
        source_mu: torch.Tensor,
        source_logvar: torch.Tensor,
        target_mu: torch.Tensor,
        reliability_components: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        mean_weight: float = 1.0,
        return_predictions: bool = False,
    ):
        condition = self.condition(source_mu, source_logvar, reliability_components)
        residual = self.normalized_residual(source_mu, target_mu)
        mean_residual = self.mean_network(condition)
        rho = geometric_reliability(reliability_components)
        if sample_weight is None:
            sample_weight = torch.ones_like(rho)
        else:
            sample_weight = sample_weight.reshape(-1)

        max_timestep = (rho * float(self.timesteps - 1)).round().long().clamp_min(0)
        known_bridge = (reliability_components[:, 1:] >= 1.0 - 1e-6).all(dim=1)
        max_timestep = torch.where(
            known_bridge,
            torch.full_like(max_timestep, self.timesteps - 1),
            max_timestep,
        )
        random_fraction = torch.rand(len(source_mu), device=source_mu.device)
        timestep = (random_fraction * (max_timestep + 1).float()).floor().long()
        alpha = self.alpha_bar[timestep, None]
        noise = torch.randn_like(residual)
        noisy = alpha.sqrt() * residual + (1.0 - alpha).sqrt() * noise
        predicted_clean = self.denoise(noisy, timestep, condition, mean_residual)

        diffusion_cell = F.smooth_l1_loss(
            predicted_clean, residual, reduction="none"
        )
        mean_cell = F.smooth_l1_loss(mean_residual, residual, reduction="none")
        diffusion_loss = self._weighted_mean(diffusion_cell, sample_weight)
        mean_loss = self._weighted_mean(mean_cell, sample_weight)
        loss = diffusion_loss + float(mean_weight) * mean_loss
        statistics = {
            "loss": float(loss.detach()),
            "diffusion": float(diffusion_loss.detach()),
            "mean": float(mean_loss.detach()),
            "rho": float(rho.mean().detach()),
            "timestep": float(timestep.float().mean().detach()),
        }
        if not return_predictions:
            return loss, statistics
        mean_target = source_mu + self.residual_mean + self.residual_std * mean_residual
        diffusion_target = (
            source_mu + self.residual_mean
            + self.residual_std * predicted_clean.clamp(-8.0, 8.0)
        )
        return loss, statistics, diffusion_target, mean_target

    def _sample_group(
        self,
        condition: torch.Tensor,
        mean_residual: torch.Tensor,
        start_timestep: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        alpha = self.alpha_bar[start_timestep]
        noise = torch.randn(
            mean_residual.shape, device=mean_residual.device, generator=generator
        )
        value = alpha.sqrt() * mean_residual + (1.0 - alpha).sqrt() * noise
        for current in range(start_timestep, -1, -1):
            timestep = torch.full(
                (len(value),), current, device=value.device, dtype=torch.long
            )
            alpha = self.alpha_bar[current]
            predicted_clean = self.denoise(
                value, timestep, condition, mean_residual
            ).clamp(-8.0, 8.0)
            if current == 0:
                value = predicted_clean
            else:
                predicted_noise = (
                    value - alpha.sqrt() * predicted_clean
                ) / (1.0 - alpha).sqrt().clamp_min(1e-6)
                next_alpha = self.alpha_bar[current - 1]
                value = (
                    next_alpha.sqrt() * predicted_clean
                    + (1.0 - next_alpha).sqrt() * predicted_noise
                )
        return value

    @torch.no_grad()
    def predict_distribution(
        self,
        source_mu: torch.Tensor,
        source_logvar: torch.Tensor,
        reliability_components: torch.Tensor,
        max_timestep: int = 9,
        max_blend: float = 0.5,
        ensemble_size: int = 8,
        seed: int = 0,
        policy_reliability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        condition = self.condition(source_mu, source_logvar, reliability_components)
        mean_residual = self.mean_network(condition)
        mean_target = source_mu + self.residual_mean + self.residual_std * mean_residual
        rho = geometric_reliability(reliability_components)
        policy = rho if policy_reliability is None else policy_reliability
        policy = policy.reshape(-1).clamp(0.0, 1.0)
        cell_timestep = (policy * float(max_timestep)).round().long().clamp(0, max_timestep)
        blend = (float(max_blend) * policy).clamp(0.0, 1.0)
        generator = torch.Generator(device=source_mu.device).manual_seed(int(seed))
        samples = []
        for _ in range(max(2, int(ensemble_size))):
            sampled_residual = torch.empty_like(mean_residual)
            for timestep_value in cell_timestep.unique(sorted=True):
                selected = cell_timestep == timestep_value
                sampled_residual[selected] = self._sample_group(
                    condition[selected], mean_residual[selected], int(timestep_value), generator
                )
            sampled_target = (
                source_mu + self.residual_mean + self.residual_std * sampled_residual
            )
            samples.append(
                mean_target + blend[:, None] * (sampled_target - mean_target)
            )
        stacked = torch.stack(samples)
        return {
            "samples": stacked,
            "sample_mean": stacked.mean(dim=0),
            "sample_std": stacked.std(dim=0, unbiased=False),
            "conditional_mean": mean_target,
            "rho": rho,
            "policy_reliability": policy,
            "blend": blend,
            "timestep": cell_timestep,
        }
