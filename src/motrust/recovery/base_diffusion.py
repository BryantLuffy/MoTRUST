"""Conditional mean-residual diffusion in a frozen multimodal VAE latent space."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
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


class ConditionalResidualLatentDiffusion(nn.Module):
    """Conditionally denoise the complete cross-modal latent residual."""

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
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        betas = betas.clamp(1e-4, 0.02)
        alphas = 1.0 - betas
        self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))

    def condition(self, source_mu: torch.Tensor, source_logvar: torch.Tensor) -> torch.Tensor:
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
        time = self.time_embedding(timesteps)
        return self.denoising_network(
            torch.cat((noisy, condition, mean_residual, time), dim=1)
        )

    def training_loss(
        self,
        source_mu: torch.Tensor,
        source_logvar: torch.Tensor,
        target_mu: torch.Tensor,
        mean_weight: float = 1.0,
        return_predictions: bool = False,
    ):
        condition = self.condition(source_mu, source_logvar)
        residual = self.normalized_residual(source_mu, target_mu)
        mean_residual = self.mean_network(condition)
        timestep = torch.randint(
            0, self.timesteps, (len(source_mu),), device=source_mu.device
        )
        alpha = self.alpha_bar[timestep, None]
        noise = torch.randn_like(residual)
        noisy = alpha.sqrt() * residual + (1.0 - alpha).sqrt() * noise
        predicted_clean = self.denoise(noisy, timestep, condition, mean_residual)
        diffusion_loss = F.smooth_l1_loss(predicted_clean, residual)
        mean_loss = F.smooth_l1_loss(mean_residual, residual)
        loss = diffusion_loss + float(mean_weight) * mean_loss
        statistics = {
            "loss": float(loss.detach()),
            "diffusion": float(diffusion_loss.detach()),
            "mean": float(mean_loss.detach()),
        }
        if not return_predictions:
            return loss, statistics
        mean_target = (
            source_mu + self.residual_mean + self.residual_std * mean_residual
        )
        diffusion_target = (
            source_mu
            + self.residual_mean
            + self.residual_std * predicted_clean.clamp(-8.0, 8.0)
        )
        return loss, statistics, diffusion_target, mean_target

    @torch.no_grad()
    def predict(
        self,
        source_mu: torch.Tensor,
        source_logvar: torch.Tensor,
        diffusion_strength: float = 1.0,
        ddim_steps: int = 25,
        ensemble_size: int = 4,
        seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.condition(source_mu, source_logvar)
        mean_residual = self.mean_network(condition)
        mean_target = source_mu + self.residual_mean + self.residual_std * mean_residual
        if diffusion_strength <= 0:
            return mean_target, mean_target

        schedule = torch.linspace(
            self.timesteps - 1, 0, min(ddim_steps, self.timesteps),
            device=source_mu.device,
        ).round().long().unique(sorted=True).flip(0)
        generator = torch.Generator(device=source_mu.device).manual_seed(int(seed))
        samples = []
        for _ in range(max(1, int(ensemble_size))):
            value = torch.randn(
                source_mu.shape, device=source_mu.device, generator=generator
            )
            for position, timestep_value in enumerate(schedule):
                timestep = torch.full(
                    (len(source_mu),), int(timestep_value),
                    device=source_mu.device, dtype=torch.long,
                )
                predicted_clean = self.denoise(
                    value, timestep, condition, mean_residual
                )
                alpha = self.alpha_bar[timestep_value]
                predicted_clean = predicted_clean.clamp(-8.0, 8.0)
                predicted_noise = (
                    value - alpha.sqrt() * predicted_clean
                ) / (1.0 - alpha).sqrt().clamp_min(1e-6)
                if position == len(schedule) - 1:
                    value = predicted_clean
                else:
                    next_alpha = self.alpha_bar[schedule[position + 1]]
                    value = (
                        next_alpha.sqrt() * predicted_clean
                        + (1.0 - next_alpha).sqrt() * predicted_noise
                    )
            samples.append(value)
        sampled_residual = torch.stack(samples).mean(dim=0)
        residual = (
            (1.0 - float(diffusion_strength)) * mean_residual
            + float(diffusion_strength) * sampled_residual
        )
        target = source_mu + self.residual_mean + self.residual_std * residual
        return target, mean_target

    @torch.no_grad()
    def predict_tweedie(
        self,
        source_mu: torch.Tensor,
        source_logvar: torch.Tensor,
        noise_timestep: int,
        ensemble_size: int = 4,
        seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reverse a controlled perturbation of the conditional posterior mean."""
        condition = self.condition(source_mu, source_logvar)
        mean_residual = self.mean_network(condition)
        mean_target = source_mu + self.residual_mean + self.residual_std * mean_residual
        timestep_value = max(0, min(int(noise_timestep), self.timesteps - 1))
        generator = torch.Generator(device=source_mu.device).manual_seed(int(seed))
        clean_residuals = []
        for _ in range(max(1, int(ensemble_size))):
            noise = torch.randn(
                source_mu.shape, device=source_mu.device, generator=generator
            )
            start_alpha = self.alpha_bar[timestep_value]
            value = start_alpha.sqrt() * mean_residual + (1.0 - start_alpha).sqrt() * noise
            schedule = torch.linspace(
                timestep_value, 0, min(timestep_value + 1, 25),
                device=source_mu.device,
            ).round().long().unique(sorted=True).flip(0)
            for position, current in enumerate(schedule):
                timestep = torch.full(
                    (len(source_mu),), int(current),
                    device=source_mu.device, dtype=torch.long,
                )
                alpha = self.alpha_bar[current]
                predicted_clean = self.denoise(
                    value, timestep, condition, mean_residual
                ).clamp(-8.0, 8.0)
                if position == len(schedule) - 1:
                    value = predicted_clean
                else:
                    predicted_noise = (
                        value - alpha.sqrt() * predicted_clean
                    ) / (1.0 - alpha).sqrt().clamp_min(1e-6)
                    next_alpha = self.alpha_bar[schedule[position + 1]]
                    value = (
                        next_alpha.sqrt() * predicted_clean
                        + (1.0 - next_alpha).sqrt() * predicted_noise
                    )
            clean_residuals.append(value)
        residual = torch.stack(clean_residuals).mean(dim=0)
        target = (
            source_mu
            + self.residual_mean
            + self.residual_std * residual
        )
        return target, mean_target
