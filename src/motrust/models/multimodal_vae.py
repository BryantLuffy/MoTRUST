"""
Improved multimodal VAE with optional 3rd modality (ADT).

Backwards compatibility:
- Old 2-modality usage (RNA + ATAC) still works.
- New 3-modality usage can be enabled by `use_adt=True` and `adt_dim>0`.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from .encoders import RNAEncoder, ATACEncoder, ADTEncoder
from .decoders import RNADecoder, ATACDecoder, ADTDecoder
from .poe import ProductOfExperts
from .batch_encoder import BatchEncoder
from .batch_decoder import BatchDecoder


def _mlp_layers(input_dim: int, hidden_dims: list[int], dropout: float) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        ])
        prev_dim = hidden_dim
    return nn.Sequential(*layers)


class SharedBackboneEncoder(nn.Module):
    """Modality-specific input projection followed by a shared encoder trunk."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        shared_trunk: nn.Module,
        private_dim: int = 512,
        shared_out_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, private_dim),
            nn.LayerNorm(private_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )
        self.shared_trunk = shared_trunk
        self.fc_mu = nn.Linear(shared_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(shared_out_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.input_proj(x)
        h = self.shared_trunk(h)
        return self.fc_mu(h), self.fc_logvar(h)


class SharedRNADecoder(nn.Module):
    """Shared latent decoder trunk with modality-specific RNA output heads."""

    def __init__(
        self,
        shared_trunk: nn.Module,
        shared_out_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
        distribution: str = "zinb",
    ):
        super().__init__()
        self.shared_trunk = shared_trunk
        self.distribution = distribution.lower()
        self.private_decoder = _mlp_layers(shared_out_dim, hidden_dims, dropout)
        head_dim = hidden_dims[-1] if hidden_dims else shared_out_dim
        if self.distribution == "poisson":
            self.fc_lambda = nn.Linear(head_dim, output_dim)
        elif self.distribution == "nb":
            self.fc_mu = nn.Linear(head_dim, output_dim)
            self.fc_theta = nn.Linear(head_dim, output_dim)
        elif self.distribution == "zinb":
            self.fc_mu = nn.Linear(head_dim, output_dim)
            self.fc_theta = nn.Linear(head_dim, output_dim)
            self.fc_pi = nn.Linear(head_dim, output_dim)
        else:
            raise ValueError(f"Unsupported RNA distribution: {distribution}")

    def forward(self, z: torch.Tensor):
        h = self.private_decoder(self.shared_trunk(z))
        if self.distribution == "poisson":
            lam = F.softplus(self.fc_lambda(h)) + 1e-6
            return torch.nan_to_num(lam, nan=1e-6, posinf=1e6, neginf=1e-6).clamp(1e-6, 1e6)
        mu = F.softplus(self.fc_mu(h)) + 1e-4
        theta = F.softplus(self.fc_theta(h)) + 1e-4
        mu = torch.nan_to_num(mu, nan=1e-4, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
        theta = torch.nan_to_num(theta, nan=1e-4, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
        if self.distribution == "nb":
            return mu, theta
        pi = torch.sigmoid(self.fc_pi(h))
        pi = torch.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-6, 1 - 1e-6)
        return mu, theta, pi

    poisson_log_prob = staticmethod(RNADecoder.poisson_log_prob)
    nb_log_prob = staticmethod(RNADecoder.nb_log_prob)
    zinb_log_prob = staticmethod(RNADecoder.zinb_log_prob)


class SharedATACDecoder(nn.Module):
    """Shared latent decoder trunk with an ATAC-specific Bernoulli head."""

    def __init__(
        self,
        shared_trunk: nn.Module,
        shared_out_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_trunk = shared_trunk
        self.private_decoder = _mlp_layers(shared_out_dim, hidden_dims, dropout)
        head_dim = hidden_dims[-1] if hidden_dims else shared_out_dim
        self.fc_logits = nn.Linear(head_dim, output_dim)

    def forward(self, z: torch.Tensor):
        h = self.private_decoder(self.shared_trunk(z))
        logits = self.fc_logits(h)
        return torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)

    @staticmethod
    def get_distribution(logits: torch.Tensor):
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        return Bernoulli(logits=logits)


class SharedADTDecoder(nn.Module):
    """Shared latent decoder trunk with modality-specific ADT output heads."""

    def __init__(
        self,
        shared_trunk: nn.Module,
        shared_out_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
        distribution: str = "poisson",
    ):
        super().__init__()
        self.shared_trunk = shared_trunk
        self.distribution = distribution.lower()
        self.private_decoder = _mlp_layers(shared_out_dim, hidden_dims, dropout)
        head_dim = hidden_dims[-1] if hidden_dims else shared_out_dim
        if self.distribution == "poisson":
            self.fc_lambda = nn.Linear(head_dim, output_dim)
        elif self.distribution == "nb":
            self.fc_mu = nn.Linear(head_dim, output_dim)
            self.fc_theta = nn.Linear(head_dim, output_dim)
        else:
            raise ValueError(f"Unsupported ADT distribution: {distribution}")

    def forward(self, z: torch.Tensor):
        h = self.private_decoder(self.shared_trunk(z))
        if self.distribution == "poisson":
            lam = F.softplus(self.fc_lambda(h)) + 1e-6
            return torch.nan_to_num(lam, nan=1e-6, posinf=1e6, neginf=1e-6).clamp(1e-6, 1e6)
        mu = F.softplus(self.fc_mu(h)) + 1e-4
        theta = F.softplus(self.fc_theta(h)) + 1e-4
        mu = torch.nan_to_num(mu, nan=1e-4, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
        theta = torch.nan_to_num(theta, nan=1e-4, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
        return mu, theta

    poisson_log_prob = staticmethod(ADTDecoder.poisson_log_prob)
    nb_log_prob = staticmethod(ADTDecoder.nb_log_prob)


class ImprovedMultiModalVAE(nn.Module):
    """
    Improved multimodal VAE with:
    - Product-of-Experts (PoE) posterior fusion
    - Latent disentanglement: c (biology) + u (technical)
    - Optional ADT branch (third modality)
    """

    def __init__(
        self,
        rna_dim: int,
        atac_dim: int,
        rna_encoder_dim: Optional[int] = None,
        atac_encoder_dim: Optional[int] = None,
        adt_dim: Optional[int] = None,
        latent_dim: int = 32,
        dim_c: int = 32,
        dim_u: int = 2,
        hidden_dims: list[int] = [512, 256],
        dropout: float = 0.1,
        beta_c: float = 1.0,
        beta_u: float = 0.5,
        use_poe: bool = True,
        batch_correction: bool = False,
        rna_distribution: str = "zinb",
        use_adt: bool = False,
        adt_distribution: str = "nb",
        adt_weight: float = 1.0,
        n_batches: int = 1,
        use_batch_latent: bool = False,
        batch_latent_weight: float = 1.0,
        use_gated_poe: bool = False,
        gate_hidden_dim: int = 64,
        use_shared_backbone: bool = False,
        shared_bridge_dim: int = 0,
        shared_bridge_hidden_dim: int = 128,
    ):
        super().__init__()

        if dim_c + dim_u != latent_dim:
            raise ValueError(
                f"dim_c ({dim_c}) + dim_u ({dim_u}) must equal latent_dim ({latent_dim})"
            )

        self.rna_dim = int(rna_dim)
        self.atac_dim = int(atac_dim)
        self.rna_encoder_dim = int(rna_encoder_dim or rna_dim)
        self.atac_encoder_dim = int(atac_encoder_dim or atac_dim)
        self.adt_dim = int(adt_dim) if adt_dim is not None else None
        self.latent_dim = int(latent_dim)
        self.dim_c = int(dim_c)
        self.dim_u = int(dim_u)
        self.beta_c = float(beta_c)
        self.beta_u = float(beta_u)
        self.use_poe = bool(use_poe)
        self.batch_correction = bool(batch_correction)
        self.rna_distribution = str(rna_distribution).lower()
        self.use_adt = bool(use_adt)
        self.adt_distribution = str(adt_distribution).lower()
        self.adt_weight = float(adt_weight)
        self.n_batches = int(n_batches)
        self.use_batch_latent = bool(use_batch_latent)
        self.batch_latent_weight = float(batch_latent_weight)
        self.use_gated_poe = bool(use_gated_poe)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.use_shared_backbone = bool(use_shared_backbone)
        self.shared_bridge_dim = int(shared_bridge_dim)

        if self.use_adt and (self.adt_dim is None or self.adt_dim <= 0):
            raise ValueError("use_adt=True requires a valid adt_dim > 0")
        if self.use_adt and self.adt_distribution not in {"poisson", "nb"}:
            raise ValueError("ADT decoder supports adt_distribution='poisson' or 'nb'.")
        if self.shared_bridge_dim < 0:
            raise ValueError("shared_bridge_dim must be non-negative")

        if self.use_shared_backbone:
            if len(hidden_dims) < 2:
                raise ValueError("use_shared_backbone=True requires at least two hidden dimensions.")
            enc_private_dim = int(hidden_dims[0])
            enc_shared_dim = int(hidden_dims[-1])
            self.shared_encoder = _mlp_layers(enc_private_dim, list(hidden_dims[1:]), dropout)
            self.rna_encoder = SharedBackboneEncoder(
                self.rna_encoder_dim, self.latent_dim, self.shared_encoder, enc_private_dim, enc_shared_dim, dropout
            )
            self.atac_encoder = SharedBackboneEncoder(
                self.atac_encoder_dim, self.latent_dim, self.shared_encoder, enc_private_dim, enc_shared_dim, dropout
            )
            self.adt_encoder = (
                SharedBackboneEncoder(
                    self.adt_dim, self.latent_dim, self.shared_encoder, enc_private_dim, enc_shared_dim, dropout
                )
                if self.use_adt
                else None
            )

            dec_dims = list(hidden_dims[::-1])
            dec_shared_dim = int(dec_dims[0])
            self.shared_decoder = _mlp_layers(self.latent_dim, [dec_shared_dim], dropout)
            self.rna_decoder = SharedRNADecoder(
                self.shared_decoder,
                dec_shared_dim,
                self.rna_dim,
                dec_dims[1:],
                dropout=dropout,
                distribution=self.rna_distribution,
            )
            self.atac_decoder = SharedATACDecoder(
                self.shared_decoder,
                dec_shared_dim,
                self.atac_dim,
                dec_dims[1:],
                dropout=dropout,
            )
            self.adt_decoder = (
                SharedADTDecoder(
                    self.shared_decoder,
                    dec_shared_dim,
                    self.adt_dim,
                    dec_dims[1:],
                    dropout=dropout,
                    distribution=self.adt_distribution,
                )
                if self.use_adt
                else None
            )
        else:
            # Encoders
            self.rna_encoder = RNAEncoder(
                input_dim=self.rna_encoder_dim,
                latent_dim=self.latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
            self.atac_encoder = ATACEncoder(
                input_dim=self.atac_encoder_dim,
                latent_dim=self.latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
            if self.use_adt:
                self.adt_encoder = ADTEncoder(
                    input_dim=self.adt_dim,
                    latent_dim=self.latent_dim,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                )
            else:
                self.adt_encoder = None

            # Decoders
            self.rna_decoder = RNADecoder(
                latent_dim=self.latent_dim,
                output_dim=self.rna_dim,
                hidden_dims=hidden_dims[::-1],
                dropout=dropout,
                distribution=self.rna_distribution,
            )
            self.atac_decoder = ATACDecoder(
                latent_dim=self.latent_dim,
                output_dim=self.atac_dim,
                hidden_dims=hidden_dims[::-1],
                dropout=dropout,
            )
            if self.use_adt:
                self.adt_decoder = ADTDecoder(
                    latent_dim=self.latent_dim,
                    output_dim=self.adt_dim,
                    hidden_dims=hidden_dims[::-1],
                    dropout=dropout,
                    distribution=self.adt_distribution,
                )
            else:
                self.adt_decoder = None

        if self.shared_bridge_dim > 0:
            bridge_hidden = max(1, int(shared_bridge_hidden_dim))
            self.shared_bridge_decoder = nn.Sequential(
                nn.Linear(self.dim_c, bridge_hidden),
                nn.SiLU(),
                nn.Linear(bridge_hidden, self.shared_bridge_dim),
            )
        else:
            self.shared_bridge_decoder = None

        self.poe = ProductOfExperts() if self.use_poe else None
        if self.use_gated_poe:
            self.modality_gates = nn.ModuleDict({
                "rna": self._make_gate_network(self.latent_dim, self.gate_hidden_dim),
                "atac": self._make_gate_network(self.latent_dim, self.gate_hidden_dim),
            })
            if self.use_adt:
                self.modality_gates["adt"] = self._make_gate_network(self.latent_dim, self.gate_hidden_dim)
        else:
            self.modality_gates = None

        # Batch latent encoder/decoder (analogous to MIDAS S_Encoder/S_Decoder).
        if self.use_batch_latent:
            self.batch_encoder = BatchEncoder(
                n_batches=self.n_batches,
                latent_dim=self.latent_dim,
                hidden_dims=[128],
                dropout=dropout,
            )
            self.batch_decoder = BatchDecoder(
                n_batches=self.n_batches,
                dim_u=self.dim_u,
                hidden_dims=[64],
                dropout=dropout,
            )
        else:
            self.batch_encoder = None
            self.batch_decoder = None

        if self.batch_correction:
            self.u_centroid = nn.Parameter(torch.zeros(1, self.dim_u))
        else:
            self.register_buffer("u_centroid", torch.zeros(1, self.dim_u))

    @staticmethod
    def _make_gate_network(latent_dim: int, hidden_dim: int) -> nn.Module:
        hidden_dim = max(1, int(hidden_dim))
        gate = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(gate[-1].weight)
        nn.init.zeros_(gate[-1].bias)
        return gate

    def _gate_weight(self, name: str, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.modality_gates is None or name not in self.modality_gates:
            return torch.ones(mu.shape[0], device=mu.device)
        gate_input = torch.cat([mu.detach(), logvar.detach()], dim=1)
        logits = self.modality_gates[name](gate_input).view(-1)
        return 2.0 * torch.sigmoid(logits)

    def encode(
        self,
        rna_x: torch.Tensor,
        atac_x: torch.Tensor,
        adt_x: Optional[torch.Tensor] = None,
        batch_ids: Optional[torch.Tensor] = None,
        rna_mask: Optional[torch.Tensor] = None,
        atac_mask: Optional[torch.Tensor] = None,
        adt_mask: Optional[torch.Tensor] = None,
    ):
        rna_mu, rna_logvar = self.rna_encoder(rna_x)
        atac_mu, atac_logvar = self.atac_encoder(atac_x)

        adt_mu = None
        adt_logvar = None
        mus = [rna_mu, atac_mu]
        logvars = [rna_logvar, atac_logvar]
        gate_weights = None
        if self.use_gated_poe:
            gate_weights = [
                self._gate_weight("rna", rna_mu, rna_logvar),
                self._gate_weight("atac", atac_mu, atac_logvar),
            ]
        modality_masks = None
        if rna_mask is not None or atac_mask is not None or adt_mask is not None:
            batch_size = rna_x.shape[0]
            device = rna_x.device
            default_mask = torch.ones(batch_size, device=device)
            modality_masks = [
                default_mask if rna_mask is None else rna_mask.float().view(-1).to(device),
                default_mask if atac_mask is None else atac_mask.float().view(-1).to(device),
            ]

        if self.use_adt and adt_x is not None:
            adt_mu, adt_logvar = self.adt_encoder(adt_x)
            mus.append(adt_mu)
            logvars.append(adt_logvar)
            if gate_weights is not None:
                gate_weights.append(self._gate_weight("adt", adt_mu, adt_logvar))
            if modality_masks is not None:
                default_mask = torch.ones(rna_x.shape[0], device=rna_x.device)
                modality_masks.append(default_mask if adt_mask is None else adt_mask.float().view(-1).to(rna_x.device))

        # If batch latents are enabled, encode batch information and fuse it through PoE.
        batch_mu = None
        batch_logvar = None
        if self.use_batch_latent and batch_ids is not None:
            batch_mu, batch_logvar = self.batch_encoder(batch_ids)
            mus.append(batch_mu)
            logvars.append(batch_logvar)
            if gate_weights is not None:
                gate_weights.append(torch.ones(rna_x.shape[0], device=rna_x.device))
            if modality_masks is not None:
                modality_masks.append(torch.ones(rna_x.shape[0], device=rna_x.device))

        if self.use_poe:
            z_mu, z_logvar = ProductOfExperts.poe_with_prior(
                mus,
                logvars,
                masks=modality_masks,
                weights=gate_weights,
            )
        else:
            z_mu = torch.stack(mus, dim=0).mean(dim=0)
            z_logvar = torch.stack(logvars, dim=0).mean(dim=0)

        return z_mu, z_logvar, rna_mu, rna_logvar, atac_mu, atac_logvar, adt_mu, adt_logvar, batch_mu, batch_logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        rna_out = self.rna_decoder(z)
        atac_logits = self.atac_decoder(z)
        adt_out = self.adt_decoder(z) if self.use_adt else None
        return rna_out, atac_logits, adt_out

    def forward(
        self,
        rna_x: torch.Tensor,
        atac_x: torch.Tensor,
        adt_x: Optional[torch.Tensor] = None,
        batch_ids: Optional[torch.Tensor] = None,
        batch_correct: bool = False,
        rna_mask: Optional[torch.Tensor] = None,
        atac_mask: Optional[torch.Tensor] = None,
        adt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        (
            z_mu,
            z_logvar,
            rna_mu_enc,
            rna_logvar_enc,
            atac_mu_enc,
            atac_logvar_enc,
            adt_mu_enc,
            adt_logvar_enc,
            batch_mu_enc,
            batch_logvar_enc,
        ) = self.encode(
            rna_x,
            atac_x,
            adt_x=adt_x,
            batch_ids=batch_ids,
            rna_mask=rna_mask,
            atac_mask=atac_mask,
            adt_mask=adt_mask,
        )

        z = self.reparameterize(z_mu, z_logvar)
        z = torch.nan_to_num(z, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)

        c, u = z.split([self.dim_c, self.dim_u], dim=1)
        
        # Batch correction: replace u with the centroid during training or when batch_correct is requested.
        if batch_correct and not self.training:
            u_corrected = self.u_centroid.expand_as(u)
            z_decode = torch.cat([c, u_corrected], dim=1)
        else:
            z_decode = z

        rna_out, atac_logits, adt_out = self.decode(z_decode)
        atac_logits = torch.nan_to_num(atac_logits, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)

        rna_outputs: Dict[str, torch.Tensor] = {}
        if self.rna_distribution == "poisson":
            rna_lambda = torch.nan_to_num(rna_out, nan=1e-6, posinf=1e6, neginf=1e-6).clamp(1e-6, 1e6)
            rna_outputs = {"rna_lambda": rna_lambda}
        elif self.rna_distribution == "nb":
            rna_mu_dec, rna_theta = rna_out
            rna_mu_dec = torch.nan_to_num(rna_mu_dec, nan=1e-4, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
            rna_theta = torch.nan_to_num(rna_theta, nan=1e-4, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
            rna_outputs = {"rna_mu": rna_mu_dec, "rna_theta": rna_theta}
        else:
            rna_mu_dec, rna_theta, rna_pi = rna_out
            rna_mu_dec = torch.nan_to_num(rna_mu_dec, nan=1e-4, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
            rna_theta = torch.nan_to_num(rna_theta, nan=1e-4, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
            rna_pi = torch.nan_to_num(rna_pi, nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-6, 1 - 1e-6)
            rna_outputs = {"rna_mu": rna_mu_dec, "rna_theta": rna_theta, "rna_pi": rna_pi}

        out = {
            "z": z,
            "c": c,
            "u": u,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "rna_mu_enc": rna_mu_enc,
            "rna_logvar_enc": rna_logvar_enc,
            "atac_mu": atac_mu_enc,
            "atac_logvar": atac_logvar_enc,
            "atac_logits": atac_logits,
            **rna_outputs,
        }
        if self.shared_bridge_decoder is not None:
            out["rna_shared_bridge_mean"] = self.shared_bridge_decoder(
                rna_mu_enc[:, : self.dim_c]
            )
            out["atac_shared_bridge_mean"] = self.shared_bridge_decoder(
                atac_mu_enc[:, : self.dim_c]
            )
        
        # Batch reconstruction when batch latents are enabled.
        if self.use_batch_latent and self.batch_decoder is not None:
            batch_logits = self.batch_decoder(u)
            out["batch_logits"] = batch_logits

        if self.use_adt:
            out["adt_mu_enc"] = adt_mu_enc
            out["adt_logvar_enc"] = adt_logvar_enc
            if adt_out is not None:
                if self.adt_distribution == "poisson":
                    adt_lambda = torch.nan_to_num(adt_out, nan=1e-6, posinf=1e6, neginf=1e-6).clamp(1e-6, 1e6)
                    out["adt_lambda"] = adt_lambda
                else:
                    adt_mu_dec, adt_theta = adt_out
                    adt_mu_dec = torch.nan_to_num(adt_mu_dec, nan=1e-4, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
                    adt_theta = torch.nan_to_num(adt_theta, nan=1e-4, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
                    out["adt_mu"] = adt_mu_dec
                    out["adt_theta"] = adt_theta
        return out

    def compute_loss(
        self,
        rna_x: torch.Tensor,
        atac_x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        adt_x: Optional[torch.Tensor] = None,
        batch_ids: Optional[torch.Tensor] = None,
        rna_loss_mask: Optional[torch.Tensor] = None,
        atac_loss_mask: Optional[torch.Tensor] = None,
        adt_loss_mask: Optional[torch.Tensor] = None,
        rna_shared_bridge_target: Optional[torch.Tensor] = None,
        atac_shared_bridge_target: Optional[torch.Tensor] = None,
        shared_bridge_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        def masked_mean(loss_per_cell: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
            if mask is None:
                return loss_per_cell.mean()
            mask = mask.float().view(-1).to(loss_per_cell.device)
            denom = mask.sum().clamp_min(1.0)
            return (loss_per_cell * mask).sum() / denom

        # RNA recon
        if self.rna_distribution == "poisson":
            rna_lambda = outputs["rna_lambda"]
            rna_counts = torch.expm1(rna_x)
            rna_counts = torch.nan_to_num(rna_counts, nan=0.0, posinf=1e6, neginf=0.0)
            log_prob = self.rna_decoder.poisson_log_prob(rna_counts, rna_lambda)
            rna_recon_loss = masked_mean(-(log_prob.sum(dim=1)), rna_loss_mask)
        elif self.rna_distribution == "nb":
            rna_mu = outputs["rna_mu"]
            rna_theta = outputs["rna_theta"]
            rna_x_safe = torch.nan_to_num(rna_x, nan=0.0, posinf=1e6, neginf=0.0)
            log_prob = self.rna_decoder.nb_log_prob(rna_x_safe, rna_mu, rna_theta)
            rna_recon_loss = masked_mean(-(log_prob.sum(dim=1)), rna_loss_mask)
        else:
            rna_mu = outputs["rna_mu"]
            rna_theta = outputs["rna_theta"]
            rna_pi = outputs["rna_pi"]
            rna_x_safe = torch.nan_to_num(rna_x, nan=0.0, posinf=1e6, neginf=0.0)
            log_prob = self.rna_decoder.zinb_log_prob(rna_x_safe, rna_mu, rna_theta, rna_pi)
            rna_recon_loss = masked_mean(-(log_prob.sum(dim=1)), rna_loss_mask)

        # ATAC recon
        atac_logits = outputs["atac_logits"]
        atac_dist = self.atac_decoder.get_distribution(atac_logits)
        atac_recon_loss = masked_mean(-atac_dist.log_prob(atac_x).sum(dim=1), atac_loss_mask)

        # ADT recon (optional)
        adt_recon_loss = torch.tensor(0.0, device=rna_x.device)
        if self.use_adt and adt_x is not None:
            adt_x_safe = torch.nan_to_num(adt_x, nan=0.0, posinf=1e6, neginf=0.0)
            if self.adt_distribution == "poisson" and "adt_lambda" in outputs:
                adt_counts = torch.expm1(adt_x_safe)
                adt_counts = torch.nan_to_num(adt_counts, nan=0.0, posinf=1e6, neginf=0.0).clamp(min=0.0)
                log_prob = self.adt_decoder.poisson_log_prob(adt_counts, outputs["adt_lambda"])
                adt_recon_loss = masked_mean(-(log_prob.sum(dim=1)), adt_loss_mask)
            elif ("adt_mu" in outputs) and ("adt_theta" in outputs):
                log_prob = self.adt_decoder.nb_log_prob(adt_x_safe, outputs["adt_mu"], outputs["adt_theta"])
                adt_recon_loss = masked_mean(-(log_prob.sum(dim=1)), adt_loss_mask)

        # KL(c) + KL(u)
        z_mu = outputs["z_mu"]
        z_logvar = outputs["z_logvar"]
        mu_c, mu_u = z_mu.split([self.dim_c, self.dim_u], dim=1)
        logvar_c, logvar_u = z_logvar.split([self.dim_c, self.dim_u], dim=1)

        kld_c = -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp(), dim=1).mean()
        kld_u = -0.5 * torch.sum(1 + logvar_u - mu_u.pow(2) - logvar_u.exp(), dim=1).mean()
        kl_loss = self.beta_c * kld_c + self.beta_u * kld_u

        # Batch reconstruction loss when batch latents are enabled.
        batch_recon_loss = torch.tensor(0.0, device=rna_x.device)
        if self.use_batch_latent and batch_ids is not None and "batch_logits" in outputs:
            batch_logits = outputs["batch_logits"]
            batch_ids_flat = batch_ids.squeeze() if batch_ids.dim() > 1 else batch_ids
            batch_recon_loss = torch.nn.functional.cross_entropy(batch_logits, batch_ids_flat.long())

        shared_bridge_loss = torch.tensor(0.0, device=rna_x.device)
        bridge_targets_present = (
            rna_shared_bridge_target is not None
            or atac_shared_bridge_target is not None
        )
        if bridge_targets_present:
            if rna_shared_bridge_target is None or atac_shared_bridge_target is None:
                raise ValueError("RNA and ATAC shared-bridge targets must be provided together")
            if self.shared_bridge_decoder is None:
                raise RuntimeError("Shared-bridge targets require shared_bridge_dim > 0")
            rna_bridge_nll = 0.5 * (
                outputs["rna_shared_bridge_mean"] - rna_shared_bridge_target
            ).square().sum(dim=1)
            atac_bridge_nll = 0.5 * (
                outputs["atac_shared_bridge_mean"] - atac_shared_bridge_target
            ).square().sum(dim=1)
            shared_bridge_loss = 0.5 * (
                masked_mean(rna_bridge_nll, rna_loss_mask)
                + masked_mean(atac_bridge_nll, atac_loss_mask)
            )

        total_loss = (
            rna_recon_loss
            + atac_recon_loss
            + self.adt_weight * adt_recon_loss
            + kl_loss
            + self.batch_latent_weight * batch_recon_loss
            + float(shared_bridge_weight) * shared_bridge_loss
        )

        loss_dict = {
            "total_loss": total_loss,
            "rna_recon_loss": rna_recon_loss,
            "atac_recon_loss": atac_recon_loss,
            "adt_recon_loss": adt_recon_loss,
            "kld_c": kld_c,
            "kld_u": kld_u,
            "kl_loss": kl_loss,
            "shared_bridge_loss": shared_bridge_loss,
        }
        
        if self.use_batch_latent:
            loss_dict["batch_recon_loss"] = batch_recon_loss
        
        return loss_dict

    def get_latent_representation(
        self,
        rna_x: torch.Tensor,
        atac_x: torch.Tensor,
        use_biological_only: bool = True,
        batch_correct: bool = False,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            z_mu, _z_logvar, *_ = self.encode(rna_x, atac_x, adt_x=None, batch_ids=None)
            
            # Replace u with the centroid when batch correction is enabled.
            if batch_correct:
                c_mu, u_mu = z_mu.split([self.dim_c, self.dim_u], dim=1)
                u_corrected = self.u_centroid.expand_as(u_mu)
                z_mu = torch.cat([c_mu, u_corrected], dim=1)
            
            if use_biological_only:
                c_mu, _ = z_mu.split([self.dim_c, self.dim_u], dim=1)
                return torch.nan_to_num(c_mu, nan=0.0, posinf=10.0, neginf=-10.0)
            return torch.nan_to_num(z_mu, nan=0.0, posinf=10.0, neginf=-10.0)
    
    def compute_u_centroid(self, data_loader, device: torch.device):
        """
        Compute the centroid (mean) of u across all batches for batch correction.
        
        Args:
            data_loader: DataLoader containing (rna, atac, adt, batch_ids).
            device: Device used for computation.
        
        Returns:
            u_centroid: (1, dim_u) tensor
        """
        self.eval()
        u_list = []
        with torch.no_grad():
            for batch_data in data_loader:
                if len(batch_data) == 4:
                    batch_rna, batch_atac, batch_adt, batch_ids = batch_data
                else:
                    batch_rna, batch_atac, batch_adt = batch_data[:3]
                    batch_ids = None
                
                batch_rna = batch_rna.to(device)
                batch_atac = batch_atac.to(device)
                batch_adt = batch_adt.to(device) if batch_adt is not None else None
                batch_ids = batch_ids.to(device) if batch_ids is not None else None
                
                outputs = self.forward(batch_rna, batch_atac, batch_adt, batch_ids=batch_ids)
                u = outputs["u"]
                u_list.append(u.cpu())
        
        u_all = torch.cat(u_list, dim=0)
        u_centroid = u_all.mean(dim=0, keepdim=True)
        
        # Update the u_centroid parameter or buffer.
        if self.batch_correction:
            with torch.no_grad():
                self.u_centroid.data = u_centroid.to(device)
        else:
            self.register_buffer("u_centroid", u_centroid.to(device))
        
        return u_centroid
