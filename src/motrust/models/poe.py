"""
Product of Experts (PoE) fusion module.
Combine posterior distributions of multimodal latent representations.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class ProductOfExperts(nn.Module):
    """
    Fuse multimodal posterior distributions using a Product of Experts (PoE).
    
    Formulation:
    q(z|x1, x2, ...) ∝ ∏_m q(z|x_m)
    
    For Gaussian distributions:
    - Fused mean: μ_fused = (Σ_m Precision_m * μ_m) / (Σ_m Precision_m)
    - Fused variance: σ²_fused = 1 / (Σ_m Precision_m)
    """
    
    @staticmethod
    def poe_fusion(
        mus: List[torch.Tensor],
        logvars: List[torch.Tensor],
        masks: Optional[List[torch.Tensor]] = None,
        weights: Optional[List[torch.Tensor]] = None,
        eps: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse multimodal posterior distributions.
        
        Parameters:
            mus : List[torch.Tensor]
                Mean of each modality, with shape [batch_size, latent_dim].
            logvars : List[torch.Tensor]
                Log variance of each modality, with shape [batch_size, latent_dim].
            masks : List[torch.Tensor], optional
                Availability mask of each modality, with shape [batch_size], for missing modalities.
            eps : float
                Constant for numerical stability.
        
        Returns:
            fused_mu : torch.Tensor [batch_size, latent_dim]
                Fused mean.
            fused_logvar : torch.Tensor [batch_size, latent_dim]
                Fused log variance.
        """
        # Compute precision (the inverse of variance).
        precisions = [torch.exp(-logvar) for logvar in logvars]
        if weights is not None:
            precisions = [
                prec * weight.view(-1, 1).to(prec.device).clamp_min(0.0)
                for prec, weight in zip(precisions, weights)
            ]
        
        if masks is None:
            # All modalities are available.
            fused_precision = sum(precisions)
            weighted_means = sum([mu * prec for mu, prec in zip(mus, precisions)])
            fused_mu = weighted_means / (fused_precision + eps)
        else:
            # Apply availability masks to handle missing modalities.
            # masks: [batch_size] -> [batch_size, 1] for broadcasting
            fused_precision = sum([
                prec * mask.unsqueeze(-1) 
                for prec, mask in zip(precisions, masks)
            ])
            weighted_means = sum([
                mu * prec * mask.unsqueeze(-1) 
                for mu, prec, mask in zip(mus, precisions, masks)
            ])
            fused_mu = weighted_means / (fused_precision + eps)
        
        # Compute the fused log variance.
        fused_logvar = -torch.log(fused_precision + eps)
        
        return fused_mu, fused_logvar
    
    @staticmethod
    def poe_with_prior(
        mus: List[torch.Tensor],
        logvars: List[torch.Tensor],
        prior_mu: Optional[torch.Tensor] = None,
        prior_logvar: Optional[torch.Tensor] = None,
        masks: Optional[List[torch.Tensor]] = None,
        weights: Optional[List[torch.Tensor]] = None,
        eps: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform PoE fusion including a prior distribution.
        
        Including a prior gives:
        q(z|x1, x2, ...) ∝ p(z) * ∏_m q(z|x_m)
        
        Parameters:
            mus : List[torch.Tensor]
                Mean of each modality.
            logvars : List[torch.Tensor]
                Log variance of each modality.
            prior_mu : torch.Tensor, optional
                Prior mean, typically zero.
            prior_logvar : torch.Tensor, optional
                Prior log variance, typically zero (unit variance).
            masks : List[torch.Tensor], optional
                Modality availability masks.
            eps : float
                Constant for numerical stability.
        
        Returns:
            fused_mu : torch.Tensor
                Fused mean.
            fused_logvar : torch.Tensor
                Fused log variance.
        """
        # Add the prior.
        if prior_mu is not None and prior_logvar is not None:
            all_mus = [prior_mu] + mus
            all_logvars = [prior_logvar] + logvars
            if weights is not None:
                prior_weight = torch.ones(prior_mu.shape[0], device=prior_mu.device)
                weights = [prior_weight] + weights
        else:
            # Default prior: standard normal distribution N(0, I).
            batch_size = mus[0].shape[0]
            latent_dim = mus[0].shape[1]
            device = mus[0].device
            prior_mu = torch.zeros(batch_size, latent_dim, device=device)
            prior_logvar = torch.zeros(batch_size, latent_dim, device=device)
            all_mus = [prior_mu] + mus
            all_logvars = [prior_logvar] + logvars
            if weights is not None:
                prior_weight = torch.ones(batch_size, device=device)
                weights = [prior_weight] + weights
            
            if masks is not None:
                # The prior is always available.
                prior_mask = torch.ones(batch_size, device=device)
                masks = [prior_mask] + masks
        
        return ProductOfExperts.poe_fusion(all_mus, all_logvars, masks=masks, weights=weights, eps=eps)


def poe_fusion(
    mus: List[torch.Tensor],
    logvars: List[torch.Tensor],
    masks: Optional[List[torch.Tensor]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper for PoE fusion.
    
    Parameters:
        mus : List[torch.Tensor]
            Mean of each modality, with shape [batch_size, latent_dim].
        logvars : List[torch.Tensor]
            Log variance of each modality, with shape [batch_size, latent_dim].
        masks : List[torch.Tensor], optional
            Availability mask of each modality, with shape [batch_size].
    
    Returns:
        fused_mu : torch.Tensor [batch_size, latent_dim]
        fused_logvar : torch.Tensor [batch_size, latent_dim]
    """
    return ProductOfExperts.poe_fusion(mus, logvars, masks)







