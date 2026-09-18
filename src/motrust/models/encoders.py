"""
Encoder modules for RNA and ATAC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEncoder(nn.Module):
    """Base encoder."""
    
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build the encoder network.
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output layers for the mean and log variance.
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class RNAEncoder(BaseEncoder):
    """RNA encoder."""
    
    def __init__(self, input_dim, latent_dim, hidden_dims=[512, 256], dropout=0.1):
        super().__init__(input_dim, hidden_dims, latent_dim, dropout)


class ATACEncoder(BaseEncoder):
    """ATAC encoder."""
    
    def __init__(self, input_dim, latent_dim, hidden_dims=[512, 256], dropout=0.1):
        super().__init__(input_dim, hidden_dims, latent_dim, dropout)


class ADTEncoder(BaseEncoder):
    """
    ADT encoder.
    
    Reuse the BaseEncoder architecture shared by the RNA and ATAC encoders
    to add protein/ADT measurements as a third modality in the multimodal VAE.
    """
    
    def __init__(self, input_dim, latent_dim, hidden_dims=[512, 256], dropout=0.1):
        super().__init__(input_dim, hidden_dims, latent_dim, dropout)