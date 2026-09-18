"""
Batch discriminator for adversarial batch-effect removal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BatchDiscriminator(nn.Module):
    """
    Batch discriminator that distinguishes latent representations from different batches.
    
    Used for adversarial training:
    - Discriminator: learn to distinguish latent representations from different batches.
    - VAE: learn latent representations that do not reveal batch identity.
    
    Parameters:
        latent_dim : int
            Input latent dimension, usually dim_c (the biological space).
        n_batches : int
            Number of batches.
        hidden_dims : list
            Hidden-layer dimensions.
        dropout : float
            Dropout probability.
    """
    
    def __init__(self, 
                 latent_dim: int,
                 n_batches: int,
                 hidden_dims: list = [128, 64],
                 dropout: float = 0.1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_batches = n_batches
        
        # Build the discriminator network.
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer for batch classification.
        layers.append(nn.Linear(prev_dim, n_batches))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
            z : torch.Tensor [batch_size, latent_dim]
                Latent representation, usually the biological component c.
        
        Returns:
            logits : torch.Tensor [batch_size, n_batches]
                Batch classification logits.
        """
        return self.net(z)
    
    def predict_batch(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict batch labels.
        
        Parameters:
            z : torch.Tensor [batch_size, latent_dim]
        
        Returns:
            batch_pred : torch.Tensor [batch_size]
                Predicted batch labels.
        """
        logits = self.forward(z)
        return logits.argmax(dim=1)
    
    def compute_loss(self, z: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the discriminator loss.
        
        Parameters:
            z : torch.Tensor [batch_size, latent_dim]
                Latent representation.
            batch_ids : torch.Tensor [batch_size]
                True batch labels from 0 to n_batches - 1.
        
        Returns:
            loss : torch.Tensor
                Cross-entropy loss.
        """
        logits = self.forward(z)
        return F.cross_entropy(logits, batch_ids)


class AdversarialTrainer:
    """
    Helper for adversarial training.
    
    Manage adversarial training of the discriminator and VAE.
    """
    
    def __init__(self, 
                 discriminator: BatchDiscriminator,
                 disc_optimizer: torch.optim.Optimizer,
                 n_disc_iter: int = 1):
        """
        Parameters:
            discriminator : BatchDiscriminator
                Batch discriminator.
            disc_optimizer : torch.optim.Optimizer
                Discriminator optimizer.
            n_disc_iter : int
                Number of discriminator updates before each VAE update.
        """
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer
        self.n_disc_iter = n_disc_iter
    
    def train_discriminator(self, 
                           z: torch.Tensor, 
                           batch_ids: torch.Tensor) -> float:
        """
        Train the discriminator.
        
        Parameters:
            z : torch.Tensor [batch_size, latent_dim]
                Detached latent representation, excluded from gradient computation.
            batch_ids : torch.Tensor [batch_size]
                True batch labels.
        
        Returns:
            disc_loss : float
                Discriminator loss.
        """
        self.discriminator.train()
        self.disc_optimizer.zero_grad()
        
        # Compute the loss.
        disc_loss = self.discriminator.compute_loss(z.detach(), batch_ids)
        
        # Backpropagation.
        disc_loss.backward()
        self.disc_optimizer.step()
        
        return disc_loss.item()
    
    def compute_adversarial_loss(self, 
                                 z: torch.Tensor, 
                                 batch_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the adversarial loss for VAE training.
        
        The VAE aims to prevent the discriminator from distinguishing batches:
        - Correct discriminator predictions should penalize the VAE.
        - Incorrect discriminator predictions should favor the VAE.
        
        Implementation: encourage equal predicted probabilities for all batches.
        Alternatively, negate the discriminator loss to maximize it.
        
        Parameters:
            z : torch.Tensor [batch_size, latent_dim]
                Latent representation requiring gradients.
            batch_ids : torch.Tensor [batch_size]
                True batch labels.
        
        Returns:
            adv_loss : torch.Tensor
                Adversarial loss to minimize.
        """
        self.discriminator.eval()  # Use the discriminator in evaluation mode.
        
        # Obtain discriminator predictions.
        logits = self.discriminator(z)
        
        # Method 1: encourage uniform predicted probabilities across batches.
        # Target: each batch has probability 1/n_batches.
        target_probs = torch.ones_like(logits) / self.discriminator.n_batches
        adv_loss = F.kl_div(
            F.log_softmax(logits, dim=1),
            target_probs,
            reduction='batchmean'
        )
        
        # Method 2: negate the discriminator loss to maximize it.
        # adv_loss = -self.discriminator.compute_loss(z, batch_ids)
        
        return adv_loss

