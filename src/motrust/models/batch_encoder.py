"""
Batch encoder: encode batch information into the latent space.
Corresponds to the MIDAS S_Encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchEncoder(nn.Module):
    """
    Encode batch information into the latent space.
    
    Encode discrete batch IDs (0, 1, 2, ...) as Gaussian latent parameters (mu, logvar).
    Fuse these parameters into the main latent space through PoE.
    
    Args:
        n_batches: Number of batches, for example 8.
        latent_dim: Latent dimension, for example 32 = 28 + 4.
        hidden_dims: Hidden-layer dimensions; default [128].
        dropout: Dropout probability; default 0.1.
    
    Example:
        >>> batch_encoder = BatchEncoder(n_batches=8, latent_dim=32)
        >>> batch_id = torch.tensor([0, 1, 2, 0, 1])  # shape: (5,)
        >>> mu, logvar = batch_encoder(batch_id)
        >>> print(mu.shape, logvar.shape)  # torch.Size([5, 32]), torch.Size([5, 32])
    """
    
    def __init__(self, n_batches, latent_dim, hidden_dims=[128], dropout=0.1):
        super().__init__()
        self.n_batches = n_batches
        self.latent_dim = latent_dim
        
        # Build the MLP encoder.
        layers = []
        in_dim = n_batches  # Dimension of the one-hot encoding.
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        # Output mu and logvar, requiring latent_dim * 2 units.
        layers.append(nn.Linear(in_dim, latent_dim * 2))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, batch_id):
        """
        Forward pass.
        
        Args:
            batch_id: Integer tensor with shape (batch_size,) or (batch_size, 1).
                      Each element is a batch index in [0, n_batches).
        
        Returns:
            mu: Mean with shape (batch_size, latent_dim).
            logvar: Log variance with shape (batch_size, latent_dim).
        """
        # Ensure the tensor is one-dimensional.
        if batch_id.dim() == 2:
            batch_id = batch_id.squeeze(1)
        
        # One-hot encoding.
        # Example: batch_id=[0,1,2,0] -> one_hot=[[1,0,0],[0,1,0],[0,0,1],[1,0,0]].
        one_hot = F.one_hot(batch_id, num_classes=self.n_batches).float()
        
        # Pass through the encoder.
        mu_logvar = self.encoder(one_hot)
        
        # Split mu and logvar.
        mu, logvar = mu_logvar.chunk(2, dim=1)
        
        return mu, logvar


if __name__ == '__main__':
    # Standalone smoke test.
    print("=" * 60)
    print("测试 BatchEncoder")
    print("=" * 60)
    
    # Create the encoder.
    batch_encoder = BatchEncoder(n_batches=8, latent_dim=32, hidden_dims=[128])
    
    # Test inputs.
    batch_id = torch.tensor([0, 1, 2, 0, 1, 3, 4, 5])
    print(f"\n输入 batch_id: {batch_id.tolist()}")
    print(f"batch_id shape: {batch_id.shape}")
    
    # Forward pass.
    mu, logvar = batch_encoder(batch_id)
    
    print(f"\n输出 mu shape: {mu.shape}")
    print(f"输出 logvar shape: {logvar.shape}")
    print(f"\nmu 示例值 (前3个):\n{mu[:3]}")
    print(f"\nlogvar 示例值 (前3个):\n{logvar[:3]}")
    
    # Test gradient backpropagation.
    loss = mu.sum()
    loss.backward()
    print(f"\n✓ 梯度反向传播成功")
    
    print("\n" + "=" * 60)
    print("BatchEncoder 测试通过！")
    print("=" * 60)

