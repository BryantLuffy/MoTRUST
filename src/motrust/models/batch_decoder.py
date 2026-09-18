"""
Batch decoder: decode batch information from the latent space.
Corresponds to the MIDAS S_Decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchDecoder(nn.Module):
    """
    Decode batch information from the latent space.
    
    Decode latent technical factors u (dim_u) into batch ID logits (n_batches).
    
    Args:
        n_batches: Number of batches, for example 4.
        dim_u: Technical-factor dimension, for example 2.
        hidden_dims: Hidden-layer dimensions; default [64].
        dropout: Dropout probability; default 0.1.
    
    Example:
        >>> batch_decoder = BatchDecoder(n_batches=4, dim_u=2)
        >>> u = torch.randn(5, 2)  # shape: (5, 2)
        >>> batch_logits = batch_decoder(u)
        >>> print(batch_logits.shape)  # torch.Size([5, 4])
    """
    
    def __init__(self, n_batches, dim_u, hidden_dims=[64], dropout=0.1):
        super().__init__()
        self.n_batches = n_batches
        self.dim_u = dim_u
        
        # Build the MLP decoder.
        layers = []
        in_dim = dim_u
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        # Output layer for batch logits (n_batches).
        layers.append(nn.Linear(in_dim, n_batches))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, u):
        """
        Forward pass.
        
        Args:
            u: Technical-factor tensor with shape (batch_size, dim_u).
        
        Returns:
            batch_logits: Batch ID logits with shape (batch_size, n_batches).
        """
        return self.decoder(u)


if __name__ == '__main__':
    # Standalone smoke test.
    print("=" * 60)
    print("测试 BatchDecoder")
    print("=" * 60)
    
    # Create the decoder.
    batch_decoder = BatchDecoder(n_batches=4, dim_u=2, hidden_dims=[64])
    
    # Test inputs.
    u = torch.randn(5, 2)
    print(f"\n输入 u shape: {u.shape}")
    
    # Forward pass.
    batch_logits = batch_decoder(u)
    
    print(f"\n输出 batch_logits shape: {batch_logits.shape}")
    print(f"\nbatch_logits 示例值:\n{batch_logits}")
    
    # Convert logits to probabilities.
    batch_probs = F.softmax(batch_logits, dim=1)
    print(f"\n批次概率 (前3个):\n{batch_probs[:3]}")
    
    # Test gradient backpropagation.
    loss = batch_logits.sum()
    loss.backward()
    print(f"\n✓ 梯度反向传播成功")
    
    print("\n" + "=" * 60)
    print("BatchDecoder 测试通过！")
    print("=" * 60)
