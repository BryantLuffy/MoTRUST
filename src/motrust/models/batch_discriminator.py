"""
Batch discriminator: classify batch information.
Corresponds to the MIDAS Discriminator for adversarial batch-effect removal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchDiscriminator(nn.Module):
    """
    Batch discriminator.
    
    Predict batch labels from the biological representation c.
    During adversarial training:
    - The discriminator aims to predict batches correctly (maximize accuracy).
    - The VAE aims to confuse the discriminator (minimize its accuracy).
    This adversarial objective discourages batch information in c.
    
    Args:
        dim_c: Biological representation dimension, for example 28.
        n_batches: Number of batches, for example 8.
        hidden_dims: Hidden-layer dimensions; default [128, 64].
        dropout: Dropout probability; default 0.1.
    
    Example:
        >>> discriminator = BatchDiscriminator(dim_c=28, n_batches=8)
        >>> c = torch.randn(32, 28)  # 32 samples with 28 biological dimensions.
        >>> batch_id = torch.randint(0, 8, (32,))
        >>> 
        >>> # Discriminator predictions.
        >>> logits = discriminator(c)
        >>> print(logits.shape)  # torch.Size([32, 8])
        >>> 
        >>> # Compute the loss.
        >>> loss = discriminator.compute_loss(c, batch_id)
        >>> print(loss.item())  # Cross-entropy loss.
    """
    
    def __init__(self, dim_c, n_batches, hidden_dims=[128, 64], dropout=0.1):
        super().__init__()
        self.dim_c = dim_c
        self.n_batches = n_batches
        
        # Build the MLP discriminator.
        layers = []
        in_dim = dim_c
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        # Output layer for batch prediction (n_batches classes).
        layers.append(nn.Linear(in_dim, n_batches))
        
        self.predictor = nn.Sequential(*layers)
        
        # Loss function.
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, c):
        """
        Forward pass.
        
        Args:
            c: Biological representation with shape (batch_size, dim_c).
        
        Returns:
            logits: Batch prediction logits with shape (batch_size, n_batches).
        """
        return self.predictor(c)
    
    def compute_loss(self, c, batch_id):
        """
        Compute the discriminator loss.
        
        Args:
            c: Biological representation with shape (batch_size, dim_c).
            batch_id: True batch labels with shape (batch_size,) or (batch_size, 1).
        
        Returns:
            loss: Cross-entropy loss (scalar).
        """
        # Ensure batch_id is one-dimensional.
        if batch_id.dim() == 2:
            batch_id = batch_id.squeeze(1)
        
        # Predict.
        logits = self.forward(c)
        
        # Compute the cross-entropy loss.
        loss = self.cross_entropy(logits, batch_id)
        
        return loss
    
    def predict(self, c):
        """
        Predict batch labels during inference.
        
        Args:
            c: Biological representation with shape (batch_size, dim_c).
        
        Returns:
            predictions: Predicted batch labels with shape (batch_size,).
            probabilities: Prediction probabilities with shape (batch_size, n_batches).
        """
        with torch.no_grad():
            logits = self.forward(c)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions, probabilities
    
    def compute_accuracy(self, c, batch_id):
        """
        Compute discriminator accuracy.
        
        Args:
            c: Biological representation with shape (batch_size, dim_c).
            batch_id: True batch labels with shape (batch_size,) or (batch_size, 1).
        
        Returns:
            accuracy: Accuracy between 0 and 1.
        """
        if batch_id.dim() == 2:
            batch_id = batch_id.squeeze(1)
        
        predictions, _ = self.predict(c)
        accuracy = (predictions == batch_id).float().mean().item()
        
        return accuracy


if __name__ == '__main__':
    # Standalone smoke test.
    print("=" * 60)
    print("测试 BatchDiscriminator")
    print("=" * 60)
    
    # Create the discriminator.
    discriminator = BatchDiscriminator(dim_c=28, n_batches=8, hidden_dims=[128, 64])
    
    # Test inputs.
    batch_size = 32
    c = torch.randn(batch_size, 28)  # Biological representation.
    batch_id = torch.randint(0, 8, (batch_size,))  # Batch labels.
    
    print(f"\n输入 c shape: {c.shape}")
    print(f"输入 batch_id: {batch_id.tolist()}")
    
    # Test the forward pass.
    logits = discriminator(c)
    print(f"\n输出 logits shape: {logits.shape}")
    print(f"logits 示例值 (前3个):\n{logits[:3]}")
    
    # Test loss computation.
    loss = discriminator.compute_loss(c, batch_id)
    print(f"\n判别损失: {loss.item():.4f}")
    
    # Test prediction.
    predictions, probabilities = discriminator.predict(c)
    print(f"\n预测 shape: {predictions.shape}")
    print(f"预测值 (前10个): {predictions[:10].tolist()}")
    print(f"真实值 (前10个): {batch_id[:10].tolist()}")
    
    # Test accuracy.
    accuracy = discriminator.compute_accuracy(c, batch_id)
    print(f"\n判别准确率: {accuracy:.2%}")
    
    # Test gradient backpropagation.
    loss = discriminator.compute_loss(c, batch_id)
    loss.backward()
    print(f"\n✓ 梯度反向传播成功")
    
    print("\n" + "=" * 60)
    print("BatchDiscriminator 测试通过！")
    print("=" * 60)

