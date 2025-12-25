#!/usr/bin/env python3
"""
Perceiver model following the original paper (arXiv:2103.03206).
- Position coords normalized to [-1, 1]
- Deterministic log-spaced Fourier frequency bands
- Weight sharing: first cross-attn/latent-transformer unique, rest share weights
"""

import math
import torch
import torch.nn as nn


def create_fourier_features(coords: torch.Tensor, num_bands: int, max_resolution: int) -> torch.Tensor:
    """Create deterministic Fourier positional encoding with log-spaced frequency bands.
    
    Args:
        coords: (batch, num_tokens, num_dims) coordinates in [-1, 1]
        num_bands: Number of frequency bands per dimension
        max_resolution: Maximum resolution for frequency scaling
    
    Returns:
        (batch, num_tokens, num_dims * num_bands * 2) Fourier features
    """
    # Log-spaced frequencies from 1 to max_resolution/2
    # freq_bands shape: (num_bands,)
    freq_bands = torch.linspace(
        1.0,
        max_resolution / 2.0,
        num_bands,
        device=coords.device,
        dtype=coords.dtype,
    )
    
    # coords: (batch, num_tokens, num_dims) -> (batch, num_tokens, num_dims, 1)
    # freq_bands: (num_bands,) -> (1, 1, 1, num_bands)
    coords = coords.unsqueeze(-1)
    freq_bands = freq_bands.view(1, 1, 1, num_bands)
    
    # (batch, num_tokens, num_dims, num_bands)
    angles = coords * freq_bands * math.pi
    
    # Concatenate sin and cos: (batch, num_tokens, num_dims * num_bands * 2)
    fourier_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    fourier_features = fourier_features.view(coords.size(0), coords.size(1), -1)
    
    return fourier_features


class LatentSelfAttentionBlock(nn.Module):
    """One block of latent Transformer: self-attention + MLP, with residuals & layer norm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, N, D)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h)   # self-attention on latents
        x = x + attn_out                   # residual connection

        h = self.ln2(x)
        mlp_out = self.mlp(h)
        x = x + mlp_out                    # residual connection
        return x


class LatentTransformer(nn.Module):
    """Stack of latent self-attention blocks."""

    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            LatentSelfAttentionBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, N, D)
        for blk in self.blocks:
            x = blk(x)
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with layer norm, projections, and dropout.
    
    Per Perceiver paper: K/V project directly from input_dim to latent_dim,
    with separate LayerNorms for latents (Q) and inputs (K/V).
    """

    def __init__(self, latent_dim: int, input_dim: int, dropout: float):
        super().__init__()
        self.ln_latents = nn.LayerNorm(latent_dim)
        self.ln_inputs = nn.LayerNorm(input_dim)
        self.q = nn.Linear(latent_dim, latent_dim)
        self.k = nn.Linear(input_dim, latent_dim)
        self.v = nn.Linear(input_dim, latent_dim)
        self.out = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = latent_dim ** -0.5

    def forward(self, latents: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        # latents: (batch, N, D), input_tokens: (batch, M, C)
        q = self.q(self.ln_latents(latents))
        k = self.k(self.ln_inputs(input_tokens))
        v = self.v(self.ln_inputs(input_tokens))
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = self.out(attn @ v)
        return latents + out  # residual connection


class Perceiver(nn.Module):
    """Perceiver model following the original paper architecture.
    
    Key design choices per paper:
    - Position coords normalized to [-1, 1]
    - Deterministic log-spaced Fourier frequency bands  
    - Weight sharing: first cross-attn/latent-transformer unique, rest share weights
    """

    def __init__(
        self,
        num_classes: int,
        num_fourier_bands: int,
        latent_size: int,
        latent_channels: int,
        num_cross_attn_iterations: int,
        latent_transformer_depth: int,
        latent_transformer_num_heads: int,
        dropout: float,
        image_size: int,
    ) -> None:
        super(Perceiver, self).__init__()
        self.num_fourier_bands = num_fourier_bands
        self.image_size = image_size
        self.num_cross_attn_iterations = num_cross_attn_iterations
        
        # Learnable latent array with positional embedding
        self.latents = nn.Parameter(torch.randn(latent_size, latent_channels))
        self.latent_pos = nn.Parameter(torch.randn(latent_size, latent_channels))
        
        # Each token: RGB (3) + Fourier features (2 dims * num_bands * 2 for sin/cos)
        self.token_dim = 3 + 2 * num_fourier_bands * 2
        self.num_tokens = self.image_size * self.image_size
        
        # First cross-attention and latent transformer (unique weights)
        self.first_cross_attn = CrossAttentionBlock(
            latent_dim=latent_channels, input_dim=self.token_dim, dropout=dropout
        )
        self.first_latent_transformer = LatentTransformer(
            dim=latent_channels,
            depth=latent_transformer_depth,
            num_heads=latent_transformer_num_heads,
            mlp_ratio=1,  # No bottleneck per paper
            dropout=dropout,
        )
        
        # Shared cross-attention and latent transformer (for iterations 2+)
        self.shared_cross_attn = CrossAttentionBlock(
            latent_dim=latent_channels, input_dim=self.token_dim, dropout=dropout
        ) if num_cross_attn_iterations > 1 else None
        self.shared_latent_transformer = LatentTransformer(
            dim=latent_channels,
            depth=latent_transformer_depth,
            num_heads=latent_transformer_num_heads,
            mlp_ratio=1,  # No bottleneck per paper
            dropout=dropout,
        ) if num_cross_attn_iterations > 1 else None

        # Classification head
        self.classifier = nn.Linear(latent_channels, num_classes)

    def create_positional_encoding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create Fourier positional encoding with coords in [-1, 1] per paper."""
        # Create coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.image_size, device=device),
            torch.arange(self.image_size, device=device),
            indexing='ij'
        )
        
        # Normalize coordinates to [-1, 1] per paper
        x_norm = (x_coords.float() / (self.image_size - 1)) * 2.0 - 1.0
        y_norm = (y_coords.float() / (self.image_size - 1)) * 2.0 - 1.0
        
        # Stack coordinates: (image_size, image_size, 2)
        coords = torch.stack([x_norm, y_norm], dim=-1)
        
        # Flatten to (num_tokens, 2) and expand for batch
        coords = coords.view(-1, 2).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply deterministic Fourier features
        fourier_features = create_fourier_features(coords, self.num_fourier_bands, self.image_size)
        
        return fourier_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        latents = self.latents + self.latent_pos  # shape (N, D)
        latents = latents.unsqueeze(0).expand(batch_size, -1, -1)  # broadcast to batch
        
        # Reshape to tokens: (batch_size, 3, H, W) -> (batch_size, num_tokens, 3)
        rgb_tokens = x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 3)
        
        # Get positional encoding: (batch_size, num_tokens, 2 * num_bands * 2)
        pos_encoding = self.create_positional_encoding(batch_size, device)
        
        # Concatenate RGB with positional encoding
        input_tokens = torch.cat([rgb_tokens, pos_encoding], dim=-1)
        
        # First iteration (unique weights)
        latents = self.first_cross_attn(latents, input_tokens)
        latents = self.first_latent_transformer(latents)
        
        # Subsequent iterations (shared weights)
        for _ in range(self.num_cross_attn_iterations - 1):
            latents = self.shared_cross_attn(latents, input_tokens)
            latents = self.shared_latent_transformer(latents)

        # Global average pooling over latents and classify
        pooled = latents.mean(dim=1)  # (batch_size, latent_channels)
        logits = self.classifier(pooled)  # (batch_size, num_classes)
        
        return logits

