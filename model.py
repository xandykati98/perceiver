#!/usr/bin/env python3
"""
Perceiver-style model with per-pixel tokens (RGB + Fourier positional encoding).
"""

import math
import torch
import torch.nn as nn


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
    """Perceiver-style model with per-pixel tokens (RGB + positional encoding)."""

    def __init__(
        self,
        num_classes: int,
        num_fourier_features: int,
        latent_size: int,
        latent_channels: int,
        num_cross_attn_layers: int,
        latent_transformer_depth: int,
        latent_transformer_num_heads: int,
        latent_transformer_mlp_ratio: int,
        dropout: float,
        image_size: int,
    ) -> None:
        super(Perceiver, self).__init__()
        self.num_fourier_features = num_fourier_features
        self.image_size = image_size
        self.latents = nn.Parameter(torch.randn(latent_size, latent_channels))  # learnable latent array
        self.latent_pos = nn.Parameter(torch.randn(latent_size, latent_channels))  # positional embedding for latents
        self.latent_transformer = LatentTransformer(
            dim=latent_channels,
            depth=latent_transformer_depth,
            num_heads=latent_transformer_num_heads,
            mlp_ratio=latent_transformer_mlp_ratio,
            dropout=dropout,
        )
        # Generate random Fourier feature matrix for 2D positional encoding
        # Shape: (2, num_fourier_features) for (x, y) coordinates
        self.register_buffer('fourier_matrix', torch.randn(2, num_fourier_features))
        
        # Each token: RGB (3) + Fourier features (2 * num_fourier_features)
        self.token_dim = 3 + 2 * num_fourier_features
        self.num_tokens = self.image_size * self.image_size
        
        # Cross-attention blocks project directly from input dim to latent dim (per paper)
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(latent_dim=latent_channels, input_dim=self.token_dim, dropout=dropout)
            for _ in range(num_cross_attn_layers)
        ])

        # Classification head
        self.classifier = nn.Linear(latent_channels, num_classes)

    def create_positional_encoding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create Fourier positional encoding for 2D coordinates."""
        # Create coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.image_size, device=device),
            torch.arange(self.image_size, device=device),
            indexing='ij'
        )
        
        # Normalize coordinates to [0, 1]
        x_coords = x_coords.float() / (self.image_size - 1)
        y_coords = y_coords.float() / (self.image_size - 1)
        
        # Stack coordinates: (image_size, image_size, 2)
        coords = torch.stack([x_coords, y_coords], dim=-1)
        
        # Flatten to (num_tokens, 2) and expand for batch
        coords = coords.view(-1, 2).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply Fourier features: (batch_size, num_tokens, 2) @ (2, num_fourier_features)
        fourier_proj = torch.matmul(coords, self.fourier_matrix)
        
        # Create sine and cosine features
        fourier_features = torch.cat([
            torch.cos(2 * math.pi * fourier_proj),
            torch.sin(2 * math.pi * fourier_proj)
        ], dim=-1)  # (batch_size, num_tokens, 2 * num_fourier_features)
        
        return fourier_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        latents = self.latents + self.latent_pos  # shape (N, D)
        latents = latents.unsqueeze(0).expand(batch_size, -1, -1)  # broadcast to batch
        
        # Reshape to tokens: (batch_size, 3, H, W) -> (batch_size, num_tokens, 3)
        rgb_tokens = x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 3)
        
        # Get positional encoding: (batch_size, num_tokens, 2 * num_fourier_features)
        pos_encoding = self.create_positional_encoding(batch_size, device)
        
        # Concatenate RGB with positional encoding: (batch_size, num_tokens, 3 + 2*num_fourier_features)
        input_tokens = torch.cat([rgb_tokens, pos_encoding], dim=-1)
        
        for cross_attn in self.cross_attn_blocks:
            latents = cross_attn(latents, input_tokens)  # (batch_size, N, D)
            latents = self.latent_transformer(latents)  # (batch_size, N, D)

        # Global average pooling over latents and classify
        pooled = latents.mean(dim=1)  # (batch_size, latent_channels)
        logits = self.classifier(pooled)  # (batch_size, num_classes)
        
        return logits

