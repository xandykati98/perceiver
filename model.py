#!/usr/bin/env python3
"""
Perceiver model following the original paper (arXiv:2103.03206).
- Position coords normalized to [-1, 1]
- Deterministic Fourier frequency bands (linear spacing + raw coords)
- Weight sharing: first cross-attn/latent-transformer unique, rest share weights
"""

import math
import torch
import torch.nn as nn


def create_fourier_features(coords: torch.Tensor, num_bands: int, max_freq: float) -> torch.Tensor:
    """Create deterministic Fourier positional encoding.
    
    Args:
        coords: (batch, num_tokens, num_dims) coordinates in [-1, 1]
        num_bands: Number of frequency bands per dimension
        max_freq: Maximum frequency for band scaling
    
    Returns:
        (batch, num_tokens, num_dims * ((num_bands * 2) + 1)) Fourier features
    """
    if max_freq <= 0.0:
        raise ValueError(f"max_freq must be > 0, got {max_freq}")

    # Linear-spaced frequencies from 1 to max_freq/2 (inclusive)
    freq_bands = torch.linspace(
        1.0,
        max_freq / 2.0,
        num_bands,
        device=coords.device,
        dtype=coords.dtype,
    )
    
    # coords: (batch, num_tokens, num_dims) -> (batch, num_tokens, num_dims, 1)
    # freq_bands: (num_bands,) -> (1, 1, 1, num_bands)
    coords = coords.unsqueeze(-1)
    orig_coords = coords
    freq_bands = freq_bands.view(1, 1, 1, num_bands)
    
    # (batch, num_tokens, num_dims, num_bands)
    angles = coords * freq_bands * math.pi
    
    # Concatenate sin and cos, then append raw coords (per-axis): (..., (num_bands*2)+1)
    fourier_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    fourier_features = torch.cat([fourier_features, orig_coords], dim=-1)
    fourier_features = fourier_features.view(fourier_features.size(0), fourier_features.size(1), -1)
    
    return fourier_features


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_val, gates = x.chunk(2, dim=-1)
        return x_val * nn.functional.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if heads <= 0:
            raise ValueError(f"heads must be > 0, got {heads}")
        if query_dim % heads != 0:
            raise ValueError(f"query_dim ({query_dim}) must be divisible by heads ({heads})")

        self.heads = heads
        self.dim_head = query_dim // heads
        inner_dim = self.dim_head * heads

        self.scale = self.dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # x: (b, n, d), context: (b, m, c), mask: (b, m) bool, True = keep
        b, n, _ = x.shape
        _, m, _ = context.shape
        h = self.heads

        q = self.to_q(x)  # (b, n, h*dh)
        kv = self.to_kv(context)  # (b, m, 2*h*dh)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(b, n, h, self.dim_head).transpose(1, 2).contiguous().view(b * h, n, self.dim_head)
        k = k.view(b, m, h, self.dim_head).transpose(1, 2).contiguous().view(b * h, m, self.dim_head)
        v = v.view(b, m, h, self.dim_head).transpose(1, 2).contiguous().view(b * h, m, self.dim_head)

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            if mask.dtype != torch.bool:
                raise TypeError(f"mask must be bool tensor, got {mask.dtype}")
            if mask.shape != (b, m):
                raise ValueError(f"mask must have shape (batch, context_len)=({b}, {m}), got {tuple(mask.shape)}")
            mask_expanded = mask.unsqueeze(1).repeat(h, 1, 1)  # (b*h, 1, m)
            max_neg = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~mask_expanded, max_neg)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = out.view(b, h, n, self.dim_head).transpose(1, 2).contiguous().view(b, n, h * self.dim_head)
        return self.to_out(out)


class PreNormCrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)
        self.attn = Attention(query_dim=dim, context_dim=context_dim, heads=heads, dropout=dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.attn(self.norm(x), context=self.norm_context(context), mask=mask)


class PreNormSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(query_dim=dim, context_dim=dim, heads=heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        return self.attn(x_norm, context=x_norm, mask=None)


class PreNormFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, mult=mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(self.norm(x))


class CrossAttentionBlock(nn.Module):
    """Compatibility wrapper for older naming in this repo.
    cross-attn + residual, then FFN + residual.
    """

    def __init__(self, latent_dim: int, input_dim: int, cross_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = PreNormCrossAttention(dim=latent_dim, context_dim=input_dim, heads=cross_heads, dropout=dropout)
        self.cross_ff = PreNormFeedForward(dim=latent_dim, mult=4, dropout=dropout)

    def forward(self, latents: torch.Tensor, input_tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = self.cross_attn(latents, context=input_tokens, mask=mask) + latents
        x = self.cross_ff(x) + x
        return x


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
        cross_heads: int,
        dropout: float,
        image_size: int,
        max_freq: float,
    ) -> None:
        super(Perceiver, self).__init__()
        self.num_fourier_bands = num_fourier_bands
        self.image_size = image_size
        self.num_cross_attn_iterations = num_cross_attn_iterations
        self.max_freq = max_freq
        
        # Learnable latent array with positional embedding
        self.latents = nn.Parameter(torch.randn(latent_size, latent_channels))
        self.latent_pos = nn.Parameter(torch.randn(latent_size, latent_channels))
        
        # Each token: RGB (3) + Fourier features (2 dims * ((num_bands*2)+1))
        self.token_dim = 3 + 2 * ((num_fourier_bands * 2) + 1)
        self.num_tokens = self.image_size * self.image_size
        
        # First cross-attention and latent transformer (unique weights)
        self.first_cross_attn = CrossAttentionBlock(
            latent_dim=latent_channels, input_dim=self.token_dim, cross_heads=cross_heads, dropout=dropout
        )
        self.first_self_attn_blocks = nn.ModuleList([
            nn.ModuleList([
                PreNormSelfAttention(
                    dim=latent_channels,
                    heads=latent_transformer_num_heads,
                    dropout=dropout,
                ),
                PreNormFeedForward(dim=latent_channels, mult=4, dropout=dropout),
            ])
            for _ in range(latent_transformer_depth)
        ])
        
        # Shared cross-attention and latent transformer (for iterations 2+)
        self.shared_cross_attn = CrossAttentionBlock(
            latent_dim=latent_channels, input_dim=self.token_dim, cross_heads=cross_heads, dropout=dropout
        ) if num_cross_attn_iterations > 1 else None
        self.shared_self_attn_blocks = nn.ModuleList([
            nn.ModuleList([
                PreNormSelfAttention(
                    dim=latent_channels,
                    heads=latent_transformer_num_heads,
                    dropout=dropout,
                ),
                PreNormFeedForward(dim=latent_channels, mult=4, dropout=dropout),
            ])
            for _ in range(latent_transformer_depth)
        ]) if num_cross_attn_iterations > 1 else None

        # Classification head
        self.classifier_norm = nn.LayerNorm(latent_channels)
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
        fourier_features = create_fourier_features(coords, self.num_fourier_bands, self.max_freq)
        
        return fourier_features

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        latents = self.latents + self.latent_pos  # shape (N, D)
        latents = latents.unsqueeze(0).expand(batch_size, -1, -1)  # broadcast to batch
        
        # Accept either NCHW or NHWC; normalize to NHWC for tokenization
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got shape {tuple(x.shape)}")

        if x.shape[1] == 3:
            # NCHW -> NHWC
            x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        elif x.shape[-1] == 3:
            # already NHWC
            x_nhwc = x
        else:
            raise ValueError(f"Expected input with 3 channels, got shape {tuple(x.shape)}")

        h = x_nhwc.size(1)
        w = x_nhwc.size(2)
        if h != self.image_size or w != self.image_size:
            raise ValueError(f"Expected input spatial size ({self.image_size}, {self.image_size}), got ({h}, {w})")

        rgb_tokens = x_nhwc.view(batch_size, -1, 3)
        
        # Get positional encoding: (batch_size, num_tokens, 2 * ((num_bands*2)+1))
        pos_encoding = self.create_positional_encoding(batch_size, device)
        
        # Concatenate RGB with positional encoding
        input_tokens = torch.cat([rgb_tokens, pos_encoding], dim=-1)

        # Normalize mask to (batch, num_tokens) if provided (True = keep)
        flat_mask: torch.Tensor | None
        if mask is None:
            flat_mask = None
        else:
            if mask.dtype != torch.bool:
                raise TypeError(f"mask must be bool tensor, got {mask.dtype}")
            if mask.ndim == 2 and mask.shape == (batch_size, self.num_tokens):
                flat_mask = mask
            elif mask.ndim == 3 and mask.shape[0] == batch_size:
                flat_mask = mask.contiguous().view(batch_size, -1)
                if flat_mask.shape[1] != self.num_tokens:
                    raise ValueError(f"mask flattened length must be {self.num_tokens}, got {flat_mask.shape[1]}")
            elif mask.ndim == 4 and mask.shape[0] == batch_size:
                flat_mask = mask.contiguous().view(batch_size, -1)
                if flat_mask.shape[1] != self.num_tokens:
                    raise ValueError(f"mask flattened length must be {self.num_tokens}, got {flat_mask.shape[1]}")
            else:
                raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}; expected (b, tokens) or (b, h, w)")
        
        # First iteration (unique weights)
        latents = self.first_cross_attn(latents, input_tokens, mask=flat_mask)
        for self_attn, self_ff in self.first_self_attn_blocks:
            latents = self_attn(latents) + latents
            latents = self_ff(latents) + latents
        
        # Subsequent iterations (shared weights)
        for _ in range(self.num_cross_attn_iterations - 1):
            if self.shared_cross_attn is None or self.shared_self_attn_blocks is None:
                raise RuntimeError("num_cross_attn_iterations > 1 but shared blocks were not initialized")
            latents = self.shared_cross_attn(latents, input_tokens, mask=flat_mask)
            for self_attn, self_ff in self.shared_self_attn_blocks:
                latents = self_attn(latents) + latents
                latents = self_ff(latents) + latents

        # Global average pooling over latents and classify
        pooled = latents.mean(dim=1)  # (batch_size, latent_channels)
        logits = self.classifier(self.classifier_norm(pooled))  # (batch_size, num_classes)
        
        return logits

