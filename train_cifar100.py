#!/usr/bin/env python3
"""
Minimal CIFAR-100 neural network training script with MLflow logging.
Uses Perceiver-style input: each pixel as token with RGB + Fourier positional encoding.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import mlflow
import mlflow.pytorch
import math
from typing import Dict, Tuple


model_name = "model_cifar100"

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentSelfAttentionBlock(nn.Module):
    """One block of latent Transformer: self-attention + MLP, with residuals & layer norm."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.1):
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

    def __init__(self, dim: int, depth: int = 6, num_heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.1):
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

class Perceiver(nn.Module):
    """Perceiver-style model with per-pixel tokens (RGB + positional encoding) for CIFAR-100."""

    def __init__(self, num_fourier_features: int = 64, latent_size: int = 256, latent_channels: int = 64) -> None:
        super(Perceiver, self).__init__()
        self.num_fourier_features = num_fourier_features
        self.image_size = 32  # CIFAR-100 images are 32x32
        self.latents = nn.Parameter(torch.randn(latent_size, latent_channels))  # learnable latent array
        self.latent_pos = nn.Parameter(torch.randn(latent_size, latent_channels))  # positional embedding for latents
        self.latent_transformer = LatentTransformer(latent_channels, depth=2)
        # Generate random Fourier feature matrix for 2D positional encoding
        # Shape: (2, num_fourier_features) for (x, y) coordinates
        self.register_buffer('fourier_matrix', torch.randn(2, num_fourier_features))
        
        # Each token: RGB (3) + Fourier features (2 * num_fourier_features)
        self.token_dim = 3 + 2 * num_fourier_features
        self.num_tokens = self.image_size * self.image_size  # 1024 tokens
        
        # Project input tokens to latent dimension
        self.input_projection = nn.Linear(self.token_dim, latent_channels)
        
        self.cross_attn_q = nn.Linear(latent_channels, latent_channels)
        self.cross_attn_k = nn.Linear(latent_channels, latent_channels)
        self.cross_attn_v = nn.Linear(latent_channels, latent_channels)
        self.cross_attn_out = nn.Linear(latent_channels, latent_channels)

        # Classification head
        self.classifier = nn.Linear(latent_channels, 100)

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
        
        # Stack coordinates: (32, 32, 2)
        coords = torch.stack([x_coords, y_coords], dim=-1)
        
        # Flatten to (1024, 2) and expand for batch
        coords = coords.view(-1, 2).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply Fourier features: (batch_size, 1024, 2) @ (2, num_fourier_features)
        fourier_proj = torch.matmul(coords, self.fourier_matrix)
        
        # Create sine and cosine features
        fourier_features = torch.cat([
            torch.cos(2 * math.pi * fourier_proj),
            torch.sin(2 * math.pi * fourier_proj)
        ], dim=-1)  # (batch_size, 1024, 2 * num_fourier_features)
        
        return fourier_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        latents = self.latents + self.latent_pos  # shape (N, D)
        latents = latents.unsqueeze(0).expand(batch_size, -1, -1)  # broadcast to batch
        
        # Reshape to tokens: (batch_size, 3, 32, 32) -> (batch_size, 1024, 3)
        rgb_tokens = x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 3)
        
        # Get positional encoding: (batch_size, 1024, 2 * num_fourier_features)
        pos_encoding = self.create_positional_encoding(batch_size, device)
        
        # Concatenate RGB with positional encoding: (batch_size, 1024, 3 + 2*num_fourier_features)
        input_tokens = torch.cat([rgb_tokens, pos_encoding], dim=-1)
        
        # Project input tokens to latent dimension
        input_tokens = self.input_projection(input_tokens)  # (batch_size, 1024, latent_channels)
        
        for i in range(8):
            q = self.cross_attn_q(latents)
            k = self.cross_attn_k(input_tokens)
            v = self.cross_attn_v(input_tokens)
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(latents.size(-1))
            attn = torch.softmax(attn, dim=-1)
            latents = self.cross_attn_out(attn @ v) # (batch_size, N, D)

            latents = self.latent_transformer(latents) # (batch_size, N, D)

        # Global average pooling over latents and classify
        pooled = latents.mean(dim=1)  # (batch_size, latent_channels)
        logits = self.classifier(pooled)  # (batch_size, 100)
        
        return logits


def create_data_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-100 train and test data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_model_architecture_info(model: nn.Module) -> Dict[str, str]:
    """
    Extract model architecture information for logging to MLflow.
    This helps track which architecture configuration was used for each run.
    """
    architecture_info: Dict[str, str] = {}

    architecture_info["model_structure"] = str(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    architecture_info["total_parameters"] = str(total_params)
    architecture_info["trainable_parameters"] = str(trainable_params)

    layer_info = []
    param_counts = []

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            module_str = str(module)
            if name:
                layer_info.append(f"{name}: {module_str}")
                layer_params = sum(p.numel() for p in module.parameters())
                if layer_params > 0:
                    param_counts.append(f"{name}: {layer_params:,}")

    architecture_info["layer_details"] = " | ".join(layer_info)
    architecture_info["layer_param_counts"] = " | ".join(param_counts)

    return architecture_info


def calculate_model_statistics(model: nn.Module) -> Dict[str, float]:
    """Calculate min, max, avg, and std of all weights and biases in the model."""
    all_weights = []
    all_biases = []

    for param_name, param in model.named_parameters():
        if param.requires_grad:
            param_data = param.data.flatten()
            if 'weight' in param_name:
                all_weights.extend(param_data.tolist())
            elif 'bias' in param_name:
                all_biases.extend(param_data.tolist())

    weights_tensor = torch.tensor(all_weights) if all_weights else torch.tensor([0.0])
    biases_tensor = torch.tensor(all_biases) if all_biases else torch.tensor([0.0])

    stats: Dict[str, float] = {
        'weights_min': weights_tensor.min().item(),
        'weights_max': weights_tensor.max().item(),
        'weights_mean': weights_tensor.mean().item(),
        'weights_std': weights_tensor.std().item(),
        'biases_min': biases_tensor.min().item(),
        'biases_max': biases_tensor.max().item(),
        'biases_mean': biases_tensor.mean().item(),
        'biases_std': biases_tensor.std().item(),
    }

    return stats


def calculate_gradient_statistics(model: nn.Module) -> Dict[str, float]:
    """
    Calculate gradient statistics to monitor training health and learning dynamics.
    """
    all_gradients = []
    grad_norms_per_layer = []

    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            grad_data = param.grad.data.flatten()
            all_gradients.extend(grad_data.tolist())
            layer_grad_norm = param.grad.data.norm().item()
            grad_norms_per_layer.append(layer_grad_norm)

    if not all_gradients:
        return {}

    gradients_tensor = torch.tensor(all_gradients)

    grad_min = gradients_tensor.min().item()
    grad_max = gradients_tensor.max().item()
    grad_mean = gradients_tensor.mean().item()
    grad_std = gradients_tensor.std().item()
    total_grad_norm = gradients_tensor.norm().item()

    stats: Dict[str, float] = {
        'gradients_min': grad_min,
        'gradients_max': grad_max,
        'gradients_mean': grad_mean,
        'gradients_std': grad_std,
        'total_grad_norm': total_grad_norm,
        'avg_layer_grad_norm': sum(grad_norms_per_layer) / len(grad_norms_per_layer) if grad_norms_per_layer else 0.0,
    }

    return stats


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, num_epochs: int) -> None:
    """Train the model and log metrics to MLflow."""
    model.train()

    batch_count = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            if batch_idx % 100 == 0:
                grad_stats = calculate_gradient_statistics(model)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            batch_count += 1
            if batch_idx % 100 == 0:
                mlflow.log_metric("batch_train_loss", loss.item(), step=batch_count)

                model_stats = calculate_model_statistics(model)
                for stat_name, stat_value in model_stats.items():
                    mlflow.log_metric("model_stats/batch_" + stat_name, stat_value, step=batch_count)

                for stat_name, stat_value in grad_stats.items():
                    mlflow.log_metric("gradients/batch_" + stat_name, stat_value, step=batch_count)

                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')

            # Evaluate test accuracy every 200 batches
            if batch_idx % 200 == 0:
                test_accuracy = evaluate_model(model, test_loader, device)
                mlflow.log_metric("test_accuracy_batch", test_accuracy, step=batch_count)
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Test Accuracy: {test_accuracy:.2f}%')
                model.train()  # Set back to training mode after evaluation

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate the model and return accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main() -> None:
    """Main training function for CIFAR-100."""
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("cifar100-training")

    with mlflow.start_run():
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("device", str(device))

        train_loader, test_loader = create_data_loaders(batch_size)

        model: nn.Module = Perceiver().to(device)
        # Print and verify parameter count
        total_params: int = sum(p.numel() for p in model.parameters())
        print(f'Total parameters: {total_params}')

        criterion: nn.Module = nn.CrossEntropyLoss()
        optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        mlflow.log_param("architecture/optimizer", optimizer.__class__.__name__)
        mlflow.log_param("architecture/criterion", criterion.__class__.__name__)
        arch_info = get_model_architecture_info(model)
        for param_name, param_value in arch_info.items():
            mlflow.log_param(f"architecture/{param_name}", param_value)

        print("Starting training...")
        train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs)

        print("Evaluating model...")
        test_accuracy = evaluate_model(model, test_loader, device)
        mlflow.log_metric("test_accuracy", test_accuracy)
        print(f'Test Accuracy: {test_accuracy:.2f}%')

        mlflow.pytorch.log_model(model, name=model_name)

        print("Training completed and logged to MLflow!")


if __name__ == "__main__":
    main()


