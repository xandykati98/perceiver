#!/usr/bin/env python3
"""
Minimal CIFAR-100 neural network training script with Weights & Biases logging.
Uses Perceiver-style input: each pixel as token with RGB + Fourier positional encoding.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import wandb
from typing import Dict, Tuple
import modal

from model import Perceiver


# Modal configuration
app = modal.App("cifar100-perceiver")
image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "wandb",
).env({
    "WANDB_PROJECT": "perceiver-100",
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY")
}).add_local_python_source("model")


model_name = "model_cifar100"


def get_required_env_var(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value


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
    """Train the model and log metrics to Weights & Biases."""
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
                metrics: Dict[str, float] = {"batch_train_loss": loss.item()}
                model_stats = calculate_model_statistics(model)
                metrics.update({f"model_stats/batch_{k}": v for k, v in model_stats.items()})

                metrics.update({f"gradients/batch_{k}": v for k, v in grad_stats.items()})
                wandb.log(metrics, step=batch_count)

                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')

            # Evaluate test accuracy every 200 batches
            if batch_idx % 200 == 0:
                test_accuracy = evaluate_model(model, test_loader, device)
                wandb.log({"test_accuracy_batch": test_accuracy}, step=batch_count)
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Test Accuracy: {test_accuracy:.2f}%')
                model.train()  # Set back to training mode after evaluation

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        wandb.log(
            {"train_loss": epoch_loss, "train_accuracy": epoch_acc, "epoch": float(epoch)},
            step=batch_count,
        )

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


@app.function(image=image, gpu="any", timeout=7200)
def main() -> None:
    """Main training function for CIFAR-100."""
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    wandb_project = get_required_env_var("WANDB_PROJECT")
    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME")
    wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name,
        config={
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "device": str(device),
            "dataset": "cifar100",
        },
    )
    if wandb_run is None:
        raise RuntimeError("wandb.init() returned None")

    try:
        train_loader, test_loader = create_data_loaders(batch_size)

        model: nn.Module = Perceiver(
            num_classes=100,
            num_fourier_bands=64,
            latent_size=512,
            latent_channels=1024,
            num_cross_attn_iterations=8,
            latent_transformer_depth=6,
            latent_transformer_num_heads=8,
            dropout=0.1,
            image_size=32,
        ).to(device)
        # Print and verify parameter count
        total_params: int = sum(p.numel() for p in model.parameters())
        print(f'Total parameters: {total_params}')

        criterion: nn.Module = nn.CrossEntropyLoss()
        optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        arch_info = get_model_architecture_info(model)
        wandb.config.update(
            {
                "architecture": arch_info,
                "architecture/optimizer": optimizer.__class__.__name__,
                "architecture/criterion": criterion.__class__.__name__,
                "architecture/total_parameters": float(total_params),
            },
            allow_val_change=True,
        )

        print("Starting training...")
        train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs)

        print("Evaluating model...")
        test_accuracy = evaluate_model(model, test_loader, device)
        wandb.log({"test_accuracy": test_accuracy})
        print(f'Test Accuracy: {test_accuracy:.2f}%')

        model_path = f"{model_name}.pt"
        torch.save(model.state_dict(), model_path)
        model_artifact = wandb.Artifact(model_name, type="model")
        model_artifact.add_file(model_path)
        wandb_run.log_artifact(model_artifact)

        print("Training completed and logged to Weights & Biases!")
    finally:
        wandb_run.finish()


@app.local_entrypoint()
def run() -> None:
    """Entrypoint for 'modal run'."""
    main.remote()


if __name__ == "__main__":
    main()
