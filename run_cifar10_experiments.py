#!/usr/bin/env python3
"""
Sequential CIFAR-10 training script with multiple configurations.
Runs each configuration one after another and logs results to Weights & Biases.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import wandb
from typing import Dict, List, Tuple
from pydantic import BaseModel
import modal

from model import Perceiver


# Modal configuration
app = modal.App("cifar10-perceiver-experiments")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch",
    "torchvision",
    "wandb",
    "pydantic",
).env({
    "WANDB_PROJECT": "perceiver-s-experiments",
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY")
}).add_local_python_source("model")


def get_required_env_var(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value


class ModelConfig(BaseModel):
    """Configuration for Perceiver model architecture."""
    num_fourier_bands: int
    latent_size: int
    latent_channels: int
    num_cross_attn_iterations: int
    latent_transformer_depth: int
    latent_transformer_num_heads: int
    dropout: float


class TrainConfig(BaseModel):
    """Configuration for training hyperparameters."""
    batch_size: int
    learning_rate: float
    num_epochs: int


class ExperimentConfig(BaseModel):
    """Full experiment configuration combining model and training configs."""
    name: str
    model: ModelConfig
    train: TrainConfig


# Define experiment configurations
EXPERIMENTS: List[ExperimentConfig] = [
    ExperimentConfig(
        name="tiny",
        model=ModelConfig(
            num_fourier_bands=4,
            latent_size=64,
            latent_channels=32,
            num_cross_attn_iterations=2,
            latent_transformer_depth=2,
            latent_transformer_num_heads=1,
            dropout=0.1,
        ),
        train=TrainConfig(
            batch_size=64,
            learning_rate=0.001,
            num_epochs=50,
        ),
    ),
    ExperimentConfig(
        name="small",
        model=ModelConfig(
            num_fourier_bands=16,
            latent_size=128,
            latent_channels=64,
            num_cross_attn_iterations=4,
            latent_transformer_depth=4,
            latent_transformer_num_heads=2,
            dropout=0.1,
        ),
        train=TrainConfig(
            batch_size=64,
            learning_rate=0.001,
            num_epochs=100,
        ),
    ),
    ExperimentConfig(
        name="medium",
        model=ModelConfig(
            num_fourier_bands=32,
            latent_size=256,
            latent_channels=256,
            num_cross_attn_iterations=6,
            latent_transformer_depth=6,
            latent_transformer_num_heads=4,
            dropout=0.1,
        ),
        train=TrainConfig(
            batch_size=32,
            learning_rate=0.0005,
            num_epochs=150,
        ),
    ),
    ExperimentConfig(
        name="large",
        model=ModelConfig(
            num_fourier_bands=64,
            latent_size=512,
            latent_channels=512,
            num_cross_attn_iterations=8,
            latent_transformer_depth=6,
            latent_transformer_num_heads=8,
            dropout=0.1,
        ),
        train=TrainConfig(
            batch_size=32,
            learning_rate=0.0003,
            num_epochs=200,
        ),
    ),
]


def create_data_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and test data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


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
    """Calculate gradient statistics to monitor training health."""
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

    stats: Dict[str, float] = {
        'gradients_min': gradients_tensor.min().item(),
        'gradients_max': gradients_tensor.max().item(),
        'gradients_mean': gradients_tensor.mean().item(),
        'gradients_std': gradients_tensor.std().item(),
        'total_grad_norm': gradients_tensor.norm().item(),
        'avg_layer_grad_norm': sum(grad_norms_per_layer) / len(grad_norms_per_layer) if grad_norms_per_layer else 0.0,
    }

    return stats


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
) -> float:
    """Train the model and log metrics to MLflow. Returns final test accuracy."""
    model.train()

    batch_count = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data, mask=None)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            batch_count += 1

            if batch_idx % 100 == 0 and batch_idx > 0:
                grad_stats = calculate_gradient_statistics(model)
                model_stats = calculate_model_statistics(model)
                metrics: Dict[str, float] = {"batch_train_loss": loss.item()}
                metrics.update({f"model_stats/batch_{k}": v for k, v in model_stats.items()})
                metrics.update({f"gradients/batch_{k}": v for k, v in grad_stats.items()})
                wandb.log(metrics, step=batch_count)

                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')

            if batch_idx % 500 == 0 and batch_idx > 0:
                test_accuracy = evaluate_model(model, test_loader, device)
                wandb.log({"test_accuracy_batch": test_accuracy}, step=batch_count)
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Test Accuracy: {test_accuracy:.2f}%')
                model.train()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        wandb.log(
            {"train_loss": epoch_loss, "train_accuracy": epoch_acc, "epoch": float(epoch)},
            step=batch_count,
        )

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    final_accuracy = evaluate_model(model, test_loader, device)
    return final_accuracy


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate the model and return accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, mask=None)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def run_experiment(config: ExperimentConfig, device: torch.device) -> float:
    """Run a single experiment with the given configuration. Returns test accuracy."""
    print(f"\n{'='*60}")
    print(f"Starting experiment: {config.name}")
    print(f"{'='*60}")

    wandb_project = get_required_env_var("WANDB_PROJECT")
    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=config.name,
        config={
            "config_name": config.name,
            "batch_size": config.train.batch_size,
            "learning_rate": config.train.learning_rate,
            "num_epochs": config.train.num_epochs,
            "device": str(device),
            **{f"model/{k}": v for k, v in config.model.model_dump().items()},
        },
        reinit=True,
    )
    if wandb_run is None:
        raise RuntimeError("wandb.init() returned None")

    try:
        train_loader, test_loader = create_data_loaders(config.train.batch_size)

        model: nn.Module = Perceiver(
            num_classes=10,
            num_fourier_bands=config.model.num_fourier_bands,
            latent_size=config.model.latent_size,
            latent_channels=config.model.latent_channels,
            num_cross_attn_iterations=config.model.num_cross_attn_iterations,
            latent_transformer_depth=config.model.latent_transformer_depth,
            latent_transformer_num_heads=config.model.latent_transformer_num_heads,
            cross_heads=1,
            dropout=config.model.dropout,
            image_size=32,
            max_freq=32.0,
        ).to(device)

        total_params: int = sum(p.numel() for p in model.parameters())
        print(f'Total parameters: {total_params:,}')
        wandb.config.update({"total_parameters": float(total_params)}, allow_val_change=True)

        criterion: nn.Module = nn.CrossEntropyLoss()
        optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)
        wandb.config.update(
            {
                "optimizer": optimizer.__class__.__name__,
                "criterion": criterion.__class__.__name__,
            },
            allow_val_change=True,
        )

        print("Starting training...")
        test_accuracy = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=config.train.num_epochs,
        )

        wandb.log({"final_test_accuracy": test_accuracy})
        print(f'Final Test Accuracy: {test_accuracy:.2f}%')

        model_path = f"model_{config.name}.pt"
        torch.save(model.state_dict(), model_path)
        model_artifact = wandb.Artifact(f"model_{config.name}", type="model")
        model_artifact.add_file(model_path)
        wandb_run.log_artifact(model_artifact)

        print(f"Experiment '{config.name}' completed!")
        return test_accuracy
    finally:
        wandb_run.finish()


@app.function(image=image, gpu="any", timeout=7200)
def main() -> None:
    """Run all experiments sequentially."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    results: Dict[str, float] = {}

    for config in EXPERIMENTS:
        accuracy = run_experiment(config=config, device=device)
        results[config.name] = accuracy

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for name, accuracy in results.items():
        print(f"{name:20s}: {accuracy:.2f}%")
    print("="*60)

    best_name = max(results, key=results.get)
    print(f"Best configuration: {best_name} with {results[best_name]:.2f}% accuracy")


if __name__ == "__main__":
    main()


@app.local_entrypoint()
def run() -> None:
    """Entrypoint for 'modal run'."""
    main.remote()

