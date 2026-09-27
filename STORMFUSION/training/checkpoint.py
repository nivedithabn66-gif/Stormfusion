"""STORMFUSION Model Checkpoint Utilities.

Provides structured saving and loading of model weights, optimizer state, scheduler state,
epoch progress, best validation loss, and runtime configuration metadata.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    filepath: str | Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    best_val_loss: float = float("inf"),
    config: Optional[Dict[str, Any]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Saves model checkpoint with state dicts and configuration metadata.

    Args:
        filepath: Target save file path (.pt / .pth)
        model: PyTorch model instance
        optimizer: PyTorch optimizer instance (optional)
        scheduler: PyTorch learning rate scheduler (optional)
        epoch: Current training epoch number
        best_val_loss: Best validation loss achieved
        config: Model configuration dictionary
        extra_meta: Optional extra metadata dictionary

    Returns:
        Path object of saved checkpoint
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,
        "config": config or {},
        "extra_meta": extra_meta or {},
    }

    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    filepath: str | Path,
    model: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Loads checkpoint, restores model/optimizer state, and verifies compatibility.

    Args:
        filepath: Source checkpoint path
        model: PyTorch model instance to load weights into (optional)
        optimizer: PyTorch optimizer instance to load state into (optional)
        scheduler: PyTorch scheduler instance to load state into (optional)
        device: Device to map tensors to

    Returns:
        Loaded checkpoint dictionary
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    map_location = torch.device(device)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
