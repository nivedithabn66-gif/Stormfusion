"""STORMFUSION Optimizer Factory.

Builds configurable PyTorch optimizers (default AdamW) filtering strictly for trainable parameters.
"""

from typing import Iterable
import torch
import torch.nn as nn
import torch.optim as optim


def build_optimizer(
    model_or_params: nn.Module | Iterable[nn.Parameter],
    name: str = "adamw",
    learning_rate: float = 0.0001,
    weight_decay: float = 0.0001,
) -> optim.Optimizer:
    """Constructs optimizer filtering parameters with requires_grad=True.

    Args:
        model_or_params: Model instance or parameter iterable
        name: Optimizer algorithm name ('adamw', 'adam', 'sgd')
        learning_rate: Initial learning rate
        weight_decay: L2 penalty / weight decay

    Returns:
        PyTorch Optimizer instance
    """
    if isinstance(model_or_params, nn.Module):
        trainable_params = [p for p in model_or_params.parameters() if p.requires_grad]
    else:
        trainable_params = [p for p in model_or_params if p.requires_grad]

    if not trainable_params:
        raise ValueError("Cannot create optimizer: no trainable parameters found (requires_grad=True).")

    name_lower = str(name).lower().strip()
    if name_lower == "adamw":
        return optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif name_lower == "adam":
        return optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif name_lower == "sgd":
        return optim.SGD(trainable_params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer '{name}'. Expected 'adamw', 'adam', or 'sgd'.")
