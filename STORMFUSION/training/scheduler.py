"""STORMFUSION Learning Rate Scheduler Factory.

Constructs warmup + cosine annealing learning rate scheduler with strict schedule compliance.
"""

import torch.optim as optim
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
    _LRScheduler,
)


def build_scheduler(
    optimizer: optim.Optimizer,
    name: str = "cosine",
    warmup_epochs: int = 5,
    total_epochs: int = 50,
    min_lr: float = 0.000001,
) -> _LRScheduler:
    """Builds a learning rate scheduler with initial linear warmup followed by cosine decay.

    Args:
        optimizer: Configured PyTorch Optimizer
        name: Scheduler strategy ('cosine')
        warmup_epochs: Number of initial linear warmup epochs
        total_epochs: Total planned training epochs
        min_lr: Minimum learning rate floor for cosine decay

    Returns:
        PyTorch LRScheduler instance
    """
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)

    # 1. Warmup scheduler: from 1/10th base LR to base LR over warmup_epochs
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # 2. Cosine scheduler: over remaining epochs
    cosine_epochs = max(1, total_epochs - warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=min_lr,
    )

    # Combine warmup + cosine
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
