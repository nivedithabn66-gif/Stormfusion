"""Detection prediction head for STORMFUSION.

Predicts tropical cyclone presence/detection logit [B, 1] from the shared multi-task trunk.
"""

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """Detection Head returning raw binary classification logit [B, 1].

    Note: Training uses BCEWithLogitsLoss. Sigmoid is not applied internally.
    """

    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Shared trunk representation [B, in_dim]

        Returns:
            Detection logit tensor [B, 1]
        """
        return self.fc(x)
