"""Pattern classification prediction head for STORMFUSION.

Predicts cyclone cloud pattern logits [B, num_classes] from the shared multi-task trunk.
"""

import torch
import torch.nn as nn


class PatternClassificationHead(nn.Module):
    """Pattern Classification Head returning raw multi-class logits [B, num_classes].

    Note: Training uses CrossEntropyLoss. Softmax is not applied internally.
    Final pattern class definitions must be aligned with authoritative IMD/RSMC guidelines.
    """

    def __init__(self, in_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Shared trunk representation [B, in_dim]

        Returns:
            Pattern classification logits [B, num_classes]
        """
        return self.fc(x)
