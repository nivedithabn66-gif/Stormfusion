"""Intensity prediction head for STORMFUSION.

Predicts future cyclone intensity (maximum sustained wind speed) regression values [B, 4]
for forecast lead times (+3h, +6h, +12h, +24h) from the shared multi-task trunk.
"""

import torch
import torch.nn as nn


class IntensityHead(nn.Module):
    """Intensity Head returning continuous wind speed forecasts [B, num_horizons].

    Note: Training uses Huber loss without output activation clamping.
    """

    def __init__(self, in_dim: int = 64, num_horizons: int = 4):
        super().__init__()
        self.num_horizons = num_horizons
        self.fc = nn.Linear(in_dim, num_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Shared trunk representation [B, in_dim]

        Returns:
            Intensity forecast predictions [B, num_horizons]
        """
        return self.fc(x)
