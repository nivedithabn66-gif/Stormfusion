"""Minimum sea-level pressure prediction head for STORMFUSION.

Predicts future central sea-level pressure regression values [B, 4] for forecast lead times
(+3h, +6h, +12h, +24h) from the shared multi-task trunk.
"""

import torch
import torch.nn as nn


class PressureHead(nn.Module):
    """Pressure Head returning central pressure forecasts [B, num_horizons].

    Note: Historical pressure data may contain missing observations. Missing values
    must be masked during loss evaluation using pressure_valid_mask.
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
            Central pressure predictions [B, num_horizons]
        """
        return self.fc(x)
