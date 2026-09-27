"""Future cyclone track prediction head for STORMFUSION.

Predicts multi-horizon geographic displacements [B, 4, 2] (delta_latitude, delta_longitude)
for forecast lead times (+3h, +6h, +12h, +24h) from the shared multi-task trunk.
"""

import torch
import torch.nn as nn


class TrackPredictionHead(nn.Module):
    """Track Prediction Head returning coordinate displacements [B, num_horizons, 2].

    Note: Predicts relative displacement from reference current position rather than
    absolute coordinates directly. Absolute coordinates are reconstructed as:
    future_lat = current_lat + delta_lat, future_lon = current_lon + delta_lon.
    """

    def __init__(self, in_dim: int = 64, num_horizons: int = 4):
        super().__init__()
        self.num_horizons = num_horizons
        self.fc = nn.Linear(in_dim, num_horizons * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Shared trunk representation [B, in_dim]

        Returns:
            Displacement predictions tensor [B, num_horizons, 2]
        """
        out = self.fc(x)  # [B, num_horizons * 2]
        return out.view(-1, self.num_horizons, 2)
