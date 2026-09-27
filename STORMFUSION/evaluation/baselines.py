"""Scientific Baseline Forecast Implementations for STORMFUSION (Step 20).

Provides benchmark baselines for cyclone track, intensity, and pressure forecasting:
1. Persistence Baseline (CLIPER-Zero): Assumes zero motion delta, zero intensity change, zero pressure change.
2. Linear Extrapolation Baseline: Extrapolates recent 3-hour motion vector and trend to future forecast lead times (+3h, +6h, +12h, +24h).
3. Track-Only GRU Baseline: Single-modality track baseline.

CRITICAL RULE:
- Evaluation must be executed strictly on identical test set splits and lead times.
- Baselines must NOT use future observations (t > t0).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch


class PersistenceBaseline:
    """Persistence Forecast Baseline (CLIPER-Zero).

    Assumes cyclone position, intensity, and pressure remain constant at reference time t0 values.
    """

    def predict(self, track_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generates persistence predictions from 8-timestep track history [B, 8, 9]."""
        B = track_features.shape[0]

        # Historical t7 is at index 7
        t7_obs = track_features[:, 7, :]  # [B, 9]
        curr_wind = t7_obs[:, 6:7]         # [B, 1] (intensity at t7)
        curr_press = t7_obs[:, 7:8]        # [B, 1] (pressure at t7)

        # Track delta: 0.0 for all 4 horizons (+3h, +6h, +12h, +24h)
        track_delta = torch.zeros(B, 4, 2, dtype=torch.float32, device=track_features.device)

        # Intensity & Pressure: constant across 4 horizons
        intensity = curr_wind.repeat(1, 4)   # [B, 4]
        pressure = curr_press.repeat(1, 4)   # [B, 4]

        # Detection: 1.0 (active cyclone)
        detection = torch.ones(B, 1, dtype=torch.float32, device=track_features.device)

        return {
            "detection": detection,
            "intensity": intensity,
            "pressure": pressure,
            "track_delta": track_delta,
            "baseline_name": "Persistence (CLIPER-Zero)",
        }


class LinearExtrapolationBaseline:
    """Linear Extrapolation Forecast Baseline.

    Extrapolates recent 3-hour motion vector (t6 -> t7) into future forecast horizons.
    """

    def predict(self, track_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generates linear extrapolation predictions from track history [B, 8, 9]."""
        B = track_features.shape[0]

        # t6 is index 6, t7 is index 7
        t6_lat = track_features[:, 6, 0]
        t6_lon = track_features[:, 6, 1]

        t7_lat = track_features[:, 7, 0]
        t7_lon = track_features[:, 7, 1]

        dlat_3h = t7_lat - t6_lat  # [B]
        dlon_3h = t7_lon - t6_lon  # [B]

        # Horizons: +3h (1x), +6h (2x), +12h (4x), +24h (8x)
        multipliers = [1.0, 2.0, 4.0, 8.0]

        track_delta_list = []
        for m in multipliers:
            h_dlat = dlat_3h * m
            h_dlon = dlon_3h * m
            track_delta_list.append(torch.stack([h_dlat, h_dlon], dim=-1))  # [B, 2]

        track_delta = torch.stack(track_delta_list, dim=1)  # [B, 4, 2]

        # Intensity and Pressure persistence
        curr_wind = track_features[:, 7, 6:7]  # [B, 1]
        curr_press = track_features[:, 7, 7:8] # [B, 1]

        intensity = curr_wind.repeat(1, 4)   # [B, 4]
        pressure = curr_press.repeat(1, 4)   # [B, 4]
        detection = torch.ones(B, 1, dtype=torch.float32, device=track_features.device)

        return {
            "detection": detection,
            "intensity": intensity,
            "pressure": pressure,
            "track_delta": track_delta,
            "baseline_name": "Linear Extrapolation",
        }
