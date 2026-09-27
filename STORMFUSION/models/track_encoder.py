"""Track GRU Encoder Module for STORMFUSION (Step 8B).

Processes 8-timestep historical cyclone track features (t-21h to t) using a GRU neural network.

Input:
  track_features: [B, 8, 8] (8 physical features: latitude, longitude, delta_lat, delta_lon,
                             motion_speed, motion_direction, intensity, pressure)
  track_valid_mask: [B, 8] (validity mask per timestep: 1.0 = valid observation, 0.0 = missing)

Output:
  Dict containing:
    "track_embedding": [B, 64] (64-dimensional track trajectory embedding)
    "track_valid_mask": [B, 8]

Anti-Leakage Guarantee:
  - Sequence processed strictly in chronological order (t_0 to t_7).
  - Uses ONLY observations at or before prediction time t_7.
  - Invalid timesteps are masked out to prevent invalid values from corrupting the embedding.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class TrackGRUEncoder(nn.Module):
    """STORMFUSION Track GRU Encoder Module."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_layers: int = 1):
        super(TrackGRUEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Single-layer GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(
        self,
        track_features: torch.Tensor,
        track_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass for 8-timestep track feature sequence.

        Args:
            track_features: [B, 8, 8] physical track features
            track_valid_mask: Optional [B, 8] validity mask per timestep

        Returns:
            Dict containing "track_embedding" [B, 64] and "track_valid_mask" [B, 8]
        """
        # If input has 9 columns (including dataset valid_flag), slice to first 8 physical features
        if track_features.size(-1) == 9:
            physical_features = track_features[..., :8]
            if track_valid_mask is None:
                track_valid_mask = track_features[..., 8]
        else:
            physical_features = track_features

        batch_size, seq_len, feat_dim = physical_features.size()

        if track_valid_mask is not None:
            # Mask physical features: zero out invalid timesteps before GRU
            mask_expanded = track_valid_mask.unsqueeze(-1)  # [B, 8, 1]
            masked_inputs = physical_features * mask_expanded
        else:
            masked_inputs = physical_features
            track_valid_mask = torch.ones(batch_size, seq_len, device=physical_features.device, dtype=physical_features.dtype)

        # Run GRU over sequence -> output_seq [B, 8, 64], h_last [1, B, 64]
        output_seq, h_last = self.gru(masked_inputs)

        # Mask-aware temporal aggregation: select the last valid timestep output for each sample
        # If a sample has all zeros in valid mask, fallback to the final GRU state h_last
        track_embedding_list = []
        for b in range(batch_size):
            valid_indices = (track_valid_mask[b] > 0.5).nonzero(as_tuple=False)
            if len(valid_indices) > 0:
                last_valid_idx = valid_indices[-1].item()
                track_embedding_list.append(output_seq[b, last_valid_idx, :])
            else:
                track_embedding_list.append(h_last[-1, b, :])

        track_embedding = torch.stack(track_embedding_list, dim=0)  # [B, 64]

        return {
            "track_embedding": track_embedding,
            "track_valid_mask": track_valid_mask,
        }
