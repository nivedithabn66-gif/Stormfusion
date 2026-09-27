"""Baseline Satellite Encoder Model for STORMFUSION (Step 8A).

Combines:
  1. ResNet18 Spatial Encoder (512 spatial feature channels)
  2. ConvLSTM Temporal Encoder (64 hidden channels, 1 layer)
  3. Global Average Pooling (GAP over spatial dimensions)
  4. Linear Projection Layer (producing 128-dimensional satellite embedding)

Preserves satellite_valid_mask and satellite_modality_mask in output dictionary.
Does NOT convert missing or placeholder observations into valid physical observations.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from models.satellite_encoder import ResNet18SpatialEncoder
from models.convlstm import ConvLSTM


class BaselineSatelliteModel(nn.Module):
    """STORMFUSION Baseline Satellite Encoder Model (ResNet18 + ConvLSTM)."""

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = False,
        convlstm_hidden: int = 64,
        convlstm_layers: int = 1,
        convlstm_kernel: int = 3,
        feature_dim: int = 128,
    ):
        super(BaselineSatelliteModel, self).__init__()

        self.in_channels = in_channels
        self.pretrained = pretrained
        self.convlstm_hidden = convlstm_hidden
        self.feature_dim = feature_dim

        # 1. Spatial Encoder (ResNet18)
        self.spatial_encoder = ResNet18SpatialEncoder(in_channels=in_channels, pretrained=pretrained)

        # 2. Temporal Encoder (ConvLSTM)
        self.convlstm = ConvLSTM(
            in_channels=self.spatial_encoder.out_channels,
            hidden_channels=convlstm_hidden,
            kernel_size=convlstm_kernel,
            num_layers=convlstm_layers,
            batch_first=True,
        )

        # 3. Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Projection Layer to final satellite embedding dimension (128)
        self.projection = nn.Linear(convlstm_hidden, feature_dim)

    def forward(
        self,
        satellite_tensor: torch.Tensor,
        satellite_valid_mask: Optional[torch.Tensor] = None,
        satellite_modality_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass for 5D satellite sequence tensor.

        Args:
            satellite_tensor: [B, T, C, H, W] where T=8, C=3, H=500, W=500
            satellite_valid_mask: Optional [B, T] validity mask per timestep
            satellite_modality_mask: Optional [B, 1] overall satellite modality mask

        Returns:
            Dict containing:
              - "satellite_embedding": [B, 128]
              - "satellite_valid_mask": [B, T]
              - "satellite_modality_mask": [B, 1]
        """
        batch_size, seq_len, num_channels, height, width = satellite_tensor.size()

        # 1. Reshape sequence tensor [B, T, C, H, W] -> [B * T, C, H, W] for single-pass spatial extraction
        x_reshaped = satellite_tensor.view(batch_size * seq_len, num_channels, height, width)

        # 2. ResNet18 Spatial Feature Extraction -> [B * T, 512, H_feat, W_feat]
        spatial_features = self.spatial_encoder(x_reshaped)

        _, feat_channels, feat_h, feat_w = spatial_features.size()

        # 3. Reshape back to 5D sequence -> [B, T, 512, H_feat, W_feat]
        spatial_seq = spatial_features.view(batch_size, seq_len, feat_channels, feat_h, feat_w)

        # 4. ConvLSTM Temporal Modeling -> final hidden state [B, 64, H_feat, W_feat]
        _, (h_last, c_last) = self.convlstm(spatial_seq)

        # 5. Global Average Pooling over spatial dimensions -> [B, 64, 1, 1]
        pooled = self.global_pool(h_last)
        pooled_flat = torch.flatten(pooled, start_dim=1)  # [B, 64]

        # 6. Linear Projection -> [B, 128]
        embedding = self.projection(pooled_flat)

        return {
            "satellite_embedding": embedding,
            "satellite_valid_mask": satellite_valid_mask,
            "satellite_modality_mask": satellite_modality_mask,
        }
