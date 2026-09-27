"""STORMFUSION Step 9 — Shared Multi-Task Prediction Model.

Consumes the 64-dimensional fused embedding from Step 8C, passes it through a shared
MLP representation trunk, and feeds the resulting feature vector into 5 task-specific
prediction heads:
1. Detection Head [B, 1]
2. Pattern Classification Head [B, 4]
3. Intensity Head [B, 4]
4. Pressure Head [B, 4]
5. Track Prediction Head [B, 4, 2]
"""

from typing import Dict
import torch
import torch.nn as nn

from models.heads import (
    DetectionHead,
    PatternClassificationHead,
    IntensityHead,
    PressureHead,
    TrackPredictionHead,
)


class StormFusionMultiTaskModel(nn.Module):
    """STORMFUSION Multi-Task Model with Shared Trunk & 5 Task Prediction Heads."""

    def __init__(
        self,
        fused_dim: int = 64,
        shared_dim: int = 64,
        dropout: float = 0.2,
        pattern_classes: int = 4,
        num_horizons: int = 4,
    ):
        super().__init__()
        self.fused_dim = fused_dim
        self.shared_dim = shared_dim
        self.pattern_classes = pattern_classes
        self.num_horizons = num_horizons

        # 1. Lightweight Shared Trunk
        self.shared_trunk = nn.Sequential(
            nn.Linear(fused_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
        )

        # 2. Task Prediction Heads
        self.detection_head = DetectionHead(in_dim=shared_dim)
        self.pattern_head = PatternClassificationHead(in_dim=shared_dim, num_classes=pattern_classes)
        self.intensity_head = IntensityHead(in_dim=shared_dim, num_horizons=num_horizons)
        self.pressure_head = PressureHead(in_dim=shared_dim, num_horizons=num_horizons)
        self.track_head = TrackPredictionHead(in_dim=shared_dim, num_horizons=num_horizons)

    def forward(self, fused_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through shared trunk and task heads.

        Args:
            fused_embedding: Fused multimodal embedding tensor [B, 64]

        Returns:
            Dict containing:
                - detection_logits: [B, 1]
                - pattern_logits:   [B, 4]
                - intensity:        [B, 4]
                - pressure:         [B, 4]
                - track_delta:      [B, 4, 2]
        """
        # Shared trunk representation [B, shared_dim]
        shared_repr = self.shared_trunk(fused_embedding)

        # Task predictions
        detection_logits = self.detection_head(shared_repr)
        pattern_logits = self.pattern_head(shared_repr)
        intensity_pred = self.intensity_head(shared_repr)
        pressure_pred = self.pressure_head(shared_repr)
        track_delta_pred = self.track_head(shared_repr)

        return {
            "detection_logits": detection_logits,
            "pattern_logits": pattern_logits,
            "intensity": intensity_pred,
            "pressure": pressure_pred,
            "track_delta": track_delta_pred,
        }
