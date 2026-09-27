"""STORMFUSION -- Complete End-to-End Integrated Model (Step 16).

Wires together all five already-tested subsystems:
  1. BaselineSatelliteModel  (ResNet18 + ConvLSTM)          -> [B, 128]
  2. TrackGRUEncoder         (GRU over 8-timestep history)  -> [B, 64]
  3. ERA5Encoder             (lightweight CNN)               -> [B, 64]
  4. AvailabilityAwareGatedFusion                           -> [B, 64]
  5. StormFusionMultiTaskModel (shared trunk + 5 heads)     -> 5 tensors

CRITICAL SAFETY RULES:
  - DO NOT perform scientific model training.
  - DO NOT use synthetic satellite imagery as training data.
  - DO NOT fabricate physical observations.
  - Pattern labels remain PENDING; 4-class head is an architectural placeholder.
  - When ALL modalities are missing, return insufficient_input=True and NaN predictions.
    Do NOT generate a fake low-confidence prediction from zero information.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import yaml

from models.baseline_satellite_model import BaselineSatelliteModel
from models.track_encoder import TrackGRUEncoder
from models.era5_encoder import ERA5Encoder
from models.gated_fusion import AvailabilityAwareGatedFusion
from models.multitask_model import StormFusionMultiTaskModel


class StormFusionModel(nn.Module):
    """Complete STORMFUSION multi-modal, multi-task cyclone prediction model.

    Architecture:
        Satellite (WV_062+IR_108+IR_120)  Track history (8 x 9)  ERA5 (8 x 4 x 32 x 32)
                |                                  |                       |
        BaselineSatelliteModel   TrackGRUEncoder          ERA5Encoder
                |                      |                       |
             [B,128]               [B,64]                  [B,64]
                \\                     |                      /
                 \\                    |                    /
                  ---> AvailabilityAwareGatedFusion <---
                                  [B,64]
                                     |
                          StormFusionMultiTaskModel
                          (shared trunk + 5 heads)
                                     |
          detection / pattern / intensity / pressure / track_delta
    """

    def __init__(
        self,
        # Satellite encoder params
        in_channels: int = 3,
        convlstm_hidden: int = 64,
        convlstm_layers: int = 1,
        convlstm_kernel: int = 3,
        sat_feature_dim: int = 128,
        # Track encoder params
        track_input_dim: int = 8,
        track_hidden_dim: int = 64,
        track_num_layers: int = 1,
        # ERA5 encoder params
        era5_input_channels: int = 4,
        era5_hidden_channels: int = 32,
        era5_embedding_dim: int = 64,
        # Fusion params
        fusion_dim: int = 64,
        gate_hidden_dim: int = 32,
        fusion_epsilon: float = 1.0e-8,
        # Multi-task head params
        shared_dim: int = 64,
        dropout: float = 0.2,
        pattern_classes: int = 4,
        num_horizons: int = 4,
    ):
        super().__init__()

        # ── Sub-modules ────────────────────────────────────────────────────
        self.satellite_encoder = BaselineSatelliteModel(
            in_channels=in_channels,
            pretrained=False,
            convlstm_hidden=convlstm_hidden,
            convlstm_layers=convlstm_layers,
            convlstm_kernel=convlstm_kernel,
            feature_dim=sat_feature_dim,
        )

        self.track_encoder = TrackGRUEncoder(
            input_dim=track_input_dim,
            hidden_dim=track_hidden_dim,
            num_layers=track_num_layers,
        )

        self.era5_encoder = ERA5Encoder(
            input_channels=era5_input_channels,
            hidden_channels=era5_hidden_channels,
            embedding_dim=era5_embedding_dim,
        )

        self.fusion = AvailabilityAwareGatedFusion(
            sat_dim=sat_feature_dim,
            track_dim=track_hidden_dim,
            era5_dim=era5_embedding_dim,
            fusion_dim=fusion_dim,
            gate_hidden_dim=gate_hidden_dim,
            epsilon=fusion_epsilon,
        )

        self.multitask = StormFusionMultiTaskModel(
            fused_dim=fusion_dim,
            shared_dim=shared_dim,
            dropout=dropout,
            pattern_classes=pattern_classes,
            num_horizons=num_horizons,
        )

        # Store config for parameter_summary()
        self._cfg = dict(
            in_channels=in_channels,
            sat_feature_dim=sat_feature_dim,
            track_hidden_dim=track_hidden_dim,
            era5_embedding_dim=era5_embedding_dim,
            fusion_dim=fusion_dim,
            shared_dim=shared_dim,
            pattern_classes=pattern_classes,
            num_horizons=num_horizons,
        )

    # ── Class method: instantiate from model_config.yaml ──────────────────

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None) -> "StormFusionModel":
        """Instantiate from configs/model_config.yaml."""
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "configs/model_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        sat_cfg   = cfg.get("satellite_encoder", {})
        clstm_cfg = cfg.get("convlstm", {})
        trk_cfg   = cfg.get("track_encoder", {})
        era5_cfg  = cfg.get("era5_encoder", {})
        fuse_cfg  = cfg.get("fusion", {})
        mt_cfg    = cfg.get("multitask", {})

        return cls(
            in_channels=sat_cfg.get("input_channels", 3),
            convlstm_hidden=clstm_cfg.get("hidden_channels", 64),
            convlstm_layers=clstm_cfg.get("num_layers", 1),
            convlstm_kernel=clstm_cfg.get("kernel_size", 3),
            sat_feature_dim=sat_cfg.get("feature_dim", 128),
            track_input_dim=trk_cfg.get("input_dim", 8),
            track_hidden_dim=trk_cfg.get("hidden_dim", 64),
            track_num_layers=trk_cfg.get("num_layers", 1),
            era5_input_channels=era5_cfg.get("input_channels", 4),
            era5_hidden_channels=era5_cfg.get("hidden_channels", 32),
            era5_embedding_dim=era5_cfg.get("embedding_dim", 64),
            fusion_dim=fuse_cfg.get("embedding_dim", 64),
            gate_hidden_dim=fuse_cfg.get("gate_hidden_dim", 32),
            fusion_epsilon=fuse_cfg.get("epsilon", 1.0e-8),
            shared_dim=mt_cfg.get("shared_dim", 64),
            dropout=mt_cfg.get("dropout", 0.2),
            pattern_classes=mt_cfg.get("pattern_classes", 4),
            num_horizons=len(mt_cfg.get("forecast_horizons", [3, 6, 12, 24])),
        )

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        # Satellite branch
        satellite_tensor: torch.Tensor,            # [B, 8, C, H, W]
        satellite_valid_mask: torch.Tensor,        # [B, 8]
        satellite_modality_mask: torch.Tensor,     # [B, 1]
        # Track branch
        track_features: torch.Tensor,             # [B, 8, 9]
        track_valid_mask: torch.Tensor,           # [B, 8]
        track_modality_mask: torch.Tensor,        # [B, 1]
        # ERA5 branch
        era5_tensor: torch.Tensor,                # [B, 8, 4, 32, 32]
        era5_valid_mask: torch.Tensor,            # [B, 8]
        era5_modality_mask: torch.Tensor,         # [B, 1]
    ) -> Dict[str, Any]:
        """End-to-end forward pass through all STORMFUSION components.

        Returns:
            Dict with keys:
              detection_logits   [B, 1]
              pattern_logits     [B, 4]
              intensity          [B, 4]
              pressure           [B, 4]
              track_delta        [B, 4, 2]
              satellite_gate     [B, 1]
              track_gate         [B, 1]
              era5_gate          [B, 1]
              modality_mask      [B, 3]
              fusion_valid_mask  [B, 1]
              insufficient_input [B, 1]  bool — True where ALL modalities absent
        """
        # ── 1. Satellite encoding ──────────────────────────────────────────
        sat_out = self.satellite_encoder(
            satellite_tensor,
            satellite_valid_mask=satellite_valid_mask,
            satellite_modality_mask=satellite_modality_mask,
        )
        sat_emb = sat_out["satellite_embedding"]   # [B, 128]

        # ── 2. Track encoding ──────────────────────────────────────────────
        trk_out = self.track_encoder(
            track_features,
            track_valid_mask=track_valid_mask,
        )
        trk_emb = trk_out["track_embedding"]       # [B, 64]

        # ── 3. ERA5 encoding ───────────────────────────────────────────────
        era_out = self.era5_encoder(
            era5_tensor,
            era5_valid_mask=era5_valid_mask,
            era5_modality_mask=era5_modality_mask,
        )
        era_emb = era_out["era5_embedding"]         # [B, 64]

        # ── 4. Availability-aware gated fusion ────────────────────────────
        fuse_out = self.fusion(
            satellite_embedding=sat_emb,
            track_embedding=trk_emb,
            era5_embedding=era_emb,
            satellite_modality_mask=satellite_modality_mask,
            track_modality_mask=track_modality_mask,
            era5_modality_mask=era5_modality_mask,
        )
        fused      = fuse_out["fused_embedding"]    # [B, 64]
        fuse_valid = fuse_out["fusion_valid_mask"]  # [B, 1]

        # ── 5. All-missing sentinel ────────────────────────────────────────
        # Where ALL modalities are absent, mark as insufficient_input.
        # Predictions for these samples are NaN — do NOT generate fake values.
        insufficient = (fuse_valid == 0)            # [B, 1]  bool

        # ── 6. Multi-task predictions ──────────────────────────────────────
        mt_out = self.multitask(fused)

        # Apply NaN to predictions where all modalities absent (no information)
        if insufficient.any():
            nan_val = float("nan")
            for key in ["detection_logits", "pattern_logits", "intensity", "pressure"]:
                mt_out[key] = torch.where(
                    insufficient.expand_as(mt_out[key]),
                    torch.full_like(mt_out[key], nan_val),
                    mt_out[key],
                )
            # track_delta: [B, 4, 2] — expand mask
            insuf_trk = insufficient.unsqueeze(-1).expand_as(mt_out["track_delta"])
            mt_out["track_delta"] = torch.where(
                insuf_trk,
                torch.full_like(mt_out["track_delta"], nan_val),
                mt_out["track_delta"],
            )

        return {
            # Predictions
            "detection_logits":  mt_out["detection_logits"],   # [B, 1]
            "pattern_logits":    mt_out["pattern_logits"],     # [B, 4]
            "intensity":         mt_out["intensity"],          # [B, 4]
            "pressure":          mt_out["pressure"],           # [B, 4]
            "track_delta":       mt_out["track_delta"],        # [B, 4, 2]
            # Fusion internals
            "satellite_gate":    fuse_out["satellite_gate"],   # [B, 1]
            "track_gate":        fuse_out["track_gate"],       # [B, 1]
            "era5_gate":         fuse_out["era5_gate"],        # [B, 1]
            "modality_mask":     fuse_out["modality_mask"],    # [B, 3]
            "fusion_valid_mask": fuse_valid,                   # [B, 1]
            "insufficient_input": insufficient,                # [B, 1] bool
        }

    # ── Parameter summary ─────────────────────────────────────────────────

    def parameter_summary(self) -> Dict[str, Any]:
        """Return per-component and total parameter counts."""
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        summary = {
            "satellite_encoder_params": _count(self.satellite_encoder),
            "track_encoder_params":     _count(self.track_encoder),
            "era5_encoder_params":      _count(self.era5_encoder),
            "fusion_params":            _count(self.fusion),
            "shared_trunk_params":      _count(self.multitask.shared_trunk),
            "detection_head_params":    _count(self.multitask.detection_head),
            "pattern_head_params":      _count(self.multitask.pattern_head),
            "intensity_head_params":    _count(self.multitask.intensity_head),
            "pressure_head_params":     _count(self.multitask.pressure_head),
            "track_head_params":        _count(self.multitask.track_head),
            "total_params":             _count(self),
        }
        return summary
