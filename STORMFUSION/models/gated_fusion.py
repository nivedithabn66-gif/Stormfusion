"""STORMFUSION Step 8C — Availability-Aware Gated Multimodal Fusion Module.

Combines satellite, track, and ERA5 embeddings using learned modality gates
and explicit modality availability masks. Ensures graceful degradation when
modalities are missing without leaking zero-placeholder signals into fusion.
"""

from typing import Dict
import torch
import torch.nn as nn


class AvailabilityAwareGatedFusion(nn.Module):
    """Availability-Aware Gated Fusion Module for STORMFUSION.

    Projects satellite (128d), track (64d), and ERA5 (64d) embeddings into a common
    fusion dimension (default 64d), computes learned per-modality contribution gates
    gated by explicit modality availability masks, and produces a normalized fused
    embedding [B, 64].

    Note on Gate Values:
    The returned gate values represent learned contribution weights under availability
    constraints, NOT causal physical importance scores. Actual modality usefulness
    must be evaluated via ablation experiments.
    """

    def __init__(
        self,
        sat_dim: int = 128,
        track_dim: int = 64,
        era5_dim: int = 64,
        fusion_dim: int = 64,
        gate_hidden_dim: int = 32,
        epsilon: float = 1.0e-8,
    ):
        super().__init__()
        self.sat_dim = sat_dim
        self.track_dim = track_dim
        self.era5_dim = era5_dim
        self.fusion_dim = fusion_dim
        self.gate_hidden_dim = gate_hidden_dim
        self.epsilon = epsilon

        # 1. Projections to common fusion dimension (64)
        self.sat_proj = nn.Linear(sat_dim, fusion_dim)
        self.track_proj = nn.Linear(track_dim, fusion_dim)
        self.era5_proj = nn.Linear(era5_dim, fusion_dim)

        # 2. Gate networks: input projected_dim (64) + mask_dim (1) = 65 -> hidden (32) -> 1 -> Sigmoid
        self.sat_gate_net = nn.Sequential(
            nn.Linear(fusion_dim + 1, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.track_gate_net = nn.Sequential(
            nn.Linear(fusion_dim + 1, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.era5_gate_net = nn.Sequential(
            nn.Linear(fusion_dim + 1, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        satellite_embedding: torch.Tensor,
        track_embedding: torch.Tensor,
        era5_embedding: torch.Tensor,
        satellite_modality_mask: torch.Tensor,
        track_modality_mask: torch.Tensor,
        era5_modality_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for availability-aware gated fusion.

        Args:
            satellite_embedding: Tensor [B, sat_dim] (typically [B, 128])
            track_embedding: Tensor [B, track_dim] (typically [B, 64])
            era5_embedding: Tensor [B, era5_dim] (typically [B, 64])
            satellite_modality_mask: Tensor [B, 1] (1 = available, 0 = missing)
            track_modality_mask: Tensor [B, 1] (1 = available, 0 = missing)
            era5_modality_mask: Tensor [B, 1] (1 = available, 0 = missing)

        Returns:
            Dict containing:
                - fused_embedding: Tensor [B, fusion_dim] ([B, 64])
                - satellite_gate: Tensor [B, 1]
                - track_gate: Tensor [B, 1]
                - era5_gate: Tensor [B, 1]
                - modality_mask: Tensor [B, 3]
                - fusion_valid_mask: Tensor [B, 1]
        """
        # Ensure mask shapes [B, 1]
        sat_mask = satellite_modality_mask.view(-1, 1).float()
        track_mask = track_modality_mask.view(-1, 1).float()
        era5_mask = era5_modality_mask.view(-1, 1).float()

        # Sanitize input embeddings for missing modalities (if mask == 0, nan_to_num prevents NaN propagation into gate network)
        sat_emb_clean = torch.where(sat_mask == 1, satellite_embedding, torch.nan_to_num(satellite_embedding, nan=0.0, posinf=0.0, neginf=0.0))
        track_emb_clean = torch.where(track_mask == 1, track_embedding, torch.nan_to_num(track_embedding, nan=0.0, posinf=0.0, neginf=0.0))
        era5_emb_clean = torch.where(era5_mask == 1, era5_embedding, torch.nan_to_num(era5_embedding, nan=0.0, posinf=0.0, neginf=0.0))

        # Project modalities to common dimension
        sat_proj_emb = self.sat_proj(sat_emb_clean)  # [B, 64]
        track_proj_emb = self.track_proj(track_emb_clean)  # [B, 64]
        era5_proj_emb = self.era5_proj(era5_emb_clean)  # [B, 64]


        # Compute raw gate outputs using embedding + mask
        sat_gate_input = torch.cat([sat_proj_emb, sat_mask], dim=-1)  # [B, 65]
        track_gate_input = torch.cat([track_proj_emb, track_mask], dim=-1)  # [B, 65]
        era5_gate_input = torch.cat([era5_proj_emb, era5_mask], dim=-1)  # [B, 65]

        raw_sat_gate = self.sat_gate_net(sat_gate_input)  # [B, 1]
        raw_track_gate = self.track_gate_net(track_gate_input)  # [B, 1]
        raw_era5_gate = self.era5_gate_net(era5_gate_input)  # [B, 1]

        # Mask gates explicitly: missing modality -> gate = 0
        sat_gate = raw_sat_gate * sat_mask
        track_gate = raw_track_gate * track_mask
        era5_gate = raw_era5_gate * era5_mask

        # Mask projected embeddings so missing modality features cannot leak
        sat_proj_masked = sat_proj_emb * sat_mask
        track_proj_masked = track_proj_emb * track_mask
        era5_proj_masked = era5_proj_emb * era5_mask

        # Calculate availability and weighted sum
        fusion_valid_mask = ((sat_mask + track_mask + era5_mask) > 0).float()  # [B, 1]

        weighted_sum = (
            sat_gate * sat_proj_masked
            + track_gate * track_proj_masked
            + era5_gate * era5_proj_masked
        )  # [B, 64]

        gate_sum = sat_gate + track_gate + era5_gate  # [B, 1]

        # Normalize safely
        gate_sum_safe = torch.where(
            fusion_valid_mask == 1,
            gate_sum + self.epsilon,
            torch.ones_like(gate_sum),
        )

        fused_embedding = torch.where(
            fusion_valid_mask == 1,
            weighted_sum / gate_sum_safe,
            torch.zeros_like(weighted_sum),
        )

        modality_mask = torch.cat([sat_mask, track_mask, era5_mask], dim=1)  # [B, 3]

        return {
            "fused_embedding": fused_embedding,
            "satellite_gate": sat_gate,
            "track_gate": track_gate,
            "era5_gate": era5_gate,
            "modality_mask": modality_mask,
            "fusion_valid_mask": fusion_valid_mask,
        }
