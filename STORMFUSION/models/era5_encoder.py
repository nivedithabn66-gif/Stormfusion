"""ERA5 Environmental Encoder Module for STORMFUSION (Step 8B).

Processes 8-timestep 4-channel atmospheric reanalysis fields (u10, v10, msl, t2m) over a 32x32 spatial grid.

Architecture:
  [B, 8, 4, 32, 32] -> Reshape [B*8, 4, 32, 32] -> Conv2d(4, 16) -> ReLU -> Conv2d(16, 32) -> ReLU
  -> AdaptiveAvgPool2d(1, 1) -> Linear(32, 64) -> Reshape [B, 8, 64] -> Mask-Weighted Temporal Aggregation -> [B, 64]

Mask Handling:
  - Accepts era5_valid_mask [B, 8] and era5_modality_mask [B, 1].
  - When real ERA5 data is unavailable (era5_modality_mask == 0), returns zero-filled placeholder embedding [B, 64].
  - Masks MUST remain 0 to prevent downstream models from treating placeholders as physical measurements.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class ERA5Encoder(nn.Module):
    """STORMFUSION Lightweight ERA5 Environmental Encoder Module."""

    def __init__(self, input_channels: int = 4, hidden_channels: int = 32, embedding_dim: int = 64):
        super(ERA5Encoder, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim

        # Lightweight CNN spatial encoder
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, hidden_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels, embedding_dim)

    def forward(
        self,
        era5_tensor: torch.Tensor,
        era5_valid_mask: Optional[torch.Tensor] = None,
        era5_modality_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass for 5D ERA5 environmental sequence tensor.

        Args:
            era5_tensor: [B, 8, 4, 32, 32] environmental sequence tensor
            era5_valid_mask: Optional [B, 8] validity mask per timestep
            era5_modality_mask: Optional [B, 1] overall modality mask

        Returns:
            Dict containing "era5_embedding" [B, 64], "era5_valid_mask", "era5_modality_mask"
        """
        batch_size, seq_len, num_channels, height, width = era5_tensor.size()

        if era5_valid_mask is None:
            era5_valid_mask = torch.ones(batch_size, seq_len, device=era5_tensor.device, dtype=era5_tensor.dtype)
        if era5_modality_mask is None:
            era5_modality_mask = torch.ones(batch_size, 1, device=era5_tensor.device, dtype=era5_tensor.dtype)

        # 1. Reshape sequence tensor [B, 8, 4, 32, 32] -> [B * 8, 4, 32, 32]
        x_reshaped = era5_tensor.view(batch_size * seq_len, num_channels, height, width)

        # 2. Lightweight CNN spatial feature extraction
        x = self.relu1(self.conv1(x_reshaped))
        x = self.relu2(self.conv2(x))

        # 3. Global pooling -> [B * 8, 32, 1, 1] -> flatten -> Linear -> [B * 8, 64]
        x_pooled = self.global_pool(x)
        x_flat = torch.flatten(x_pooled, start_dim=1)
        step_embeddings = self.fc(x_flat)  # [B * 8, 64]

        # 4. Reshape back to sequence -> [B, 8, 64]
        step_seq = step_embeddings.view(batch_size, seq_len, self.embedding_dim)

        # 5. Mask-aware temporal aggregation
        era5_embedding_list = []
        for b in range(batch_size):
            mod_val = era5_modality_mask[b].item() if era5_modality_mask.dim() == 2 else era5_modality_mask[b].item()
            if mod_val < 0.5:
                # Modality unavailable: return zero placeholder
                era5_embedding_list.append(torch.zeros(self.embedding_dim, device=era5_tensor.device, dtype=era5_tensor.dtype))
            else:
                valid_mask_b = era5_valid_mask[b]  # [8]
                valid_count = (valid_mask_b > 0.5).sum().item()
                if valid_count > 0:
                    mask_expanded = valid_mask_b.unsqueeze(-1)  # [8, 1]
                    valid_steps = step_seq[b] * mask_expanded
                    mean_emb = valid_steps.sum(dim=0) / valid_count
                    era5_embedding_list.append(mean_emb)
                else:
                    era5_embedding_list.append(torch.zeros(self.embedding_dim, device=era5_tensor.device, dtype=era5_tensor.dtype))

        era5_embedding = torch.stack(era5_embedding_list, dim=0)  # [B, 64]

        return {
            "era5_embedding": era5_embedding,
            "era5_valid_mask": era5_valid_mask,
            "era5_modality_mask": era5_modality_mask,
        }
