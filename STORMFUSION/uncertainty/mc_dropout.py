"""Monte Carlo Dropout Engine for STORMFUSION (Step 18).

Executes stochastic forward passes by enabling dropout layers during inference
while keeping normalization layers (e.g. BatchNorm) in evaluation mode.

CRITICAL RULES:
- Does NOT perform model weight updates or training.
- Runs N stochastic forward passes (default N=20).
- Collects predictions for classification and regression UQ estimation.
"""

from typing import Dict, Any, List, Optional
import copy
import torch
import torch.nn as nn


def enable_dropout_only(model: nn.Module) -> None:
    """Enables dropout layers for stochastic inference without altering BatchNorm eval state."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train()


class MCDropoutEstimator:
    """Monte Carlo Dropout Stochastic Inference Engine."""

    def __init__(self, model: nn.Module, num_samples: int = 20):
        self.model = model
        self.num_samples = num_samples

    def predict_stochastic(
        self,
        satellite_tensor: Optional[torch.Tensor] = None,
        satellite_valid_mask: Optional[torch.Tensor] = None,
        satellite_modality_mask: Optional[torch.Tensor] = None,
        track_features: Optional[torch.Tensor] = None,
        track_valid_mask: Optional[torch.Tensor] = None,
        track_modality_mask: Optional[torch.Tensor] = None,
        era5_tensor: Optional[torch.Tensor] = None,
        era5_valid_mask: Optional[torch.Tensor] = None,
        era5_modality_mask: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Executes N stochastic forward passes under Monte Carlo dropout."""
        N = num_samples if num_samples is not None else self.num_samples

        # Ensure model is in eval mode, then enable dropout modules only
        self.model.eval()
        enable_dropout_only(self.model)

        sample_outputs: List[Dict[str, torch.Tensor]] = []

        with torch.no_grad():
            for _ in range(N):
                out = self.model(
                    satellite_tensor=satellite_tensor,
                    satellite_valid_mask=satellite_valid_mask,
                    satellite_modality_mask=satellite_modality_mask,
                    track_features=track_features,
                    track_valid_mask=track_valid_mask,
                    track_modality_mask=track_modality_mask,
                    era5_tensor=era5_tensor,
                    era5_valid_mask=era5_valid_mask,
                    era5_modality_mask=era5_modality_mask,
                )
                # Clone outputs to ensure detach from computational graph
                cloned_out = {
                    k: (v.clone() if isinstance(v, torch.Tensor) else v)
                    for k, v in out.items()
                }
                sample_outputs.append(cloned_out)

        # Restore model to clean eval state
        self.model.eval()

        return sample_outputs
