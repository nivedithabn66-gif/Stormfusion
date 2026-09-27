"""Grad-CAM Explainability Module for STORMFUSION (Step 19).

Extracts spatial attribution heatmaps (Class Activation Maps) for STORMFUSION using
gradient-weighted feature activations from convolutional layers (ResNet18 / Conv2D encoders).

CRITICAL SCIENTIFIC DISCLAIMER:
- Grad-CAM highlights spatial feature regions that influence model predictions.
- Grad-CAM provides feature attribution, NOT physical causality.
- Do NOT interpret Grad-CAM heatmaps as causal meteorological proofs.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMHook:
    """Attaches forward and backward hooks to a target convolutional layer."""

    def __init__(self, target_layer: nn.Module):
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> None:
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        """Removes registered hooks."""
        if hasattr(self, "forward_handle"):
            self.forward_handle.remove()
        if hasattr(self, "backward_handle"):
            self.backward_handle.remove()



class StormFusionGradCAM:
    """Grad-CAM attribution map generator for STORMFUSION."""

    def __init__(self, model: nn.Module, target_layer_name: Optional[str] = None):
        self.model = model
        self.target_layer_name = target_layer_name or self._auto_discover_target_layer()
        self.target_layer = self._get_module_by_name(self.target_layer_name)

    def _auto_discover_target_layer(self) -> str:
        """Discovers the primary convolutional feature extraction layer."""
        if hasattr(self.model, "satellite_encoder"):
            sat = self.model.satellite_encoder
            if hasattr(sat, "resnet") and hasattr(sat.resnet, "layer4"):
                return "satellite_encoder.resnet.layer4"
            elif hasattr(sat, "spatial_conv"):
                return "satellite_encoder.spatial_conv"

        if hasattr(self.model, "era5_encoder"):
            era5 = self.model.era5_encoder
            if hasattr(era5, "conv2"):
                return "era5_encoder.conv2"

        # Fallback search for last Conv2d layer
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return name

        raise ValueError("No suitable Conv2d layer found in StormFusion model.")

    def _get_module_by_name(self, name: str) -> nn.Module:
        """Retrieves submodule by dot-separated string path."""
        parts = name.split(".")
        curr = self.model
        for part in parts:
            if hasattr(curr, part):
                curr = getattr(curr, part)
            else:
                raise AttributeError(f"Module {curr} has no submodule '{part}' in path '{name}'.")
        return curr

    def generate_cam(
        self,
        inputs: Dict[str, torch.Tensor],
        target_head: str = "detection",
        target_index: int = 0,
        lead_time_index: int = 0,
    ) -> Dict[str, Any]:
        """Generates normalized Grad-CAM spatial activation map for specified target head and sample.

        Args:
            inputs: Model input dictionary (satellite, track, era5 tensors & masks).
            target_head: Target head name ('detection', 'intensity', 'pressure', 'track_delta').
            target_index: Batch index or class index.
            lead_time_index: Lead time index (0: 3h, 1: 6h, 2: 12h, 3: 24h) for regression.

        Returns:
            Dict containing raw CAM, normalized CAM [0..1], target metadata, and attribution info.
        """
        self.model.eval()

        # Attach hooks
        hook = GradCAMHook(self.target_layer)

        # Clone input tensors and enable gradients on inputs where needed
        req_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                req_inputs[k] = v.detach().clone()
            else:
                req_inputs[k] = v

        # Forward pass with gradient recording on model outputs
        self.model.zero_grad()
        outputs = self.model(**req_inputs)

        if target_head not in outputs:
            hook.remove()
            raise KeyError(f"Target head '{target_head}' not found in model outputs: {list(outputs.keys())}")

        output_tensor = outputs[target_head]  # Shape depends on head

        # Select target scalar score for backward pass
        if target_head == "detection_logits":
            score = output_tensor[target_index, 0]
        elif target_head in ["intensity", "pressure"]:
            score = output_tensor[target_index, lead_time_index]
        elif target_head == "track_delta":
            score = output_tensor[target_index, lead_time_index, 0]  # default delta_lat component
        else:
            if output_tensor.ndim >= 2:
                score = output_tensor[target_index, 0]
            else:
                score = output_tensor[target_index]

        # Backward pass to compute gradients
        score.backward(retain_graph=True)

        activations = hook.activations  # [B*T, C, H, W] or [B, C, H, W] or similar
        gradients = hook.gradients

        hook.remove()

        if activations is None or gradients is None:
            raise RuntimeError(f"Grad-CAM failed to capture activations or gradients for layer {self.target_layer_name}.")

        # Handle 4D or 5D activation shapes
        # If activations are 4D: [N_batch, C, H_feat, W_feat]
        if activations.ndim == 4:
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [N_batch, C, 1, 1]
            cam = torch.sum(weights * activations, dim=1)              # [N_batch, H_feat, W_feat]
            cam = F.relu(cam)
        else:
            cam = F.relu(torch.sum(activations, dim=1))

        cam_np = cam.detach().cpu().numpy()

        # Min-max normalization per sample to [0.0, 1.0]
        cam_min = np.min(cam_np)
        cam_max = np.max(cam_np)
        if cam_max > cam_min:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam_norm = np.zeros_like(cam_np)

        return {
            "cam_raw": cam_np,
            "cam_normalized": cam_norm,
            "target_layer": self.target_layer_name,
            "target_head": target_head,
            "target_index": target_index,
            "lead_time_index": lead_time_index,
            "min_val": float(cam_min),
            "max_val": float(cam_max),
            "causality_disclaimer": "Grad-CAM heatmaps represent gradient-weighted spatial feature attribution, NOT physical causality.",
        }
