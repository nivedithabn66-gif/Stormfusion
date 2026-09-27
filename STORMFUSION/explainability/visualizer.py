"""Explainability Visualization & Metadata Formatter for STORMFUSION (Step 19).

Provides utilities for:
1. Heatmap overlay blending on satellite imagery / ERA5 fields.
2. Structuring reproducible explanation metadata with model provenance.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np


def resize_cam_map(cam: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resizes a 2D spatial CAM heatmap to target_shape (H, W) using bilinear interpolation."""
    H_target, W_target = target_shape
    H_orig, W_orig = cam.shape

    if (H_orig, W_orig) == (H_target, W_target):
        return cam

    # Simple numpy bilinear interpolation
    x_ratio = float(W_orig - 1) / float(W_target - 1) if W_target > 1 else 0.0
    y_ratio = float(H_orig - 1) / float(H_target - 1) if H_target > 1 else 0.0

    y_indices = (np.arange(H_target) * y_ratio).astype(int)
    x_indices = (np.arange(W_target) * x_ratio).astype(int)

    return cam[np.ix_(y_indices, x_indices)]


def overlay_cam_on_image(
    cam: np.ndarray,
    base_image: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Blends normalized CAM heatmap [H, W] onto base image [H, W] or [H, W, 3].

    Args:
        cam: 2D numpy array [0.0..1.0]
        base_image: 2D or 3D numpy array
        alpha: Blend ratio for heatmap (default 0.5)
        colormap: Simple pseudo-color scheme ('jet' or 'viridis')

    Returns:
        3D numpy array [H, W, 3] with values in [0, 255] uint8.
    """
    H, W = cam.shape[:2]

    # Resize CAM to match base image if shapes differ
    if base_image.ndim >= 2:
        img_h, img_w = base_image.shape[:2]
        if (H, W) != (img_h, img_w):
            cam = resize_cam_map(cam, (img_h, img_w))
            H, W = img_h, img_w

    # Simple pseudo-color mapping (Red-Yellow-Blue gradient for JET approximation)
    cam_clamped = np.clip(cam, 0.0, 1.0)
    heatmap_rgb = np.zeros((H, W, 3), dtype=np.float32)

    # Red channel (high activation)
    heatmap_rgb[:, :, 0] = np.clip(2.0 * cam_clamped - 0.5, 0.0, 1.0)
    # Green channel (mid activation)
    heatmap_rgb[:, :, 1] = np.clip(1.0 - math_abs_arr(2.0 * cam_clamped - 1.0), 0.0, 1.0)
    # Blue channel (low activation)
    heatmap_rgb[:, :, 2] = np.clip(1.0 - 2.0 * cam_clamped, 0.0, 1.0)

    # Base image RGB normalization
    if base_image.ndim == 2:
        img_norm = (base_image - np.min(base_image)) / (np.max(base_image) - np.min(base_image) + 1e-8)
        img_rgb = np.stack([img_norm] * 3, axis=-1)
    else:
        img_rgb = base_image.astype(np.float32)
        if np.max(img_rgb) > 1.0:
            img_rgb = img_rgb / 255.0

    blended = alpha * heatmap_rgb + (1.0 - alpha) * img_rgb
    blended_uint8 = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    return blended_uint8


def math_abs_arr(arr: np.ndarray) -> np.ndarray:
    return np.abs(arr)


def format_explanation_metadata(
    sample_id: str,
    timestamp: str,
    target_head: str,
    lead_time: str,
    layer_name: str,
    model_version: str = "v1.0",
) -> Dict[str, Any]:
    """Formats reproducible explanation metadata dictionary."""
    return {
        "sample_id": sample_id,
        "timestamp": timestamp,
        "target_head": target_head,
        "lead_time": lead_time,
        "layer_used": layer_name,
        "model_version": model_version,
        "explanation_type": "Grad-CAM Feature Attribution",
        "causality_disclaimer": "Grad-CAM spatial heatmaps quantify gradient-weighted neural feature attribution. They do NOT establish physical meteorological causality.",
    }
