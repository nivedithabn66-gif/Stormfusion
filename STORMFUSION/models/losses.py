"""STORMFUSION Loss Functions for Multi-Task Prediction.

Implements detection, pattern, intensity, pressure, track, and total multi-task loss calculation
with mask-aware missing target handling and safe gradient graph preservation.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def detection_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCEWithLogitsLoss for binary detection task [B, 1]."""
    logits_flat = logits.view(-1, 1)
    target_flat = target.view(-1, 1).float()
    unmasked = F.binary_cross_entropy_with_logits(logits_flat, target_flat, reduction="none")

    if valid_mask is not None:
        mask_flat = valid_mask.view(-1, 1).float()
        valid_count = mask_flat.sum()
        if valid_count == 0:
            return (logits * 0.0).sum()
        return (unmasked * mask_flat).sum() / valid_count.clamp(min=1.0)
    return unmasked.mean()


def pattern_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CrossEntropyLoss for pattern classification task [B, num_classes]."""
    target_long = target.view(-1).long()
    target_clamped = target_long.clamp(min=0)
    unmasked = F.cross_entropy(logits, target_clamped, weight=class_weights, reduction="none")

    if valid_mask is not None:
        mask_flat = valid_mask.view(-1).float()
        valid_count = mask_flat.sum()
        if valid_count == 0:
            return (logits * 0.0).sum()
        return (unmasked * mask_flat).sum() / valid_count.clamp(min=1.0)
    return unmasked.mean()



def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    delta: float = 1.0,
) -> torch.Tensor:
    """Masked Huber loss for continuous regression targets (intensity, pressure)."""
    target_clean = torch.nan_to_num(target, nan=0.0)
    unmasked = F.huber_loss(pred, target_clean, delta=delta, reduction="none")

    if mask is not None:
        mask_float = mask.float()
        valid_count = mask_float.sum()
        if valid_count == 0:
            return (pred * 0.0).sum()
        return (unmasked * mask_float).sum() / valid_count.clamp(min=1.0)
    return unmasked.mean()



def track_loss(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    delta: float = 1.0,
) -> torch.Tensor:
    """Masked Huber loss for track displacements [B, 4, 2]."""
    if mask is not None:
        if mask.dim() == 2 and pred_delta.dim() == 3:
            # Expand horizon mask [B, 4] across coordinate dim -> [B, 4, 2]
            mask = mask.unsqueeze(-1).expand_as(pred_delta)
    return masked_huber_loss(pred_delta, target_delta, mask=mask, delta=delta)


def total_multi_task_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
    huber_delta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calculates weighted multi-task total loss across all 5 prediction tasks.

    Args:
        predictions: Dict containing detection_logits, pattern_logits, intensity, pressure, track_delta
        targets: Dict containing detection_target, pattern_target, intensity_target, pressure_target, track_delta_target
        masks: Dict containing detection_valid_mask, pattern_valid_mask, intensity_valid_mask, pressure_valid_mask, track_valid_mask
        weights: Task loss weights dict
        huber_delta: Huber loss threshold parameter

    Returns:
        Tuple of (total_loss_tensor, loss_values_dict)
    """
    if weights is None:
        weights = {
            "detection": 1.0,
            "pattern": 1.0,
            "intensity": 1.0,
            "pressure": 1.0,
            "track": 1.0,
        }

    # 1. Individual task losses
    l_det = detection_loss(
        predictions["detection_logits"],
        targets["detection_target"],
        valid_mask=masks.get("detection_valid_mask"),
    )

    l_pat = pattern_loss(
        predictions["pattern_logits"],
        targets["pattern_target"],
        valid_mask=masks.get("pattern_valid_mask"),
    )

    l_int = masked_huber_loss(
        predictions["intensity"],
        targets["intensity_target"],
        mask=masks.get("intensity_valid_mask"),
        delta=huber_delta,
    )

    l_press = masked_huber_loss(
        predictions["pressure"],
        targets["pressure_target"],
        mask=masks.get("pressure_valid_mask"),
        delta=huber_delta,
    )

    l_trk = track_loss(
        predictions["track_delta"],
        targets["track_delta_target"],
        mask=masks.get("track_valid_mask"),
        delta=huber_delta,
    )

    # 2. Weighted summation
    total = (
        weights.get("detection", 1.0) * l_det
        + weights.get("pattern", 1.0) * l_pat
        + weights.get("intensity", 1.0) * l_int
        + weights.get("pressure", 1.0) * l_press
        + weights.get("track", 1.0) * l_trk
    )

    loss_dict = {
        "total_loss": total.item(),
        "detection_loss": l_det.item(),
        "pattern_loss": l_pat.item(),
        "intensity_loss": l_int.item(),
        "pressure_loss": l_press.item(),
        "track_loss": l_trk.item(),
    }

    return total, loss_dict
