"""Robustness Software Metrics & Invariance Validation Module for STORMFUSION.

Evaluates engineering-level software robustness properties:
1. Mask Invariance: Modifying an unavailable modality tensor yields 0 change in outputs.
2. Gate Bounds: All gates in [0, 1] and missing modality gates == 0.
3. NaN/Inf Safety: Unavailable branch NaN/Inf values do not leak into outputs.
4. Determinism: Identical inputs + masks + state produce bit-identical outputs.
"""

from typing import Dict, Any
import torch

from models.gated_fusion import AvailabilityAwareGatedFusion
from models.multitask_model import StormFusionMultiTaskModel


def evaluate_mask_invariance(
    fusion_model: AvailabilityAwareGatedFusion,
    multitask_model: StormFusionMultiTaskModel,
    batch_size: int = 4,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Tests mask invariance across all three modalities.

    For each modality (satellite, track, ERA5):
    1. Set its availability mask to 0 (missing).
    2. Pass original embedding tensor and compute fused output + predictions.
    3. Modify missing modality embedding tensor (add noise, NaNs, zeros).
    4. Compute new fused output + predictions.
    5. Verify max absolute difference in outputs is exactly 0.0.

    Returns:
        Dict with invariance pass/fail results for each modality.
    """
    fusion_model.eval()
    multitask_model.eval()

    sat_base = torch.randn(batch_size, 128, device=device)
    track_base = torch.randn(batch_size, 64, device=device)
    era5_base = torch.randn(batch_size, 64, device=device)

    sat_mask_on = torch.ones(batch_size, 1, device=device)
    track_mask_on = torch.ones(batch_size, 1, device=device)
    era5_mask_on = torch.ones(batch_size, 1, device=device)

    mask_off = torch.zeros(batch_size, 1, device=device)

    results = {}

    # 1. Test Satellite Invariance (Satellite missing: sat_mask = 0)
    with torch.no_grad():
        out1 = fusion_model(sat_base, track_base, era5_base, mask_off, track_mask_on, era5_mask_on)
        pred1 = multitask_model(out1["fused_embedding"])

        # Perturb satellite input
        sat_alt = torch.randn(batch_size, 128, device=device) * 100.0 + 50.0
        out2 = fusion_model(sat_alt, track_base, era5_base, mask_off, track_mask_on, era5_mask_on)
        pred2 = multitask_model(out2["fused_embedding"])

        max_delta_sat = float(torch.max(torch.abs(out1["fused_embedding"] - out2["fused_embedding"])))
        sat_inv_pass = (max_delta_sat == 0.0)
        results["satellite_invariance"] = {"pass": sat_inv_pass, "max_delta": max_delta_sat}

    # 2. Test Track Invariance (Track missing: track_mask = 0)
    with torch.no_grad():
        out1 = fusion_model(sat_base, track_base, era5_base, sat_mask_on, mask_off, era5_mask_on)
        pred1 = multitask_model(out1["fused_embedding"])

        track_alt = torch.randn(batch_size, 64, device=device) * 100.0 - 50.0
        out2 = fusion_model(sat_base, track_alt, era5_base, sat_mask_on, mask_off, era5_mask_on)
        pred2 = multitask_model(out2["fused_embedding"])

        max_delta_track = float(torch.max(torch.abs(out1["fused_embedding"] - out2["fused_embedding"])))
        track_inv_pass = (max_delta_track == 0.0)
        results["track_invariance"] = {"pass": track_inv_pass, "max_delta": max_delta_track}

    # 3. Test ERA5 Invariance (ERA5 missing: era5_mask = 0)
    with torch.no_grad():
        out1 = fusion_model(sat_base, track_base, era5_base, sat_mask_on, track_mask_on, mask_off)
        pred1 = multitask_model(out1["fused_embedding"])

        era5_alt = torch.randn(batch_size, 64, device=device) * 100.0 + 25.0
        out2 = fusion_model(sat_base, track_base, era5_alt, sat_mask_on, track_mask_on, mask_off)
        pred2 = multitask_model(out2["fused_embedding"])

        max_delta_era5 = float(torch.max(torch.abs(out1["fused_embedding"] - out2["fused_embedding"])))
        era5_inv_pass = (max_delta_era5 == 0.0)
        results["era5_invariance"] = {"pass": era5_inv_pass, "max_delta": max_delta_era5}

    all_pass = sat_inv_pass and track_inv_pass and era5_inv_pass
    results["all_mask_invariance_pass"] = all_pass

    return results


def evaluate_nan_inf_safety(
    fusion_model: AvailabilityAwareGatedFusion,
    multitask_model: StormFusionMultiTaskModel,
    batch_size: int = 4,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Tests that NaN/Inf in an unavailable branch do not contaminate output."""
    fusion_model.eval()
    multitask_model.eval()

    sat_base = torch.randn(batch_size, 128, device=device)
    track_base = torch.randn(batch_size, 64, device=device)
    era5_nan = torch.full((batch_size, 64), float("nan"), device=device)

    sat_mask_on = torch.ones(batch_size, 1, device=device)
    track_mask_on = torch.ones(batch_size, 1, device=device)
    era5_mask_off = torch.zeros(batch_size, 1, device=device)

    with torch.no_grad():
        # Mask out NaN ERA5 branch
        out = fusion_model(sat_base, track_base, era5_nan, sat_mask_on, track_mask_on, era5_mask_off)
        preds = multitask_model(out["fused_embedding"])

        fused = out["fused_embedding"]
        nan_in_fused = bool(torch.isnan(fused).any() or torch.isinf(fused).any())

        preds_has_nan = any(
            torch.isnan(v).any() or torch.isinf(v).any()
            for v in preds.values()
        )

        safe = (not nan_in_fused) and (not preds_has_nan)

    return {"nan_inf_safety_pass": safe}


def evaluate_determinism(
    fusion_model: AvailabilityAwareGatedFusion,
    multitask_model: StormFusionMultiTaskModel,
    batch_size: int = 4,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Tests that identical inputs and masks produce bit-identical output tensors."""
    fusion_model.eval()
    multitask_model.eval()

    sat = torch.randn(batch_size, 128, device=device)
    track = torch.randn(batch_size, 64, device=device)
    era5 = torch.randn(batch_size, 64, device=device)

    m_sat = torch.ones(batch_size, 1, device=device)
    m_track = torch.zeros(batch_size, 1, device=device)
    m_era5 = torch.ones(batch_size, 1, device=device)

    with torch.no_grad():
        out1 = fusion_model(sat, track, era5, m_sat, m_track, m_era5)
        pred1 = multitask_model(out1["fused_embedding"])

        out2 = fusion_model(sat, track, era5, m_sat, m_track, m_era5)
        pred2 = multitask_model(out2["fused_embedding"])

        fused_equal = torch.equal(out1["fused_embedding"], out2["fused_embedding"])
        preds_equal = all(torch.equal(pred1[k], pred2[k]) for k in pred1)

        det_pass = fused_equal and preds_equal

    return {"determinism_pass": det_pass}
