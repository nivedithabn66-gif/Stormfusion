"""Modality Degradation Runner for STORMFUSION.

Executes inference across modality availability scenarios under torch.no_grad(),
recording gate outputs, fusion embeddings, multi-task predictions, and validity flags
without altering model parameters.
"""

from typing import Dict, Any, Optional
import torch

from models.gated_fusion import AvailabilityAwareGatedFusion
from models.multitask_model import StormFusionMultiTaskModel
from robustness.modality_scenarios import get_modality_scenarios


def check_tensor_nan_inf(tensor_dict: Dict[str, torch.Tensor]) -> bool:
    """Returns True if any tensor in dictionary contains NaN or Inf values."""
    for key, t in tensor_dict.items():
        if isinstance(t, torch.Tensor):
            if torch.isnan(t).any() or torch.isinf(t).any():
                return True
    return False


def run_degradation_scenario(
    fusion_model: AvailabilityAwareGatedFusion,
    multitask_model: StormFusionMultiTaskModel,
    sat_emb: torch.Tensor,
    track_emb: torch.Tensor,
    era5_emb: torch.Tensor,
    scenario_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Executes forward pass for a single modality scenario under torch.no_grad().

    Args:
        fusion_model: AvailabilityAwareGatedFusion instance
        multitask_model: StormFusionMultiTaskModel instance
        sat_emb: Satellite embedding [B, 128]
        track_emb: Track embedding [B, 64]
        era5_emb: ERA5 embedding [B, 64]
        scenario_info: Scenario dictionary from get_modality_scenarios

    Returns:
        Dict containing execution results and diagnostic flags.
    """
    fusion_model.eval()
    multitask_model.eval()

    sat_mask = scenario_info["satellite_mask"]
    track_mask = scenario_info["track_mask"]
    era5_mask = scenario_info["era5_mask"]

    with torch.no_grad():
        fusion_out = fusion_model(
            satellite_embedding=sat_emb,
            track_embedding=track_emb,
            era5_embedding=era5_emb,
            satellite_modality_mask=sat_mask,
            track_modality_mask=track_mask,
            era5_modality_mask=era5_mask,
        )

        fused_emb = fusion_out["fused_embedding"]
        preds = multitask_model(fused_emb)

    # Check NaN / Inf safety across all outputs
    all_tensors = {**fusion_out, **preds}
    has_nan_inf = check_tensor_nan_inf(all_tensors)

    # Check gate bounds: 0 <= gate <= 1
    sat_gate = fusion_out["satellite_gate"]
    track_gate = fusion_out["track_gate"]
    era5_gate = fusion_out["era5_gate"]

    gate_bounds_valid = (
        (sat_gate >= 0.0).all() and (sat_gate <= 1.0).all()
        and (track_gate >= 0.0).all() and (track_gate <= 1.0).all()
        and (era5_gate >= 0.0).all() and (era5_gate <= 1.0).all()
    )

    # Check missing modality gate is strictly 0
    sat_v, trk_v, era_v = scenario_info["mask_values"]
    mask_respected = (
        (sat_v == 1 or (sat_gate == 0.0).all())
        and (trk_v == 1 or (track_gate == 0.0).all())
        and (era_v == 1 or (era5_gate == 0.0).all())
    )

    # Output validity check
    expected_valid = scenario_info["expected_valid"]
    actual_valid = bool((fusion_out["fusion_valid_mask"] == 1.0).all()) if expected_valid else bool((fusion_out["fusion_valid_mask"] == 0.0).all())

    return {
        "scenario_name": scenario_info["name"],
        "mask_values": scenario_info["mask_values"],
        "inference_completed": True,
        "fusion_output": fusion_out,
        "predictions": preds,
        "has_nan_inf": has_nan_inf,
        "gate_bounds_valid": bool(gate_bounds_valid),
        "mask_respected": bool(mask_respected),
        "output_validity": actual_valid,
    }


def evaluate_all_degradation_scenarios(
    fusion_model: AvailabilityAwareGatedFusion,
    multitask_model: StormFusionMultiTaskModel,
    batch_size: int = 4,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Runs all 8 degradation scenarios for given model instances.

    Returns:
        Dict mapping scenario name to execution result dict.
    """
    scenarios = get_modality_scenarios(batch_size=batch_size, device=device)

    sat_emb = torch.randn(batch_size, 128, device=device)
    track_emb = torch.randn(batch_size, 64, device=device)
    era5_emb = torch.randn(batch_size, 64, device=device)

    results = {}
    for name, sc_info in scenarios.items():
        res = run_degradation_scenario(
            fusion_model, multitask_model, sat_emb, track_emb, era5_emb, sc_info
        )
        results[name] = res

    return results
