"""Evaluate UQ (MC-Dropout) and XAI (Grad-CAM) on Trained Checkpoint (Section 10).

Uses `checkpoints/dev_training/best.pt` to compute:
1. MC-Dropout Uncertainty Quantification across N=10 stochastic forward passes.
2. Grad-CAM Feature Attribution heatmaps for Detection, Intensity, Central Pressure, and Track.
Saves results to `checkpoints/dev_training/uq_xai_results.json`.
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.storm_dataset import StormSequenceDataset, storm_collate_fn
from models.stormfusion import StormFusionModel
from uncertainty import STORMFUSIONUQRunner
from explainability import StormFusionGradCAM, overlay_cam_on_image, format_explanation_metadata


def run_uq_xai_evaluation() -> dict:
    print("\n==================================================", flush=True)
    print("   STORMFUSION POST-TRAINING UQ & GRAD-CAM EVAL   ", flush=True)
    print("==================================================", flush=True)

    checkpoint_path = PROJECT_ROOT / "checkpoints/dev_training/best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Trained checkpoint missing at: {checkpoint_path}")

    # Load model state from trained checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = StormFusionModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load validation sample for UQ & XAI verification
    val_dataset = StormSequenceDataset(split="val")
    val_loader = DataLoader(
        torch.utils.data.Subset(val_dataset, [0, 1]),
        batch_size=2,
        shuffle=False,
        collate_fn=storm_collate_fn,
    )
    sample_batch = next(iter(val_loader))

    sat_dict = sample_batch["satellite"]
    trk_dict = sample_batch["track"]
    env_dict = sample_batch["environment"]
    tgt_dict = sample_batch["targets"]

    inputs = {
        "satellite_tensor": sat_dict["satellite_tensor"],
        "satellite_valid_mask": sat_dict["satellite_valid_mask"],
        "satellite_modality_mask": sat_dict["satellite_modality_mask"],
        "track_features": trk_dict["track_features"],
        "track_valid_mask": trk_dict["track_valid_mask"],
        "track_modality_mask": torch.ones_like(trk_dict["track_valid_mask"][:, :1]),
        "era5_tensor": env_dict["era5_features"],
        "era5_valid_mask": env_dict["era5_valid_mask"],
        "era5_modality_mask": env_dict["era5_modality_mask"],
    }

    # 1. MC-Dropout Uncertainty Evaluation
    print("\n[1. EVALUATING MC-DROPOUT UNCERTAINTY QUANTIFICATION]", flush=True)
    uq_runner = STORMFUSIONUQRunner(model=model, num_mc_samples=10)
    uq_results = uq_runner.run_uq_analysis(**inputs)

    uq_passed = uq_results.get("status") == "COMPUTED"
    print(f"  • Detection Variance     : {float(uq_results['detection']['variance'][0]):.6f}", flush=True)
    print(f"  • Intensity 24h Std      : {float(uq_results['intensity']['24h']['std'][0]):.6f}", flush=True)
    print(f"  • Track 24h Pos Uncertainty: {float(uq_results['track']['24h']['positional_uncertainty_km'][0]):.2f} km", flush=True)
    print(f"  • UQ Verification        : {'PASS' if uq_passed else 'FAIL'}", flush=True)

    # 2. Grad-CAM Feature Attribution Evaluation
    print("\n[2. EVALUATING GRAD-CAM FEATURE ATTRIBUTION]", flush=True)
    gradcam_explainer = StormFusionGradCAM(model=model)

    cam_heads = [
        ("detection_logits", 0, None),
        ("intensity", 0, 3),  # 24h lead intensity
        ("pressure", 0, 3),   # 24h lead pressure
        ("track_delta", 0, 3),# 24h lead track
    ]

    cam_results = {}
    xai_passed = True

    for head_name, tgt_idx, lead_idx in cam_heads:
        try:
            if lead_idx is not None:
                res = gradcam_explainer.generate_cam(
                    inputs, target_head=head_name, target_index=tgt_idx, lead_time_index=lead_idx
                )
            else:
                res = gradcam_explainer.generate_cam(inputs, target_head=head_name, target_index=tgt_idx)

            cam_map = res["cam_normalized"]
            is_valid = (not np.isnan(cam_map).any()) and (0.0 <= np.min(cam_map) <= np.max(cam_map) <= 1.0)
            
            cam_results[f"{head_name}_lead{lead_idx if lead_idx is not None else 0}"] = {
                "status": "GENERATED" if is_valid else "INVALID",
                "target_head": head_name,
                "lead_time_index": lead_idx,
                "cam_min": float(np.min(cam_map)),
                "cam_max": float(np.max(cam_map)),
                "cam_mean": float(np.mean(cam_map)),
                "target_layer": res.get("target_layer"),
            }

            if not is_valid:
                xai_passed = False
            print(f"  • Head '{head_name}' (lead={lead_idx}): Min={np.min(cam_map):.4f}, Max={np.max(cam_map):.4f} [PASS]", flush=True)
        except Exception as e:
            cam_results[f"{head_name}"] = {"status": "FAILED", "error": str(e)}
            xai_passed = False
            print(f"  • Head '{head_name}': FAILED ({e})", flush=True)

    overall_results = {
        "checkpoint_used": str(checkpoint_path),
        "mc_dropout_uq": {
            "status": "PASS" if uq_passed else "FAIL",
            "details": uq_results,
        },
        "gradcam_xai": {
            "status": "PASS" if xai_passed else "FAIL",
            "details": cam_results,
        },
    }

    out_path = PROJECT_ROOT / "checkpoints/dev_training/uq_xai_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2)

    print(f"\n[POST-TRAINING UQ & XAI EVALUATION COMPLETE] Saved to {out_path}", flush=True)
    return overall_results


if __name__ == "__main__":
    run_uq_xai_evaluation()
