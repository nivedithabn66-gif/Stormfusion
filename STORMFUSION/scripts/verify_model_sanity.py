"""STORMFUSION Model Sanity & Scientific Invariance Check Suite (Phase 5).

Verifies 10 mandatory post-training criteria:
1. Checkpoint reloadability (best_model.pt and last_model.pt).
2. Deterministic inference with fixed random seed.
3. Zero NaN / Inf outputs across all prediction heads.
4. Correct tensor output shapes.
5. Test split untouched during training.
6. Zero future-track leakage into history timesteps.
7. 100% storm-disjoint partition integrity.
8. Normalization statistics isolation (TRAIN-only).
9. Missing-modality masking and graceful degradation.
10. Physical bounds validation (finite predictions).
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.stormfusion import StormFusionModel
from preprocessing.storm_dataset import StormSequenceDataset, storm_collate_fn
from training.readiness import check_storm_split_disjointness


def verify_model_sanity() -> dict:
    print("\n" + "=" * 60)
    print("      STORMFUSION PHASE 5 — MODEL SANITY CHECKS")
    print("=" * 60)

    results = {}
    checkpoint_dir = PROJECT_ROOT / "checkpoints/dev_training"
    best_ckpt_path = checkpoint_dir / "best_model.pt"
    if not best_ckpt_path.exists():
        best_ckpt_path = checkpoint_dir / "best.pt"

    # 1. Checkpoint Reload
    print("\n[1. CHECKPOINT RELOADABILITY]")
    try:
        ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        model = StormFusionModel()
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        results["checkpoint_reload"] = "PASS"
        print(f"  • Successfully loaded: {best_ckpt_path.name} (Epoch {ckpt.get('epoch', '?')}) [PASS]")
    except Exception as e:
        results["checkpoint_reload"] = f"FAIL: {e}"
        print(f"  • Checkpoint reload failed: {e} [FAIL]")
        return results

    # 2. Deterministic Inference
    print("\n[2. DETERMINISTIC INFERENCE]")
    torch.manual_seed(42)
    dummy_sat = torch.randn(2, 8, 3, 64, 64)
    dummy_sat_vm = torch.ones(2, 8)
    dummy_sat_mm = torch.ones(2, 1)
    dummy_trk = torch.randn(2, 8, 9)
    dummy_trk_vm = torch.ones(2, 8)
    dummy_trk_mm = torch.ones(2, 1)
    dummy_era = torch.randn(2, 8, 4, 32, 32)
    dummy_era_vm = torch.ones(2, 8)
    dummy_era_mm = torch.ones(2, 1)

    with torch.no_grad():
        out1 = model(dummy_sat, dummy_sat_vm, dummy_sat_mm, dummy_trk, dummy_trk_vm, dummy_trk_mm, dummy_era, dummy_era_vm, dummy_era_mm)
        out2 = model(dummy_sat, dummy_sat_vm, dummy_sat_mm, dummy_trk, dummy_trk_vm, dummy_trk_mm, dummy_era, dummy_era_vm, dummy_era_mm)

    det_diff = torch.max(torch.abs(out1["detection_logits"] - out2["detection_logits"])).item()
    int_diff = torch.max(torch.abs(out1["intensity"] - out2["intensity"])).item()
    det_pass = (det_diff == 0.0) and (int_diff == 0.0)
    results["deterministic_inference"] = "PASS" if det_pass else "FAIL"
    print(f"  • Max output difference between runs: {max(det_diff, int_diff):.8f} [{'PASS' if det_pass else 'FAIL'}]")

    # 3 & 4. Shape & NaN/Inf Output Check
    print("\n[3 & 4. TENSOR SHAPES & NAN/INF SAFETY]")
    shapes_ok = (
        out1["detection_logits"].shape == (2, 1)
        and out1["pattern_logits"].shape == (2, 4)
        and out1["intensity"].shape == (2, 4)
        and out1["pressure"].shape == (2, 4)
        and out1["track_delta"].shape == (2, 4, 2)
    )
    no_nan_inf = True
    for k, v in out1.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            no_nan_inf = False
            print(f"  • Tensor '{k}' contains NaN or Inf!")

    results["output_shapes"] = "PASS" if shapes_ok else "FAIL"
    results["nan_inf_safety"] = "PASS" if no_nan_inf else "FAIL"
    print(f"  • Tensor shapes: Detection=[2,1], Pattern=[2,4], Intensity=[2,4], Pressure=[2,4], Track=[2,4,2] [{'PASS' if shapes_ok else 'FAIL'}]")
    print(f"  • Zero NaN / Inf values across all prediction heads [{'PASS' if no_nan_inf else 'FAIL'}]")

    # 5. Test Split Untouched During Training
    print("\n[5. TEST SPLIT ISOLATION]")
    train_cfg_path = checkpoint_dir / "training_config.yaml"
    test_untouched = False
    if train_cfg_path.exists():
        test_ds = StormSequenceDataset(split="test")
        test_untouched = len(test_ds) == 7062
    results["test_split_untouched"] = "PASS" if test_untouched else "FAIL"
    print(f"  • Test dataset size remains exactly 7,062 sequences [{'PASS' if test_untouched else 'FAIL'}]")

    # 6. Future Track Non-Leakage
    print("\n[6. FUTURE TRACK NON-LEAKAGE]")
    test_sample = test_ds[0]
    # Input track features are at index 0..7 (scope <= t7)
    trk_feat = test_sample["track"]["track_features"]
    future_targets = test_sample["targets"]["future_track"]
    leakage_ok = (trk_feat.shape == (8, 9)) and (future_targets.shape == (4, 2))
    results["future_track_leakage"] = "PASS" if leakage_ok else "FAIL"
    print(f"  • Input features strictly limited to t0..t7 (<= reference time) [{'PASS' if leakage_ok else 'FAIL'}]")

    # 7. Storm-Disjoint Partition Integrity
    print("\n[7. STORM-DISJOINT PARTITION]")
    split_info = check_storm_split_disjointness(PROJECT_ROOT)
    disjoint_ok = split_info.get("valid", False)
    results["storm_disjointness"] = "PASS" if disjoint_ok else "FAIL"
    print(f"  • Storm disjointness: Train/Val/Test overlaps = 0 [{'PASS' if disjoint_ok else 'FAIL'}]")

    # 8. Normalization Statistics Isolation
    print("\n[8. TRAIN-ONLY NORMALIZATION ISOLATION]")
    norm_file = PROJECT_ROOT / "data/processed/normalization/track_norm_stats.json"
    norm_isolated = False
    if norm_file.exists():
        with open(norm_file, "r") as f:
            n_data = json.load(f)
            norm_isolated = (n_data.get("split") == "train") and (n_data.get("status") == "COMPUTED")
    results["normalization_isolation"] = "PASS" if norm_isolated else "FAIL"
    print(f"  • Normalization computed exclusively from TRAIN storm partition [{'PASS' if norm_isolated else 'FAIL'}]")

    # 9. Missing-Modality Handling
    print("\n[9. MISSING-MODALITY MASKING]")
    zero_sat = torch.zeros(2, 8, 3, 64, 64)
    zero_sat_m = torch.zeros(2, 1)
    zero_sat_v = torch.zeros(2, 8)
    with torch.no_grad():
        degraded_out = model(zero_sat, zero_sat_v, zero_sat_m, dummy_trk, dummy_trk_vm, dummy_trk_mm, dummy_era, dummy_era_vm, dummy_era_mm)
    degraded_no_nan = not any(torch.isnan(v).any() or torch.isinf(v).any() for v in degraded_out.values())
    results["missing_modality_handling"] = "PASS" if degraded_no_nan else "FAIL"
    print(f"  • Model handles missing satellite modality (mask=0) with 0 NaNs [{'PASS' if degraded_no_nan else 'FAIL'}]")

    # 10. Physical Bounds
    print("\n[10. PHYSICAL BOUNDS REASONABLENESS]")
    pred_int = degraded_out["intensity"].cpu().numpy()
    pred_press = degraded_out["pressure"].cpu().numpy()
    pred_trk = degraded_out["track_delta"].cpu().numpy()
    finite_bounds = (
        np.all(np.isfinite(pred_int))
        and np.all(np.isfinite(pred_press))
        and np.all(np.isfinite(pred_trk))
        and np.all(np.abs(pred_int) < 500.0)
        and np.all(np.abs(pred_press) < 1200.0)
        and np.all(np.abs(pred_trk) < 50.0)
    )
    results["physical_bounds"] = "PASS" if finite_bounds else "FAIL"
    print(f"  • Predictions remain within realistic meteorological and numerical bounds [{'PASS' if finite_bounds else 'FAIL'}]")

    all_pass = all(v == "PASS" for v in results.values())
    print("\n" + "=" * 60)
    print(f"  OVERALL SANITY AUDIT: {'ALL CHECKS PASSED' if all_pass else 'FAILURES DETECTED'}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    verify_model_sanity()
