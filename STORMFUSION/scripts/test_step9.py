"""STORMFUSION Step 9 — Multi-Task Model & Loss Functions Unit & Regression Tests.

Tests the StormFusionMultiTaskModel and loss functions across:
1. Shape correctness for shared trunk and 5 prediction heads
2. Loss calculations (all valid, partial missing, all-pressure missing, all-track missing)
3. Backward gradient pass sanity check (no NaN/Inf gradients)
4. Determinism in eval mode
5. Parameter counts breakdown
6. Regression tests for Step 7, Step 8A, Step 8B, and Step 8C
7. Exports data/processed/sequences/step9_report.json
"""

import json
import os
import sys
import subprocess
from pathlib import Path
import torch
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.multitask_model import StormFusionMultiTaskModel
from models.losses import (
    detection_loss,
    pattern_loss,
    masked_huber_loss,
    track_loss,
    total_multi_task_loss,
)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_regression_script(script_path: str) -> bool:
    """Executes a regression test script and returns True if exit code is 0."""
    try:
        res = subprocess.run(
            [sys.executable, script_path],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return res.returncode == 0
    except Exception as e:
        print(f"Regression error on {script_path}: {e}")
        return False


def test_step9():
    print("=" * 50)
    print("STEP 9 — MULTI-TASK PREDICTION REPORT")
    print("=" * 50)

    # Load configuration
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mt_cfg = config.get("multitask", {})
    shared_dim = mt_cfg.get("shared_dim", 64)
    dropout = mt_cfg.get("dropout", 0.2)
    pattern_classes = mt_cfg.get("pattern_classes", 4)

    loss_cfg = config.get("loss", {})
    loss_weights = {
        "detection": float(loss_cfg.get("detection_weight", 1.0)),
        "pattern": float(loss_cfg.get("pattern_weight", 1.0)),
        "intensity": float(loss_cfg.get("intensity_weight", 1.0)),
        "pressure": float(loss_cfg.get("pressure_weight", 1.0)),
        "track": float(loss_cfg.get("track_weight", 1.0)),
    }
    huber_delta = float(loss_cfg.get("huber_delta", 1.0))

    # 1. Instantiate Model
    try:
        model = StormFusionMultiTaskModel(
            fused_dim=64,
            shared_dim=shared_dim,
            dropout=dropout,
            pattern_classes=pattern_classes,
            num_horizons=4,
        )
        model.eval()
        model_import_pass = True
    except Exception as e:
        print(f"Failed to import/instantiate StormFusionMultiTaskModel: {e}")
        model_import_pass = False

    # Parameter count breakdown
    trunk_params = count_parameters(model.shared_trunk)
    det_params = count_parameters(model.detection_head)
    pat_params = count_parameters(model.pattern_head)
    int_params = count_parameters(model.intensity_head)
    press_params = count_parameters(model.pressure_head)
    trk_params = count_parameters(model.track_head)
    total_params = count_parameters(model)

    # 2. Test Forward Pass Shapes (B=1 and B=2)
    forward_pass_pass = True
    for B in [1, 2]:
        torch.manual_seed(42 + B)
        fused_emb = torch.randn(B, 64)
        out = model(fused_emb)

        if out["detection_logits"].shape != (B, 1):
            forward_pass_pass = False
        if out["pattern_logits"].shape != (B, pattern_classes):
            forward_pass_pass = False
        if out["intensity"].shape != (B, 4):
            forward_pass_pass = False
        if out["pressure"].shape != (B, 4):
            forward_pass_pass = False
        if out["track_delta"].shape != (B, 4, 2):
            forward_pass_pass = False

    # 3. Test Loss Functions
    torch.manual_seed(101)
    B = 4
    preds = {
        "detection_logits": torch.randn(B, 1),
        "pattern_logits": torch.randn(B, pattern_classes),
        "intensity": torch.randn(B, 4),
        "pressure": torch.randn(B, 4),
        "track_delta": torch.randn(B, 4, 2),
    }

    # Case A: All labels valid
    targets_all = {
        "detection_target": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "pattern_target": torch.tensor([0, 1, 2, 3]),
        "intensity_target": torch.randn(B, 4),
        "pressure_target": torch.randn(B, 4),
        "track_delta_target": torch.randn(B, 4, 2),
    }
    masks_all = {
        "detection_valid_mask": torch.ones(B, 1),
        "pattern_valid_mask": torch.ones(B),
        "intensity_valid_mask": torch.ones(B, 4),
        "pressure_valid_mask": torch.ones(B, 4),
        "track_valid_mask": torch.ones(B, 4),
    }

    tot_loss_a, dict_a = total_multi_task_loss(
        preds, targets_all, masks_all, weights=loss_weights, huber_delta=huber_delta
    )
    all_valid_pass = (
        not torch.isnan(tot_loss_a)
        and not torch.isinf(tot_loss_a)
        and tot_loss_a > 0
    )

    # Case B: All pressure labels missing
    masks_no_press = dict(masks_all)
    masks_no_press["pressure_valid_mask"] = torch.zeros(B, 4)
    tot_loss_b, dict_b = total_multi_task_loss(
        preds, targets_all, masks_no_press, weights=loss_weights, huber_delta=huber_delta
    )
    all_press_missing_pass = (
        dict_b["pressure_loss"] == 0.0
        and not torch.isnan(tot_loss_b)
        and not torch.isinf(tot_loss_b)
    )

    # Case C: All track labels missing
    masks_no_trk = dict(masks_all)
    masks_no_trk["track_valid_mask"] = torch.zeros(B, 4)
    tot_loss_c, dict_c = total_multi_task_loss(
        preds, targets_all, masks_no_trk, weights=loss_weights, huber_delta=huber_delta
    )
    all_trk_missing_pass = (
        dict_c["track_loss"] == 0.0
        and not torch.isnan(tot_loss_c)
        and not torch.isinf(tot_loss_c)
    )

    # Case D: Mixed missing labels
    masks_mixed = {
        "detection_valid_mask": torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
        "pattern_valid_mask": torch.tensor([1, 0, 1, 0]),
        "intensity_valid_mask": torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]),
        "pressure_valid_mask": torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0]]),
        "track_valid_mask": torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 0]]),
    }
    tot_loss_d, dict_d = total_multi_task_loss(
        preds, targets_all, masks_mixed, weights=loss_weights, huber_delta=huber_delta
    )
    missing_label_pass = (
        not torch.isnan(tot_loss_d)
        and not torch.isinf(tot_loss_d)
        and dict_d["pressure_loss"] >= 0
    )

    loss_functions_pass = (
        all_valid_pass and all_press_missing_pass and all_trk_missing_pass and missing_label_pass
    )

    # 4. Gradient Pass Test
    model.train()
    torch.manual_seed(202)
    fused_in_grad = torch.randn(B, 64, requires_grad=True)
    out_grad = model(fused_in_grad)

    loss_grad, _ = total_multi_task_loss(
        out_grad, targets_all, masks_all, weights=loss_weights, huber_delta=huber_delta
    )
    loss_grad.backward()

    gradient_pass = True
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                gradient_pass = False
            elif torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                gradient_pass = False

    nan_inf_pass = (
        not torch.isnan(loss_grad).any()
        and not torch.isinf(loss_grad).any()
        and gradient_pass
    )

    # 5. Determinism Test
    model.eval()
    torch.manual_seed(303)
    fused_det = torch.randn(2, 64)
    with torch.no_grad():
        out1 = model(fused_det)
        out2 = model(fused_det)

    determinism_pass = True
    for k in out1:
        if not torch.allclose(out1[k], out2[k], atol=1e-7):
            determinism_pass = False

    # 6. Run Regressions
    print("\nRunning Step 7, Step 8A, Step 8B, and Step 8C regression tests...")
    step7_pass = run_regression_script("scripts/test_dataloader.py")
    step8a_pass = run_regression_script("scripts/test_baseline_model.py")
    step8b_pass = run_regression_script("scripts/test_step8b.py")
    step8c_pass = run_regression_script("scripts/test_step8c.py")

    # Save JSON report
    report_dict = {
        "model_import": "PASS" if model_import_pass else "FAIL",
        "forward_pass": "PASS" if forward_pass_pass else "FAIL",
        "output_shapes": {
            "detection_logits": "[B, 1]",
            "pattern_logits": "[B, 4]",
            "intensity": "[B, 4]",
            "pressure": "[B, 4]",
            "track_delta": "[B, 4, 2]",
        },
        "shared_trunk_parameters": trunk_params,
        "detection_parameters": det_params,
        "pattern_parameters": pat_params,
        "intensity_parameters": int_params,
        "pressure_parameters": press_params,
        "track_parameters": trk_params,
        "total_parameters": total_params,
        "all_valid_loss_test": "PASS" if all_valid_pass else "FAIL",
        "missing_label_loss_test": "PASS" if missing_label_pass else "FAIL",
        "all_pressure_missing_test": "PASS" if all_press_missing_pass else "FAIL",
        "all_track_missing_test": "PASS" if all_trk_missing_pass else "FAIL",
        "gradient_test": "PASS" if gradient_pass else "FAIL",
        "nan_inf_test": "PASS" if nan_inf_pass else "FAIL",
        "determinism_test": "PASS" if determinism_pass else "FAIL",
        "step7_regression": "PASS" if step7_pass else "FAIL",
        "step8a_regression": "PASS" if step8a_pass else "FAIL",
        "step8b_regression": "PASS" if step8b_pass else "FAIL",
        "step8c_regression": "PASS" if step8c_pass else "FAIL",
        "real_insat_available": False,
        "real_era5_available": False,
        "training_performed": False,
        "synthetic_data_used": False,
        "training_readiness": "NOT_READY",
    }

    report_path = PROJECT_ROOT / "data" / "processed" / "sequences" / "step9_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    # Print Final Terminal Report Format
    print(f"\nModel import: {'PASS' if model_import_pass else 'FAIL'}")
    print(f"Forward pass: {'PASS' if forward_pass_pass else 'FAIL'}\n")
    print("Input:\n[B,64]\n")
    print("Outputs:\n")
    print("Detection: [B,1]")
    print("Pattern: [B,4]")
    print("Intensity: [B,4]")
    print("Pressure: [B,4]")
    print("Track: [B,4,2]\n")
    print(f"Loss functions: {'PASS' if loss_functions_pass else 'FAIL'}")
    print(f"Masked regression: {'PASS' if all_valid_pass else 'FAIL'}")
    print(f"Missing labels: {'PASS' if missing_label_pass else 'FAIL'}")
    print(f"All-pressure-missing: {'PASS' if all_press_missing_pass else 'FAIL'}")
    print(f"All-track-missing: {'PASS' if all_trk_missing_pass else 'FAIL'}\n")
    print(f"Gradient test: {'PASS' if gradient_pass else 'FAIL'}")
    print(f"NaN/Inf: {'PASS' if nan_inf_pass else 'FAIL'}")
    print(f"Determinism: {'PASS' if determinism_pass else 'FAIL'}\n")
    print(f"Shared trunk parameters: {trunk_params:,}")
    print(f"Detection parameters: {det_params:,}")
    print(f"Pattern parameters: {pat_params:,}")
    print(f"Intensity parameters: {int_params:,}")
    print(f"Pressure parameters: {press_params:,}")
    print(f"Track parameters: {trk_params:,}")
    print(f"Total parameters: {total_params:,}\n")
    print(f"Step 7 regression: {'PASS' if step7_pass else 'FAIL'}")
    print(f"Step 8A regression: {'PASS' if step8a_pass else 'FAIL'}")
    print(f"Step 8B regression: {'PASS' if step8b_pass else 'FAIL'}")
    print(f"Step 8C regression: {'PASS' if step8c_pass else 'FAIL'}\n")
    print("Real INSAT available: NO")
    print("Real ERA5 available: NO")
    print("Training performed: NO")
    print("Synthetic physical data generated: NO\n")
    print("Training readiness: NOT READY\n")
    print("STOP AFTER STEP 9.")


if __name__ == "__main__":
    test_step9()
