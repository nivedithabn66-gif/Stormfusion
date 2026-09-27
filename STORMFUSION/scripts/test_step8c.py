"""STORMFUSION Step 8C — Availability-Aware Gated Fusion Unit & Regression Tests.

Tests the AvailabilityAwareGatedFusion module across:
1. Shape and forward pass correctness
2. Cases A-H missing-modality combinations
3. Mask invariance (unavailable modality feature changes do not alter output)
4. Gate availability and range [0, 1]
5. Finite values (no NaN/Inf) across batch sizes and masks
6. Determinism in eval mode
7. Parameter counts
8. Regression tests for Step 7, Step 8A, and Step 8B
9. Exports data/processed/sequences/step8c_report.json
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

from models.gated_fusion import AvailabilityAwareGatedFusion


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


def test_step8c():
    print("=" * 50)
    print("STEP 8C — AVAILABILITY-AWARE GATED FUSION REPORT")
    print("=" * 50)

    # Load configuration
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    fusion_cfg = config.get("fusion", {})
    fusion_dim = fusion_cfg.get("embedding_dim", 64)
    gate_hidden_dim = fusion_cfg.get("gate_hidden_dim", 32)
    epsilon = float(fusion_cfg.get("epsilon", 1.0e-8))

    # 1. Instantiate module
    try:
        fusion_model = AvailabilityAwareGatedFusion(
            sat_dim=128,
            track_dim=64,
            era5_dim=64,
            fusion_dim=fusion_dim,
            gate_hidden_dim=gate_hidden_dim,
            epsilon=epsilon,
        )
        fusion_model.eval()
        fusion_import_pass = True
    except Exception as e:
        print(f"Failed to import/instantiate AvailabilityAwareGatedFusion: {e}")
        fusion_import_pass = False

    # Parameter counts
    sat_proj_params = count_parameters(fusion_model.sat_proj)
    track_proj_params = count_parameters(fusion_model.track_proj)
    era5_proj_params = count_parameters(fusion_model.era5_proj)

    sat_gate_params = count_parameters(fusion_model.sat_gate_net)
    track_gate_params = count_parameters(fusion_model.track_gate_net)
    era5_gate_params = count_parameters(fusion_model.era5_gate_net)

    total_fusion_params = count_parameters(fusion_model)

    # Prepare standard random inputs (B=4)
    torch.manual_seed(42)
    B = 4
    sat_emb = torch.randn(B, 128)
    track_emb = torch.randn(B, 64)
    era5_emb = torch.randn(B, 64)

    # 2. Test Cases A - H
    results = {}

    # Case A: All available
    m_sat = torch.ones(B, 1)
    m_track = torch.ones(B, 1)
    m_era5 = torch.ones(B, 1)
    out_a = fusion_model(sat_emb, track_emb, era5_emb, m_sat, m_track, m_era5)
    case_a_pass = (
        out_a["fused_embedding"].shape == (B, 64)
        and (out_a["satellite_gate"] >= 0).all() and (out_a["satellite_gate"] <= 1).all()
        and (out_a["track_gate"] >= 0).all() and (out_a["track_gate"] <= 1).all()
        and (out_a["era5_gate"] >= 0).all() and (out_a["era5_gate"] <= 1).all()
        and (out_a["fusion_valid_mask"] == 1).all()
    )
    results["all_modalities_test"] = "PASS" if case_a_pass else "FAIL"

    # Case B: ERA5 missing
    m_sat = torch.ones(B, 1)
    m_track = torch.ones(B, 1)
    m_era5 = torch.zeros(B, 1)
    out_b = fusion_model(sat_emb, track_emb, era5_emb, m_sat, m_track, m_era5)
    case_b_pass = (
        (out_b["era5_gate"] == 0).all()
        and (out_b["satellite_gate"] >= 0).all()
        and (out_b["track_gate"] >= 0).all()
        and (out_b["fusion_valid_mask"] == 1).all()
        and not torch.isnan(out_b["fused_embedding"]).any()
        and not torch.isinf(out_b["fused_embedding"]).any()
    )
    results["missing_era5_test"] = "PASS" if case_b_pass else "FAIL"

    # Case C: Satellite missing
    m_sat = torch.zeros(B, 1)
    m_track = torch.ones(B, 1)
    m_era5 = torch.ones(B, 1)
    out_c = fusion_model(sat_emb, track_emb, era5_emb, m_sat, m_track, m_era5)
    case_c_pass = (
        (out_c["satellite_gate"] == 0).all()
        and (out_c["fusion_valid_mask"] == 1).all()
        and not torch.isnan(out_c["fused_embedding"]).any()
    )
    results["missing_satellite_test"] = "PASS" if case_c_pass else "FAIL"

    # Case D: Track missing
    m_sat = torch.ones(B, 1)
    m_track = torch.zeros(B, 1)
    m_era5 = torch.ones(B, 1)
    out_d = fusion_model(sat_emb, track_emb, era5_emb, m_sat, m_track, m_era5)
    case_d_pass = (
        (out_d["track_gate"] == 0).all()
        and (out_d["fusion_valid_mask"] == 1).all()
        and not torch.isnan(out_d["fused_embedding"]).any()
    )
    results["missing_track_test"] = "PASS" if case_d_pass else "FAIL"

    # Case E: Only Satellite
    m_sat = torch.ones(B, 1)
    m_track = torch.zeros(B, 1)
    m_era5 = torch.zeros(B, 1)
    out_e = fusion_model(sat_emb, track_emb, era5_emb, m_sat, m_track, m_era5)
    case_e_pass = (
        (out_e["satellite_gate"] >= 0).all() and (out_e["satellite_gate"] <= 1).all()
        and (out_e["track_gate"] == 0).all()
        and (out_e["era5_gate"] == 0).all()
        and (out_e["fusion_valid_mask"] == 1).all()
    )
    results["only_satellite_test"] = "PASS" if case_e_pass else "FAIL"

    # Case F: Only Track
    m_sat = torch.zeros(B, 1)
    m_track = torch.ones(B, 1)
    m_era5 = torch.zeros(B, 1)
    out_f = fusion_model(sat_emb, track_emb, era5_emb, m_sat, m_track, m_era5)
    case_f_pass = (
        (out_f["satellite_gate"] == 0).all()
        and (out_f["track_gate"] >= 0).all() and (out_f["track_gate"] <= 1).all()
        and (out_f["era5_gate"] == 0).all()
        and (out_f["fusion_valid_mask"] == 1).all()
    )
    results["only_track_test"] = "PASS" if case_f_pass else "FAIL"

    # Case G: Only ERA5
    m_sat = torch.zeros(B, 1)
    m_track = torch.zeros(B, 1)
    m_era5 = torch.ones(B, 1)
    out_g = fusion_model(sat_emb, track_emb, era5_emb, m_sat, m_track, m_era5)
    case_g_pass = (
        (out_g["satellite_gate"] == 0).all()
        and (out_g["track_gate"] == 0).all()
        and (out_g["era5_gate"] >= 0).all() and (out_g["era5_gate"] <= 1).all()
        and (out_g["fusion_valid_mask"] == 1).all()
    )
    results["only_era5_test"] = "PASS" if case_g_pass else "FAIL"

    # Case H: No modalities
    m_sat = torch.zeros(B, 1)
    m_track = torch.zeros(B, 1)
    m_era5 = torch.zeros(B, 1)
    out_h = fusion_model(sat_emb, track_emb, era5_emb, m_sat, m_track, m_era5)
    case_h_pass = (
        (out_h["satellite_gate"] == 0).all()
        and (out_h["track_gate"] == 0).all()
        and (out_h["era5_gate"] == 0).all()
        and (out_h["fused_embedding"] == 0).all()
        and (out_h["fusion_valid_mask"] == 0).all()
        and not torch.isnan(out_h["fused_embedding"]).any()
        and not torch.isinf(out_h["fused_embedding"]).any()
    )
    results["no_modalities_test"] = "PASS" if case_h_pass else "FAIL"

    # 3. Mask Invariance Test
    # For an unavailable modality (mask = 0), changing its embedding must NOT alter fused output
    m_sat = torch.ones(B, 1)
    m_track = torch.ones(B, 1)
    m_era5 = torch.zeros(B, 1)  # ERA5 unavailable
    era5_emb_alt = torch.randn(B, 64) * 50.0 + 100.0  # Totally different embedding values

    out_inv1 = fusion_model(sat_emb, track_emb, era5_emb, m_sat, m_track, m_era5)
    out_inv2 = fusion_model(sat_emb, track_emb, era5_emb_alt, m_sat, m_track, m_era5)

    mask_inv_pass = torch.allclose(
        out_inv1["fused_embedding"], out_inv2["fused_embedding"], atol=1e-6
    ) and torch.allclose(out_inv1["era5_gate"], out_inv2["era5_gate"], atol=1e-6)
    results["mask_invariance_test"] = "PASS" if mask_inv_pass else "FAIL"

    # 4. Gate Range Test
    gate_range_pass = True
    for out_case in [out_a, out_b, out_c, out_d, out_e, out_f, out_g, out_h]:
        for g_name in ["satellite_gate", "track_gate", "era5_gate"]:
            g = out_case[g_name]
            if (g < 0).any() or (g > 1).any():
                gate_range_pass = False
    results["gate_range_test"] = "PASS" if gate_range_pass else "FAIL"

    # 5. NaN / Inf Test across B=1, B=2 and all mask combinations
    nan_inf_pass = True
    for b_size in [1, 2]:
        for sat_m in [0, 1]:
            for trk_m in [0, 1]:
                for e5_m in [0, 1]:
                    s_emb = torch.randn(b_size, 128)
                    t_emb = torch.randn(b_size, 64)
                    e_emb = torch.randn(b_size, 64)
                    sm = torch.tensor([[sat_m]] * b_size, dtype=torch.float32)
                    tm = torch.tensor([[trk_m]] * b_size, dtype=torch.float32)
                    em = torch.tensor([[e5_m]] * b_size, dtype=torch.float32)
                    out_chk = fusion_model(s_emb, t_emb, e_emb, sm, tm, em)
                    if torch.isnan(out_chk["fused_embedding"]).any() or torch.isinf(out_chk["fused_embedding"]).any():
                        nan_inf_pass = False
    results["nan_inf_test"] = "PASS" if nan_inf_pass else "FAIL"

    # 6. Determinism Test
    torch.manual_seed(999)
    s_test = torch.randn(2, 128)
    t_test = torch.randn(2, 64)
    e_test = torch.randn(2, 64)
    sm_test = torch.tensor([[1.0], [0.0]])
    tm_test = torch.tensor([[1.0], [1.0]])
    em_test = torch.tensor([[0.0], [1.0]])

    det_out1 = fusion_model(s_test, t_test, e_test, sm_test, tm_test, em_test)
    det_out2 = fusion_model(s_test, t_test, e_test, sm_test, tm_test, em_test)
    determinism_pass = torch.allclose(
        det_out1["fused_embedding"], det_out2["fused_embedding"], atol=1e-7
    )
    results["determinism_test"] = "PASS" if determinism_pass else "FAIL"

    # Overall fusion forward pass check
    fusion_forward_pass = (
        fusion_import_pass
        and all(results[k] == "PASS" for k in [
            "all_modalities_test", "missing_era5_test", "missing_satellite_test",
            "missing_track_test", "only_satellite_test", "only_track_test",
            "only_era5_test", "no_modalities_test", "mask_invariance_test",
            "gate_range_test", "nan_inf_test", "determinism_test"
        ])
    )

    # 7. Regressions
    print("\nRunning Step 7, Step 8A, and Step 8B regression tests...")
    step7_pass = run_regression_script("scripts/test_dataloader.py")
    step8a_pass = run_regression_script("scripts/test_baseline_model.py")
    step8b_pass = run_regression_script("scripts/test_step8b.py")

    results["step7_regression"] = "PASS" if step7_pass else "FAIL"
    results["step8a_regression"] = "PASS" if step8a_pass else "FAIL"
    results["step8b_regression"] = "PASS" if step8b_pass else "FAIL"

    # Save JSON report
    report_dict = {
        "fusion_input_dimensions": {
            "satellite": 128,
            "track": 64,
            "era5": 64,
        },
        "fusion_output_dimension": fusion_dim,
        "parameter_count": {
            "satellite_proj": sat_proj_params,
            "track_proj": track_proj_params,
            "era5_proj": era5_proj_params,
            "satellite_gate": sat_gate_params,
            "track_gate": track_gate_params,
            "era5_gate": era5_gate_params,
            "total_fusion_params": total_fusion_params,
        },
        "all_modalities_test": results["all_modalities_test"],
        "missing_era5_test": results["missing_era5_test"],
        "missing_satellite_test": results["missing_satellite_test"],
        "missing_track_test": results["missing_track_test"],
        "only_satellite_test": results["only_satellite_test"],
        "only_track_test": results["only_track_test"],
        "only_era5_test": results["only_era5_test"],
        "no_modalities_test": results["no_modalities_test"],
        "mask_invariance_test": results["mask_invariance_test"],
        "gate_range_test": results["gate_range_test"],
        "nan_inf_test": results["nan_inf_test"],
        "determinism_test": results["determinism_test"],
        "step7_regression": results["step7_regression"],
        "step8a_regression": results["step8a_regression"],
        "step8b_regression": results["step8b_regression"],
        "real_insat_available": False,
        "real_era5_available": False,
        "training_performed": False,
        "synthetic_data_used": False,
        "training_readiness": "NOT_READY",
    }

    report_path = PROJECT_ROOT / "data" / "processed" / "sequences" / "step8c_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    # Print Final Terminal Report Format
    print(f"\nFusion import: {'PASS' if fusion_import_pass else 'FAIL'}")
    print(f"Fusion forward pass: {'PASS' if fusion_forward_pass else 'FAIL'}\n")
    print("Satellite input: [B,128]")
    print("Track input: [B,64]")
    print("ERA5 input: [B,64]\n")
    print(f"Projected dimension: {fusion_dim}")
    print(f"Fused output: [B,{fusion_dim}]\n")
    print(f"All modalities: {results['all_modalities_test']}")
    print(f"ERA5 missing: {results['missing_era5_test']}")
    print(f"Satellite missing: {results['missing_satellite_test']}")
    print(f"Track missing: {results['missing_track_test']}")
    print(f"Only satellite: {results['only_satellite_test']}")
    print(f"Only track: {results['only_track_test']}")
    print(f"Only ERA5: {results['only_era5_test']}")
    print(f"No modalities: {results['no_modalities_test']}\n")
    print(f"Mask invariance: {results['mask_invariance_test']}")
    print(f"Gate range: {results['gate_range_test']}")
    print(f"NaN/Inf: {results['nan_inf_test']}")
    print(f"Determinism: {results['determinism_test']}\n")
    print(f"Fusion parameters: {total_fusion_params:,}\n")
    print(f"Step 7 regression: {results['step7_regression']}")
    print(f"Step 8A regression: {results['step8a_regression']}")
    print(f"Step 8B regression: {results['step8b_regression']}\n")
    print("Real INSAT available: NO")
    print("Real ERA5 available: NO")
    print("Training performed: NO")
    print("Synthetic physical data generated: NO\n")
    print("Training readiness: NOT READY\n")
    print("STOP AFTER STEP 8C.")


if __name__ == "__main__":
    test_step8c()
