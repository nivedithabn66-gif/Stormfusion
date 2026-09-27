"""Step 8B Unit Test & Verification Runner for STORMFUSION.

Executes software unit tests for TrackGRUEncoder and ERA5Encoder:
  1. Tests Track GRU import and forward pass ([B, 8, 8] -> [B, 64]).
  2. Tests ERA5 CNN import and forward pass ([B, 8, 4, 32, 32] -> [B, 64]).
  3. Tests batch sizes B=1 and B=2.
  4. Asserts zero NaN or Inf values in output embeddings.
  5. Verifies CUDA/CPU device compatibility.
  6. Computes and reports parameter counts (Track GRU, ERA5 CNN, Total Step 8B).
  7. Asserts 100% deterministic inference.
  8. Verifies mask interfaces (track_valid_mask, era5_valid_mask, era5_modality_mask).
  9. Asserts mask influence (invalid timesteps do not corrupt output representations).
 10. Confirms real_insat_available=false, real_era5_available=false, training_performed=false, synthetic_data_used=false.
 11. Exports data/processed/sequences/step8b_report.json and prints formal terminal report.

IMPORTANT: Random in-memory tensors are used ONLY within this test script to verify tensor shapes
and execution correctness. They are NOT saved to disk, NOT entered into the dataset, and NOT used for training.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import yaml

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch

from models.track_encoder import TrackGRUEncoder
from models.era5_encoder import ERA5Encoder


def count_params(model: torch.nn.Module) -> int:
    """Computes total parameter count of a module."""
    return sum(p.numel() for p in model.parameters())


def run_step8b_unit_tests():
    """Executes Step 8B unit tests."""
    torch.manual_seed(42)
    np.random.seed(42)

    # 0. Device Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = project_root / "configs/model_config.yaml"
    model_cfg = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)

    tr_cfg = model_cfg.get("track_encoder", {})
    e5_cfg = model_cfg.get("era5_encoder", {})

    print("\n==================================================")
    print("STARTING STEP 8B — TRACK + ERA5 ENCODER UNIT TEST")
    print(f"Device: {device} (CUDA available: {torch.cuda.is_available()})")
    print("==================================================")

    # 1. Imports Check
    track_import_pass = True
    era5_import_pass = True
    print("\n[1. MODULE IMPORTS CHECK]")
    print(f"  • TrackGRUEncoder class imported : {track_import_pass}")
    print(f"  • ERA5Encoder class imported     : {era5_import_pass}")

    # 2. Instantiate Models
    track_model = TrackGRUEncoder(
        input_dim=tr_cfg.get("input_dim", 8),
        hidden_dim=tr_cfg.get("hidden_dim", 64),
        num_layers=tr_cfg.get("num_layers", 1),
    ).to(device)
    track_model.eval()

    era5_model = ERA5Encoder(
        input_channels=e5_cfg.get("input_channels", 4),
        hidden_channels=e5_cfg.get("hidden_channels", 32),
        embedding_dim=e5_cfg.get("embedding_dim", 64),
    ).to(device)
    era5_model.eval()

    track_params = count_params(track_model)
    era5_params = count_params(era5_model)
    total_step8b_params = track_params + era5_params

    print(f"\n[2. PARAMETER COUNT REPORT]")
    print(f"  • Track GRU Parameters : {track_params:,}")
    print(f"  • ERA5 CNN Parameters  : {era5_params:,}")
    print(f"  • Total Step 8B Params : {total_step8b_params:,}")

    # 3. Batch Size 1 Tests
    b1_track_pass = False
    b1_era5_pass = False
    b1_tr_shape = None
    b1_e5_shape = None

    with torch.no_grad():
        # Track input: [1, 8, 8]
        tr_x1 = torch.randn(1, 8, 8, device=device, dtype=torch.float32)
        tr_vmask1 = torch.ones(1, 8, device=device, dtype=torch.float32)
        tr_out1 = track_model(tr_x1, track_valid_mask=tr_vmask1)
        tr_emb1 = tr_out1["track_embedding"]
        b1_tr_shape = list(tr_emb1.shape)
        if b1_tr_shape == [1, 64]:
            b1_track_pass = True

        # ERA5 input: [1, 8, 4, 32, 32]
        e5_x1 = torch.randn(1, 8, 4, 32, 32, device=device, dtype=torch.float32)
        e5_vmask1 = torch.ones(1, 8, device=device, dtype=torch.float32)
        e5_mmask1 = torch.ones(1, 1, device=device, dtype=torch.float32)
        e5_out1 = era5_model(e5_x1, era5_valid_mask=e5_vmask1, era5_modality_mask=e5_mmask1)
        e5_emb1 = e5_out1["era5_embedding"]
        b1_e5_shape = list(e5_emb1.shape)
        if b1_e5_shape == [1, 64]:
            b1_era5_pass = True

    print(f"\n[3. BATCH SIZE 1 TEST]")
    print(f"  • Track Input [1, 8, 8]           -> Output {b1_tr_shape} | Status: {'PASS' if b1_track_pass else 'FAIL'}")
    print(f"  • ERA5 Input  [1, 8, 4, 32, 32]   -> Output {b1_e5_shape} | Status: {'PASS' if b1_era5_pass else 'FAIL'}")

    # 4. Batch Size 2 Tests
    b2_track_pass = False
    b2_era5_pass = False
    b2_tr_shape = None
    b2_e5_shape = None

    with torch.no_grad():
        tr_x2 = torch.randn(2, 8, 8, device=device, dtype=torch.float32)
        tr_vmask2 = torch.ones(2, 8, device=device, dtype=torch.float32)
        tr_out2 = track_model(tr_x2, track_valid_mask=tr_vmask2)
        tr_emb2 = tr_out2["track_embedding"]
        b2_tr_shape = list(tr_emb2.shape)
        if b2_tr_shape == [2, 64]:
            b2_track_pass = True

        e5_x2 = torch.randn(2, 8, 4, 32, 32, device=device, dtype=torch.float32)
        e5_vmask2 = torch.ones(2, 8, device=device, dtype=torch.float32)
        e5_mmask2 = torch.ones(2, 1, device=device, dtype=torch.float32)
        e5_out2 = era5_model(e5_x2, era5_valid_mask=e5_vmask2, era5_modality_mask=e5_mmask2)
        e5_emb2 = e5_out2["era5_embedding"]
        b2_e5_shape = list(e5_emb2.shape)
        if b2_e5_shape == [2, 64]:
            b2_era5_pass = True

    print(f"\n[4. BATCH SIZE 2 TEST]")
    print(f"  • Track Input [2, 8, 8]           -> Output {b2_tr_shape} | Status: {'PASS' if b2_track_pass else 'FAIL'}")
    print(f"  • ERA5 Input  [2, 8, 4, 32, 32]   -> Output {b2_e5_shape} | Status: {'PASS' if b2_era5_pass else 'FAIL'}")

    # 5. NaN / Inf Check
    nan_inf_track = torch.isnan(tr_emb1).any().item() or torch.isinf(tr_emb1).any().item() or torch.isnan(tr_emb2).any().item() or torch.isinf(tr_emb2).any().item()
    nan_inf_era5 = torch.isnan(e5_emb1).any().item() or torch.isinf(e5_emb1).any().item() or torch.isnan(e5_emb2).any().item() or torch.isinf(e5_emb2).any().item()
    nan_inf_pass = not (nan_inf_track or nan_inf_era5)

    print(f"\n[5. NAN / INF CHECK]")
    print(f"  • Track NaN/Inf present : {nan_inf_track}")
    print(f"  • ERA5 NaN/Inf present  : {nan_inf_era5}")
    print(f"  • Status                : {'PASS' if nan_inf_pass else 'FAIL'}")

    # 6. Determinism Check
    with torch.no_grad():
        tr_out1_repeat = track_model(tr_x1, track_valid_mask=tr_vmask1)
        e5_out1_repeat = era5_model(e5_x1, era5_valid_mask=e5_vmask1, era5_modality_mask=e5_mmask1)
        det_tr = torch.equal(tr_emb1, tr_out1_repeat["track_embedding"])
        det_e5 = torch.equal(e5_emb1, e5_out1_repeat["era5_embedding"])
        determinism_pass = det_tr and det_e5

    print(f"\n[6. DETERMINISM CHECK]")
    print(f"  • Track determinism : {det_tr}")
    print(f"  • ERA5 determinism  : {det_e5}")
    print(f"  • Status            : {'PASS' if determinism_pass else 'FAIL'}")

    # 7. Mask Interface & Influence Verification
    # Test Track Mask Influence: changing physical values at an invalid timestep (mask=0) should NOT alter output embedding
    with torch.no_grad():
        tr_x_base = torch.randn(1, 8, 8, device=device, dtype=torch.float32)
        tr_mask_partial = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]], device=device, dtype=torch.float32) # last step invalid

        out_orig = track_model(tr_x_base, track_valid_mask=tr_mask_partial)["track_embedding"]

        # Modify physical values at timestep 7 (which is masked out with 0.0)
        tr_x_corrupted = tr_x_base.clone()
        tr_x_corrupted[0, 7, :] += 999.0

        out_corrupted = track_model(tr_x_corrupted, track_valid_mask=tr_mask_partial)["track_embedding"]
        track_mask_influence_pass = torch.allclose(out_orig, out_corrupted, atol=1e-5)

        # Test ERA5 Modality Mask: when modality_mask = 0, output embedding must be zero tensor
        e5_mmask_zero = torch.tensor([[0.0]], device=device, dtype=torch.float32)
        out_e5_missing = era5_model(e5_x1, era5_valid_mask=e5_vmask1, era5_modality_mask=e5_mmask_zero)["era5_embedding"]
        era5_mask_influence_pass = (out_e5_missing.abs().max().item() == 0.0)

    track_mask_pass = (tr_out1["track_valid_mask"] is not None) and track_mask_influence_pass
    era5_mask_pass = (e5_out1["era5_valid_mask"] is not None) and (e5_out1["era5_modality_mask"] is not None) and era5_mask_influence_pass

    print(f"\n[7. MASK INTERFACE & INFLUENCE CHECK]")
    print(f"  • Track invalid timestep isolation : {track_mask_influence_pass}")
    print(f"  • ERA5 missing modality zero-mask   : {era5_mask_influence_pass}")
    print(f"  • Track mask status                : {'PASS' if track_mask_pass else 'FAIL'}")
    print(f"  • ERA5 mask status                 : {'PASS' if era5_mask_pass else 'FAIL'}")

    track_gru_pass = track_import_pass and b1_track_pass and b2_track_pass and track_mask_pass
    era5_cnn_pass = era5_import_pass and b1_era5_pass and b2_era5_pass and era5_mask_pass
    track_fw_pass = b1_track_pass and b2_track_pass
    era5_fw_pass = b1_era5_pass and b2_era5_pass

    # Save JSON Report
    report_json = {
        "step": "STEP_8B",
        "track_encoder_import": "PASS" if track_import_pass else "FAIL",
        "track_gru": "PASS" if track_gru_pass else "FAIL",
        "track_forward_pass": "PASS" if track_fw_pass else "FAIL",
        "track_input_shape": [None, 8, 8],
        "track_output_shape": [None, 64],
        "track_parameters": track_params,
        "era5_encoder_import": "PASS" if era5_import_pass else "FAIL",
        "era5_cnn": "PASS" if era5_cnn_pass else "FAIL",
        "era5_forward_pass": "PASS" if era5_fw_pass else "FAIL",
        "era5_input_shape": [None, 8, 4, 32, 32],
        "era5_output_shape": [None, 64],
        "era5_parameters": era5_params,
        "total_step8b_parameters": total_step8b_params,
        "mask_interface": "PASS" if (track_mask_pass and era5_mask_pass) else "FAIL",
        "nan_inf_check": "PASS" if nan_inf_pass else "FAIL",
        "determinism_check": "PASS" if determinism_pass else "FAIL",
        "batch_size_tests": {
            "batch_size_1": "PASS" if (b1_track_pass and b1_era5_pass) else "FAIL",
            "batch_size_2": "PASS" if (b2_track_pass and b2_era5_pass) else "FAIL",
        },
        "device_used": str(device),
        "real_insat_available": False,
        "real_era5_available": False,
        "training_performed": False,
        "synthetic_data_used": False,
        "training_readiness": "NOT_READY",
    }

    report_output_path = project_root / "data/processed/sequences/step8b_report.json"
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    print(f"\n[EXPORT] Saved Step 8B report to: {report_output_path}")

    # Print Final Step 8B Terminal Report
    print("\n==================================================")
    print("STEP 8B — TRACK + ERA5 ENCODER REPORT")
    print("==================================================")
    print()
    print(f"Track encoder import: {'PASS' if track_import_pass else 'FAIL'}")
    print(f"Track GRU: {'PASS' if track_gru_pass else 'FAIL'}")
    print(f"Track forward pass: {'PASS' if track_fw_pass else 'FAIL'}")
    print()
    print("Track input:")
    print("[B, 8, 8]")
    print()
    print("Track output:")
    print("[B, 64]")
    print()
    print(f"ERA5 encoder import: {'PASS' if era5_import_pass else 'FAIL'}")
    print(f"ERA5 CNN: {'PASS' if era5_cnn_pass else 'FAIL'}")
    print(f"ERA5 forward pass: {'PASS' if era5_fw_pass else 'FAIL'}")
    print()
    print("ERA5 input:")
    print("[B, 8, 4, 32, 32]")
    print()
    print("ERA5 output:")
    print("[B, 64]")
    print()
    print(f"Track mask: {'PASS' if track_mask_pass else 'FAIL'}")
    print(f"ERA5 mask: {'PASS' if era5_mask_pass else 'FAIL'}")
    print()
    print(f"Batch size 1: {'PASS' if (b1_track_pass and b1_era5_pass) else 'FAIL'}")
    print(f"Batch size 2: {'PASS' if (b2_track_pass and b2_era5_pass) else 'FAIL'}")
    print()
    print(f"NaN/Inf: {'PASS' if nan_inf_pass else 'FAIL'}")
    print(f"Determinism: {'PASS' if determinism_pass else 'FAIL'}")
    print()
    print(f"Track parameters: {track_params:,}")
    print(f"ERA5 parameters: {era5_params:,}")
    print(f"Total parameters: {total_step8b_params:,}")
    print()
    print("Training performed: NO")
    print("Real INSAT available: NO")
    print("Real ERA5 available: NO")
    print("Synthetic physical data generated: NO")
    print()
    print("Training readiness: NOT READY")
    print("==================================================\n")

    return report_json


if __name__ == "__main__":
    run_step8b_unit_tests()
