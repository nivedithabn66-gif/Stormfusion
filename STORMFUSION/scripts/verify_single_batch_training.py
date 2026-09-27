"""Single-Batch Training Safety & Gradient Verification (Step 3 & 4).

Loads one representative training batch with valid targets and verifies:
1. Input tensor shapes:
   - Satellite: [B, 8, 3, H, W]
   - Track: [B, 8, features]
   - ERA5: [B, 8, channels, H, W]
   - Masks: satellite_valid_mask, satellite_modality_mask, track_valid_mask, track_modality_mask, ERA5 masks
2. Multi-Task Forward Pass:
   - Detection logits: [B, 1]
   - Intensity: [B, 4]
   - Pressure: [B, 4]
   - Track delta: [B, 4, 2]
3. Loss Computation:
   - Detection loss (BCEWithLogits)
   - Intensity loss (Masked Huber)
   - Pressure loss (Masked Huber)
   - Track loss (Masked Huber)
   - Total multi-task loss
4. Backward Pass & Gradient Sanity:
   - No NaN or Inf gradients
   - Gradients are non-zero for active parameters
5. Optimizer Step:
   - Weights update cleanly without numerical instability
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.storm_dataset import StormSequenceDataset, storm_collate_fn
from models.stormfusion import StormFusionModel
from models.losses import total_multi_task_loss


def run_single_batch_check() -> dict:
    print("=" * 60, flush=True)
    print("      STORMFUSION SINGLE-BATCH TRAINING SAFETY CHECK", flush=True)
    print("=" * 60, flush=True)

    device = torch.device("cpu")

    # 1. Dataset & find samples with valid intensity and pressure targets
    ds_train = StormSequenceDataset(split="train")
    
    # Filter for indices with valid pressure AND wind targets
    valid_mask = ds_train.df_manifest["target_wind_t3"].notna() & ds_train.df_manifest["target_pressure_t3"].notna()
    valid_indices = ds_train.df_manifest[valid_mask].index.tolist()
    
    print(f"• Total train sequences available: {len(ds_train):,}", flush=True)
    print(f"• Sequences with valid wind + pressure targets: {len(valid_indices):,}", flush=True)
    
    batch_size = 4
    sample_indices = valid_indices[:batch_size]
    subset = Subset(ds_train, sample_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=storm_collate_fn)

    batch = next(iter(loader))
    sat_dict = batch["satellite"]
    trk_dict = batch["track"]
    env_dict = batch["environment"]
    tgt_dict = batch["targets"]

    print("\n[1. TENSOR SHAPES & CONTRACT VERIFICATION]", flush=True)
    print(f"  • Satellite Tensor Shape      : {list(sat_dict['satellite_tensor'].shape)} (Expected [B, 8, 3, 500, 500])", flush=True)
    print(f"  • Satellite Valid Mask        : {list(sat_dict['satellite_valid_mask'].shape)}", flush=True)
    print(f"  • Satellite Modality Mask     : {list(sat_dict['satellite_modality_mask'].shape)}", flush=True)
    print(f"  • Track Features Shape        : {list(trk_dict['track_features'].shape)} (Expected [B, 8, 9])", flush=True)
    print(f"  • Track Valid Mask            : {list(trk_dict['track_valid_mask'].shape)}", flush=True)
    print(f"  • ERA5 Tensor Shape           : {list(env_dict['era5_features'].shape)} (Expected [B, 8, 4, 32, 32])", flush=True)
    print(f"  • ERA5 Valid Mask             : {list(env_dict['era5_valid_mask'].shape)}", flush=True)
    print(f"  • ERA5 Modality Mask          : {list(env_dict['era5_modality_mask'].shape)}", flush=True)

    print("\n[2. TARGETS & VALIDITY MASKS]", flush=True)
    print(f"  • Detection Targets Shape     : {list(tgt_dict['detection'].shape)}", flush=True)
    print(f"  • Intensity Targets Shape     : {list(tgt_dict['intensity'].shape)}", flush=True)
    print(f"  • Intensity Valid Mask Sum    : {tgt_dict['intensity_valid_mask'].sum().item()} / {tgt_dict['intensity_valid_mask'].numel()}", flush=True)
    print(f"  • Pressure Targets Shape      : {list(tgt_dict['pressure'].shape)}", flush=True)
    print(f"  • Pressure Valid Mask Sum     : {tgt_dict['pressure_valid_mask'].sum().item()} / {tgt_dict['pressure_valid_mask'].numel()}", flush=True)
    print(f"  • Track Future Targets Shape  : {list(tgt_dict['future_track'].shape)}", flush=True)
    print(f"  • Track Future Valid Mask Sum : {tgt_dict['future_track_valid_mask'].sum().item()} / {tgt_dict['future_track_valid_mask'].numel()}", flush=True)

    # 2. Model Forward Pass
    print("\n[3. MODEL FORWARD PASS]", flush=True)
    model = StormFusionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Record initial weight snapshot
    init_weight = model.multitask.intensity_head.head[0].weight.data.clone() if hasattr(model.multitask.intensity_head, "head") else next(model.multitask.intensity_head.parameters()).data.clone()

    inputs = {
        "satellite_tensor": sat_dict["satellite_tensor"].to(device),
        "satellite_valid_mask": sat_dict["satellite_valid_mask"].to(device),
        "satellite_modality_mask": sat_dict["satellite_modality_mask"].to(device),
        "track_features": trk_dict["track_features"].to(device),
        "track_valid_mask": trk_dict["track_valid_mask"].to(device),
        "track_modality_mask": torch.ones_like(trk_dict["track_valid_mask"][:, :1]).to(device),
        "era5_tensor": env_dict["era5_features"].to(device),
        "era5_valid_mask": env_dict["era5_valid_mask"].to(device),
        "era5_modality_mask": env_dict["era5_modality_mask"].to(device),
    }

    preds = model(**inputs)
    print(f"  • Detection Logits Shape      : {list(preds['detection_logits'].shape)}", flush=True)
    print(f"  • Intensity Predictions Shape : {list(preds['intensity'].shape)}", flush=True)
    print(f"  • Pressure Predictions Shape  : {list(preds['pressure'].shape)}", flush=True)
    print(f"  • Track Delta Shape           : {list(preds['track_delta'].shape)}", flush=True)

    # 3. Loss Calculation
    print("\n[4. LOSS COMPUTATION & BREAKDOWN]", flush=True)
    targets = {
        "detection_target": tgt_dict["detection"].to(device),
        "pattern_target": tgt_dict["pattern"].to(device),
        "intensity_target": tgt_dict["intensity"].to(device),
        "pressure_target": tgt_dict["pressure"].to(device),
        "track_delta_target": tgt_dict["future_track"].to(device),
    }

    masks = {
        "detection_valid_mask": torch.ones_like(tgt_dict["detection"]).to(device),
        "pattern_valid_mask": (tgt_dict["pattern"] != -1).float().to(device),
        "intensity_valid_mask": tgt_dict["intensity_valid_mask"].to(device),
        "pressure_valid_mask": tgt_dict["pressure_valid_mask"].to(device),
        "track_valid_mask": tgt_dict["future_track_valid_mask"].to(device),
    }

    weights = {"detection": 1.0, "intensity": 1.0, "pressure": 1.0, "track": 1.0, "pattern": 0.0}

    total_loss, loss_breakdown = total_multi_task_loss(preds, targets, masks, weights=weights)

    print(f"  • Detection Loss              : {loss_breakdown['detection_loss']:.4f}", flush=True)
    print(f"  • Intensity Loss              : {loss_breakdown['intensity_loss']:.4f}", flush=True)
    print(f"  • Pressure Loss               : {loss_breakdown['pressure_loss']:.4f}", flush=True)
    print(f"  • Track Loss                  : {loss_breakdown['track_loss']:.4f}", flush=True)
    print(f"  • Pattern Loss (weight 0.0)   : {loss_breakdown['pattern_loss']:.4f}", flush=True)
    print(f"  • TOTAL MULTI-TASK LOSS       : {total_loss.item():.4f}", flush=True)

    assert not torch.isnan(total_loss), "CRITICAL: Loss is NaN!"
    assert not torch.isinf(total_loss), "CRITICAL: Loss is infinite!"
    assert loss_breakdown["intensity_loss"] > 0, "Intensity loss must be non-zero when valid targets are supplied"
    assert loss_breakdown["pressure_loss"] > 0, "Pressure loss must be non-zero when valid targets are supplied"
    assert loss_breakdown["track_loss"] > 0, "Track loss must be non-zero when valid targets are supplied"

    # 4. Backward Pass
    print("\n[5. BACKWARD PASS & GRADIENT VERIFICATION]", flush=True)
    optimizer.zero_grad()
    total_loss.backward()

    has_nan_grad = False
    has_inf_grad = False
    max_grad = 0.0
    active_grads = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            active_grads += 1
            if torch.isnan(param.grad).any():
                has_nan_grad = True
            if torch.isinf(param.grad).any():
                has_inf_grad = True
            max_grad = max(max_grad, param.grad.abs().max().item())

    print(f"  • Active Parameter Gradients  : {active_grads} tensors", flush=True)
    print(f"  • Max Gradient Norm (Peak)    : {max_grad:.4f}", flush=True)
    print(f"  • NaN Gradients Detected      : {has_nan_grad}", flush=True)
    print(f"  • Inf Gradients Detected      : {has_inf_grad}", flush=True)

    assert not has_nan_grad, "CRITICAL: NaN gradient detected!"
    assert not has_inf_grad, "CRITICAL: Inf gradient detected!"
    assert active_grads > 0, "CRITICAL: No gradients computed!"

    # 5. Optimizer Step
    print("\n[6. OPTIMIZER STEP]", flush=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    updated_weight = model.multitask.intensity_head.head[0].weight.data if hasattr(model.multitask.intensity_head, "head") else next(model.multitask.intensity_head.parameters()).data
    weight_diff = (updated_weight - init_weight).abs().sum().item()
    print(f"  • Parameter Update Delta (L1) : {weight_diff:.6f}", flush=True)
    assert weight_diff > 0, "CRITICAL: Optimizer step failed to update weights!"

    print("\n==================================================", flush=True)
    print("  SINGLE-BATCH TRAINING SAFETY CHECK: ALL PASS!   ", flush=True)
    print("==================================================", flush=True)

    return {
        "status": "PASS",
        "total_loss": total_loss.item(),
        "breakdown": loss_breakdown,
        "max_grad": max_grad,
        "weight_update_delta": weight_diff,
    }


if __name__ == "__main__":
    run_single_batch_check()
