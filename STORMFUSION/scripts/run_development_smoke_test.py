"""STORMFUSION Development Training End-to-End Smoke Test.

Executes a small development training experiment (2 epochs) on real sequence batches to verify:
DataLoader -> Satellite Provider -> ResNet18 -> ConvLSTM -> ERA5 Encoder -> Track GRU -> Gated Fusion -> Multi-task heads -> Loss computation -> Backpropagation -> Optimizer -> Validation -> Checkpointing.
"""

import json
import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.storm_dataset import StormSequenceDataset, storm_collate_fn
from models.stormfusion import StormFusionModel
from models.losses import total_multi_task_loss
from training.trainer import StormFusionTrainer
from training.logger import StructuredLogger


def run_development_smoke_test() -> dict:
    print("==================================================")
    print("      STORMFUSION DEVELOPMENT SMOKE TEST          ")
    print("==================================================")

    # 1. Dataset & DataLoader Initialization (Subset slice for fast development smoke test)
    train_dataset = StormSequenceDataset(split="train")
    val_dataset = StormSequenceDataset(split="val")

    # Limit to 12 train sequences (3 batches) and 4 val sequences (1 batch) for fast CPU execution
    train_subset = torch.utils.data.Subset(train_dataset, range(min(12, len(train_dataset))))
    val_subset = torch.utils.data.Subset(val_dataset, range(min(4, len(val_dataset))))


    print(f"• Total Train sequences available      : {len(train_dataset)}")
    print(f"• Smoke test Train subset sequences    : {len(train_subset)}")
    print(f"• Total Validation sequences available : {len(val_dataset)}")
    print(f"• Smoke test Val subset sequences      : {len(val_subset)}")

    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, collate_fn=storm_collate_fn)
    val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, collate_fn=storm_collate_fn)


    # 2. Model & Optimizer Initialization
    model = StormFusionModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    checkpoint_dir = PROJECT_ROOT / "checkpoints/smoke_test"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(log_dir=checkpoint_dir)

    trainer = StormFusionTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=total_multi_task_loss,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        logger=logger,
    )

    # 3. Execute 2-Epoch Smoke Test
    print("\n[SMOKE TEST] Launching 2-epoch development fit loop...")
    try:
        trainer.fit(train_loader, val_loader, epochs=2, bypass_readiness_check=True)
        smoke_test_passed = True
        error_msg = None
        print("\n[SMOKE TEST] End-to-end forward, backward, validation, and checkpointing PASSED!")
    except Exception as e:
        smoke_test_passed = False
        error_msg = str(e)
        print(f"\n[SMOKE TEST FAILED] Error: {e}")

    # Check best checkpoint file existence
    best_ckpt = checkpoint_dir / "best.pt"
    ckpt_saved = best_ckpt.exists()

    summary = {
        "smoke_test_passed": smoke_test_passed,
        "epochs_completed": trainer.current_epoch,
        "best_val_loss": float(trainer.best_val_loss) if trainer.best_val_loss != float("inf") else None,
        "checkpoint_saved": ckpt_saved,
        "checkpoint_path": str(best_ckpt) if ckpt_saved else None,
        "error_message": error_msg,
    }

    return summary


if __name__ == "__main__":
    res = run_development_smoke_test()
    print(json.dumps(res, indent=2))
