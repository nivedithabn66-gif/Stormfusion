"""STORMFUSION Full Development Training Engine (Section 6).

Executes full development training using real NOAA development data, ERA5, and historical track data.
Saves reproducible checkpoints and artifacts to `checkpoints/dev_training/`.
"""

import json
import time
import sys
from pathlib import Path
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.storm_dataset import StormSequenceDataset, storm_collate_fn
from models.stormfusion import StormFusionModel
from models.losses import total_multi_task_loss
from training.trainer import StormFusionTrainer
from training.logger import StructuredLogger


def run_development_training(
    epochs: int = 2,
    batch_size: int = 8,
    num_workers: int = 0,
    max_train_samples: int = 32,
    max_val_samples: int = 16,
) -> dict:
    print("\n==================================================", flush=True)
    print("      STORMFUSION FULL DEVELOPMENT TRAINING       ", flush=True)
    print("==================================================", flush=True)

    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Dataset & DataLoader Initialization
    print("\n[1. DATASET INITIALIZATION]", flush=True)
    train_dataset_full = StormSequenceDataset(split="train")
    val_dataset_full = StormSequenceDataset(split="val")

    if max_train_samples and max_train_samples < len(train_dataset_full):
        train_dataset = torch.utils.data.Subset(train_dataset_full, range(max_train_samples))
    else:
        train_dataset = train_dataset_full

    if max_val_samples and max_val_samples < len(val_dataset_full):
        val_dataset = torch.utils.data.Subset(val_dataset_full, range(max_val_samples))
    else:
        val_dataset = val_dataset_full

    print(f"  • Total Train Sequences        : {len(train_dataset_full):,}", flush=True)
    print(f"  • Dev-Training Train Sequences  : {len(train_dataset):,}", flush=True)
    print(f"  • Total Validation Sequences   : {len(val_dataset_full):,}", flush=True)
    print(f"  • Dev-Training Val Sequences    : {len(val_dataset):,}", flush=True)
    print(f"  • Seed                         : {seed}", flush=True)
    print(f"  • Batch Size                   : {batch_size}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=storm_collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=storm_collate_fn,
        num_workers=num_workers,
    )

    # 2. Setup Output Directory
    checkpoint_dir = PROJECT_ROOT / "checkpoints/dev_training"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 3. Model, Optimizer, Scheduler Setup
    print("\n[2. MODEL & OPTIMIZER SETUP]")
    model = StormFusionModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    logger = StructuredLogger(log_dir=checkpoint_dir)

    training_config = {
        "model_name": "StormFusionModel",
        "development_mode": True,
        "satellite_source": "NOAA",
        "hardware": {
            "device": "cpu",
            "cpu_cores": 18,
            "torch_version": torch.__version__,
            "cuda_available": False,
        },
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "sequence_length": 8,
            "image_dimensions": [500, 500],
            "satellite_channels": 3,
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "scheduler": "StepLR",
            "step_size": 2,
            "gamma": 0.5,
            "dropout": 0.2,
            "gradient_clipping": 1.0,
            "mixed_precision": False,
            "seed": seed,
        },
        "task_loss_weights": {
            "detection": 1.0,
            "intensity": 1.0,
            "pressure": 1.0,
            "track": 1.0,
            "pattern": 0.0,
        },
        "pattern_training_status": "PENDING_LABELS",
    }

    # Save training_config.yaml
    config_yaml_path = checkpoint_dir / "training_config.yaml"
    with open(config_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(training_config, f, default_flow_style=False)

    # 4. Save Normalization References & Manifest Reference
    norm_refs = {
        "satellite_norm_stats": str(PROJECT_ROOT / "data/processed/normalization/noaa_norm_stats.json"),
        "era5_norm_stats": str(PROJECT_ROOT / "data/processed/normalization/era5_norm_stats.json"),
        "track_norm_stats": str(PROJECT_ROOT / "data/processed/normalization/track_norm_stats.json"),
        "sequence_manifest": str(PROJECT_ROOT / "data/processed/sequences/sequence_manifest.csv"),
        "split_metadata": str(PROJECT_ROOT / "data/splits/split_metadata.json"),
    }
    with open(checkpoint_dir / "normalization_references.json", "w", encoding="utf-8") as f:
        json.dump(norm_refs, f, indent=2)

    # 5. Initialize Trainer
    trainer = StormFusionTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=total_multi_task_loss,
        device="cpu",
        config=training_config,
        checkpoint_dir=checkpoint_dir,
        logger=logger,
        gradient_clip_norm=1.0,
    )

    # 6. Fit Model over Epochs
    print(f"\n[3. FIT LOOP STARTING — {epochs} EPOCHS]")
    start_time = time.time()
    epoch_metrics_history = []

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate_epoch(val_loader)
        scheduler.step()
        ep_elapsed = time.time() - ep_start

        trainer.current_epoch = epoch
        val_loss = val_metrics.get("val_loss", 0.0)

        # Save latest checkpoint
        latest_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_metrics.get("train_loss", 0.0),
            "val_loss": val_loss,
            "config": training_config,
        }
        torch.save(latest_payload, checkpoint_dir / "latest.pt")
        torch.save(latest_payload, checkpoint_dir / "last_model.pt")

        # Save best checkpoint
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            best_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_metrics.get("train_loss", 0.0),
                "val_loss": val_loss,
                "config": training_config,
            }
            torch.save(best_payload, checkpoint_dir / "best.pt")
            torch.save(best_payload, checkpoint_dir / "best_model.pt")
            is_best = True
        else:
            is_best = False

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics.get("train_loss", 0.0),
            "val_loss": val_loss,
            "train_task_losses": train_metrics,
            "val_task_losses": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": ep_elapsed,
            "is_best": is_best,
        }
        epoch_metrics_history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_elapsed:.1f}s] | "
            f"Train Loss: {train_metrics.get('train_loss', 0.0):.4f} | "
            f"Val Loss: {val_loss:.4f} {'(BEST)' if is_best else ''}",
            flush=True,
        )

    total_training_time = time.time() - start_time
    print(f"\n[DEVELOPMENT TRAINING COMPLETE] Total Time: {total_training_time:.1f}s", flush=True)
    print(f"  • Best Validation Loss : {trainer.best_val_loss:.4f}", flush=True)
    print(f"  • Checkpoints Saved to  : {checkpoint_dir}", flush=True)

    # 7. Save Metrics and Summary JSON
    metrics_data = {
        "total_training_time_sec": total_training_time,
        "epochs_completed": epochs,
        "best_epoch": int(min(epoch_metrics_history, key=lambda x: x["val_loss"])["epoch"]),
        "best_val_loss": float(trainer.best_val_loss),
        "final_train_loss": float(epoch_metrics_history[-1]["train_loss"]),
        "final_val_loss": float(epoch_metrics_history[-1]["val_loss"]),
        "epoch_history": epoch_metrics_history,
    }
    with open(checkpoint_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    summary_data = {
        "status": "COMPLETED",
        "training_type": "DEVELOPMENT",
        "real_eumetsat_hrseviri": False,
        "dataset_provenance": "NOAA_NCEI_IBTrACS_v04r01_Historical_Catalog_with_Development_Fixtures",
        "model": "STORMFUSION NOAA DEVELOPMENT PRETRAINING MODEL",
        "scientific_claim": "NOAA-based DEVELOPMENT / PRETRAINING MODEL ONLY (NOT NIO OPERATIONAL)",
        "readiness_flags": {
            "NOAA_PIPELINE_READY": True,
            "NOAA_REAL_DATA_AVAILABLE": True,
            "EUMETSAT_REAL_DATA_AVAILABLE": False,
            "MOSDAC": "REMOVED",
            "NIO_SATELLITE_TRAINING_READY": False,
            "SCIENTIFIC_TRAINING_READY": False,
        },
        "best_epoch": metrics_data["best_epoch"],
        "best_val_loss": metrics_data["best_val_loss"],
        "checkpoints": {
            "best": str(checkpoint_dir / "best_model.pt"),
            "last": str(checkpoint_dir / "last_model.pt"),
            "best_pt": str(checkpoint_dir / "best.pt"),
            "latest_pt": str(checkpoint_dir / "latest.pt"),
        },
    }
    with open(checkpoint_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


if __name__ == "__main__":
    run_development_training(epochs=2, batch_size=8)
