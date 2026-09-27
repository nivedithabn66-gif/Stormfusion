"""STORMFUSION Trainer Engine.

Orchestrates multi-task model training, validation epochs, gradient clipping, learning rate
scheduling, structured logging, early stopping, and checkpointing while enforcing the
readiness safeguard gate before execution.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim

from training.device import select_device
from training.checkpoint import save_checkpoint, load_checkpoint
from training.logger import StructuredLogger
from training.readiness import check_training_readiness


class EarlyStopping:
    """Monitors validation loss and signals when to stop training early."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0001, enabled: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.enabled = enabled
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if not self.enabled:
            return False

        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class StormFusionTrainer:
    """STORMFUSION Multi-Task Training Infrastructure Engine."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[Any] = None,
        device: str | torch.device = "auto",
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: Optional[str | Path] = None,
        logger: Optional[StructuredLogger] = None,
        gradient_clip_norm: float = 1.0,
    ):
        self.config = config or {}
        self.device = select_device(str(device)) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.gradient_clip_norm = gradient_clip_norm

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.logger = logger or StructuredLogger(log_dir=self.checkpoint_dir)

        es_cfg = self.config.get("training", {}).get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 10),
            min_delta=float(es_cfg.get("min_delta", 0.0001)),
            enabled=es_cfg.get("enabled", True),
        )

        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Runs a single training epoch over dataloader.

        Returns:
            Dict containing mean train loss and task loss breakdowns
        """
        self.model.train()
        total_loss = 0.0
        task_loss_totals: Dict[str, float] = {}
        batch_count = 0

        for batch in dataloader:
            batch_count += 1
            self.optimizer.zero_grad()

            # Handle batch format
            if isinstance(batch, dict) and "satellite" in batch and "track" in batch and "environment" in batch:
                sat_dict = batch["satellite"]
                trk_dict = batch["track"]
                env_dict = batch["environment"]
                tgt_dict = batch["targets"]

                inputs = {
                    "satellite_tensor": sat_dict["satellite_tensor"].to(self.device),
                    "satellite_valid_mask": sat_dict["satellite_valid_mask"].to(self.device),
                    "satellite_modality_mask": sat_dict["satellite_modality_mask"].to(self.device),
                    "track_features": trk_dict["track_features"].to(self.device),
                    "track_valid_mask": trk_dict["track_valid_mask"].to(self.device),
                    "track_modality_mask": trk_dict.get("track_modality_mask", torch.ones_like(trk_dict["track_valid_mask"][:, :1])).to(self.device),
                    "era5_tensor": env_dict["era5_features"].to(self.device),
                    "era5_valid_mask": env_dict["era5_valid_mask"].to(self.device),
                    "era5_modality_mask": env_dict["era5_modality_mask"].to(self.device),
                }

                targets = {
                    "detection_target": tgt_dict["detection"].to(self.device),
                    "pattern_target": tgt_dict["pattern"].to(self.device),
                    "intensity_target": tgt_dict["intensity"].to(self.device),
                    "pressure_target": tgt_dict["pressure"].to(self.device),
                    "track_delta_target": tgt_dict["future_track"].to(self.device),
                }

                masks = {
                    "detection_valid_mask": torch.ones_like(tgt_dict["detection"]).to(self.device),
                    "pattern_valid_mask": (tgt_dict["pattern"] != -1).float().to(self.device),
                    "intensity_valid_mask": tgt_dict["intensity_valid_mask"].to(self.device),
                    "pressure_valid_mask": tgt_dict["pressure_valid_mask"].to(self.device),
                    "track_valid_mask": tgt_dict["future_track_valid_mask"].to(self.device),
                }
            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 3:
                    inputs, targets, masks = batch[0], batch[1], batch[2]
                elif len(batch) == 2:
                    inputs, targets, masks = batch[0], batch[1], {}
                else:
                    inputs, targets, masks = batch[0], {}, {}
            elif isinstance(batch, dict):
                inputs = batch.get("inputs", batch.get("fused_embedding"))
                targets = batch.get("targets", {})
                masks = batch.get("masks", {})
            else:
                inputs = batch
                targets = {}
                masks = {}

            # Move inputs to device if tensor or dict
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            if isinstance(targets, dict):
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
            if isinstance(masks, dict):
                masks = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in masks.items()}

            # Forward pass
            if isinstance(inputs, dict):
                preds = self.model(**inputs)
            else:
                preds = self.model(inputs)

            # Compute loss
            if self.loss_fn is not None:
                loss, loss_dict = self.loss_fn(preds, targets, masks)
            else:
                loss = preds.sum()
                loss_dict = {"total_loss": loss.item()}

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)

            self.optimizer.step()

            total_loss += loss.item()
            for k, v in loss_dict.items():
                task_loss_totals[k] = task_loss_totals.get(k, 0.0) + v

        avg_loss = total_loss / max(1, batch_count)
        avg_task_losses = {k: v / max(1, batch_count) for k, v in task_loss_totals.items()}
        avg_task_losses["train_loss"] = avg_loss

        return avg_task_losses

    def validate_epoch(self, dataloader) -> Dict[str, float]:
        """Runs a single validation epoch over dataloader without updating parameters.

        Returns:
            Dict containing mean val loss and task loss breakdowns
        """
        self.model.eval()
        total_loss = 0.0
        task_loss_totals: Dict[str, float] = {}
        batch_count = 0

        with torch.no_grad():
            for batch in dataloader:
                batch_count += 1

                if isinstance(batch, dict) and "satellite" in batch and "track" in batch and "environment" in batch:
                    sat_dict = batch["satellite"]
                    trk_dict = batch["track"]
                    env_dict = batch["environment"]
                    tgt_dict = batch["targets"]

                    inputs = {
                        "satellite_tensor": sat_dict["satellite_tensor"].to(self.device),
                        "satellite_valid_mask": sat_dict["satellite_valid_mask"].to(self.device),
                        "satellite_modality_mask": sat_dict["satellite_modality_mask"].to(self.device),
                        "track_features": trk_dict["track_features"].to(self.device),
                        "track_valid_mask": trk_dict["track_valid_mask"].to(self.device),
                        "track_modality_mask": trk_dict.get("track_modality_mask", torch.ones_like(trk_dict["track_valid_mask"][:, :1])).to(self.device),
                        "era5_tensor": env_dict["era5_features"].to(self.device),
                        "era5_valid_mask": env_dict["era5_valid_mask"].to(self.device),
                        "era5_modality_mask": env_dict["era5_modality_mask"].to(self.device),
                    }

                    targets = {
                        "detection_target": tgt_dict["detection"].to(self.device),
                        "pattern_target": tgt_dict["pattern"].to(self.device),
                        "intensity_target": tgt_dict["intensity"].to(self.device),
                        "pressure_target": tgt_dict["pressure"].to(self.device),
                        "track_delta_target": tgt_dict["future_track"].to(self.device),
                    }

                    masks = {
                        "detection_valid_mask": torch.ones_like(tgt_dict["detection"]).to(self.device),
                        "pattern_valid_mask": (tgt_dict["pattern"] != -1).float().to(self.device),
                        "intensity_valid_mask": tgt_dict["intensity_valid_mask"].to(self.device),
                        "pressure_valid_mask": tgt_dict["pressure_valid_mask"].to(self.device),
                        "track_valid_mask": tgt_dict["future_track_valid_mask"].to(self.device),
                    }
                elif isinstance(batch, (list, tuple)):
                    if len(batch) >= 3:
                        inputs, targets, masks = batch[0], batch[1], batch[2]
                    elif len(batch) == 2:
                        inputs, targets, masks = batch[0], batch[1], {}
                    else:
                        inputs, targets, masks = batch[0], {}, {}
                elif isinstance(batch, dict):
                    inputs = batch.get("inputs", batch.get("fused_embedding"))
                    targets = batch.get("targets", {})
                    masks = batch.get("masks", {})
                else:
                    inputs = batch
                    targets = {}
                    masks = {}

                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                elif isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                if isinstance(targets, dict):
                    targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
                if isinstance(masks, dict):
                    masks = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in masks.items()}

                if isinstance(inputs, dict):
                    preds = self.model(**inputs)
                else:
                    preds = self.model(inputs)

                if self.loss_fn is not None:
                    loss, loss_dict = self.loss_fn(preds, targets, masks)
                else:
                    loss = preds.sum()
                    loss_dict = {"total_loss": loss.item()}

                total_loss += loss.item()
                for k, v in loss_dict.items():
                    task_loss_totals[k] = task_loss_totals.get(k, 0.0) + v

        avg_loss = total_loss / max(1, batch_count)
        avg_task_losses = {k: v / max(1, batch_count) for k, v in task_loss_totals.items()}
        avg_task_losses["val_loss"] = avg_loss

        return avg_task_losses


    def fit(self, train_loader, val_loader, epochs: int = 10, bypass_readiness_check: bool = False):
        """Runs complete training and validation fit loop.

        Enforces training readiness gate unless bypass_readiness_check=True (used only in software unit tests).

        Raises:
            PermissionError: If data readiness gate is not satisfied.
        """
        if not bypass_readiness_check:
            readiness = check_training_readiness(self.config)
            if not readiness["training_allowed"]:
                reasons_list = readiness.get("blocking_reasons", readiness.get("reasons", []))
                reasons_str = "; ".join(reasons_list)
                raise PermissionError(
                    f"Scientific training blocked by readiness safeguard gate. Reasons: {reasons_str}"
                )


        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start_time

            val_loss = val_metrics.get("val_loss", val_metrics.get("total_loss", 0.0))

            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_metrics.get("train_loss", 0.0),
                val_loss=val_loss,
                task_losses=val_metrics,
                lr=current_lr,
                elapsed_seconds=elapsed,
            )

            # Check best loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.checkpoint_dir and bypass_readiness_check:
                    save_checkpoint(
                        self.checkpoint_dir / "best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        best_val_loss=self.best_val_loss,
                        config=self.config,
                    )

            if self.early_stopping.step(val_loss):
                print(f"Early stopping triggered at epoch {epoch}")
                break
