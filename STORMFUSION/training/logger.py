"""STORMFUSION Structured Logger.

Provides structured logging for training and validation epoch events, tracking losses,
learning rates, task metrics, and elapsed time.
"""

from typing import Dict, Any, List
from pathlib import Path
import json
import time


class StructuredLogger:
    """Structured event logger recording training progress per epoch."""

    def __init__(self, log_dir: str | Path | None = None, experiment_name: str = "stormfusion_experiment"):
        self.log_dir = Path(log_dir) if log_dir else None
        self.experiment_name = experiment_name
        self.history: List[Dict[str, Any]] = []

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / f"{experiment_name}_history.json"
        else:
            self.log_file = None

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        task_losses: Dict[str, float],
        lr: float,
        elapsed_seconds: float,
        val_metrics: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Logs epoch metrics and appends to structured history."""
        entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "task_losses": task_losses,
            "learning_rate": float(lr),
            "elapsed_seconds": float(elapsed_seconds),
            "val_metrics": val_metrics or {},
            "timestamp": time.time(),
        }

        self.history.append(entry)

        if self.log_file:
            with open(self.log_file, "w") as f:
                json.dump(self.history, f, indent=2)

        return entry

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
