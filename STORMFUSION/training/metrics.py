"""STORMFUSION Evaluation Metrics Interfaces.

Computes software-level task performance metrics:
- Binary detection: Precision, Recall, F1 score
- Multi-class pattern classification: Accuracy, Macro-F1 score
- Continuous regression (Intensity, Pressure, Track): MAE, RMSE, Mean Bias
"""

from typing import Dict, Any
import numpy as np
import torch


def compute_binary_metrics(
    logits: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """Calculates Precision, Recall, and F1 score for binary detection logits.

    Args:
        logits: Raw detection logits [N, 1] or [N]
        targets: Binary ground truth labels [N, 1] or [N]
        threshold: Decision threshold for raw logits (default 0.0)

    Returns:
        Dict of {'precision', 'recall', 'f1'}
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    preds = (logits.ravel() >= threshold).astype(int)
    targs = targets.ravel().astype(int)

    tp = float(np.sum((preds == 1) & (targs == 1)))
    fp = float(np.sum((preds == 1) & (targs == 0)))
    fn = float(np.sum((preds == 0) & (targs == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_pattern_metrics(
    logits: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
) -> Dict[str, float]:
    """Calculates Accuracy and Macro-F1 for multi-class pattern classification.

    Args:
        logits: Pattern class logits [N, num_classes]
        targets: Class index targets [N]

    Returns:
        Dict of {'accuracy', 'macro_f1'}
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    preds = np.argmax(logits, axis=1)
    targs = targets.ravel().astype(int)

    accuracy = float(np.mean(preds == targs))

    classes = np.unique(np.concatenate([preds, targs]))
    f1s = []
    for c in classes:
        tp = float(np.sum((preds == c) & (targs == c)))
        fp = float(np.sum((preds == c) & (targs != c)))
        fn = float(np.sum((preds != c) & (targs == c)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_c = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1_c)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def compute_regression_metrics(
    preds: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray | None = None,
) -> Dict[str, float]:
    """Calculates MAE, RMSE, and Mean Bias for regression targets.

    Args:
        preds: Prediction tensor
        targets: Target tensor
        mask: Optional binary validity mask

    Returns:
        Dict of {'mae', 'rmse', 'bias'}
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if mask is not None:
        valid_idx = (mask.ravel() > 0)
        p = preds.ravel()[valid_idx]
        t = targets.ravel()[valid_idx]
    else:
        p = preds.ravel()
        t = targets.ravel()

    if len(p) == 0:
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0}

    diff = p - t
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
    }
