"""Calibration Evaluation Infrastructure for STORMFUSION (Step 18).

Provides:
1. Expected Calibration Error (ECE) for classification.
2. Brier score for classification.
3. Reliability curve binned output.
4. Empirical coverage evaluation for regression prediction intervals.
5. Strict test set isolation policy enforcement.

CRITICAL RULE:
- Test set MUST NEVER be used to fit calibration parameters.
- Coverage evaluation requires genuine held-out evaluation ground truth; returns NOT_AVAILABLE when targets are absent.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np


def compute_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculates Expected Calibration Error (ECE) for binary classification.

    Args:
        probabilities: 1D array of predicted probabilities [M]
        labels: 1D array of binary ground truth labels [M]
        n_bins: Number of equal-width probability bins

    Returns:
        float: Expected Calibration Error
    """
    if len(probabilities) == 0 or len(labels) == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idxs = np.digitize(probabilities, bins) - 1
    bin_idxs = np.clip(bin_idxs, 0, n_bins - 1)

    ece = 0.0
    total_samples = len(probabilities)

    for b in range(n_bins):
        mask = bin_idxs == b
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_conf = np.mean(probabilities[mask])
            bin_acc = np.mean(labels[mask])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Calculates Brier Score for binary classification."""
    if len(probabilities) == 0 or len(labels) == 0:
        return float("nan")
    return float(np.mean((probabilities - labels) ** 2))


def compute_reliability_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Generates reliability diagram data (bin confidences, bin accuracies, bin counts)."""
    if len(probabilities) == 0 or len(labels) == 0:
        return {"bin_confidences": [], "bin_accuracies": [], "bin_counts": []}

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idxs = np.digitize(probabilities, bins) - 1
    bin_idxs = np.clip(bin_idxs, 0, n_bins - 1)

    confidences = []
    accuracies = []
    counts = []

    for b in range(n_bins):
        mask = bin_idxs == b
        b_count = int(np.sum(mask))
        counts.append(b_count)
        if b_count > 0:
            confidences.append(float(np.mean(probabilities[mask])))
            accuracies.append(float(np.mean(labels[mask])))
        else:
            confidences.append(float((bins[b] + bins[b + 1]) / 2.0))
            accuracies.append(0.0)

    return {
        "bin_confidences": confidences,
        "bin_accuracies": accuracies,
        "bin_counts": counts,
        "n_bins": n_bins,
    }


def evaluate_prediction_interval_coverage(
    interval_bounds: List[Tuple[float, float]],
    true_targets: Optional[np.ndarray] = None,
    nominal_level: float = 0.90,
) -> Dict[str, Any]:
    """Evaluates empirical coverage of prediction intervals against ground truth targets.

    Returns NOT_AVAILABLE when true targets are absent (never fabricates fake coverage).
    """
    if true_targets is None or len(true_targets) == 0:
        return {
            "status": "NOT_AVAILABLE",
            "nominal_level": nominal_level,
            "empirical_coverage": None,
            "reason": "Genuine ground truth evaluation targets absent.",
        }

    y_true = np.array(true_targets).flatten()
    n_samples = len(y_true)

    if len(interval_bounds) != n_samples:
        return {
            "status": "ERROR",
            "reason": f"Mismatch between interval bounds count ({len(interval_bounds)}) and ground truth count ({n_samples}).",
        }

    covered = 0
    for i in range(n_samples):
        lower, upper = interval_bounds[i]
        if lower <= y_true[i] <= upper:
            covered += 1

    emp_coverage = covered / n_samples if n_samples > 0 else 0.0

    return {
        "status": "COMPUTED",
        "nominal_level": nominal_level,
        "empirical_coverage": float(emp_coverage),
        "covered_samples": covered,
        "total_samples": n_samples,
    }


def enforce_calibration_split_isolation(split: str) -> None:
    """Enforces that calibration parameters must NEVER be fitted on the test split."""
    if split.lower() == "test":
        raise ValueError(
            "CALIBRATION POLICY VIOLATION: Test split MUST NOT be used to fit calibration parameters. "
            "Fit calibration parameters on validation split only, and reserve test split for final evaluation."
        )
