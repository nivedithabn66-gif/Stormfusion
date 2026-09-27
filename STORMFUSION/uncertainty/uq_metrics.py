"""Predictive Uncertainty Metrics Calculator for STORMFUSION (Step 18).

Calculates:
1. Classification predictive mean, variance, entropy, and max probability.
2. Multi-horizon regression predictive mean, std, and prediction intervals (50%, 80%, 90%).
3. Track positional uncertainty in degrees and kilometer approximations.
"""

from typing import Dict, Any, List, Tuple, Optional
import math
import numpy as np
import torch


def compute_classification_uq(
    logits_samples: torch.Tensor,
    eps: float = 1e-7,
) -> Dict[str, Any]:
    """Calculates classification uncertainty metrics across MC samples.

    Args:
        logits_samples: Tensor of shape [N, B, 1] or [N, B] containing detection logits.

    Returns:
        Dict containing predictive mean, variance, entropy, and max probability.
    """
    # Convert logits to probabilities
    probs = torch.sigmoid(logits_samples)  # [N, B, 1] or [N, B]
    if probs.ndim == 3:
        probs = probs.squeeze(-1)  # [N, B]

    # Mean probability across MC samples N
    mean_prob = torch.mean(probs, dim=0)  # [B]
    var_prob = torch.var(probs, dim=0, unbiased=True) if probs.shape[0] > 1 else torch.zeros_like(mean_prob)

    # Binary entropy H(p) = - [p log(p) + (1-p) log(1-p)]
    p_clamped = torch.clamp(mean_prob, eps, 1.0 - eps)
    entropy = - (p_clamped * torch.log(p_clamped) + (1.0 - p_clamped) * torch.log(1.0 - p_clamped))

    # Max probability (uncalibrated confidence indicator)
    max_prob = torch.maximum(mean_prob, 1.0 - mean_prob)

    return {
        "mean_probability": mean_prob,
        "variance": var_prob,
        "entropy": entropy,
        "max_probability": max_prob,
    }


def compute_regression_interval(
    samples: np.ndarray,
    interval_level: float = 0.90,
) -> Tuple[float, float]:
    """Calculates predictive interval bounds for a 1D array of MC regression samples."""
    if len(samples) == 0 or np.isnan(samples).all():
        return (float("nan"), float("nan"))

    alpha = (1.0 - interval_level) / 2.0
    lower = float(np.percentile(samples, alpha * 100))
    upper = float(np.percentile(samples, (1.0 - alpha) * 100))
    return (lower, upper)


def compute_regression_uq(
    samples_tensor: torch.Tensor,
    intervals: List[float] = [0.50, 0.80, 0.90],
) -> Dict[str, Any]:
    """Calculates predictive mean, std, and predictive intervals across MC samples.

    Args:
        samples_tensor: Tensor of shape [N, B, 4] for 4 lead times (+3h, +6h, +12h, +24h).

    Returns:
        Dict containing per-horizon statistics.
    """
    horizons = ["3h", "6h", "12h", "24h"]
    # Convert to numpy [N, B, 4]
    arr = samples_tensor.detach().cpu().numpy()
    N, B, H = arr.shape

    results = {}
    for h_idx, h_name in enumerate(horizons):
        sub_arr = arr[:, :, h_idx]  # [N, B]
        h_dict = {}

        # Per batch sample statistics
        batch_means = np.mean(sub_arr, axis=0)  # [B]
        batch_stds = np.std(sub_arr, axis=0, ddof=1) if N > 1 else np.zeros(B)  # [B]

        batch_intervals = {}
        for lvl in intervals:
            lvl_key = f"interval_{int(lvl * 100)}"
            b_bounds = []
            for b in range(B):
                b_bounds.append(compute_regression_interval(sub_arr[:, b], lvl))
            batch_intervals[lvl_key] = b_bounds

        h_dict["mean"] = batch_means.tolist()
        h_dict["std"] = batch_stds.tolist()
        h_dict["predictive_intervals"] = batch_intervals
        results[h_name] = h_dict

    return results


def compute_track_positional_uq(
    track_delta_samples: torch.Tensor,
    ref_lats: Optional[List[float]] = None,
    intervals: List[float] = [0.50, 0.80, 0.90],
) -> Dict[str, Any]:
    """Calculates track positional uncertainty in degrees and kilometer approximations.

    Args:
        track_delta_samples: Tensor of shape [N, B, 4, 2] containing [delta_lat, delta_lon].
        ref_lats: Reference latitudes for longitude-to-km conversion (default 15.0 deg if None).

    Returns:
        Dict containing per-horizon track positional uncertainty metrics.
    """
    horizons = ["3h", "6h", "12h", "24h"]
    arr = track_delta_samples.detach().cpu().numpy()  # [N, B, 4, 2]
    N, B, H, C = arr.shape

    if ref_lats is None:
        ref_lats = [15.0] * B

    results = {}
    KM_PER_DEG_LAT = 111.12

    for h_idx, h_name in enumerate(horizons):
        lat_samples = arr[:, :, h_idx, 0]  # [N, B]
        lon_samples = arr[:, :, h_idx, 1]  # [N, B]

        lat_means = np.mean(lat_samples, axis=0)  # [B]
        lon_means = np.mean(lon_samples, axis=0)  # [B]

        lat_stds = np.std(lat_samples, axis=0, ddof=1) if N > 1 else np.zeros(B)  # [B]
        lon_stds = np.std(lon_samples, axis=0, ddof=1) if N > 1 else np.zeros(B)  # [B]

        pos_uncertainty_km = []
        for b in range(B):
            ref_lat_rad = math.radians(ref_lats[b])
            km_per_deg_lon = KM_PER_DEG_LAT * math.cos(ref_lat_rad)

            lat_err_km = lat_stds[b] * KM_PER_DEG_LAT
            lon_err_km = lon_stds[b] * km_per_deg_lon
            comb_km = math.sqrt(lat_err_km**2 + lon_err_km**2)
            pos_uncertainty_km.append(comb_km)

        results[h_name] = {
            "delta_lat_mean": lat_means.tolist(),
            "delta_lon_mean": lon_means.tolist(),
            "delta_lat_std": lat_stds.tolist(),
            "delta_lon_std": lon_stds.tolist(),
            "latitude_uncertainty_deg": lat_stds.tolist(),
            "longitude_uncertainty_deg": lon_stds.tolist(),
            "positional_uncertainty_km": pos_uncertainty_km,
            "approximation_note": "Positional uncertainty in km approximated using WGS84 1 deg lat = 111.12 km and 1 deg lon = 111.12 * cos(lat) km.",
        }

    return results
