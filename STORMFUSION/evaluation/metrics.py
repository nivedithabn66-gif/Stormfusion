"""Evaluation Metrics Calculator for STORMFUSION (Step 20).

Calculates deterministic and probabilistic evaluation metrics:
1. Trajectory Position Error: Haversine distance error (km / nautical miles) for lead times (+3h, +6h, +12h, +24h).
2. Intensity Regression: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R2.
3. Pressure Regression: MAE, RMSE, R2.
4. Detection Classification: Accuracy, Precision, Recall, F1, ROC-AUC.
5. UQ Calibration Integration: Expected Calibration Error (ECE), Brier Score, prediction interval empirical coverage.
"""

from typing import Dict, Any, List, Tuple, Optional
import math
import numpy as np


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates Haversine great-circle distance in kilometers between two lat/lon points."""
    if np.isnan(lat1) or np.isnan(lon1) or np.isnan(lat2) or np.isnan(lon2):
        return float("nan")

    R_EARTH_KM = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(R_EARTH_KM * c)


def compute_track_position_errors(
    pred_track_deltas: np.ndarray,  # [M, 4, 2]
    true_track_deltas: np.ndarray,  # [M, 4, 2]
    valid_masks: Optional[np.ndarray] = None,  # [M, 4]
) -> Dict[str, Any]:
    """Calculates trajectory position error in kilometers for lead times (+3h, +6h, +12h, +24h)."""
    horizons = ["3h", "6h", "12h", "24h"]
    M, H, C = pred_track_deltas.shape

    if valid_masks is None:
        valid_masks = np.ones((M, H), dtype=bool)
    else:
        valid_masks = valid_masks.astype(bool)

    results = {}
    for h_idx, h_name in enumerate(horizons):
        pred_h = pred_track_deltas[:, h_idx, :]  # [M, 2]
        true_h = true_track_deltas[:, h_idx, :]  # [M, 2]
        mask_h = valid_masks[:, h_idx]

        errors_km = []
        for i in range(M):
            if mask_h[i]:
                # Haversine distance error between predicted delta lat/lon and true delta lat/lon
                d_km = haversine_distance_km(0.0, 0.0, pred_h[i, 0] - true_h[i, 0], pred_h[i, 1] - true_h[i, 1])
                if not math.isnan(d_km):
                    errors_km.append(d_km)

        if len(errors_km) > 0:
            results[h_name] = {
                "mean_error_km": float(np.mean(errors_km)),
                "median_error_km": float(np.median(errors_km)),
                "rmse_km": float(np.sqrt(np.mean(np.square(errors_km)))),
                "std_km": float(np.std(errors_km)),
                "count": len(errors_km),
            }
        else:
            results[h_name] = {
                "mean_error_km": float("nan"),
                "median_error_km": float("nan"),
                "rmse_km": float("nan"),
                "std_km": float("nan"),
                "count": 0,
            }

    return results


def compute_regression_metrics(
    pred_vals: np.ndarray,  # [M, 4]
    true_vals: np.ndarray,  # [M, 4]
    valid_masks: Optional[np.ndarray] = None,  # [M, 4]
) -> Dict[str, Any]:
    """Calculates MAE, RMSE, and R2 for regression outputs across forecast horizons."""
    horizons = ["3h", "6h", "12h", "24h"]
    M, H = pred_vals.shape

    if valid_masks is None:
        valid_masks = np.ones((M, H), dtype=bool)
    else:
        valid_masks = valid_masks.astype(bool)

    results = {}
    for h_idx, h_name in enumerate(horizons):
        p_h = pred_vals[:, h_idx]
        t_h = true_vals[:, h_idx]
        m_h = valid_masks[:, h_idx]

        p_valid = p_h[m_h]
        t_valid = t_h[m_h]

        if len(t_valid) > 0:
            mae = float(np.mean(np.abs(p_valid - t_valid)))
            rmse = float(np.sqrt(np.mean(np.square(p_valid - t_valid))))

            ss_res = np.sum((t_valid - p_valid) ** 2)
            ss_tot = np.sum((t_valid - np.mean(t_valid)) ** 2)
            r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

            results[h_name] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "count": len(t_valid),
            }
        else:
            results[h_name] = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "count": 0,
            }

    return results
