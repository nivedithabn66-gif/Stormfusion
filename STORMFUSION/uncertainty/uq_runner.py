"""STORMFUSION Uncertainty Quantification Runner (Step 18).

Coordinates:
1. MC-Dropout stochastic inference execution.
2. Classification UQ computation (detection & pattern pending gate).
3. Multi-horizon regression predictive intervals (intensity, pressure).
4. Track positional uncertainty (degrees & km).
5. Missing-modality UQ scenario evaluation.
6. All-missing sentinel handling (INSUFFICIENT_INPUT).
7. Structured, versioned output schema generation.
"""

from typing import Dict, Any, List, Optional
import time
import numpy as np
import torch
import torch.nn as nn

from uncertainty.mc_dropout import MCDropoutEstimator
from uncertainty.uq_metrics import (
    compute_classification_uq,
    compute_regression_uq,
    compute_track_positional_uq,
)
from uncertainty.calibration import (
    compute_ece,
    compute_brier_score,
    compute_reliability_curve,
    evaluate_prediction_interval_coverage,
)


class STORMFUSIONUQRunner:
    """End-to-End Uncertainty Quantification Runner."""

    def __init__(self, model: nn.Module, num_mc_samples: int = 20):
        self.model = model
        self.num_mc_samples = num_mc_samples
        self.estimator = MCDropoutEstimator(model, num_samples=num_mc_samples)

    def run_uq_analysis(
        self,
        satellite_tensor: Optional[torch.Tensor] = None,
        satellite_valid_mask: Optional[torch.Tensor] = None,
        satellite_modality_mask: Optional[torch.Tensor] = None,
        track_features: Optional[torch.Tensor] = None,
        track_valid_mask: Optional[torch.Tensor] = None,
        track_modality_mask: Optional[torch.Tensor] = None,
        era5_tensor: Optional[torch.Tensor] = None,
        era5_valid_mask: Optional[torch.Tensor] = None,
        era5_modality_mask: Optional[torch.Tensor] = None,
        true_targets: Optional[Dict[str, torch.Tensor]] = None,
        intervals: List[float] = [0.50, 0.80, 0.90],
    ) -> Dict[str, Any]:
        """Executes full UQ analysis pipeline and returns versioned UQ output schema."""
        start_time = time.perf_counter()

        # Check modality availability
        sat_avail = bool(satellite_modality_mask is not None and (satellite_modality_mask == 1).any())
        trk_avail = bool(track_modality_mask is not None and (track_modality_mask == 1).any())
        era_avail = bool(era5_modality_mask is not None and (era5_modality_mask == 1).any())

        all_missing = not (sat_avail or trk_avail or era_avail)

        if all_missing:
            elapsed = time.perf_counter() - start_time
            return {
                "schema_version": "1.0",
                "status": "INSUFFICIENT_INPUT",
                "insufficient_input": True,
                "num_mc_samples": self.num_mc_samples,
                "modality_availability": {
                    "satellite": False,
                    "track": False,
                    "era5": False,
                },
                "execution_time_seconds": float(elapsed),
                "detection": {
                    "probability": None,
                    "variance": None,
                    "entropy": None,
                    "max_probability": None,
                },
                "pattern": {
                    "status": "PATTERN_LABEL_PENDING",
                    "pattern_labels_final": False,
                },
                "intensity": None,
                "pressure": None,
                "track": None,
                "calibration": {
                    "status": "NOT_AVAILABLE",
                    "reason": "Input modalities all missing (INSUFFICIENT_INPUT).",
                },
                "scientific_training_performed": False,
            }

        # Execute MC-Dropout stochastic sampling
        samples = self.estimator.predict_stochastic(
            satellite_tensor=satellite_tensor,
            satellite_valid_mask=satellite_valid_mask,
            satellite_modality_mask=satellite_modality_mask,
            track_features=track_features,
            track_valid_mask=track_valid_mask,
            track_modality_mask=track_modality_mask,
            era5_tensor=era5_tensor,
            era5_valid_mask=era5_valid_mask,
            era5_modality_mask=era5_modality_mask,
            num_samples=self.num_mc_samples,
        )

        # Stack outputs across MC samples [N, B, ...]
        det_logits_list = [s["detection_logits"] for s in samples if "detection_logits" in s]
        intensity_list = [s["intensity"] for s in samples if "intensity" in s]
        pressure_list = [s["pressure"] for s in samples if "pressure" in s]
        track_delta_list = [s["track_delta"] for s in samples if "track_delta" in s]

        # 1. Detection Classification UQ
        if len(det_logits_list) > 0:
            det_stack = torch.stack(det_logits_list, dim=0)  # [N, B, 1]
            det_uq = compute_classification_uq(det_stack)
            det_out = {
                "probability": det_uq["mean_probability"].detach().cpu().numpy().tolist(),
                "variance": det_uq["variance"].detach().cpu().numpy().tolist(),
                "entropy": det_uq["entropy"].detach().cpu().numpy().tolist(),
                "max_probability": det_uq["max_probability"].detach().cpu().numpy().tolist(),
            }
        else:
            det_out = None

        # 2. Pattern Classification UQ (PATTERN_LABEL_PENDING = -1)
        pattern_out = {
            "status": "PATTERN_LABEL_PENDING",
            "pattern_labels_final": False,
            "note": "Pattern classification labels are pending. Pattern head UQ will activate once labels are finalized.",
        }

        # 3. Intensity Regression UQ
        if len(intensity_list) > 0:
            int_stack = torch.stack(intensity_list, dim=0)  # [N, B, 4]
            int_out = compute_regression_uq(int_stack, intervals=intervals)
        else:
            int_out = None

        # 4. Pressure Regression UQ
        if len(pressure_list) > 0:
            press_stack = torch.stack(pressure_list, dim=0)  # [N, B, 4]
            press_out = compute_regression_uq(press_stack, intervals=intervals)
        else:
            press_out = None

        # 5. Track Positional UQ
        if len(track_delta_list) > 0:
            trk_stack = torch.stack(track_delta_list, dim=0)  # [N, B, 4, 2]
            trk_out = compute_track_positional_uq(trk_stack, intervals=intervals)
        else:
            trk_out = None

        # 6. Calibration Infrastructure
        calib_out = {
            "ece": float("nan"),
            "brier_score": float("nan"),
            "interval_coverage": {
                "status": "NOT_AVAILABLE",
                "reason": "Held-out evaluation targets absent in software testing mode.",
            },
        }

        if true_targets is not None:
            if "detection_target" in true_targets and det_out is not None:
                probs_np = np.array(det_out["probability"])
                lbls_np = true_targets["detection_target"].detach().cpu().numpy().flatten()
                calib_out["ece"] = compute_ece(probs_np, lbls_np)
                calib_out["brier_score"] = compute_brier_score(probs_np, lbls_np)

        elapsed = time.perf_counter() - start_time

        return {
            "schema_version": "1.0",
            "status": "COMPUTED",
            "insufficient_input": False,
            "num_mc_samples": self.num_mc_samples,
            "modality_availability": {
                "satellite": sat_avail,
                "track": trk_avail,
                "era5": era_avail,
            },
            "execution_time_seconds": float(elapsed),
            "detection": det_out,
            "pattern": pattern_out,
            "intensity": int_out,
            "pressure": press_out,
            "track": trk_out,
            "calibration": calib_out,
            "scientific_training_performed": False,
        }
