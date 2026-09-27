"""Robustness Experiments Module for STORMFUSION (Step 21).

Evaluates controlled real-world input perturbations:
1. Frame dropout / temporal sequence gaps (0%, 25%, 50%, 75%, 100% missing historical timesteps).
2. Input noise addition (Gaussian noise scaling sigma = 0.01 to 0.50).
3. Missing modality degradation scenarios (all_available, satellite_missing, track_missing, era5_missing, etc.).
4. Uncertainty response audit under input degradation (evaluating if predictive variance/entropy responds appropriately).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from robustness.degradation_runner import check_tensor_nan_inf
from uncertainty.uq_runner import STORMFUSIONUQRunner


def add_input_gaussian_noise(tensor: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Adds zero-mean Gaussian noise N(0, sigma^2) to input tensor."""
    if sigma <= 0.0:
        return tensor.clone()
    noise = torch.randn_like(tensor) * sigma
    return tensor + noise


def apply_frame_dropout(
    satellite_tensor: torch.Tensor,
    satellite_valid_mask: torch.Tensor,
    dropout_ratio: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulates satellite frame dropout by setting valid_mask to 0 for dropped timesteps."""
    B, T = satellite_valid_mask.shape
    new_tensor = satellite_tensor.clone()
    new_mask = satellite_valid_mask.clone()

    if dropout_ratio <= 0.0:
        return new_tensor, new_mask

    # Determine timesteps to drop (up to T-1 to preserve reference time t7)
    max_drop = max(1, T - 1)
    num_drop = min(int(round((T - 1) * dropout_ratio)), max_drop)
    if num_drop > 0:
        drop_indices = np.random.choice(T - 1, size=num_drop, replace=False)  # keep t7 intact
        for b in range(B):
            for t_idx in drop_indices:
                new_mask[b, t_idx] = 0.0
                new_tensor[b, t_idx] = 0.0


    return new_tensor, new_mask


class RobustnessExperimentRunner:
    """Orchestrates robustness degradation experiments and UQ response analysis."""

    def __init__(self, model: nn.Module, num_mc_samples: int = 10):
        self.model = model
        self.uq_runner = STORMFUSIONUQRunner(model, num_mc_samples=num_mc_samples)

    def run_noise_robustness_experiment(
        self,
        inputs: Dict[str, torch.Tensor],
        sigma_levels: List[float] = [0.0, 0.05, 0.10, 0.25, 0.50],
    ) -> Dict[str, Any]:
        """Evaluates model prediction stability and UQ response under increasing Gaussian noise."""
        results = {}

        for sig in sigma_levels:
            noisy_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and "tensor" in k:
                    noisy_inputs[k] = add_input_gaussian_noise(v, sigma=sig)
                else:
                    noisy_inputs[k] = v.clone() if isinstance(v, torch.Tensor) else v

            uq_res = self.uq_runner.run_uq_analysis(**noisy_inputs)
            results[f"sigma_{sig}"] = {
                "noise_level": sig,
                "status": uq_res["status"],
                "detection_variance_mean": float(np.mean(uq_res["detection"]["variance"])) if uq_res.get("detection") else float("nan"),
                "detection_entropy_mean": float(np.mean(uq_res["detection"]["entropy"])) if uq_res.get("detection") else float("nan"),
            }

        return results

    def run_frame_dropout_experiment(
        self,
        inputs: Dict[str, torch.Tensor],
        dropout_ratios: List[float] = [0.0, 0.25, 0.50, 0.75, 1.0],
    ) -> Dict[str, Any]:
        """Evaluates model performance and UQ response under frame dropout ratios."""
        results = {}

        sat_tensor = inputs["satellite_tensor"]
        sat_vmask = inputs["satellite_valid_mask"]

        for ratio in dropout_ratios:
            mod_sat, mod_mask = apply_frame_dropout(sat_tensor, sat_vmask, dropout_ratio=ratio)
            mod_inputs = dict(inputs)
            mod_inputs["satellite_tensor"] = mod_sat
            mod_inputs["satellite_valid_mask"] = mod_mask

            uq_res = self.uq_runner.run_uq_analysis(**mod_inputs)
            results[f"dropout_{int(ratio*100)}pct"] = {
                "dropout_ratio": ratio,
                "status": uq_res["status"],
                "detection_variance_mean": float(np.mean(uq_res["detection"]["variance"])) if uq_res.get("detection") else float("nan"),
            }

        return results
