"""STORMFUSION Robustness & Graceful-Degradation Package."""

from robustness.modality_scenarios import get_modality_scenarios
from robustness.degradation_runner import (
    run_degradation_scenario,
    evaluate_all_degradation_scenarios,
)
from robustness.robustness_metrics import (
    evaluate_mask_invariance,
    evaluate_nan_inf_safety,
    evaluate_determinism,
)
from robustness.robustness_report import build_robustness_report

__all__ = [
    "get_modality_scenarios",
    "run_degradation_scenario",
    "evaluate_all_degradation_scenarios",
    "evaluate_mask_invariance",
    "evaluate_nan_inf_safety",
    "evaluate_determinism",
    "build_robustness_report",
]
