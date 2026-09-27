"""Robustness Evaluation Report Aggregator for STORMFUSION.

Aggregates scenario results, mask invariance, gate bounds, NaN/Inf safety,
and determinism tests into a structured report dictionary.
"""

from typing import Dict, Any


def build_robustness_report(
    scenario_results: Dict[str, Dict[str, Any]],
    invariance_results: Dict[str, Any],
    nan_inf_results: Dict[str, Any],
    determinism_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Aggregates all robustness evaluations into a final summary dict.

    Args:
        scenario_results: Dict mapping scenario name to degradation runner output
        invariance_results: Dict from evaluate_mask_invariance
        nan_inf_results: Dict from evaluate_nan_inf_safety
        determinism_results: Dict from evaluate_determinism

    Returns:
        Structured report dictionary.
    """
    scenario_statuses = {}
    all_scenarios_pass = True

    for name, res in scenario_results.items():
        sc_pass = (
            res.get("inference_completed", False)
            and res.get("gate_bounds_valid", False)
            and res.get("mask_respected", False)
            and res.get("output_validity", False)
            and not res.get("has_nan_inf", True)
        )
        scenario_statuses[name] = "PASS" if sc_pass else "FAIL"
        if not sc_pass:
            all_scenarios_pass = False

    mask_validation_pass = all(
        res.get("mask_respected", False) for res in scenario_results.values()
    )
    gate_validation_pass = all(
        res.get("gate_bounds_valid", False) for res in scenario_results.values()
    )
    output_validity_pass = all(
        res.get("output_validity", False) for res in scenario_results.values()
    )

    invariance_pass = invariance_results.get("all_mask_invariance_pass", False)
    nan_inf_pass = nan_inf_results.get("nan_inf_safety_pass", False)
    determinism_pass = determinism_results.get("determinism_pass", False)

    return {
        "scenario_generation": "PASS" if len(scenario_results) == 8 else "FAIL",
        "mask_validation": "PASS" if mask_validation_pass else "FAIL",
        "mask_invariance": "PASS" if invariance_pass else "FAIL",
        "gate_validation": "PASS" if gate_validation_pass else "FAIL",
        "nan_inf_safety": "PASS" if nan_inf_pass else "FAIL",
        "determinism": "PASS" if determinism_pass else "FAIL",
        "output_validity": "PASS" if output_validity_pass else "FAIL",
        "scenario_statuses": scenario_statuses,
        "all_scenarios_pass": all_scenarios_pass,
    }
