"""STORMFUSION Scientific Evaluation & Baseline Framework Package (Step 20)."""

from evaluation.metrics import (
    haversine_distance_km,
    compute_track_position_errors,
    compute_regression_metrics,
)
from evaluation.baselines import (
    PersistenceBaseline,
    LinearExtrapolationBaseline,
)
from evaluation.pipeline import STORMFUSIONEvaluator

__all__ = [
    "haversine_distance_km",
    "compute_track_position_errors",
    "compute_regression_metrics",
    "PersistenceBaseline",
    "LinearExtrapolationBaseline",
    "STORMFUSIONEvaluator",
]
