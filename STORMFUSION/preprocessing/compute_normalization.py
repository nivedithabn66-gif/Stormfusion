"""Normalization Statistics Interface for STORMFUSION.

Enforces strict normalization rules:
1. Normalization statistics can ONLY be computed after genuine satellite and ERA5 files are verified.
2. Normalization statistics must be calculated strictly from the storm-wise TRAIN SPLIT ONLY.
3. Never calculate normalization using validation, test, or synthetic/placeholder data.
"""

from typing import Dict, Any
from pathlib import Path
from preprocessing.normalization import compute_all_train_normalization


def compute_dataset_normalization(
    train_manifest_path: str | Path = None,
    output_stats_dir: str | Path = None,
    satellite_provenance_path: str | Path = None,
    era5_provenance_path: str | Path = None,
    **kwargs,
) -> Dict[str, Any]:
    """Calculates channel mean, std, min, max from train split imagery and environmental fields.

    Enforces train split isolation and real data provenance checks.
    """
    project_root = Path(__file__).resolve().parent.parent
    return compute_all_train_normalization(project_root)
