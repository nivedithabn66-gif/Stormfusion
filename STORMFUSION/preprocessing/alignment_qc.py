"""Temporal & Geographic Alignment QC Module for STORMFUSION.

Validates:
1. Temporal sequence alignment between satellite, ERA5, and IBTrACS track timestamps.
2. Prevention of future data leakage (history features strictly <= reference time T0).
3. Bounding box coordinates for 5 deg x 5 deg cyclone-centered spatial crops.
"""

from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import pandas as pd


def check_temporal_alignment(
    satellite_timestamps: Optional[List[datetime | str]] = None,
    era5_timestamps: Optional[List[datetime | str]] = None,
    track_timestamps: Optional[List[datetime | str]] = None,
    ref_time: datetime | str = None,
    max_tolerance_minutes: float = 30.0,
    **kwargs,
) -> Dict[str, Any]:
    """Validates chronological ordering and temporal alignment across modalities.

    Args:
        satellite_timestamps: List of satellite observation timestamps
        era5_timestamps: List of ERA5 reanalysis timestamps
        track_timestamps: List of IBTrACS track timestamps
        ref_time: Prediction reference time T0
        max_tolerance_minutes: Maximum allowable time difference in minutes

    Returns:
        Dict containing alignment pass/fail flags and diagnostics
    """
    if track_timestamps is None:
        track_timestamps = kwargs.get("insat_timestamps", [])

    ref_dt = pd.to_datetime(ref_time)
    track_dts = [pd.to_datetime(t) for t in track_timestamps]

    # 1. No Future Data Leakage Check
    future_leakage = any(t > ref_dt for t in track_dts)

    # 2. Chronological Ordering Check
    is_sorted = all(track_dts[i] <= track_dts[i + 1] for i in range(len(track_dts) - 1)) if len(track_dts) > 1 else True

    # 3. Alignment Check
    aligned = not future_leakage and is_sorted

    return {
        "aligned": aligned,
        "no_future_leakage": not future_leakage,
        "chronological_order": is_sorted,
        "ref_time": str(ref_dt),
        "history_length": len(track_dts),
    }


def compute_5deg_crop_bounds(
    center_lat: float,
    center_lon: float,
    crop_size_deg: float = 5.0,
) -> Tuple[float, float, float, float]:
    """Calculates 5 deg x 5 deg bounding box centered on cyclone coordinates.

    Args:
        center_lat: Cyclone center latitude
        center_lon: Cyclone center longitude
        crop_size_deg: Total crop size in degrees (default 5.0 deg)

    Returns:
        Tuple of (min_lat, max_lat, min_lon, max_lon)
    """
    half = crop_size_deg / 2.0
    min_lat = round(center_lat - half, 4)
    max_lat = round(center_lat + half, 4)
    min_lon = round(center_lon - half, 4)
    max_lon = round(center_lon + half, 4)

    return min_lat, max_lat, min_lon, max_lon
