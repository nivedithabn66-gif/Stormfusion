"""Track History & Target Feature Extractor for STORMFUSION (Step 7).

Extracts sequence-level track history features (t-21h to t: 8 timesteps) strictly from
observations at or before the reference timestamp t_7.

CRITICAL SAFETY RULE:
  - Future observations (t > t_7) MUST NOT enter input track features.
  - Future track positions, wind, and pressure belong ONLY in target labels (+3h, +6h, +12h, +24h).

Features per timestep k in {0..7}:
  1. latitude (degrees N)
  2. longitude (degrees E)
  3. delta_lat (lat_k - lat_{k-1})
  4. delta_lon (lon_k - lon_{k-1})
  5. motion_speed (knots)
  6. motion_direction (heading angle degrees 0..360)
  7. intensity (max sustained wind in knots)
  8. pressure (minimum central pressure in hPa/mb)
  9. valid_flag (1.0 if observation present, 0.0 if missing)
"""

import math
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def calculate_motion(lat1: float, lon1: float, lat2: float, lon2: float, dt_hours: float = 3.0) -> Tuple[float, float]:
    """Calculates motion speed (knots) and direction (heading degrees 0..360) between two lat/lon points."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2) or dt_hours <= 0:
        return 0.0, 0.0

    # Haversine distance in nautical miles (1 deg lat = 60 nm)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    mean_lat = math.radians((lat1 + lat2) / 2.0)

    dx_nm = dlon * 60.0 * math.cos(mean_lat)
    dy_nm = dlat * 60.0

    dist_nm = math.sqrt(dx_nm**2 + dy_nm**2)
    speed_knots = dist_nm / dt_hours

    # Heading angle clockwise from North (0..360)
    angle_rad = math.atan2(dx_nm, dy_nm)
    heading_deg = (math.degrees(angle_rad)) % 360.0

    return float(speed_knots), float(heading_deg)


class TrackAndTargetLoader:
    """Extracts 8-timestep history features and target labels for a sequence."""

    def __init__(self, project_root: Path = None):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = project_root

        self.tracks_path = self.project_root / "data/processed/tracks/ibtracs_nio_clean.csv"
        if not self.tracks_path.exists():
            raise FileNotFoundError(f"Clean tracks file not found at: {self.tracks_path}")

        df_tracks = pd.read_csv(self.tracks_path)
        df_tracks["dt"] = pd.to_datetime(df_tracks["timestamp"])
        self.df_tracks = df_tracks

        # Fast lookup index: storm_id -> sorted track dataframe
        self.storm_tracks = {}
        for sid, group in df_tracks.groupby("storm_id"):
            self.storm_tracks[sid] = group.sort_values(by="dt").reset_index(drop=True)

    def extract_sequence_track_features(self, storm_id: str, reference_time_str: str) -> Dict[str, Any]:
        """Extracts 8 history timesteps (t_0 to t_7) spaced 3 hours apart.

        Strictly enforces reference_time_str as t_7 and ensures no observations > t_7 enter inputs.
        """
        ref_dt = pd.to_datetime(reference_time_str)
        
        # History timestamps: t_0 = ref_dt - 21h, t_1 = ref_dt - 18h, ..., t_7 = ref_dt
        history_dts = [ref_dt - pd.Timedelta(hours=3 * (7 - k)) for k in range(8)]

        if storm_id not in self.storm_tracks:
            # Storm track not found, return empty placeholder
            return self._get_empty_track_features()

        df_storm = self.storm_tracks[storm_id]

        # Anti-leakage assertion: Filter to observations strictly <= ref_dt
        df_valid_history = df_storm[df_storm["dt"] <= ref_dt]

        # Extract features for each of the 8 history timesteps
        features_list = []
        valid_flags = np.zeros(8, dtype=np.float32)

        prev_lat, prev_lon = None, None

        for k, h_dt in enumerate(history_dts):
            # Assert no future leakage
            assert h_dt <= ref_dt, f"Future leakage error: history timestamp {h_dt} > reference time {ref_dt}"

            # Search for exact observation matching history timestep h_dt (within 15 min tolerance)
            obs_match = df_valid_history[(df_valid_history["dt"] - h_dt).abs() <= pd.Timedelta(minutes=15)]

            if not obs_match.empty:
                row = obs_match.iloc[0]
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                wind = float(row["wind"]) if pd.notna(row["wind"]) else 0.0
                press = float(row["pressure"]) if pd.notna(row["pressure"]) else 0.0
                is_obs_valid = 1.0

                dlat = (lat - prev_lat) if prev_lat is not None else 0.0
                dlon = (lon - prev_lon) if prev_lon is not None else 0.0
                speed, direction = calculate_motion(prev_lat, prev_lon, lat, lon) if prev_lat is not None else (0.0, 0.0)

                prev_lat, prev_lon = lat, lon
            else:
                # Missing history observation
                lat = prev_lat if prev_lat is not None else 0.0
                lon = prev_lon if prev_lon is not None else 0.0
                wind, press = 0.0, 0.0
                dlat, dlon, speed, direction = 0.0, 0.0, 0.0, 0.0
                is_obs_valid = 0.0

            valid_flags[k] = is_obs_valid
            features_list.append([lat, lon, dlat, dlon, speed, direction, wind, press, is_obs_valid])

        track_arr = np.array(features_list, dtype=np.float32) # [8, 9]

        return {
            "track_features": torch.from_numpy(track_arr),
            "track_valid_mask": torch.from_numpy(valid_flags),
        }

    def _get_empty_track_features(self) -> Dict[str, Any]:
        """Returns zero placeholder track features [8, 9] with valid_mask = 0."""
        return {
            "track_features": torch.zeros((8, 9), dtype=torch.float32),
            "track_valid_mask": torch.zeros(8, dtype=torch.float32),
        }

    def extract_targets(self, sequence_row: pd.Series) -> Dict[str, Any]:
        """Extracts targets and target validity masks for lead times (+3h, +6h, +12h, +24h)."""
        ref_lat = float(sequence_row["center_lat"])
        ref_lon = float(sequence_row["center_lon"])

        future_track = []
        future_track_mask = []

        intensity = []
        intensity_mask = []

        pressure = []
        pressure_mask = []

        for h in ["t3", "t6", "t12", "t24"]:
            t_lat = sequence_row.get(f"target_lat_{h}", np.nan)
            t_lon = sequence_row.get(f"target_lon_{h}", np.nan)
            t_wind = sequence_row.get(f"target_wind_{h}", np.nan)
            t_press = sequence_row.get(f"target_pressure_{h}", np.nan)

            # Track delta target
            if pd.notna(t_lat) and pd.notna(t_lon):
                dlat = float(t_lat) - ref_lat
                dlon = float(t_lon) - ref_lon
                future_track.append([dlat, dlon])
                future_track_mask.append(1.0)
            else:
                future_track.append([0.0, 0.0])
                future_track_mask.append(0.0)

            # Intensity (wind) target
            if pd.notna(t_wind):
                intensity.append(float(t_wind))
                intensity_mask.append(1.0)
            else:
                intensity.append(0.0)
                intensity_mask.append(0.0)

            # Pressure target
            if pd.notna(t_press):
                pressure.append(float(t_press))
                pressure_mask.append(1.0)
            else:
                pressure.append(0.0)
                pressure_mask.append(0.0)

        # Detection target: 1.0 for active cyclone sequence
        detection = torch.tensor(1.0, dtype=torch.float32)

        # Pattern target: -1.0 placeholder (categorical class label)
        pattern = torch.tensor(-1, dtype=torch.long)

        return {
            "detection": detection,
            "pattern": pattern,
            "intensity": torch.tensor(intensity, dtype=torch.float32), # [4]
            "intensity_valid_mask": torch.tensor(intensity_mask, dtype=torch.float32), # [4]
            "pressure": torch.tensor(pressure, dtype=torch.float32), # [4]
            "pressure_valid_mask": torch.tensor(pressure_mask, dtype=torch.float32), # [4]
            "future_track": torch.tensor(future_track, dtype=torch.float32), # [4, 2]
            "future_track_valid_mask": torch.tensor(future_track_mask, dtype=torch.float32), # [4]
        }
