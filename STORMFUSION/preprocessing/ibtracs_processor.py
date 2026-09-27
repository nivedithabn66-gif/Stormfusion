"""IBTrACS NetCDF Processor for STORMFUSION.

Parses official IBTrACS v04 NetCDF data, inspects schema and basin codes,
filters for North Indian Ocean (NIO) storms, applies quality control,
and exports clean observation tracks and a storm catalog CSV.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import xarray as xr
import yaml


class IBTrACSProcessor:
    """Processor for IBTrACS v04 NetCDF tropical cyclone dataset."""

    def __init__(self, config_path: Path = Path("configs/data_config.yaml")):
        self.project_root = Path(__file__).resolve().parent.parent
        self.config_path = self.project_root / config_path
        self.config = self._load_config()

        ibtracs_cfg = self.config["ibtracs"]
        self.raw_path = self.project_root / ibtracs_cfg["raw_dir"] / ibtracs_cfg["raw_filename"]
        self.processed_dir = self.project_root / ibtracs_cfg["processed_dir"]
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.catalog_path = self.processed_dir / ibtracs_cfg["catalog_filename"]
        self.clean_tracks_path = self.processed_dir / ibtracs_cfg["clean_tracks_filename"]
        self.metadata_path = self.processed_dir / ibtracs_cfg["metadata_filename"]

        self.nio_basin_codes = [b.encode("utf-8") if isinstance(b, str) else b for b in ibtracs_cfg["nio_basins"]]
        self.qc_limits = ibtracs_cfg["qc"]

    def _load_config(self) -> dict:
        """Load configuration YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _decode_bytes_array(self, arr: np.ndarray) -> np.ndarray:
        """Helper to convert byte character arrays to Python strings safely."""
        if arr.dtype.kind in ["S", "V", "O"]:
            try:
                # Handle 2D character arrays (e.g., char arrays representing strings)
                if arr.ndim == 2 and arr.dtype.kind in ["S", "V"]:
                    return np.array(["".join([c.decode("utf-8", errors="ignore") for c in row]).strip() for row in arr])
                elif arr.ndim == 1:
                    return np.array([
                        elem.decode("utf-8", errors="ignore").strip() if isinstance(elem, bytes) else str(elem).strip()
                        for elem in arr
                    ])
            except Exception:
                pass
        return arr

    def load_dataset(self) -> xr.Dataset:
        """Load NetCDF dataset and inspect variable structures."""
        if not self.raw_path.exists():
            raise FileNotFoundError(f"IBTrACS raw file not found at {self.raw_path}. Run download_ibtracs.py first.")

        print(f"[INFO] Loading IBTrACS NetCDF from: {self.raw_path}")
        ds = xr.open_dataset(self.raw_path, decode_times=False)
        print(f"[INFO] NetCDF Dimensions: {dict(ds.dims)}")
        print(f"[INFO] NetCDF Data Variables: {list(ds.data_vars.keys())[:15]} ... (total: {len(ds.data_vars)})")
        return ds

    def inspect_basins(self, ds: xr.Dataset) -> List[str]:
        """Inspect and return all unique basin codes present in the dataset."""
        if "basin" not in ds.data_vars:
            raise KeyError("Expected 'basin' variable not found in IBTrACS dataset.")

        raw_basin = ds["basin"].values
        # Flatten and decode byte strings
        flat_basin = raw_basin.flatten()
        decoded_basins = set()
        for b in flat_basin:
            if isinstance(b, bytes):
                decoded = b.decode("utf-8", errors="ignore").strip()
            elif isinstance(b, np.bytes_):
                decoded = b.tobytes().decode("utf-8", errors="ignore").strip()
            else:
                decoded = str(b).strip()
            if decoded:
                decoded_basins.add(decoded)

        basin_list = sorted(list(decoded_basins))
        print(f"[INFO] Discovered unique Basin codes in IBTrACS: {basin_list}")
        
        # Check subbasin variable if present
        if "subbasin" in ds.data_vars:
            raw_subbasin = ds["subbasin"].values
            flat_subbasin = raw_subbasin.flatten()
            decoded_subbasins = set()
            for sb in flat_subbasin:
                if isinstance(sb, (bytes, np.bytes_)):
                    decoded = sb.decode("utf-8", errors="ignore").strip() if isinstance(sb, bytes) else sb.tobytes().decode("utf-8", errors="ignore").strip()
                else:
                    decoded = str(sb).strip()
                if decoded:
                    decoded_subbasins.add(decoded)
            print(f"[INFO] Discovered unique Sub-basin codes in IBTrACS: {sorted(list(decoded_subbasins))}")

        return basin_list

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Process IBTrACS dataset to extract NIO observations and build storm catalog."""
        ds = self.load_dataset()
        discovered_basins = self.inspect_basins(ds)

        # Validate NIO basin availability
        nio_targets = set(self.config["ibtracs"]["nio_basins"])
        matched_basins = [b for b in discovered_basins if b in nio_targets]
        print(f"[INFO] Target NIO Basins configured: {sorted(list(nio_targets))}")
        print(f"[INFO] Matched NIO Basins in dataset: {matched_basins}")

        if not matched_basins:
            print(f"[WARNING] No direct match for configured NIO basins {nio_targets} in discovered basins {discovered_basins}.")

        # Extract storm level variables
        sid_raw = ds["sid"].values # (storm,) or (storm, char)
        name_raw = ds["name"].values # (storm,) or (storm, char)
        season_raw = ds["season"].values # (storm,)

        # Convert sid/name
        if sid_raw.ndim == 2:
            sids = np.array(["".join([c.decode("utf-8", errors="ignore") for c in row]).strip() for row in sid_raw])
        else:
            sids = np.array([s.decode("utf-8", errors="ignore").strip() if isinstance(s, bytes) else str(s).strip() for s in sid_raw])

        if name_raw.ndim == 2:
            names = np.array(["".join([c.decode("utf-8", errors="ignore") for c in row]).strip() for row in name_raw])
        else:
            names = np.array([n.decode("utf-8", errors="ignore").strip() if isinstance(n, bytes) else str(n).strip() for n in name_raw])

        seasons = season_raw

        # Extract 2D track variables (storm, date_time)
        time_raw = ds["iso_time"].values # (storm, date_time, char)
        lat_raw = ds["lat"].values # (storm, date_time)
        lon_raw = ds["lon"].values # (storm, date_time)
        basin_raw = ds["basin"].values # (storm, date_time, char) or (storm, date_time)

        # Wind & Pressure: Check for usa_wind/wmo_wind and usa_pres/wmo_pres
        if "usa_wind" in ds.data_vars:
            wind_raw = ds["usa_wind"].values
        elif "wmo_wind" in ds.data_vars:
            wind_raw = ds["wmo_wind"].values
        else:
            wind_raw = np.full_like(lat_raw, np.nan)

        if "usa_pres" in ds.data_vars:
            pres_raw = ds["usa_pres"].values
        elif "wmo_pres" in ds.data_vars:
            pres_raw = ds["wmo_pres"].values
        else:
            pres_raw = np.full_like(lat_raw, np.nan)

        if "nature" in ds.data_vars:
            nature_raw = ds["nature"].values
        else:
            nature_raw = None

        num_storms, max_times = lat_raw.shape
        print(f"[INFO] Extracting track records across {num_storms} total storms ({max_times} max time slots per storm)...")

        records = []
        total_obs = 0

        for i in range(num_storms):
            storm_id = sids[i]
            storm_name = names[i]
            season = int(seasons[i]) if not np.isnan(seasons[i]) else None

            for t in range(max_times):
                lat = float(lat_raw[i, t])
                lon = float(lon_raw[i, t])

                # Skip completely empty/masked time slots
                if np.isnan(lat) and np.isnan(lon):
                    continue

                total_obs += 1

                # Decode ISO time string
                raw_time_slot = time_raw[i, t]
                if raw_time_slot.ndim > 0:
                    time_str = "".join([c.decode("utf-8", errors="ignore") for c in raw_time_slot]).strip()
                elif isinstance(raw_time_slot, (bytes, np.bytes_)):
                    time_str = raw_time_slot.decode("utf-8", errors="ignore").strip()
                else:
                    time_str = str(raw_time_slot).strip()

                if not time_str or time_str == "nan":
                    continue

                # Decode basin string for this observation
                raw_b_slot = basin_raw[i, t]
                if raw_b_slot.ndim > 0:
                    b_str = "".join([c.decode("utf-8", errors="ignore") for c in raw_b_slot]).strip()
                elif isinstance(raw_b_slot, (bytes, np.bytes_)):
                    b_str = raw_b_slot.decode("utf-8", errors="ignore").strip()
                else:
                    b_str = str(raw_b_slot).strip()

                # Decode nature if present
                nature_str = ""
                if nature_raw is not None:
                    raw_nat_slot = nature_raw[i, t]
                    if raw_nat_slot.ndim > 0:
                        nature_str = "".join([c.decode("utf-8", errors="ignore") for c in raw_nat_slot]).strip()
                    elif isinstance(raw_nat_slot, (bytes, np.bytes_)):
                        nature_str = raw_nat_slot.decode("utf-8", errors="ignore").strip()
                    else:
                        nature_str = str(raw_nat_slot).strip()

                wind = float(wind_raw[i, t]) if not np.isnan(wind_raw[i, t]) else np.nan
                pres = float(pres_raw[i, t]) if not np.isnan(pres_raw[i, t]) else np.nan

                records.append({
                    "storm_id": storm_id,
                    "storm_name": storm_name,
                    "season": season,
                    "timestamp": time_str,
                    "latitude": lat,
                    "longitude": lon,
                    "wind": wind,
                    "pressure": pres,
                    "basin": b_str,
                    "nature": nature_str
                })

        df_all = pd.DataFrame(records)
        print(f"[INFO] Total observations parsed across all global basins: {len(df_all)}")

        # -------------------------------------------------------------
        # Filtering for North Indian Ocean (NIO)
        # -------------------------------------------------------------
        # Filter rules: basin code matches configured NIO basin target
        df_nio = df_all[df_all["basin"].isin(matched_basins)].copy()
        
        # Also check if any storms have observation-level NIO entries
        nio_storm_ids = set(df_nio["storm_id"].unique())
        print(f"[INFO] Filtered {len(df_nio)} observation records for {len(nio_storm_ids)} NIO storms.")

        # Parse datetime objects for proper sorting and validation
        df_nio["datetime"] = pd.to_datetime(df_nio["timestamp"], errors="coerce")
        
        # Sort chronologically within each storm
        df_nio = df_nio.sort_values(by=["storm_id", "datetime"]).reset_index(drop=True)

        # -------------------------------------------------------------
        # Quality Control (QC)
        # -------------------------------------------------------------
        df_nio["quality_flag"] = "VALID"
        
        # QC rule checks
        invalid_lat = (df_nio["latitude"] < self.qc_limits["lat_min"]) | (df_nio["latitude"] > self.qc_limits["lat_max"]) | df_nio["latitude"].isna()
        invalid_lon = (df_nio["longitude"] < self.qc_limits["lon_min"]) | (df_nio["longitude"] > self.qc_limits["lon_max"]) | df_nio["longitude"].isna()
        invalid_time = df_nio["datetime"].isna()
        
        # Wind bounds check (0 - 250 kts)
        invalid_wind = (~df_nio["wind"].isna()) & ((df_nio["wind"] < self.qc_limits["wind_min_kts"]) | (df_nio["wind"] > self.qc_limits["wind_max_kts"]))
        
        # Pressure bounds check (800 - 1050 hPa)
        invalid_pres = (~df_nio["pressure"].isna()) & ((df_nio["pressure"] < self.qc_limits["pressure_min_hpa"]) | (df_nio["pressure"] > self.qc_limits["pressure_max_hpa"]))

        # Apply QC Flags without silently deleting scientific data
        df_nio.loc[invalid_lat | invalid_lon | invalid_time, "quality_flag"] = "INVALID_COORDS_OR_TIME"
        df_nio.loc[invalid_wind, "quality_flag"] = "INVALID_WIND_VALUE"
        df_nio.loc[invalid_pres, "quality_flag"] = "INVALID_PRES_VALUE"

        # Remove only observations with critical invalid time or coordinates for the clean dataset
        df_clean = df_nio[df_nio["quality_flag"] != "INVALID_COORDS_OR_TIME"].copy()
        df_clean = df_clean.drop(columns=["datetime"])

        # Write clean observations to CSV
        df_clean.to_csv(self.clean_tracks_path, index=False)
        print(f"[SUCCESS] Clean NIO tracks saved to: {self.clean_tracks_path}")

        # -------------------------------------------------------------
        # Build Storm Catalog (one row per storm)
        # -------------------------------------------------------------
        catalog_rows = []
        for sid, group in df_nio.groupby("storm_id", sort=False):
            # Sort group chronologically
            group_sorted = group.sort_values(by="datetime")
            
            sname = group_sorted["storm_name"].iloc[0]
            season = group_sorted["season"].iloc[0]
            basin = group_sorted["basin"].iloc[0]

            valid_group = group_sorted[group_sorted["quality_flag"] != "INVALID_COORDS_OR_TIME"]
            if valid_group.empty:
                continue

            start_time = valid_group["timestamp"].iloc[0]
            end_time = valid_group["timestamp"].iloc[-1]
            obs_count = len(valid_group)

            dt_start = valid_group["datetime"].iloc[0]
            dt_end = valid_group["datetime"].iloc[-1]
            duration_hours = (dt_end - dt_start).total_seconds() / 3600.0 if (pd.notna(dt_start) and pd.notna(dt_end)) else np.nan

            min_lat = float(valid_group["latitude"].min())
            max_lat = float(valid_group["latitude"].max())
            min_lon = float(valid_group["longitude"].min())
            max_lon = float(valid_group["longitude"].max())

            first_lat = float(valid_group["latitude"].iloc[0])
            first_lon = float(valid_group["longitude"].iloc[0])
            last_lat = float(valid_group["latitude"].iloc[-1])
            last_lon = float(valid_group["longitude"].iloc[-1])

            # Max wind & min pressure handling NaNs cleanly
            valid_winds = valid_group["wind"].dropna()
            max_wind = float(valid_winds.max()) if not valid_winds.empty else np.nan

            valid_pressures = valid_group["pressure"].dropna()
            min_pressure = float(valid_pressures.min()) if not valid_pressures.empty else np.nan

            catalog_rows.append({
                "storm_id": sid,
                "storm_name": sname,
                "season": season,
                "basin": basin,
                "start_time": start_time,
                "end_time": end_time,
                "observation_count": obs_count,
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon,
                "first_lat": first_lat,
                "first_lon": first_lon,
                "last_lat": last_lat,
                "last_lon": last_lon,
                "duration_hours": duration_hours,
                "max_wind": max_wind,
                "min_pressure": min_pressure
            })

        df_catalog = pd.DataFrame(catalog_rows)
        # Sort catalog by season and start_time
        df_catalog = df_catalog.sort_values(by=["season", "start_time"]).reset_index(drop=True)
        df_catalog.to_csv(self.catalog_path, index=False)
        print(f"[SUCCESS] Storm Catalog saved to: {self.catalog_path} ({len(df_catalog)} storms)")

        # -------------------------------------------------------------
        # Data Provenance Metadata
        # -------------------------------------------------------------
        metadata = {
            "dataset_name": "IBTrACS (International Best Track Archive for Climate Stewardship)",
            "version": "v04r01",
            "source": "NOAA NCEI",
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_url": self.config["ibtracs"]["url"],
            "total_global_observations": total_obs,
            "total_global_storms": num_storms,
            "nio_observations_count": len(df_clean),
            "nio_storms_count": len(df_catalog),
            "discovered_basins": discovered_basins,
            "nio_target_basins_matched": matched_basins,
            "processing_script": "preprocessing/ibtracs_processor.py",
            "qc_rules": {
                "lat_bounds": [self.qc_limits["lat_min"], self.qc_limits["lat_max"]],
                "lon_bounds": [self.qc_limits["lon_min"], self.qc_limits["lon_max"]],
                "wind_max_kts": self.qc_limits["wind_max_kts"],
                "pressure_bounds_hpa": [self.qc_limits["pressure_min_hpa"], self.qc_limits["pressure_max_hpa"]],
                "missing_values_policy": "Flagged with quality_flag column without silent deletion"
            }
        }

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"[SUCCESS] IBTrACS Metadata saved to: {self.metadata_path}")

        ds.close()
        return df_clean, df_catalog, metadata


if __name__ == "__main__":
    processor = IBTrACSProcessor()
    processor.process()
