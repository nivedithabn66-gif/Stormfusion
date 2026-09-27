"""ERA5 Test Data Download and Track Alignment Script for STORMFUSION.

Constructs CDS API requests for Cyclone FANI (IBTrACS 2019116N02090),
aligns ERA5 3-hourly timestamps with track observations,
executes download via cdsapi if credentials exist, and generates era5_test_manifest.csv.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import yaml


def load_config(config_path: Path = Path("configs/era5_config.yaml")) -> dict:
    """Load ERA5 YAML configuration."""
    project_root = Path(__file__).resolve().parent.parent
    abs_config_path = project_root / config_path
    if not abs_config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {abs_config_path}")
    with open(abs_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_era5_test(config_path: Path = Path("configs/era5_config.yaml")):
    """Align ERA5 timesteps with IBTrACS FANI track and download test dataset via CDS API."""
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(config_path)

    # 1. Load Track Data for Cyclone FANI
    tracks_path = project_root / "data/processed/tracks/ibtracs_nio_clean.csv"
    catalog_path = project_root / "data/processed/tracks/storm_catalog.csv"

    if not (tracks_path.exists() and catalog_path.exists()):
        raise FileNotFoundError("Step 2 track data not found. Run ibtracs_processor.py first.")

    df_tracks = pd.read_csv(tracks_path)
    storm_id = config["alignment"]["storm_id"]
    
    fani_obs = df_tracks[df_tracks["storm_id"] == storm_id].copy()
    if fani_obs.empty:
        raise ValueError(f"Target storm ID {storm_id} not found in ibtracs_nio_clean.csv.")

    fani_obs["datetime"] = pd.to_datetime(fani_obs["timestamp"])
    fani_obs = fani_obs.sort_values(by="datetime").reset_index(drop=True)

    # 2. Select 12 consecutive 3-hourly timesteps around peak intensity window
    peak_idx = fani_obs["wind"].idxmax() if fani_obs["wind"].notna().any() else len(fani_obs) // 2
    start_idx = max(0, peak_idx - 6)
    end_idx = min(len(fani_obs), start_idx + 12)
    selected_obs = fani_obs.iloc[start_idx:end_idx].copy()

    manifest_rows = []
    dates_needed = set()
    times_needed = set()

    for _, row in selected_obs.iterrows():
        dt = row["datetime"]
        # Round to 3-hourly grid (00, 03, 06, 09, 12, 15, 18, 21)
        era5_dt = dt.floor("3h")
        diff_hours = abs((dt - era5_dt).total_seconds()) / 3600.0

        date_str = era5_dt.strftime("%Y-%m-%d")
        time_str = era5_dt.strftime("%H:00")
        
        dates_needed.add(date_str)
        times_needed.add(time_str)

        manifest_rows.append({
            "storm_id": storm_id,
            "storm_name": row["storm_name"],
            "timestamp": era5_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "ibtracs_timestamp": row["timestamp"],
            "track_lat": row["latitude"],
            "track_lon": row["longitude"],
            "max_wind_kts": row["wind"],
            "time_diff_hours": diff_hours,
            "era5_file": config["storage"]["raw_filename"],
            "north": config["region"]["north"],
            "south": config["region"]["south"],
            "west": config["region"]["west"],
            "east": config["region"]["east"],
            "download_status": "PENDING"
        })

    df_manifest = pd.DataFrame(manifest_rows)
    processed_dir = project_root / config["storage"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = processed_dir / config["storage"]["manifest_filename"]
    
    print("=" * 70)
    print("         STORMFUSION — ERA5 TEST DATA ALIGNMENT & DOWNLOAD           ")
    print("=" * 70)
    print(f"Target Storm           : {config['alignment']['storm_name']} ({storm_id})")
    print(f"Selected Test Timesteps: {len(df_manifest)} (3-hourly intervals)")
    print(f"Date Range             : {min(dates_needed)} to {max(dates_needed)}")
    print(f"Target Bounding Box    : [{config['region']['north']}°N, {config['region']['west']}°E, {config['region']['south']}°N, {config['region']['east']}°E]")

    # 3. Check CDS Credentials
    cds_key = os.environ.get("CDS_API_KEY")
    cdsapirc_path = Path.home() / ".cdsapirc"
    has_credentials = bool(cds_key or cdsapirc_path.exists())

    raw_dir = project_root / config["storage"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file_path = raw_dir / config["storage"]["raw_filename"]

    if raw_file_path.exists() and raw_file_path.stat().st_size > 0:
        print(f"\n[INFO] Real ERA5 NetCDF file already exists at: {raw_file_path}")
        print(f"       File Size: {raw_file_path.stat().st_size / (1024 * 1024):.2f} MB")
        df_manifest["download_status"] = "SUCCESS_EXISTING"
        df_manifest.to_csv(manifest_path, index=False)
        return raw_file_path

    if not has_credentials:
        print("\n[AUTHENTICATION NOTICE] Copernicus CDS API key not detected in environment.")
        print("                         Skipping live download to adhere to Data Policy.")
        print("                         NO fake/synthetic files will be generated.")
        print("\n  To enable official ERA5 download:")
        print("  1. Obtain CDS API key from https://cds.climate.copernicus.eu/")
        print("  2. Set CDS_API_KEY in .env or create ~/.cdsapirc")
        print("  3. Re-run python scripts/download_era5_test.py")

        df_manifest["download_status"] = "PENDING_CDS_CREDENTIALS"
        df_manifest.to_csv(manifest_path, index=False)
        print(f"\n[SUCCESS] Manifest saved to: {manifest_path}")
        return None

    # 4. Execute Live CDS API Request
    try:
        import cdsapi
        print("\n[INFO] Initializing Copernicus CDS API Client...")
        c = cdsapi.Client()
        
        request_params = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
                "2m_temperature",
                "total_column_water_vapour",
            ],
            "year": sorted(list(set([d.split("-")[0] for d in dates_needed]))),
            "month": sorted(list(set([d.split("-")[1] for d in dates_needed]))),
            "day": sorted(list(set([d.split("-")[2] for d in dates_needed]))),
            "time": sorted(list(times_needed)),
            "area": [
                config["region"]["north"],
                config["region"]["west"],
                config["region"]["south"],
                config["region"]["east"],
            ],
        }

        print(f"[INFO] Sending request to Copernicus CDS for dataset '{config['dataset']['dataset_id']}'...")
        c.retrieve(config["dataset"]["dataset_id"], request_params, str(raw_file_path))
        
        print(f"[SUCCESS] Real ERA5 NetCDF downloaded to: {raw_file_path} ({raw_file_path.stat().st_size / (1024*1024):.2f} MB)")
        df_manifest["download_status"] = "SUCCESS_DOWNLOADED"
        df_manifest.to_csv(manifest_path, index=False)
        return raw_file_path

    except Exception as e:
        print(f"\n[ERROR] CDS Download failed: {e}")
        df_manifest["download_status"] = f"FAILED: {e}"
        df_manifest.to_csv(manifest_path, index=False)
        return None


if __name__ == "__main__":
    download_era5_test()
