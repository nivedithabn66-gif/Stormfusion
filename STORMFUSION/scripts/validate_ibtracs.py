"""IBTrACS Validation Script for STORMFUSION.

Validates processed North Indian Ocean (NIO) cyclone tracks and storm catalog.
Reports dataset statistics, missing value counts, top intense storms, and seasonal distributions.
"""

import json
from pathlib import Path
import pandas as pd
import yaml


def validate_ibtracs(config_path: Path = Path("configs/data_config.yaml")):
    """Run validation checks on processed IBTrACS NIO data."""
    project_root = Path(__file__).resolve().parent.parent
    abs_config_path = project_root / config_path

    with open(abs_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ibtracs_cfg = config["ibtracs"]
    processed_dir = project_root / ibtracs_cfg["processed_dir"]

    catalog_path = processed_dir / ibtracs_cfg["catalog_filename"]
    tracks_path = processed_dir / ibtracs_cfg["clean_tracks_filename"]
    metadata_path = processed_dir / ibtracs_cfg["metadata_filename"]

    if not catalog_path.exists() or not tracks_path.exists():
        raise FileNotFoundError(
            f"Processed data files not found in {processed_dir}. Run ibtracs_processor.py first."
        )

    df_catalog = pd.read_csv(catalog_path)
    df_tracks = pd.read_csv(tracks_path)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    print("=" * 70)
    print("           STORMFUSION — IBTrACS NIO DATA VALIDATION REPORT          ")
    print("=" * 70)

    # 1 & 2: Observations and Storms Count
    total_global_obs = metadata.get("total_global_observations", "N/A")
    total_global_storms = metadata.get("total_global_storms", "N/A")
    nio_obs = len(df_tracks)
    nio_storms = len(df_catalog)

    print(f"1. Total Global Observations (Parsed) : {total_global_obs}")
    print(f"2. Total Global Storms (Parsed)       : {total_global_storms}")
    print(f"3. North Indian Ocean (NIO) Obs       : {nio_obs}")
    print(f"4. North Indian Ocean (NIO) Storms    : {nio_storms}")

    # 5. Date Range
    start_dates = pd.to_datetime(df_catalog["start_time"])
    end_dates = pd.to_datetime(df_catalog["end_time"])
    min_date = start_dates.min().strftime("%Y-%m-%d %H:%M") if not start_dates.empty else "N/A"
    max_date = end_dates.max().strftime("%Y-%m-%d %H:%M") if not end_dates.empty else "N/A"
    print(f"5. NIO Date Range                     : {min_date}  to  {max_date}")

    print("-" * 70)
    # 6-9: Missing Value Statistics
    missing_lat = df_tracks["latitude"].isna().sum()
    missing_lon = df_tracks["longitude"].isna().sum()
    missing_wind = df_tracks["wind"].isna().sum()
    missing_pres = df_tracks["pressure"].isna().sum()

    print("MISSING DATA METRICS (NIO Tracks):")
    print(f"  * Missing Latitude Count   : {missing_lat} ({missing_lat/nio_obs*100:.2f}%)")
    print(f"  * Missing Longitude Count  : {missing_lon} ({missing_lon/nio_obs*100:.2f}%)")
    print(f"  * Missing Wind Count       : {missing_wind} ({missing_wind/nio_obs*100:.2f}%)")
    print(f"  * Missing Pressure Count   : {missing_pres} ({missing_pres/nio_obs*100:.2f}%)")

    print("-" * 70)
    # 10. Observations per storm stats
    obs_counts = df_catalog["observation_count"]
    print("OBSERVATIONS PER STORM STATISTICS:")
    print(f"  * Min Obs per Storm  : {obs_counts.min()}")
    print(f"  * Max Obs per Storm  : {obs_counts.max()}")
    print(f"  * Mean Obs per Storm : {obs_counts.mean():.2f}")
    print(f"  * Median Obs         : {obs_counts.median():.1f}")

    print("-" * 70)
    # 11. Top 10 storms by maximum wind speed
    top_wind_storms = df_catalog.sort_values(by="max_wind", ascending=False).head(10)
    print("TOP 10 NIO STORMS BY MAXIMUM SUSTAINED WIND (Knots):")
    print(f"{'Storm ID':<15} {'Storm Name':<16} {'Season':<8} {'Max Wind (kts)':<15} {'Min Pres (hPa)':<15}")
    print("-" * 70)
    for _, row in top_wind_storms.iterrows():
        sid = str(row["storm_id"])
        sname = str(row["storm_name"])
        season = str(int(row["season"])) if pd.notna(row["season"]) else "N/A"
        mwind = f"{row['max_wind']:.1f}" if pd.notna(row["max_wind"]) else "N/A"
        mpres = f"{row['min_pressure']:.1f}" if pd.notna(row["min_pressure"]) else "N/A"
        print(f"{sid:<15} {sname:<16} {season:<8} {mwind:<15} {mpres:<15}")

    print("-" * 70)
    # 12. Number of storms per season (recent 15 seasons summary)
    season_counts = df_catalog["season"].value_counts().sort_index(ascending=False)
    print("STORMS PER SEASON (Recent 15 Seasons):")
    for season, count in season_counts.head(15).items():
        s_int = int(season) if pd.notna(season) else "N/A"
        print(f"  * Season {s_int}: {count} storms")

    print("=" * 70)
    return {
        "nio_obs": nio_obs,
        "nio_storms": nio_storms,
        "min_date": min_date,
        "max_date": max_date,
        "missing_wind": missing_wind,
        "missing_pres": missing_pres
    }


if __name__ == "__main__":
    validate_ibtracs()
