"""NIO Basin Verification Script for STORMFUSION.

Independently inspects raw IBTrACS v04 NetCDF dataset and processed tracks
to verify basin codes, storm counts per basin, and confirm that only North Indian Ocean (NI)
storms exist in the processed dataset.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


def verify_nio_basin():
    project_root = Path(__file__).resolve().parent.parent

    raw_nc_path = project_root / "data/raw/ibtracs/IBTrACS.ALL.v04r01.nc"
    clean_csv_path = project_root / "data/processed/tracks/ibtracs_nio_clean.csv"
    catalog_csv_path = project_root / "data/processed/tracks/storm_catalog.csv"

    print("==================================================")
    print("        NIO BASIN VERIFICATION AUDIT            ")
    print("==================================================")

    if not raw_nc_path.exists():
        print(f"[ERROR] Raw IBTrACS NetCDF file not found at: {raw_nc_path}")
        return

    # 1. Load Raw NetCDF
    ds = xr.open_dataset(raw_nc_path, decode_times=False)
    raw_storms_count = len(ds["sid"])

    # Extract basin array (storm, time)
    basin_var = ds["basin"].values
    num_storms, max_times = basin_var.shape

    # Determine unique basins and storms per basin BEFORE filtering
    storm_basins = {}
    basin_storm_counts = {}
    basin_obs_counts = {}

    for i in range(num_storms):
        s_basins = set()
        for t in range(max_times):
            b_val = basin_var[i, t]
            if b_val.ndim > 0:
                b_str = "".join([c.decode("utf-8", errors="ignore") for c in b_val]).strip()
            elif isinstance(b_val, (bytes, np.bytes_)):
                b_str = b_val.decode("utf-8", errors="ignore").strip()
            else:
                b_str = str(b_val).strip()

            if b_str:
                s_basins.add(b_str)
                basin_obs_counts[b_str] = basin_obs_counts.get(b_str, 0) + 1

        for b in s_basins:
            basin_storm_counts[b] = basin_storm_counts.get(b, 0) + 1

    ds.close()

    raw_unique_basins = sorted(list(basin_storm_counts.keys()))

    print("\n1. UNIQUE BASINS BEFORE FILTERING (Raw IBTrACS):")
    print(f"   {raw_unique_basins}")

    print("\n2. STORMS PER BASIN BEFORE FILTERING (Raw IBTrACS):")
    for b in raw_unique_basins:
        print(f"   • Basin '{b}': {basin_storm_counts[b]} storms ({basin_obs_counts.get(b, 0)} observations)")

    # 3. Inspect Processed Clean Track CSV
    if not clean_csv_path.exists() or not catalog_csv_path.exists():
        print("[ERROR] Clean tracks or catalog CSV missing.")
        return

    df_clean = pd.read_csv(clean_csv_path)
    df_cat = pd.read_csv(catalog_csv_path)

    clean_unique_basins = sorted(df_clean["basin"].unique().tolist())
    catalog_unique_basins = sorted(df_cat["basin"].unique().tolist())

    processed_nio_obs = len(df_clean)
    processed_nio_storms = len(df_cat)

    print("\n3. UNIQUE BASINS AFTER FILTERING (ibtracs_nio_clean.csv):")
    print(f"   {clean_unique_basins}")

    print("\n4. STORMS RETAINED AFTER FILTERING:")
    print(f"   • Total NIO Storms       : {processed_nio_storms}")
    print(f"   • Total NIO Observations : {processed_nio_obs}")

    print("\n5. EXACT IBTrACS BASIN CODE USED FOR NORTH INDIAN OCEAN:")
    print("   • Primary Basin Code : 'NI' (North Indian Ocean)")

    # 6. Check for non-NIO basin leakage
    other_basins = [b for b in clean_unique_basins if b != "NI"]
    leakage_count = len(other_basins)

    print("\n6. NON-NIO BASIN LEAKAGE CHECK:")
    if leakage_count == 0:
        print("   • CONFIRMED: 0 non-NIO basin storms remain (SI, WP, EP, NA, SA, SP strictly excluded).")
    else:
        print(f"   • WARNING: Non-NIO basin codes detected in clean data: {other_basins}")

    print("\n7. DATASET COUNT COMPARISON FLOW:")
    print(f"   • Raw IBTrACS Global Storm Count : {raw_storms_count}")
    print(f"   • Raw 'NI' Basin Storm Count    : {basin_storm_counts.get('NI', 0)}")
    print(f"   • Processed NIO Storm Catalog   : {processed_nio_storms}")

    print("==================================================")
    is_valid = (leakage_count == 0) and (processed_nio_storms == basin_storm_counts.get("NI", 0))
    print(f"NIO BASIN VERIFICATION STATUS: {'PASS' if is_valid else 'FAIL'}")
    print("==================================================")


if __name__ == "__main__":
    verify_nio_basin()
