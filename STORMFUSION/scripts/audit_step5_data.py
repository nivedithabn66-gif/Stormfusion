"""Step 5 Data Audit Script for STORMFUSION.

Audits IBTrACS, INSAT, and ERA5 raw and processed data stores.
Verifies data provenance, record counts, and checks for synthetic/test schema files.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import h5py
import numpy as np

def audit_data():
    project_root = Path(__file__).resolve().parent.parent

    # ----------------------------------------------------
    # 1. IBTrACS Data Audit
    # ----------------------------------------------------
    ibtracs_raw_dir = project_root / "data/raw/ibtracs"
    ibtracs_raw_files = list(ibtracs_raw_dir.glob("*.nc")) if ibtracs_raw_dir.exists() else []

    tracks_clean_file = project_root / "data/processed/tracks/ibtracs_nio_clean.csv"
    catalog_file = project_root / "data/processed/tracks/storm_catalog.csv"

    ibtracs_record_count = 0
    ibtracs_storm_count = 0

    if tracks_clean_file.exists():
        df_tracks = pd.read_csv(tracks_clean_file)
        ibtracs_record_count = len(df_tracks)
    
    if catalog_file.exists():
        df_cat = pd.read_csv(catalog_file)
        ibtracs_storm_count = len(df_cat)

    ibtracs_status_str = (
        f"files: {len(ibtracs_raw_files)} raw ({', '.join(f.name for f in ibtracs_raw_files)}), "
        f"2 processed (ibtracs_nio_clean.csv, storm_catalog.csv)\n"
        f"records: {ibtracs_record_count}\n"
        f"storms: {ibtracs_storm_count}"
    )

    # ----------------------------------------------------
    # 2. INSAT Data Audit
    # ----------------------------------------------------
    insat_raw_dir = project_root / "data/raw/insat"
    insat_h5_files = []
    if insat_raw_dir.exists():
        insat_h5_files = list(insat_raw_dir.rglob("*.h5"))

    genuine_count = 0
    unverified_count = 0
    synthetic_suspected_count = 0

    # Check test_manifest.csv if present
    test_manifest_file = project_root / "data/processed/satellite/test_manifest.csv"
    sample_filenames = set()
    if test_manifest_file.exists():
        try:
            df_manifest = pd.read_csv(test_manifest_file)
            sample_rows = df_manifest[df_manifest["download_status"].str.contains("SAMPLE", na=False)]
            sample_filenames = set(sample_rows["filename"].tolist())
        except Exception:
            pass

    for file_path in insat_h5_files:
        is_synthetic = False
        if file_path.name in sample_filenames:
            is_synthetic = True
        else:
            # Inspect internal structure to check for synthetic pattern
            try:
                with h5py.File(file_path, "r") as f:
                    prod_name = str(f.attrs.get("Product_Name", ""))
                    # Known synthetic generator signature in download_insat_test.py
                    if "03MAY2019_0600" in prod_name and file_path.name != "3DIMG_03MAY2019_0600_L1B_STD.h5":
                        is_synthetic = True
                    elif "IMG_TIR1" in f:
                        data = f["IMG_TIR1"][()]
                        # Check uniform random integer characteristics or 500x500 dimension template
                        if data.shape == (500, 500) and np.min(data) >= 100 and np.max(data) <= 1024:
                            is_synthetic = True
            except Exception as e:
                unverified_count += 1
                continue

        if is_synthetic:
            synthetic_suspected_count += 1
        else:
            # Cannot guarantee MOSDAC digital signature / cryptographic provenance
            unverified_count += 1

    insat_status_str = (
        f"files: {len(insat_h5_files)}\n"
        f"genuine/verified: {genuine_count}\n"
        f"unverified: {unverified_count}\n"
        f"synthetic/test-schema suspected: {synthetic_suspected_count}"
    )

    # ----------------------------------------------------
    # 3. ERA5 Data Audit
    # ----------------------------------------------------
    era5_raw_dir = project_root / "data/raw/era5"
    era5_files = list(era5_raw_dir.glob("*.nc")) if era5_raw_dir.exists() else []
    era5_available = "YES" if len(era5_files) > 0 else "NO"
    era5_reason = "Available locally" if era5_available == "YES" else "CDS authentication unavailable"

    # ----------------------------------------------------
    # Print Formal Audit Report
    # ----------------------------------------------------
    print("==================================================")
    print("## DATA AUDIT")
    print("==================================================")
    print("\nIBTrACS:")
    print(ibtracs_status_str)
    print("\nINSAT:")
    print(insat_status_str)
    print("\nERA5:")
    print(f"available: {era5_available}")
    print(f"reason: {era5_reason}")
    print("==================================================")

    if genuine_count == 0:
        print("\nUNVERIFIED — DO NOT USE FOR TRAINING")
        print("Reason: All existing INSAT files are synthetic/test schema files or lack verified MOSDAC provenance.\n")

    return {
        "ibtracs": {
            "raw_files_count": len(ibtracs_raw_files),
            "records": ibtracs_record_count,
            "storms": ibtracs_storm_count
        },
        "insat": {
            "total_files": len(insat_h5_files),
            "genuine_verified": genuine_count,
            "unverified": unverified_count,
            "synthetic_suspected": synthetic_suspected_count,
            "provenance_status": "UNVERIFIED" if genuine_count == 0 else "VERIFIED"
        },
        "era5": {
            "available": era5_available,
            "reason": era5_reason
        }
    }

if __name__ == "__main__":
    audit_data()
