"""ERA5 File Inspection Runner Script for STORMFUSION.

Locates downloaded ERA5 test product files and runs structure inspection & physical QC.
"""

from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preprocessing.era5_inspector import ERA5Inspector


def run_file_inspection():
    """Locate downloaded ERA5 NetCDF test file and run inspection."""
    inspector = ERA5Inspector()
    raw_dir = project_root / "data/raw/era5"

    nc_files = list(raw_dir.glob("*.nc"))
    if not nc_files:
        print("=" * 70)
        print("              ERA5 FILE INSPECTION STATUS REPORT                     ")
        print("=" * 70)
        print(f"[NOTICE] No NetCDF (.nc) ERA5 files found in: {raw_dir}")
        print("         Copernicus CDS credentials are required for live download.")
        print("         Adhering to DATA POLICY: NO synthetic/fake data has been generated.")
        print("\nTo inspect a real ERA5 dataset:")
        print("  1. Configure CDS_API_KEY in .env or ~/.cdsapirc")
        print("  2. Execute: python scripts/download_era5_test.py")
        print("  3. Execute: python scripts/inspect_era5_file.py")
        print("=" * 70)
        return

    test_file = nc_files[0]
    print(f"[INFO] Inspecting ERA5 dataset file: {test_file.name}")
    inspector.inspect_file(test_file)


if __name__ == "__main__":
    run_file_inspection()
