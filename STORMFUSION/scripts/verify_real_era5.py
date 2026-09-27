"""Script to inspect raw ERA5 data under data/raw/era5/ and export era5_provenance.json."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.era5_provenance import verify_era5_directory


def main():
    raw_era5_dir = PROJECT_ROOT / "data" / "raw" / "era5"
    output_manifest = PROJECT_ROOT / "data" / "processed" / "era5_provenance.json"

    res = verify_era5_directory(raw_era5_dir, output_manifest_path=output_manifest)

    print("==================================================")
    print("ERA5 REAL DATA VERIFICATION")
    print("==================================================")
    print(f"Verified status: {res['verified']}")
    print(f"Verified real files: {res['verified_files']}")
    print(f"Total inspected files: {res['total_files_inspected']}")
    print(f"Exported provenance manifest to: {output_manifest}")
    print("==================================================")


if __name__ == "__main__":
    main()
