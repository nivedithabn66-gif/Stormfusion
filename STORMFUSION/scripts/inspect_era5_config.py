"""ERA5 Configuration and CDS API Environment Inspection Script for STORMFUSION.

Inspects configs/era5_config.yaml, verifies cdsapi client availability, checks
Copernicus CDS credentials, and outputs ERA5 acquisition parameters.
"""

import os
import sys
from pathlib import Path
import yaml


def load_config(config_path: Path = Path("configs/era5_config.yaml")) -> dict:
    """Load ERA5 YAML configuration."""
    project_root = Path(__file__).resolve().parent.parent
    abs_config_path = project_root / config_path
    if not abs_config_path.exists():
        raise FileNotFoundError(f"ERA5 configuration not found at: {abs_config_path}")
    with open(abs_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def inspect_era5_config():
    """Inspect CDS configuration, library availability, and authentication environment."""
    config = load_config()

    print("=" * 70)
    print("         STORMFUSION — ERA5 CONFIGURATION & CDS ENVIRONMENT REPORT    ")
    print("=" * 70)

    # 1. CDS Python API Client Status
    try:
        import cdsapi
        cdsapi_status = f"[INSTALLED] Version {getattr(cdsapi, '__version__', 'available')}"
    except ImportError:
        cdsapi_status = "[NOT INSTALLED] Run 'pip install cdsapi' to enable Copernicus CDS downloads."

    print(f"1. Python cdsapi Client : {cdsapi_status}")

    # 2. Credential Environment Check
    cds_key = os.environ.get("CDS_API_KEY")
    cds_url = os.environ.get("CDS_API_URL")
    cdsapirc_path = Path.home() / ".cdsapirc"

    print("\nCDS CREDENTIAL ENVIRONMENT STATUS:")
    if cds_key:
        print(f"  * CDS_API_KEY   : [PRESENT] Key set in environment variables.")
    else:
        print("  * CDS_API_KEY   : [MISSING] Not found in environment variables.")

    if cds_url:
        print(f"  * CDS_API_URL   : [PRESENT] URL: {cds_url}")
    else:
        print("  * CDS_API_URL   : [MISSING] Not found in environment (will default to CDS API endpoint).")

    if cdsapirc_path.exists():
        print(f"  * ~/.cdsapirc   : [PRESENT] Configuration file found at {cdsapirc_path}")
    else:
        print("  * ~/.cdsapirc   : [MISSING] ~/.cdsapirc file not present.")

    if not (cds_key or cdsapirc_path.exists()):
        print("\n[NOTE] Copernicus CDS credentials not active in current session.")
        print("       To enable official ERA5 downloads:")
        print("       1. Register/Login at https://cds.climate.copernicus.eu/")
        print("       2. Obtain your Personal Access Token / API Key from your profile.")
        print("       3. Set CDS_API_KEY in .env or create ~/.cdsapirc file.")

    print("-" * 70)
    # 3. Data Parameters & Target Storm
    ds_info = config["dataset"]
    region = config["region"]
    align = config["alignment"]
    sample = config["sampling"]

    print("TARGET ERA5 DATASET & REGION METADATA:")
    print(f"  • Source Dataset ID   : {ds_info['dataset_id']}")
    print(f"  • Target Storm        : {align['storm_name']} (IBTrACS ID: {align['storm_id']})")
    print(f"  • Temporal Resolution : {ds_info['temporal_resolution']} (Interval: {sample['interval_hours']} hours)")
    print(f"  • Test Timesteps       : {sample['test_timesteps']} steps (~{sample['interval_hours'] * sample['test_timesteps']} hours)")
    print("  • Bounding Region (NIO):")
    print(f"      North: {region['north']}°N | South: {region['south']}°N | West: {region['west']}°E | East: {region['east']}°E")

    print("\nCONFIGURED ATMOSPHERIC VARIABLES:")
    print("  • Surface Variables:")
    for var in config["variables"]["surface"]:
        print(f"      - {var}")

    print("  • Optional Variables:")
    for var in config["variables"].get("optional", []):
        print(f"      - {var}")

    plevels = config["pressure_levels"]
    print(f"  • Pressure Levels Enabled: {plevels['enabled']}")
    if plevels['enabled']:
        print(f"      Levels: {plevels['levels']} hPa")
        print(f"      Variables: {plevels['variables']}")

    print("=" * 70)


if __name__ == "__main__":
    inspect_era5_config()
