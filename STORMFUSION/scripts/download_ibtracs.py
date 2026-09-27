"""IBTrACS Data Download Script for STORMFUSION.

Downloads the official IBTrACS v04r01 NetCDF dataset from NOAA/NCEI
based on configurations specified in configs/data_config.yaml.
"""

import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
import yaml


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_progress_hook(block_num: int, block_size: int, total_size: int):
    """Callback function for urlretrieve to show download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded / total_size * 100)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(
            f"\rDownloading IBTrACS dataset: {percent:.1f}% ({downloaded_mb:.1f} MB / {total_mb:.1f} MB)"
        )
        sys.stdout.flush()
    else:
        downloaded_mb = downloaded / (1024 * 1024)
        sys.stdout.write(f"\rDownloading IBTrACS dataset: {downloaded_mb:.1f} MB downloaded")
        sys.stdout.flush()


def download_ibtracs(config_path: Path = Path("configs/data_config.yaml")) -> Path:
    """Download IBTrACS dataset if not already present.

    Args:
        config_path: Path to the data configuration YAML file.

    Returns:
        Path to the downloaded NetCDF file.
    """
    project_root = Path(__file__).resolve().parent.parent
    abs_config_path = project_root / config_path
    config = load_config(abs_config_path)

    ibtracs_cfg = config["ibtracs"]
    url = ibtracs_cfg["url"]
    raw_dir = project_root / ibtracs_cfg["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    dest_file = raw_dir / ibtracs_cfg["raw_filename"]

    if dest_file.exists() and dest_file.stat().st_size > 0:
        size_mb = dest_file.stat().st_size / (1024 * 1024)
        print(f"[INFO] IBTrACS file already exists at: {dest_file} ({size_mb:.2f} MB)")
        return dest_file

    print(f"[INFO] Downloading IBTrACS dataset from: {url}")
    print(f"[INFO] Target destination: {dest_file}")

    start_time = time.time()
    try:
        urllib.request.urlretrieve(url, dest_file, reporthook=download_progress_hook)
        print() # New line after progress bar
        elapsed = time.time() - start_time
        file_size_mb = dest_file.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] Download completed in {elapsed:.1f}s ({file_size_mb:.2f} MB)")
        return dest_file
    except urllib.error.URLError as e:
        if dest_file.exists():
            dest_file.unlink() # Cleanup partial download
        print(f"\n[ERROR] Failed to download IBTrACS dataset: {e}")
        raise e
    except Exception as e:
        if dest_file.exists():
            dest_file.unlink()
        print(f"\n[ERROR] Unexpected error during download: {e}")
        raise e


if __name__ == "__main__":
    download_ibtracs()
