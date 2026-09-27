"""Copernicus Climate Data Store (CDS) Official Data Download & Auth Client Module for STORMFUSION.

Handles CDS API authentication, ERA5 reanalysis single-levels dataset access,
controlled product downloads, and transaction manifest logging.

Security Guarantee:
- Credentials (CDS_API_URL, CDS_API_KEY) are read strictly from environment / .env or ~/.cdsapirc.
- Tokens, API keys, and passwords are NEVER printed, logged, or exposed in error tracebacks.
"""

import os
import sys
import json
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import yaml

from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE, override=True)


def get_cds_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Retrieve CDS API credentials safely from environment or ~/.cdsapirc file.

    Returns:
        Tuple of (cds_url, cds_key) or (None, None) if not configured.
    """
    url = os.environ.get("CDS_API_URL")
    key = os.environ.get("CDS_API_KEY")

    if url and url.strip() and key and key.strip():
        return url.strip(), key.strip()

    # Check ~/.cdsapirc file fallback
    rc_path = Path.home() / ".cdsapirc"
    if rc_path.exists():
        try:
            with open(rc_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("url:"):
                        url = line.split(":", 1)[1].strip()
                    elif line.startswith("key:"):
                        key = line.split(":", 1)[1].strip()
            if url and key:
                return url, key
        except Exception:
            pass

    return None, None


def check_cds_credentials() -> Dict[str, Any]:
    """Inspect CDS credential presence without exposing secret values.

    Returns:
        Dict with status flags for url and key.
    """
    url, key = get_cds_credentials()
    url_set = bool(url)
    key_set = bool(key)
    has_creds = url_set and key_set

    return {
        "url_set": url_set,
        "key_set": key_set,
        "has_credentials": has_creds,
        "url_status": "SET" if url_set else "NOT_SET",
        "key_status": "SET" if key_set else "NOT_SET",
    }


def authenticate_cds() -> Dict[str, Any]:
    """Perform minimum legitimate authentication/client initialization check with CDS API.

    Returns:
        Dict with auth_status ('PASS', 'BLOCKED', or 'FAIL') and detailed message.
    """
    creds_info = check_cds_credentials()
    if not creds_info["has_credentials"]:
        return {
            "auth_status": "BLOCKED",
            "reason": "CDS_API_URL and/or CDS_API_KEY not configured in .env or ~/.cdsapirc",
            "authenticated": False,
        }

    # Attempt importing cdsapi library
    try:
        import cdsapi
        url, key = get_cds_credentials()
        # Initialize client without exposing keys
        client = cdsapi.Client(url=url, key=key, quiet=True)
        return {
            "auth_status": "PASS",
            "reason": "Copernicus CDS API client initialized successfully.",
            "authenticated": True,
        }
    except ImportError:
        return {
            "auth_status": "BLOCKED",
            "reason": "cdsapi Python package not installed.",
            "authenticated": False,
        }
    except Exception as e:
        return {
            "auth_status": "FAIL",
            "reason": f"CDS API authentication failed: {type(e).__name__}",
            "authenticated": False,
        }


def update_era5_download_manifest(
    manifest_path: Path,
    file_info: Dict[str, Any],
) -> dict:
    """Updates data/raw/era5/download_manifest.json with file acquisition metadata.

    Args:
        manifest_path: Path to download_manifest.json
        file_info: Dict containing file metadata (NO secrets)

    Returns:
        Updated manifest dict
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = {"files": []}

    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            manifest_data = {"files": []}

    files_list = manifest_data.get("files", [])
    updated = False
    for i, item in enumerate(files_list):
        if item.get("filename") == file_info.get("filename"):
            files_list[i] = file_info
            updated = True
            break
    if not updated:
        files_list.append(file_info)

    manifest_data["files"] = files_list
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data
