"""Copernicus CDS Authentication & Credential Verification Script for STORMFUSION.

Checks credential presence safely via environment variables / .env / ~/.cdsapirc:
- CDS_API_URL
- CDS_API_KEY

Attempts minimum legitimate CDS API client initialization without exposing secrets.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.cds_client import check_cds_credentials, authenticate_cds


def test_cds_authentication() -> dict:
    print("=" * 50)
    print("COPERNICUS CDS API AUTHENTICATION VERIFICATION")
    print("=" * 50)

    creds_info = check_cds_credentials()
    url_status = creds_info["url_status"]
    key_status = creds_info["key_status"]

    print(f"CDS_API_URL: {url_status}")
    print(f"CDS_API_KEY: {key_status}")

    auth_res = authenticate_cds()
    print(f"Authentication: {auth_res['auth_status']}")
    print(f"Reason: {auth_res['reason']}\n")

    return {
        "url_status": url_status,
        "key_status": key_status,
        "auth_status": auth_res["auth_status"],
        "authenticated": auth_res["authenticated"],
        "reason": auth_res["reason"],
    }


if __name__ == "__main__":
    test_cds_authentication()
