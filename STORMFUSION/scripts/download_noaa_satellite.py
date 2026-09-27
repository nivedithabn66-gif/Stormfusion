"""NOAA Open Satellite Data Ingestion Script for STORMFUSION.

Acquires authentic NOAA GOES open-data observations from the official public
AWS S3 NOAA GOES open-data bucket (https://noaa-goes16.s3.amazonaws.com/)
without requiring credentials.

Strict scientific metadata:
  source = NOAA_GOES
  platform = GOES-16
  instrument = ABI
  channel = ABI_CH13 (Clean IR Longwave Window, 10.35 µm)
  coverage_status = DEVELOPMENT_ONLY
  NIO_OPERATIONAL_COMPATIBILITY = FALSE
"""

import json
import sys
import hashlib
import urllib.request
import urllib.error
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preprocessing.noaa_provenance import NOAAProvenanceAuditor
from preprocessing.noaa_inspector import NOAAInspector


# Official NOAA GOES-16 AWS Open Data S3 Public Object URL
DEFAULT_NOAA_S3_KEY = "ABI-L1b-RadM/2023/001/00/OR_ABI-L1b-RadM1-M6C13_G16_s20230010000281_e20230010000350_c20230010000384.nc"
DEFAULT_NOAA_S3_URL = f"https://noaa-goes16.s3.amazonaws.com/{DEFAULT_NOAA_S3_KEY}"


def download_noaa_satellite_samples(
    project_root: Path = None,
    s3_url: str = DEFAULT_NOAA_S3_URL,
    target_key: str = DEFAULT_NOAA_S3_KEY,
) -> Path:
    """Acquires authentic NOAA GOES satellite observation from official open data repository.

    Downloads the genuine Level-1b NetCDF4 file, validates file integrity and format,
    computes cryptographic SHA256 checksum, and records provenance.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    noaa_dir = project_root / "data/raw/noaa/goes"
    noaa_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(target_key).name
    dest_path = noaa_dir / filename

    if not dest_path.exists():
        print(f"[NOAA INGESTION] Fetching authentic NOAA GOES file from:\n  {s3_url}")
        req = urllib.request.Request(s3_url, headers={"User-Agent": "STORMFUSION-SIH26070/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP Error {resp.status} downloading NOAA observation")
                data = resp.read()

                # Reject HTML or error responses
                if b"<!DOCTYPE html" in data[:128] or b"<Error>" in data[:128]:
                    raise ValueError(f"Downloaded content from {s3_url} is an HTML/XML error page, not NetCDF4")

                with open(dest_path, "wb") as f:
                    f.write(data)
                print(f"[NOAA INGESTION] Download complete: {len(data)} bytes written to {dest_path.name}")
        except Exception as e:
            print(f"[NOAA INGESTION ERROR] Failed to fetch {s3_url}: {e}")
            raise

    # Compute SHA-256
    with open(dest_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    # Verify using NetCDF4 inspector
    inspector = NOAAInspector()
    meta = inspector.inspect_file(dest_path)
    calib = meta.get("calibration_metadata", {})

    print(f"[NOAA VALIDATION] Verified authentic NetCDF4 product: {meta.get('filename')}")
    print(f"  • Format      : {meta.get('format')}")
    print(f"  • Channels    : {meta.get('channels', [])[:5]}")
    print(f"  • Scale Factor: {calib.get('scale_factor')}")
    print(f"  • Add Offset  : {calib.get('add_offset')}")
    print(f"  • Units       : {calib.get('units')}")
    print(f"  • SHA-256     : {sha256}")

    # Run provenance audit
    auditor = NOAAProvenanceAuditor(project_root=project_root)
    prov = auditor.audit_noaa_directory()
    print(f"[NOAA PROVENANCE] Verified real NOAA files indexed: {prov.get('verified_files', 0)}")
    print(f"  • Coverage Status          : {prov.get('coverage_status')}")
    print(f"  • NIO Compatibility        : {prov.get('NIO_OPERATIONAL_COMPATIBILITY')}")

    return dest_path


if __name__ == "__main__":
    download_noaa_satellite_samples()
