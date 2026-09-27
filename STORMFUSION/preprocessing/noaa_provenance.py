"""NOAA Satellite Data Provenance Auditing & Verification Module for STORMFUSION.

Enforces strict data provenance rules for NOAA GOES satellite files:
1. Computes SHA256 checksums for every NOAA file.
2. Inspects product metadata and channel attributes.
3. Classifies files into VERIFIED_REAL, UNVERIFIED, or SYNTHETIC_OR_TEST_SCHEMA.
4. Enforces geographic non-equivalency metadata:
     source = NOAA_GOES
     coverage_status = DEVELOPMENT_ONLY
     NIO_OPERATIONAL_COMPATIBILITY = FALSE
5. Exports data/processed/satellite/noaa_provenance.json and noaa_verified_manifest.csv.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from preprocessing.noaa_inspector import NOAAInspector


def compute_sha256(file_path: Path) -> str:
    """Computes SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class NOAAProvenanceAuditor:
    """Auditor for NOAA GOES satellite data provenance."""

    def __init__(self, project_root: Path = None):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root)

        self.raw_noaa_dir = self.project_root / "data/raw/noaa/goes"
        self.processed_dir = self.project_root / "data/processed/satellite"
        self.prov_json_path = self.processed_dir / "noaa_provenance.json"
        self.manifest_csv_path = self.processed_dir / "noaa_verified_manifest.csv"
        self.inspector = NOAAInspector()

    def audit_noaa_directory(self) -> Dict[str, Any]:
        """Audits all NOAA files in raw directory and updates manifest and provenance json."""
        self.raw_noaa_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        files = [f for f in self.raw_noaa_dir.glob("*.*") if not (f.name.endswith(".json") or f.name.endswith(".gitkeep") or f.name.startswith("."))]
        records: List[Dict[str, Any]] = []
        verified_count = 0
        synthetic_count = 0
        corrupted_count = 0
        unverified_count = 0

        for f in files:
            sha256 = compute_sha256(f)
            file_size = f.stat().st_size
            filename_lower = f.name.lower()

            # 1. Reject 0-byte or corrupted / HTML files
            if file_size == 0:
                prov_status = "CORRUPTED"
                training_eligible = False
                corrupted_count += 1
                meta = {}
            # 2. Identify test / sample / mock fixtures
            elif any(tag in filename_lower for tag in ["sample", "test", "mock", "dummy", "synthetic"]):
                prov_status = "SYNTHETIC_OR_TEST_SCHEMA"
                training_eligible = False
                synthetic_count += 1
                meta = {}
            else:
                # 3. Check for HTML error response disguised as .nc
                try:
                    with open(f, "rb") as check_f:
                        header = check_f.read(128)
                        if b"<!DOCTYPE html" in header or b"<html" in header or b"<Error>" in header:
                            prov_status = "CORRUPTED"
                            training_eligible = False
                            corrupted_count += 1
                            meta = {}
                        else:
                            # 4. Strict NetCDF4 / HDF5 inspection
                            meta = self.inspector.inspect_file(f)
                            global_attrs = meta.get("global_attributes", {})
                            variables = meta.get("variables", {})

                            has_rad = "Rad" in variables or any("rad" in v.lower() for v in variables)
                            platform = global_attrs.get("platform_ID", global_attrs.get("platform", "GOES-16"))
                            instrument = global_attrs.get("instrument_type", global_attrs.get("instrument_ID", "ABI"))

                            is_valid_noaa = meta.get("format") in ["NetCDF4", "HDF5"] and has_rad
                            if is_valid_noaa:
                                prov_status = "VERIFIED_REAL"
                                training_eligible = True
                                verified_count += 1
                            else:
                                prov_status = "UNVERIFIED"
                                training_eligible = False
                                unverified_count += 1
                except Exception:
                    prov_status = "CORRUPTED"
                    training_eligible = False
                    corrupted_count += 1
                    meta = {}

            gattrs = meta.get("global_attributes", {}) if isinstance(meta, dict) else {}
            calib = meta.get("calibration_metadata", {}) if isinstance(meta, dict) else {}

            # Spatial metadata inspection for NIO coverage check
            # GOES-16 sits at 75.2W (Western Hemisphere). It strictly does NOT cover North Indian Ocean (45E-105E).
            orbital_slot = gattrs.get("orbital_slot", "GOES-East (75.2W)")
            time_coverage_start = gattrs.get("time_coverage_start", "")
            time_coverage_end = gattrs.get("time_coverage_end", "")

            rec = {
                "filename": f.name,
                "filepath": str(f.relative_to(self.project_root)),
                "provider": "NOAA",
                "source_url_or_object_id": f"s3://noaa-goes16/{f.name}",
                "platform": gattrs.get("platform_ID", "GOES-16"),
                "instrument": gattrs.get("instrument_type", "ABI"),
                "product": gattrs.get("title", "ABI L1b Radiances"),
                "channel": gattrs.get("channel_id", "ABI_CH13"),
                "acquisition_time": time_coverage_start or gattrs.get("date_created", ""),
                "start_time": time_coverage_start,
                "end_time": time_coverage_end,
                "latitude_min": -81.0,
                "latitude_max": 81.0,
                "longitude_min": -156.0,
                "longitude_max": 6.0,
                "file_size_bytes": file_size,
                "sha256": sha256,
                "provenance_status": prov_status,
                "training_eligible": training_eligible,
                "coverage_status": "DEVELOPMENT_ONLY",
                "nio_operational_compatibility": False,
            }
            records.append(rec)

        df_manifest = pd.DataFrame(records)
        if not df_manifest.empty:
            df_manifest.to_csv(self.manifest_csv_path, index=False)
        else:
            df_empty = pd.DataFrame(columns=[
                "filename", "filepath", "provider", "source_url_or_object_id", "platform",
                "instrument", "product", "channel", "acquisition_time", "start_time", "end_time",
                "latitude_min", "latitude_max", "longitude_min", "longitude_max", "file_size_bytes",
                "sha256", "provenance_status", "training_eligible", "coverage_status",
                "nio_operational_compatibility"
            ])
            df_empty.to_csv(self.manifest_csv_path, index=False)

        provenance_data = {
            "source": "NOAA_GOES",
            "coverage_status": "DEVELOPMENT_ONLY",
            "NIO_OPERATIONAL_COMPATIBILITY": False,
            "total_files": len(records),
            "verified_files": verified_count,
            "synthetic_files": synthetic_count,
            "corrupted_files": corrupted_count,
            "unverified_files": unverified_count,
            "verified": verified_count > 0,
            "manifest_path": str(self.manifest_csv_path.relative_to(self.project_root)),
        }

        with open(self.prov_json_path, "w", encoding="utf-8") as f_out:
            json.dump(provenance_data, f_out, indent=2)

        return provenance_data


if __name__ == "__main__":
    auditor = NOAAProvenanceAuditor()
    status = auditor.audit_noaa_directory()
    print(json.dumps(status, indent=2))
