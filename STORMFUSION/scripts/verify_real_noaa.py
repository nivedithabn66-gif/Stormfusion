"""NOAA Real Data Verification & Provenance Gate Script for STORMFUSION.

Executes NOAA SHA256 hashing, metadata inspection, classification, and provenance manifest generation.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preprocessing.noaa_provenance import NOAAProvenanceAuditor



def verify_real_noaa(project_root: Path = None) -> bool:
    """Executes NOAA provenance audit and prints verification report."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    auditor = NOAAProvenanceAuditor(project_root=project_root)
    status = auditor.audit_noaa_directory()

    print("==================================================")
    print("      STORMFUSION NOAA DATA PROVENANCE AUDIT      ")
    print("==================================================")
    print(f"Total NOAA Files Scanned : {status.get('total_files', 0)}")
    print(f"Verified Real Files     : {status.get('verified_files', 0)}")
    print(f"Synthetic / Test Files   : {status.get('synthetic_files', 0)}")
    print(f"Unverified Files         : {status.get('unverified_files', 0)}")
    print(f"Coverage Status          : {status.get('coverage_status', 'DEVELOPMENT_ONLY')}")
    print(f"NIO Compatibility        : {status.get('NIO_OPERATIONAL_COMPATIBILITY', False)}")
    print("==================================================")

    return status.get("verified", False)


if __name__ == "__main__":
    verify_real_noaa()
