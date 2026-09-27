"""STORMFUSION — Real Data Provenance & Geographic Compatibility Integration Test Suite.
SIH Problem Statement: SIH26070.

Explicitly tests and proves:
1. Synthetic/sample fixtures are strictly rejected from VERIFIED_REAL classification.
2. HTML/XML error responses or zero-byte files are classified as CORRUPTED.
3. Genuine downloaded NOAA GOES NetCDF4 file passes provenance validation as VERIFIED_REAL,
   but is strictly flagged NIO_OPERATIONAL_COMPATIBILITY = False (Development only).
4. Genuine downloaded EUMETSAT IODC file passes provenance validation as VERIFIED_REAL,
   and is flagged NIO_OPERATIONAL_COMPATIBILITY = True (Meteosat 45.5°E IODC).
5. Geographic non-equivalency is preserved: GOES-16 cannot be misclassified as NIO operational.
6. MOSDAC/INSAT provider is cleanly removed and rejected; EUMETSAT IODC is sole NIO provider.
"""

import os
import sys
import unittest
import tempfile
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.noaa_provenance import NOAAProvenanceAuditor
from preprocessing.noaa_inspector import NOAAInspector
from preprocessing.eumetsat_provenance import EUMETSATProvenanceAuditor


class TestRealDataProvenance(unittest.TestCase):
    """Test suite for data provenance validation and geographic compatibility gates."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = PROJECT_ROOT

    def test_01_synthetic_fixture_rejected(self):
        """Proves that synthetic or sample fixture files are strictly rejected as non-real."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fake_sample = tmp_path / "OR_ABI-L1b-RadF-M6C13_G16_s2023001000000_sample.nc"
            with open(fake_sample, "wb") as f:
                f.write(b"NOAA_GOES16_ABI_L1B_REAL_HEADER_STUB_2023001000000_VERIFIED_DATA_BLOCK\n")
                f.write(b"\x00" * 4000)

            auditor = NOAAProvenanceAuditor(project_root=tmp_path)
            auditor.raw_noaa_dir = tmp_path
            auditor.processed_dir = tmp_path / "processed"
            auditor.prov_json_path = auditor.processed_dir / "noaa_prov.json"
            auditor.manifest_csv_path = auditor.processed_dir / "noaa_manifest.csv"

            status = auditor.audit_noaa_directory()
            self.assertEqual(status["verified_files"], 0, "Synthetic fixture must NOT be marked verified_real")
            self.assertEqual(status["synthetic_files"], 1, "Synthetic fixture must be classified as SYNTHETIC_OR_TEST_SCHEMA")
            self.assertFalse(status["verified"])

    def test_02_html_error_payload_rejected(self):
        """Proves that HTTP error responses or HTML pages disguised as NetCDF are classified as CORRUPTED."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            html_file = tmp_path / "OR_ABI-L1b-RadF-M6C13_G16_error.nc"
            with open(html_file, "wb") as f:
                f.write(b"<!DOCTYPE html><html><head><title>404 Not Found</title></head><body>Error</body></html>")

            auditor = NOAAProvenanceAuditor(project_root=tmp_path)
            auditor.raw_noaa_dir = tmp_path
            auditor.processed_dir = tmp_path / "processed"
            auditor.prov_json_path = auditor.processed_dir / "noaa_prov.json"
            auditor.manifest_csv_path = auditor.processed_dir / "noaa_manifest.csv"

            status = auditor.audit_noaa_directory()
            self.assertEqual(status["verified_files"], 0)
            self.assertEqual(status["corrupted_files"], 1)

    def test_03_zero_byte_file_rejected(self):
        """Proves that zero-byte files are classified as CORRUPTED."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            zero_file = tmp_path / "empty_observation.nc"
            zero_file.touch()

            auditor = NOAAProvenanceAuditor(project_root=tmp_path)
            auditor.raw_noaa_dir = tmp_path
            auditor.processed_dir = tmp_path / "processed"
            auditor.prov_json_path = auditor.processed_dir / "noaa_prov.json"
            auditor.manifest_csv_path = auditor.processed_dir / "noaa_manifest.csv"

            status = auditor.audit_noaa_directory()
            self.assertEqual(status["verified_files"], 0)
            self.assertEqual(status["corrupted_files"], 1)

    def test_04_noaa_real_file_provenance_and_non_nio(self):
        """Proves that genuine NOAA GOES NetCDF4 files are VERIFIED_REAL but strictly NIO_OPERATIONAL_COMPATIBILITY = False."""
        auditor = NOAAProvenanceAuditor(project_root=self.project_root)
        status = auditor.audit_noaa_directory()

        self.assertIn("verified_files", status)
        self.assertGreaterEqual(status["verified_files"], 1, "At least 1 authentic NOAA file must be indexed")
        self.assertEqual(status["coverage_status"], "DEVELOPMENT_ONLY")
        self.assertFalse(status["NIO_OPERATIONAL_COMPATIBILITY"], "GOES-16 must NEVER be classified as NIO compatible")

    def test_05_eumetsat_real_file_provenance_and_nio_compatibility(self):
        """Proves that authentic EUMETSAT Meteosat IODC data is verified and NIO_OPERATIONAL_COMPATIBILITY = True."""
        auditor = EUMETSATProvenanceAuditor(project_root=self.project_root)
        status = auditor.audit_directory()

        self.assertIn("total_scanned_files", status)
        self.assertGreaterEqual(status["total_scanned_files"], 1, "At least 1 authentic EUMETSAT file must be scanned")
        self.assertGreaterEqual(status["meteorological_derived_files"], 1, "At least 1 authentic EUMETSAT IODC product must be indexed")
        self.assertEqual(status["coverage_status"], "NIO_OPERATIONAL_COVERAGE")
        self.assertTrue(status["NIO_OPERATIONAL_COMPATIBILITY"], "Meteosat-9/8 IODC at 41.5E/45.5E MUST be NIO compatible")

    def test_06_geographic_non_equivalency(self):
        """Asserts that NOAA (GOES-16 at 75.2W) and EUMETSAT (Meteosat at 45.5E) have distinct geographic roles."""
        noaa_auditor = NOAAProvenanceAuditor(project_root=self.project_root)
        eum_auditor = EUMETSATProvenanceAuditor(project_root=self.project_root)

        noaa_status = noaa_auditor.audit_noaa_directory()
        eum_status = eum_auditor.audit_directory()

        # GOES: Western Hemisphere development
        self.assertFalse(noaa_status["NIO_OPERATIONAL_COMPATIBILITY"])
        # Meteosat: North Indian Ocean operational
        self.assertTrue(eum_status["NIO_OPERATIONAL_COMPATIBILITY"])

    def test_07_mosdac_cleanly_removed(self):
        """Asserts that MOSDAC/INSAT provider is completely removed and rejected by factory."""
        from preprocessing.satellite_provider import SatelliteProviderFactory
        with self.assertRaises(ValueError):
            SatelliteProviderFactory.create("mosdac", project_root=self.project_root)
        with self.assertRaises(ValueError):
            SatelliteProviderFactory.create("insat", project_root=self.project_root)


if __name__ == "__main__":
    unittest.main()
