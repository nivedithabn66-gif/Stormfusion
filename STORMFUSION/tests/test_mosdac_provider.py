"""Unit Tests for MOSDAC Satellite Data Provider & Provenance Integration.
SIH Problem Statement: SIH26070 — Multi-Modal NIO Tropical Cyclone AI System.
"""

import os
import sys
import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.satellite_provider import SatelliteProviderFactory, SatelliteDataProvider
from preprocessing.mosdac_provider import MOSDACProvider
from preprocessing.mosdac_provenance import MOSDACProvenanceAuditor


class TestMOSDACProvider(unittest.TestCase):
    """Test suite for ISRO MOSDAC INSAT data provider integration."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = PROJECT_ROOT

    def test_01_credentials_detection(self):
        """Tests credential inspection without leaking values."""
        provider = MOSDACProvider(project_root=self.project_root)
        creds = provider.check_credentials()
        self.assertIn("credentials_available", creds)
        self.assertIn("username_configured", creds)
        self.assertIn("password_configured", creds)
        self.assertIn("status", creds)

    def test_02_factory_instantiation(self):
        """Tests that SatelliteProviderFactory instantiates MOSDACProvider when requested."""
        p1 = SatelliteProviderFactory.create("mosdac_provider", project_root=self.project_root)
        self.assertIsInstance(p1, MOSDACProvider)
        self.assertEqual(p1.get_source_name(), "MOSDAC")

        p2 = SatelliteProviderFactory.create("mosdac", project_root=self.project_root, allow_mosdac=True)
        self.assertIsInstance(p2, MOSDACProvider)

    def test_03_legacy_removal_guardrail_preserved(self):
        """Proves that default factory call without allow_mosdac raises ValueError to protect existing tests."""
        with self.assertRaises(ValueError):
            SatelliteProviderFactory.create("mosdac", project_root=self.project_root)
        with self.assertRaises(ValueError):
            SatelliteProviderFactory.create("insat", project_root=self.project_root)

    def test_04_missing_data_placeholder_contract(self):
        """Proves that missing observations return safe zero tensor with modality mask 0.0."""
        provider = MOSDACProvider(project_root=self.project_root)
        seq_row = {"lat_t0": 15.0, "lon_t0": 85.0}
        placeholder = provider.load_sequence_frames(seq_row)

        self.assertIn("satellite_tensor", placeholder)
        self.assertIn("satellite_valid_mask", placeholder)
        self.assertIn("satellite_modality_mask", placeholder)

        self.assertEqual(placeholder["satellite_tensor"].shape, (8, 3, 500, 500))
        self.assertEqual(placeholder["satellite_valid_mask"].sum().item(), 0.0)
        self.assertEqual(placeholder["satellite_modality_mask"].item(), 0.0)
        self.assertFalse(placeholder["satellite_real_data_available"])
        self.assertEqual(placeholder["satellite_source"], "MOSDAC")

    def test_05_channel_mapping_contract(self):
        """Asserts that MOSDAC channels match STORMFUSION sensor-agnostic requirements."""
        cmap = MOSDACProvider.CHANNEL_MAP
        self.assertEqual(cmap["model_channel_0"]["insat_channel"], "IMG_TIR1")
        self.assertEqual(cmap["model_channel_1"]["insat_channel"], "IMG_WV")
        self.assertEqual(cmap["model_channel_2"]["insat_channel"], "IMG_TIR2")

    def test_06_provenance_auditor(self):
        """Tests MOSDACProvenanceAuditor report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            auditor = MOSDACProvenanceAuditor(project_root=tmp_path)
            rep = auditor.audit_directory()

            self.assertEqual(rep["provider"], "MOSDAC")
            self.assertEqual(rep["agency"], "ISRO")
            self.assertTrue(rep["NIO_OPERATIONAL_COMPATIBILITY"])
            self.assertTrue(auditor.verified_manifest_path.exists())
            self.assertTrue(auditor.provenance_report_path.exists())

    def test_07_satpy_reader_registration(self):
        """Verifies that Satpy has the insat3d_img_l1b_h5 reader available in this environment."""
        import satpy
        available = satpy.available_readers()
        self.assertIn("insat3d_img_l1b_h5", available, "Satpy MUST have insat3d_img_l1b_h5 installed")


if __name__ == "__main__":
    unittest.main()
