"""Dedicated End-to-End Real Data Test for EUMETSAT IODC / MSG SEVIRI Pipeline.
SIH Problem Statement: SIH26070 — STORMFUSION (Phase 9 Master Prompt).

Validates the complete chain:
OAuth2
  ↓
Catalog
  ↓
Product Discovery
  ↓
Product Retrieval
  ↓
File Verification
  ↓
Provenance
  ↓
Decoder
  ↓
QC
  ↓
Preprocessing
  ↓
Tensor
  ↓
Model
  ↓
Predictions

CRITICAL SCIENTIFIC SAFETY:
The test MUST FAIL LOUDLY if real data is unavailable.
No fallback to synthetic/mock data is permitted.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preprocessing.eumetsat_iodc_provider import EUMETSATIODCProvider
from preprocessing.eumetsat_provenance import EUMETSATProvenanceAuditor
from models.stormfusion import StormFusionModel


class TestEUMETSATRealDataEndToEnd(unittest.TestCase):
    """Rigorous end-to-end integration test for genuine EUMETSAT observation pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = project_root
        cls.provider = EUMETSATIODCProvider(project_root=cls.project_root, crop_size=(500, 500))
        cls.auditor = EUMETSATProvenanceAuditor(project_root=cls.project_root)

    def test_01_oauth2_authentication(self):
        """Validates OAuth2 bearer token acquisition from EUMETSAT Data Store."""
        creds = self.provider.check_credentials()
        self.assertTrue(
            creds["credentials_available"],
            "EUMETSAT credentials not configured in environment (EUMETSAT_CONSUMER_KEY/SECRET)",
        )
        auth_res = self.provider.authenticate()
        self.assertTrue(auth_res["authenticated"], f"EUMETSAT authentication failed: {auth_res.get('reason')}")
        self.assertEqual(auth_res["status"], "AUTHENTICATED")
        self.assertIsNotNone(auth_res.get("token"))

    def test_02_catalog_and_product_discovery(self):
        """Validates OpenSearch catalog query and discovered genuine IODC product metadata."""
        search_res = self.provider.search_fani_period(limit=3)
        self.assertEqual(search_res["status"], "SUCCESS", f"Catalog search failed: {search_res.get('reason')}")
        self.assertGreater(search_res["count"], 0, "No products located in OpenSearch catalog")
        products = search_res["products"]
        self.assertGreaterEqual(len(products), 1)

        first_prod = products[0]
        self.assertIn("id", first_prod)
        self.assertIn("MSG1-SEVI-MSG15", first_prod["id"])
        self.assertIn("download_url", first_prod)
        self.assertTrue(first_prod["download_url"].startswith("https://api.eumetsat.int/data/download/1.0.0/"))

    def test_03_product_retrieval_status_handling(self):
        """Validates retrieval logic cleanly handles API status or GeneralLicense block per Phase 12."""
        search_res = self.provider.search_fani_period(limit=1)
        products = search_res.get("products", [])
        if products:
            prod_id = products[0]["id"]
            dl_res = self.provider.download_product(prod_id, stop_on_license_block=True)
            self.assertIn(
                dl_res["classification"],
                ["DOWNLOAD_SUCCESS", "GENERAL_LICENSE_REQUIRED"],
                f"Unexpected retrieval status: {dl_res}",
            )
            if dl_res["classification"] == "GENERAL_LICENSE_REQUIRED":
                self.assertEqual(dl_res["http_status"], 403)
                self.assertIn("GeneralLicense", dl_res["reason"])

    def test_04_cryptographic_file_verification(self):
        """Asserts at least ONE genuine Level-1.5 observation binary exists on disk and passes SHA-256 audit."""
        prov = self.auditor.audit_directory()
        verified_count = prov.get("verified_files", 0)
        self.assertGreater(
            verified_count,
            0,
            "FAIL LOUDLY: No verified real EUMETSAT files on disk. Real data pipeline requires genuine observation binary.",
        )

        real_files = list(self.provider.real_files_index.keys())
        self.assertGreater(len(real_files), 0, "No genuine files indexed in real_files_index")

        sample_name = real_files[0]
        file_path = self.provider.raw_data_dir / sample_name
        self.assertTrue(file_path.exists())
        self.assertGreater(file_path.stat().st_size, 1000000, "Observation file too small to be authentic satellite data")

        # Verify SHA-256 calculation
        sha256 = self.auditor.calculate_sha256(file_path)
        self.assertEqual(len(sha256), 64)

    def test_05_provenance_metadata(self):
        """Asserts provenance metadata accurately documents genuine EUMETSAT observations."""
        prov = self.auditor.audit_directory()
        self.assertTrue(prov["verified"])
        self.assertTrue(prov["real_data"])
        self.assertFalse(prov["synthetic"])
        self.assertEqual(prov["provider"], "EUMETSAT")
        self.assertEqual(prov["coverage_status"], "NIO_OPERATIONAL_COVERAGE")
        self.assertTrue(prov["NIO_OPERATIONAL_COMPATIBILITY"])

    def test_06_decoder_and_physical_qc(self):
        """Decodes authentic SEVIRI observation via Satpy native reader and verifies physical QC."""
        sample_name = list(self.provider.real_files_index.keys())[0]
        file_path = self.provider.raw_data_dir / sample_name

        frame, qc = self.provider.decode_and_standardize_frame(file_path, center_lat=15.0, center_lon=85.0)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (3, 500, 500))
        self.assertFalse(np.isnan(frame).any(), "Decoded frame contains NaN values")
        self.assertFalse(np.isinf(frame).any(), "Decoded frame contains Inf values")

        # Physical Kelvin bounds check
        self.assertEqual(qc["status"], "PASS", f"Quality control failed: {qc.get('reason')}")
        self.assertGreaterEqual(qc["valid_pixel_pct"], 95.0)

    def test_07_preprocessing_and_tensor_contract(self):
        """Validates standard preprocessing pipeline and tensor generation with genuine satellite data."""
        sample_name = list(self.provider.real_files_index.keys())[0]
        test_row = {
            "sequence_id": "TEST_E2E_REAL",
            "lat_t0": 15.0,
            "lon_t0": 85.0,
            "eumetsat_t0": sample_name,
            "time_t0": "2026-09-08T08:12:40Z",
        }
        seq_data = self.provider.load_sequence_frames(test_row)

        self.assertTrue(seq_data["satellite_real_data_available"])
        self.assertEqual(seq_data["satellite_status"], "REAL_OBSERVATION_LOADED")
        self.assertEqual(seq_data["satellite_provider_status"], "OPERATIONAL")
        self.assertEqual(float(seq_data["satellite_modality_mask"]), 1.0)
        self.assertEqual(seq_data["satellite_tensor"].shape, (8, 3, 500, 500))
        self.assertEqual(float(seq_data["satellite_valid_mask"][0]), 1.0)

    def test_08_end_to_end_model_inference(self):
        """Passes genuine decoded SEVIRI observation tensor through complete neural network forward pass."""
        sample_name = list(self.provider.real_files_index.keys())[0]
        test_row = {
            "sequence_id": "TEST_E2E_MODEL",
            "lat_t0": 15.0,
            "lon_t0": 85.0,
            "eumetsat_t0": sample_name,
            "time_t0": "2026-09-08T08:12:40Z",
        }
        seq_data = self.provider.load_sequence_frames(test_row)
        sat_tensor = seq_data["satellite_tensor"]

        # Scale down spatially for standard model receptive field
        b_sat = F.interpolate(sat_tensor, size=(32, 32), mode="bilinear", align_corners=False).unsqueeze(0)
        b_sat_m = seq_data["satellite_modality_mask"].unsqueeze(0) if hasattr(seq_data["satellite_modality_mask"], "unsqueeze") else torch.tensor([[1.0]])
        b_sat_v = seq_data["satellite_valid_mask"].unsqueeze(0) if hasattr(seq_data["satellite_valid_mask"], "unsqueeze") else torch.from_numpy(seq_data["satellite_valid_mask"]).unsqueeze(0)

        b_trk = torch.randn(1, 8, 8)
        b_era = torch.randn(1, 8, 4, 32, 32)

        model = StormFusionModel()
        out = model(
            satellite_tensor=b_sat,
            satellite_valid_mask=b_sat_v,
            satellite_modality_mask=b_sat_m,
            track_features=b_trk,
            track_valid_mask=torch.ones(1, 8),
            track_modality_mask=torch.ones(1, 1),
            era5_tensor=b_era,
            era5_valid_mask=torch.ones(1, 8),
            era5_modality_mask=torch.ones(1, 1),
        )

        # Assert expected multi-task prediction shapes
        self.assertEqual(out["detection_logits"].shape, torch.Size([1, 1]))
        self.assertEqual(out["intensity"].shape, torch.Size([1, 4]))
        self.assertEqual(out["pressure"].shape, torch.Size([1, 4]))
        self.assertEqual(out["track_delta"].shape, torch.Size([1, 4, 2]))


if __name__ == "__main__":
    unittest.main()
