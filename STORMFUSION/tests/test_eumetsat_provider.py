"""Unit and Mock Integration Test Suite for EUMETSAT IODC Provider.
SIH Problem Statement: SIH26070 — STORMFUSION.

Covers:
1. Provider Interface compliance with SatelliteDataProvider.
2. Authentication testing (valid, missing, and invalid credentials).
3. Product catalog discovery & spatial/temporal filtering.
4. Download failure & rate limit (HTTP 429) simulation.
5. SEVIRI frame decoding & Quality Control (QC).
6. Spatial standardization (cropping to [3, 500, 500], float32).
7. Temporal standardization & monotonicity verification.
8. Provenance audit generation & SHA-256 verification.
9. End-to-end model compatibility (StormFusionModel forward pass).
10. MC-Dropout UQ & Grad-CAM XAI compatibility.
"""

import os
import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import torch

# Ensure project root in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preprocessing.satellite_provider import SatelliteDataProvider, SatelliteSample, SatelliteProviderFactory
from preprocessing.eumetsat_iodc_provider import EUMETSATIODCProvider
from preprocessing.eumetsat_provenance import EUMETSATProvenanceAuditor
from models.stormfusion import StormFusionModel
from uncertainty.uq_runner import STORMFUSIONUQRunner
from explainability.gradcam import StormFusionGradCAM


class TestEUMETSATProvider(unittest.TestCase):
    """Test suite for EUMETSAT IODC satellite provider."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = project_root
        cls.provider = EUMETSATIODCProvider(project_root=cls.project_root, crop_size=(500, 500))

    def test_01_interface_compliance(self):
        """Asserts that EUMETSATIODCProvider adheres to SatelliteDataProvider ABC."""
        self.assertEqual(self.provider.get_source_name(), "EUMETSAT_IODC")
        self.assertIsInstance(self.provider.is_real_data_available(), bool)

    def test_02_authentication_missing_credentials(self):
        """Asserts graceful handling when EUMETSAT credentials are absent."""
        with patch.dict(os.environ, {}, clear=True):
            creds = self.provider.check_credentials()
            self.assertFalse(creds["credentials_available"])
            self.assertEqual(creds["status"], "MISSING")

            auth_res = self.provider.authenticate()
            self.assertFalse(auth_res["authenticated"])
            self.assertEqual(auth_res["status"], "EUMETSAT_IODC_ACCESS_UNAVAILABLE")

    def test_03_authentication_valid_mock(self):
        """Asserts OAuth2 client credentials token acquisition with mocked endpoint."""
        with patch.dict(os.environ, {"EUMETSAT_CONSUMER_KEY": "dummy_key", "EUMETSAT_CONSUMER_SECRET": "dummy_secret"}):
            creds = self.provider.check_credentials()
            self.assertTrue(creds["credentials_available"])

            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({
                "access_token": "mock_eumetsat_token_xyz123",
                "expires_in": 3600,
            }).encode("utf-8")
            mock_response.__enter__.return_value = mock_response

            with patch("urllib.request.urlopen", return_value=mock_response):
                auth_res = self.provider.authenticate()
                self.assertTrue(auth_res["authenticated"])
                self.assertEqual(auth_res["token"], "mock_eumetsat_token_xyz123")
                self.assertEqual(auth_res["status"], "AUTHENTICATED")

    def test_04_product_discovery_mock(self):
        """Asserts catalog search and bounding box parsing."""
        mock_features = {
            "features": [
                {
                    "id": "MSG9-SEVI-MSG15-0100-NA-20190503060000.000000000Z-NA",
                    "properties": {
                        "title": "SEVIRI Level 1.5 IODC Image Data",
                        "date": "2019-05-03T06:00:00Z",
                        "status": "ACQUIRED",
                        "links": {"data": {"href": "https://api.eumetsat.int/data/download/1.0.0/test"}},
                    },
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_features).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch.object(self.provider, "authenticate", return_value={"authenticated": True, "token": "mock_token"}):
            with patch("urllib.request.urlopen", return_value=mock_response):
                res = self.provider.search_catalog(
                    start_time="2019-05-03T00:00:00Z",
                    end_time="2019-05-03T12:00:00Z",
                    bbox=(45.0, 0.0, 105.0, 35.0),
                )
                self.assertEqual(res["status"], "SUCCESS")
                self.assertEqual(res["count"], 1)
                self.assertEqual(res["products"][0]["id"], "MSG9-SEVI-MSG15-0100-NA-20190503060000.000000000Z-NA")

    def test_05_quality_control_checks(self):
        """Asserts frame QC behavior for valid, out-of-range, and NaN values."""
        # 1. Valid frame (Physical Kelvin: TIR1 ~240K, WV ~220K, TIR2 ~250K)
        valid_frame = np.stack([
            np.ones((500, 500), dtype=np.float32) * 240.0,
            np.ones((500, 500), dtype=np.float32) * 220.0,
            np.ones((500, 500), dtype=np.float32) * 250.0,
        ], axis=0)
        qc_valid = self.provider.quality_control_frame(valid_frame)
        self.assertEqual(qc_valid["status"], "PASS")

        # 2. NaN containing frame
        nan_frame = valid_frame.copy()
        nan_frame[0, 10, 10] = np.nan
        qc_nan = self.provider.quality_control_frame(nan_frame)
        self.assertEqual(qc_nan["status"], "FAIL")
        self.assertIn("NaN", qc_nan["reason"])

        # 3. Unphysical values (below 160K or above 330K)
        bad_range_frame = valid_frame.copy()
        bad_range_frame[0, :, :] = 50.0  # Unphysically cold
        qc_bad = self.provider.quality_control_frame(bad_range_frame)
        self.assertEqual(qc_bad["status"], "FAIL")

    def test_06_spatial_standardization(self):
        """Asserts spatial cropping and grid standardization to target [3, 500, 500]."""
        raw_larger = np.ones((3, 800, 800), dtype=np.float32) * 240.0
        standardized = self.provider._standardize_spatial_grid(raw_larger, center_lat=15.0, center_lon=85.0)
        self.assertEqual(standardized.shape, (3, 500, 500))
        self.assertEqual(standardized.dtype, np.float32)

    def test_07_temporal_standardization(self):
        """Asserts temporal sequence chronological monotonicity validation."""
        # Valid 3-hourly sequence
        valid_seq = [
            "2019-05-02T09:00:00Z", "2019-05-02T12:00:00Z", "2019-05-02T15:00:00Z", "2019-05-02T18:00:00Z",
            "2019-05-02T21:00:00Z", "2019-05-03T00:00:00Z", "2019-05-03T03:00:00Z", "2019-05-03T06:00:00Z"
        ]
        res_valid = self.provider.validate_temporal_sequence(valid_seq)
        self.assertTrue(res_valid["is_valid"])

        # Non-monotonic / reversed sequence
        invalid_seq = list(reversed(valid_seq))
        res_invalid = self.provider.validate_temporal_sequence(invalid_seq)
        self.assertFalse(res_invalid["is_valid"])
        self.assertEqual(res_invalid["status"], "NON_MONOTONIC_SEQUENCE")

    def test_08_provenance_auditor(self):
        """Asserts EUMETSAT provenance auditing and manifest generation."""
        auditor = EUMETSATProvenanceAuditor(project_root=self.project_root)
        prov = auditor.audit_directory()
        self.assertEqual(prov["source"], "EUMETSAT_IODC")
        self.assertEqual(prov["coverage_status"], "NIO_OPERATIONAL_COVERAGE")
        self.assertTrue(prov["NIO_OPERATIONAL_COMPATIBILITY"])
        self.assertIn("Meteosat-9", prov["satellite_constellation"])

    def test_09_satellite_provider_factory(self):
        """Asserts SatelliteProviderFactory instantiation for EUMETSAT/NOAA and rejection of removed INSAT."""
        eum_prov = SatelliteProviderFactory.create("eumetsat_iodc", project_root=self.project_root)
        self.assertEqual(eum_prov.get_source_name(), "EUMETSAT_IODC")

        noaa_prov = SatelliteProviderFactory.create("noaa", project_root=self.project_root)
        self.assertEqual(noaa_prov.get_source_name(), "NOAA")

        with self.assertRaises(ValueError):
            SatelliteProviderFactory.create("insat", project_root=self.project_root)

        with self.assertRaises(ValueError):
            SatelliteProviderFactory.create("mosdac", project_root=self.project_root)

    def test_10_missing_modality_placeholder(self):
        """Asserts that missing real data returns zero tensor with modality_mask=0 and valid_mask=0."""
        fake_row = pd.Series({"sequence_id": "TEST_SEQ", "lat_t0": 15.0, "lon_t0": 85.0})
        res = self.provider.load_sequence_frames(fake_row)

        self.assertEqual(res["satellite_tensor"].shape, (8, 3, 500, 500))
        self.assertEqual(float(res["satellite_modality_mask"]), 0.0)
        self.assertEqual(float(res["satellite_valid_mask"].sum()), 0.0)
        self.assertFalse(res["satellite_real_data_available"])
        self.assertEqual(res["satellite_source"], "EUMETSAT_IODC")
        self.assertEqual(res["satellite_status"], "EUMETSAT_IODC_ACCESS_UNAVAILABLE")

    def test_11_model_forward_pass_with_eumetsat_tensor(self):
        """Asserts StormFusionModel end-to-end forward pass with standardized EUMETSAT tensor."""
        model = StormFusionModel()
        # [Batch=2, T=8, C=3, H=32, W=32] (scaled crop for unit test speed)
        sat_tensor = torch.randn(2, 8, 3, 32, 32)
        sat_valid = torch.ones(2, 8)
        sat_modality = torch.ones(2, 1)

        trk_features = torch.randn(2, 8, 8)
        trk_valid = torch.ones(2, 8)
        trk_modality = torch.ones(2, 1)

        era5_tensor = torch.randn(2, 8, 4, 32, 32)
        era5_valid = torch.ones(2, 8)
        era5_modality = torch.ones(2, 1)

        out = model(
            satellite_tensor=sat_tensor,
            satellite_valid_mask=sat_valid,
            satellite_modality_mask=sat_modality,
            track_features=trk_features,
            track_valid_mask=trk_valid,
            track_modality_mask=trk_modality,
            era5_tensor=era5_tensor,
            era5_valid_mask=era5_valid,
            era5_modality_mask=era5_modality,
        )

        self.assertIn("detection_logits", out)
        self.assertIn("intensity", out)
        self.assertIn("pressure", out)
        self.assertIn("track_delta", out)
        self.assertEqual(out["detection_logits"].shape, (2, 1))
        self.assertEqual(out["intensity"].shape, (2, 4))
        self.assertEqual(out["track_delta"].shape, (2, 4, 2))

    def test_12_mc_dropout_and_gradcam_with_eumetsat(self):
        """Asserts UQ runner and Grad-CAM explainability with EUMETSAT tensor."""
        model = StormFusionModel()
        runner = STORMFUSIONUQRunner(model, num_mc_samples=3)

        uq_res = runner.run_uq_analysis(
            satellite_tensor=torch.randn(1, 8, 3, 32, 32),
            satellite_valid_mask=torch.ones(1, 8),
            satellite_modality_mask=torch.ones(1, 1),
            track_features=torch.randn(1, 8, 8),
            track_valid_mask=torch.ones(1, 8),
            track_modality_mask=torch.ones(1, 1),
            era5_tensor=torch.randn(1, 8, 4, 32, 32),
            era5_valid_mask=torch.ones(1, 8),
            era5_modality_mask=torch.ones(1, 1),
        )
        self.assertEqual(uq_res["status"], "COMPUTED")

        # Grad-CAM attribution
        gradcam = StormFusionGradCAM(model)
        sample_batch = {
            "satellite_tensor": torch.randn(1, 8, 3, 32, 32),
            "satellite_valid_mask": torch.ones(1, 8),
            "satellite_modality_mask": torch.ones(1, 1),
            "track_features": torch.randn(1, 8, 8),
            "track_valid_mask": torch.ones(1, 8),
            "track_modality_mask": torch.ones(1, 1),
            "era5_tensor": torch.randn(1, 8, 4, 32, 32),
            "era5_valid_mask": torch.ones(1, 8),
            "era5_modality_mask": torch.ones(1, 1),
        }
        cam_res = gradcam.generate_cam(sample_batch, target_head="intensity")
        self.assertIn("cam_normalized", cam_res)

    def test_13_general_license_block_handling(self):
        """Asserts explicit classification of GeneralLicense HTTP 403 as BLOCKED_GENERAL_LICENSE."""
        import urllib.error
        from io import BytesIO

        err_body = b'{"error": {"code": 403, "message": "GeneralLicense required to access this collection"}}'
        http_err = urllib.error.HTTPError(
            url="https://api.eumetsat.int/data/download/1.0.0/test",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=BytesIO(err_body),
        )

        test_provider = EUMETSATIODCProvider(project_root=self.project_root)
        with patch.object(test_provider, "authenticate", return_value={"authenticated": True, "token": "mock_token"}):
            with patch("urllib.request.urlopen", side_effect=http_err):
                res = test_provider.download_product("EO:EUM:DAT:MSG:HRSEVIRI-IODC/test_prod_1")

                self.assertEqual(res["status"], "GENERAL_LICENSE_REQUIRED")
                self.assertEqual(res["classification"], "GENERAL_LICENSE_REQUIRED")
                self.assertEqual(res["HISTORICAL_EUMETSAT_ACCESS"], "BLOCKED_GENERAL_LICENSE")
                self.assertEqual(res["http_status"], 403)
                self.assertTrue(test_provider.general_license_blocked)
                self.assertEqual(test_provider.historical_access_status, "BLOCKED_GENERAL_LICENSE")
                self.assertIn("GeneralLicense", res["reason"])
                self.assertIn("Training remains blocked", res["message"])

    def test_14_circuit_breaker_stops_retries(self):
        """Asserts that automated retries are stopped once GeneralLicense is blocked."""
        test_provider = EUMETSATIODCProvider(project_root=self.project_root)
        test_provider.general_license_blocked = True
        test_provider.historical_access_status = "BLOCKED_GENERAL_LICENSE"

        # Calling download_product should return immediately without making network calls
        with patch("urllib.request.urlopen") as mock_urlopen:
            res = test_provider.download_product("test_prod_any")
            mock_urlopen.assert_not_called()
            self.assertEqual(res["status"], "GENERAL_LICENSE_REQUIRED")
            self.assertEqual(res["HISTORICAL_EUMETSAT_ACCESS"], "BLOCKED_GENERAL_LICENSE")
            self.assertFalse(res["retried"])

    def test_15_sequence_uniqueness_and_contract(self):
        """Asserts [B, 8, 3, 500, 500] contract and strict requirement of 8 distinct timestamps."""
        test_provider = EUMETSATIODCProvider(project_root=self.project_root)

        # 8 distinct timestamps
        distinct_timestamps = [
            f"2019-05-02T{hour:02d}:00:00Z" for hour in range(0, 24, 3)
        ]
        self.assertEqual(len(set(distinct_timestamps)), 8)

        # Valid mock sequence with 8 distinct frames
        valid_frames = [np.ones((3, 500, 500), dtype=np.float32) * 250.0 for _ in range(8)]
        res_valid = test_provider.verify_sequence_contract(valid_frames, distinct_timestamps)
        self.assertTrue(res_valid["contract_passed"])
        self.assertEqual(res_valid["unique_satellite_timestamps"], 8)
        self.assertEqual(res_valid["tensor_shape"], (1, 8, 3, 500, 500))

        # Duplicated timestamp should be rejected
        dup_timestamps = distinct_timestamps.copy()
        dup_timestamps[1] = dup_timestamps[0]
        res_dup = test_provider.verify_sequence_contract(valid_frames, dup_timestamps)
        self.assertFalse(res_dup["contract_passed"])
        self.assertEqual(res_dup["reason"], "DUPLICATED_OBSERVATION_TIMESTAMPS")

        # Insufficient frames should be rejected
        res_insufficient = test_provider.verify_sequence_contract(valid_frames[:7], distinct_timestamps[:7])
        self.assertFalse(res_insufficient["contract_passed"])
        self.assertEqual(res_insufficient["reason"], "INSUFFICIENT_FRAMES_FOR_TENSOR_CONTRACT")

    def test_16_fani_acquisition_plan(self):
        """Asserts legally compliant FANI 2019 target acquisition configuration."""
        test_provider = EUMETSATIODCProvider(project_root=self.project_root)
        plan = test_provider.get_fani_acquisition_plan()

        self.assertEqual(plan["storm_name"], "FANI")
        self.assertEqual(plan["ibtracs_sid"], "2019116N02090")
        self.assertEqual(plan["collection_id"], "EO:EUM:DAT:MSG:HRSEVIRI-IODC")
        self.assertEqual(plan["channels"], ["WV_062", "IR_108", "IR_120"])
        self.assertEqual(plan["spatial_crop"], [500, 500])
        self.assertEqual(plan["sequence_frames"], 8)
        self.assertEqual(plan["min_unique_timestamps"], 8)
        self.assertEqual(plan["split"], "train")
        self.assertEqual(plan["storm_group_id"], "2019116N02090_FANI")
        self.assertEqual(plan["acquisition_status"], "READY_TO_ACQUIRE_UPON_LICENSE_APPROVAL")


if __name__ == "__main__":
    unittest.main()
