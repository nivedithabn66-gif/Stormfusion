"""NOAA Satellite Integration Test Suite for STORMFUSION.

Validates:
1. Sensor-agnostic SatelliteDataProvider and SatelliteSample interface.
2. NOAA Inspector metadata parsing and physical calibration.
3. NOAA Provenance audit and SHA256 manifest generation.
4. NOAAProvider data loading with missing modality mask handling.
5. End-to-end forward pass through StormFusionModel.
6. Monte Carlo Dropout UQ integration.
7. Grad-CAM XAI spatial explainability integration.
8. Updated Training Readiness Safeguard Gate (5 status flags).
"""

import json
import sys
from pathlib import Path
import unittest
import torch
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preprocessing.satellite_provider import SatelliteDataProvider, SatelliteSample

from preprocessing.noaa_inspector import NOAAInspector
from preprocessing.noaa_provenance import NOAAProvenanceAuditor
from preprocessing.noaa_satellite_provider import NOAAProvider
from preprocessing.normalization import compute_noaa_normalization
from models.stormfusion import StormFusionModel
from uncertainty.uq_runner import STORMFUSIONUQRunner
from explainability.gradcam import StormFusionGradCAM
from training.readiness import check_training_readiness



class TestNOAAPipeline(unittest.TestCase):
    """Test suite for NOAA satellite integration."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent

    def test_01_sensor_agnostic_interface(self):
        """Asserts SatelliteSample container and SatelliteDataProvider interface."""
        sample_tensor = torch.zeros((8, 3, 500, 500), dtype=torch.float32)
        valid_mask = torch.ones(8, dtype=torch.float32)
        sample = SatelliteSample(
            image=sample_tensor,
            timestamp=["2023-01-01T00:00:00Z"] * 8,
            latitude=15.0,
            longitude=85.0,
            channels=["ABI_CH07", "ABI_CH08", "ABI_CH13"],
            source="NOAA_GOES",
            validity_mask=valid_mask,
            modality_mask=1.0,
            quality_metadata={"coverage_status": "DEVELOPMENT_ONLY"},
        )
        s_dict = sample.to_dict()
        self.assertEqual(s_dict["satellite_source"], "NOAA_GOES")
        self.assertTrue(s_dict["satellite_real_data_available"])
        self.assertEqual(s_dict["satellite_tensor"].shape, (8, 3, 500, 500))

    def test_02_noaa_inspector_and_calibration(self):
        """Asserts NOAAInspector product inspection and physical calibration."""
        inspector = NOAAInspector()
        calib_meta = {
            "units": "mW m-2 sr-1 (cm-1)-1",
            "scale_factor": 0.05,
            "add_offset": 10.0,
            "fill_value": -999.0,
        }
        raw_arr = np.ones((50, 50), dtype=np.float32) * 100.0
        cal_arr, qc = inspector.calibrate_array(raw_arr, calib_meta)
        self.assertEqual(cal_arr[0, 0], 100.0 * 0.05 + 10.0)  # 15.0
        self.assertEqual(qc["valid_pixel_percentage"], 100.0)

    def test_03_noaa_provenance_auditor(self):
        """Asserts NOAA provenance auditing and manifest generation."""
        auditor = NOAAProvenanceAuditor(project_root=self.project_root)
        status = auditor.audit_noaa_directory()
        self.assertIn("source", status)
        self.assertEqual(status["source"], "NOAA_GOES")
        self.assertEqual(status["coverage_status"], "DEVELOPMENT_ONLY")
        self.assertFalse(status["NIO_OPERATIONAL_COMPATIBILITY"])

    def test_04_noaa_provider_loading(self):
        """Asserts NOAAProvider sequence loading and missing mask handling."""
        provider = NOAAProvider(project_root=self.project_root, crop_size=(100, 100))
        self.assertEqual(provider.get_source_name(), "NOAA")

        # Fake sequence row
        row = pd.Series({"sequence_id": "SEQ_001", "lat_t0": 15.0, "lon_t0": 85.0})
        res = provider.load_sequence_frames(row)
        self.assertIn("satellite_tensor", res)
        self.assertEqual(res["satellite_tensor"].shape, (8, 3, 100, 100))
        self.assertIn("satellite_source", res)

    def test_05_model_forward_pass_with_noaa(self):
        """Asserts StormFusionModel forward pass with NOAA satellite input tensors."""
        model = StormFusionModel()
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

        self.assertEqual(out["detection_logits"].shape, (2, 1))
        self.assertEqual(out["intensity"].shape, (2, 4))
        self.assertEqual(out["track_delta"].shape, (2, 4, 2))

    def test_06_mc_dropout_uq_integration(self):
        """Asserts Monte Carlo Dropout UQ evaluator compatibility."""
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
        self.assertIn("status", uq_res)
        self.assertEqual(uq_res["status"], "COMPUTED")
        self.assertIn("intensity", uq_res)


    def test_07_gradcam_xai_integration(self):
        """Asserts Grad-CAM spatial explainability attribution map generation."""
        model = StormFusionModel()
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
        self.assertEqual(cam_res["target_head"], "intensity")


    def test_08_readiness_gate_five_status_flags(self):
        """Asserts training readiness gate 5 status flags."""
        status = check_training_readiness(project_root=self.project_root)
        self.assertIn("NOAA_PIPELINE_READY", status)
        self.assertTrue(status["NOAA_PIPELINE_READY"])
        self.assertIn("NOAA_REAL_DATA_AVAILABLE", status)
        self.assertIn("EUMETSAT_REAL_DATA_AVAILABLE", status)
        self.assertEqual(status.get("MOSDAC"), "REMOVED")
        self.assertIn("NIO_SATELLITE_TRAINING_READY", status)
        self.assertIn("SCIENTIFIC_TRAINING_READY", status)


if __name__ == "__main__":
    unittest.main()
