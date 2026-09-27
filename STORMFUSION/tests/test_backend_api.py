"""Unit and Integration Test Suite for STORMFUSION FastAPI Backend API.
SIH Problem Statement: SIH26070 — STORMFUSION.

Covers:
1. Health check endpoint (GET /health).
2. System status endpoint (GET /api/v1/model/status) verifying EUMETSAT IODC operational provenance.
3. Multi-modal prediction endpoint (POST /api/v1/predict).
4. Monte Carlo Dropout Uncertainty Quantification (POST /api/v1/predict/uq).
5. Grad-CAM Spatial Explainability (POST /api/v1/explain).
6. Clear distinction between REAL DATA and SYNTHETIC/DEMO data modes.
"""

import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.app.main import app, startup_event


class TestBackendAPI(unittest.TestCase):
    """Test suite for STORMFUSION FastAPI backend routes."""

    @classmethod
    def setUpClass(cls):
        startup_event()
        cls.client = TestClient(app)

    def test_01_health_check(self):
        """Asserts health endpoint returns operational 200 status."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "STORMFUSION API")

    def test_02_model_status_eumetsat_provenance(self):
        """Asserts model status reports verified EUMETSAT IODC provenance and operational ready state."""
        resp = self.client.get("/api/v1/model/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["sih_problem"], "SIH26070")
        self.assertEqual(data["configured_provider"], "EUMETSAT_IODC")
        self.assertEqual(data["active_data_provider"], "EUMETSAT_IODC")
        self.assertEqual(data["access_status"], "ACCESS_READY")
        self.assertEqual(data["data_mode"], "OPERATIONAL_VERIFIED")
        self.assertIn("VERIFIED", data["eumetsat_provenance"])
        self.assertIn("REMOVED", data["mosdac_status"])
        self.assertIn("VERIFIED_REAL", data["era5_provenance"])

    def test_03_predict_endpoint(self):
        """Asserts standard multi-modal forward pass returns valid 4-horizon forecasts."""
        payload = {
            "storm_id": "TEST_STORM_2026",
            "reference_timestamp": "2026-09-08T08:12:40Z",
            "satellite_available": True,
            "track_available": True,
            "era5_available": True,
        }
        resp = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["status"], "COMPUTED")
        self.assertIn("intensity_knots", data)
        self.assertIn("pressure_hpa", data)
        self.assertIn("track_delta_deg", data)
        self.assertIn("h3", data["intensity_knots"])
        self.assertIn("h6", data["intensity_knots"])
        self.assertIn("h12", data["intensity_knots"])
        self.assertIn("h24", data["intensity_knots"])
        self.assertGreaterEqual(data["detection_probability"], 0.0)
        self.assertLessEqual(data["detection_probability"], 1.0)

    def test_04_predict_uq_endpoint(self):
        """Asserts MC Dropout returns prediction mean and uncertainty bounds."""
        payload = {
            "storm_id": "TEST_UQ_STORM",
            "reference_timestamp": "2026-09-08T08:12:40Z",
            "satellite_available": True,
            "track_available": True,
            "era5_available": True,
        }
        resp = self.client.post("/api/v1/predict/uq", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["status"], "COMPUTED")
        self.assertIn("intensity_intervals", data)
        self.assertIn("pressure_intervals", data)
        self.assertIn("track_positional_uncertainty_km", data)

    def test_05_explain_endpoint(self):
        """Asserts Grad-CAM heatmap generation returns normalized 2D attribution matrix."""
        payload = {
            "storm_id": "TEST_EXPLAIN_STORM",
            "reference_timestamp": "2026-09-08T08:12:40Z",
            "target_head": "intensity",
            "lead_time_index": 0,
        }
        resp = self.client.post("/api/v1/explain", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["status"], "COMPUTED")
        self.assertEqual(data["target_head"], "intensity")
        self.assertIsInstance(data["cam_normalized_matrix"], list)
        self.assertGreaterEqual(data["min_val"], 0.0)
        self.assertLessEqual(data["max_val"], 1.0)

    def test_06_distinguish_real_vs_synthetic_data(self):
        """Asserts that missing real modalities are accurately flagged in prediction payload."""
        payload_no_sat = {
            "storm_id": "TEST_NO_SAT",
            "reference_timestamp": "2026-09-08T08:12:40Z",
            "satellite_available": False,
            "track_available": True,
            "era5_available": True,
        }
        resp = self.client.post("/api/v1/predict", json=payload_no_sat)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "COMPUTED")
        self.assertFalse(data["modality_availability"]["satellite"])


if __name__ == "__main__":
    unittest.main()
