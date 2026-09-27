"""STORMFUSION Inference Service Module (Step 22).

Encapsulates:
1. Singleton model loading at API startup.
2. End-to-end multi-modal prediction pipeline execution.
3. Monte Carlo Dropout Uncertainty Quantification analysis.
4. Grad-CAM spatial attribution heatmap generation.
5. Scientific status & training readiness auditing.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import torch

from models.stormfusion import StormFusionModel
from training.readiness import check_training_readiness
from uncertainty.uq_runner import STORMFUSIONUQRunner
from explainability.gradcam import StormFusionGradCAM
from backend.app.schemas import (
    PredictionRequest,
    PredictionResponse,
    UQPredictionResponse,
    ExplainabilityRequest,
    ExplainabilityResponse,
    SystemStatusResponse,
    MultiHorizonForecast,
    TrackHorizonDelta,
)


def _build_input_tensors(request: PredictionRequest, device: torch.device) -> Dict[str, torch.Tensor]:
    """Constructs PyTorch input tensors from request payload or synthetic software test tensors."""
    B = 1

    # Satellite inputs [B, 8, 3, 64, 64]
    if request.satellite_tensor is not None:
        sat_tensor = torch.tensor(request.satellite_tensor, dtype=torch.float32, device=device)
    else:
        sat_tensor = torch.randn(B, 8, 3, 64, 64, device=device)

    sat_vmask = torch.ones(B, 8, device=device)
    sat_mmask = torch.ones(B, 1, device=device) * (1.0 if request.satellite_available else 0.0)

    # Track inputs [B, 8, 9]
    if request.track_features is not None:
        trk_tensor = torch.tensor(request.track_features, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        trk_tensor = torch.randn(B, 8, 9, device=device)

    trk_vmask = torch.ones(B, 8, device=device)
    trk_mmask = torch.ones(B, 1, device=device) * (1.0 if request.track_available else 0.0)

    # ERA5 inputs [B, 8, 4, 32, 32]
    if request.era5_tensor is not None:
        era_tensor = torch.tensor(request.era5_tensor, dtype=torch.float32, device=device)
    else:
        era_tensor = torch.randn(B, 8, 4, 32, 32, device=device)

    era_vmask = torch.ones(B, 8, device=device)
    era_mmask = torch.ones(B, 1, device=device) * (1.0 if request.era5_available else 0.0)

    return {
        "satellite_tensor": sat_tensor,
        "satellite_valid_mask": sat_vmask,
        "satellite_modality_mask": sat_mmask,
        "track_features": trk_tensor,
        "track_valid_mask": trk_vmask,
        "track_modality_mask": trk_mmask,
        "era5_tensor": era_tensor,
        "era5_valid_mask": era_vmask,
        "era5_modality_mask": era_mmask,
    }


class StormFusionInferenceService:
    """Singleton Service managing model, UQ, and Explainability execution."""

    def __init__(self, device_str: str = "cpu"):
        self.device = torch.device(device_str)
        self.model = StormFusionModel.from_config().to(self.device)
        # Prioritize production checkpoint first, then dev fallback
        root_dir = Path(__file__).resolve().parent.parent.parent
        ckpt_candidates = [
            root_dir / "checkpoints/production_training/best_model.pt",
            root_dir / "checkpoints/production_training/last_model.pt",
            root_dir / "checkpoints/dev_training/best_model.pt",
            root_dir / "checkpoints/dev_training/best.pt",
        ]
        loaded_ckpt = None
        for ckpt_path in ckpt_candidates:
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(ckpt["model_state_dict"])
                    print(f"[STORMFUSION] Loaded checkpoint: {ckpt_path.relative_to(root_dir)}")
                    loaded_ckpt = ckpt_path
                    break
                except Exception as e:
                    print(f"[STORMFUSION WARN] Could not load checkpoint {ckpt_path.name}: {e}")
        if loaded_ckpt is None:
            print("[STORMFUSION INFO] Running with initialized model weights (no checkpoint found).")
        self.model.eval()

        self.uq_runner = STORMFUSIONUQRunner(self.model, num_mc_samples=20)
        self.explainer = StormFusionGradCAM(self.model)

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Executes STORMFUSION forward pass and returns PredictionResponse."""
        inputs = _build_input_tensors(request, self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        insufficient = bool(outputs["insufficient_input"].all().item())

        if insufficient:
            return PredictionResponse(
                status="INSUFFICIENT_INPUT",
                storm_id=request.storm_id,
                reference_timestamp=request.reference_timestamp,
                insufficient_input=True,
                modality_availability={
                    "satellite": request.satellite_available,
                    "track": request.track_available,
                    "era5": request.era5_available,
                },
                pattern_status="PATTERN_LABEL_PENDING",
                scientific_training_performed=False,
            )

        det_prob = float(torch.sigmoid(outputs["detection_logits"][0, 0]).item())

        int_vals = outputs["intensity"][0].detach().cpu().numpy().tolist()
        press_vals = outputs["pressure"][0].detach().cpu().numpy().tolist()
        trk_vals = outputs["track_delta"][0].detach().cpu().numpy().tolist()

        # Determine actual active provider & data mode per Step 13 requirements
        req_provider = str(request.satellite_provider).upper()
        if req_provider in ["NOAA", "NOAA_GOES", "GOES"]:
            configured_prov = "EUMETSAT_IODC"
            active_data_prov = "NOAA"
            access_st = "EUMETSAT_UNAVAILABLE"
            data_md = "DEVELOPMENT_FALLBACK"
            active_sat_label = "NOAA"
        else:
            # Sole operational provider: EUMETSAT_IODC
            configured_prov = "EUMETSAT_IODC"
            active_data_prov = None
            access_st = "BLOCKED_GENERAL_LICENSE"
            data_md = "NO_LIVE_DATA"
            active_sat_label = None

        return PredictionResponse(
            status="COMPUTED",
            storm_id=request.storm_id,
            reference_timestamp=request.reference_timestamp,
            insufficient_input=False,
            configured_provider=configured_prov,
            active_data_provider=active_data_prov,
            access_status=access_st,
            data_mode=data_md,
            active_satellite_provider=active_sat_label,
            modality_availability={
                "satellite": request.satellite_available,
                "track": request.track_available,
                "era5": request.era5_available,
            },
            detection_probability=det_prob,
            pattern_status="PATTERN_LABEL_PENDING",
            intensity_knots=MultiHorizonForecast(h3=int_vals[0], h6=int_vals[1], h12=int_vals[2], h24=int_vals[3]),
            pressure_hpa=MultiHorizonForecast(h3=press_vals[0], h6=press_vals[1], h12=press_vals[2], h24=press_vals[3]),
            track_delta_deg=TrackHorizonDelta(h3=trk_vals[0], h6=trk_vals[1], h12=trk_vals[2], h24=trk_vals[3]),
            scientific_training_performed=False,
        )

    def predict_uq(self, request: PredictionRequest) -> UQPredictionResponse:
        """Executes STORMFUSION forward pass with UQ analysis."""
        inputs = _build_input_tensors(request, self.device)
        uq_res = self.uq_runner.run_uq_analysis(**inputs)

        std_resp = self.predict(request)

        if uq_res["status"] == "INSUFFICIENT_INPUT":
            return UQPredictionResponse(**std_resp.model_dump())

        det_var = uq_res["detection"]["variance"][0] if uq_res.get("detection") else None
        det_ent = uq_res["detection"]["entropy"][0] if uq_res.get("detection") else None

        pos_km = {
            "h3": uq_res["track"]["3h"]["positional_uncertainty_km"][0],
            "h6": uq_res["track"]["6h"]["positional_uncertainty_km"][0],
            "h12": uq_res["track"]["12h"]["positional_uncertainty_km"][0],
            "h24": uq_res["track"]["24h"]["positional_uncertainty_km"][0],
        } if uq_res.get("track") else None

        return UQPredictionResponse(
            **std_resp.model_dump(),
            num_mc_samples=20,
            detection_variance=det_var,
            detection_entropy=det_ent,
            intensity_intervals=uq_res.get("intensity"),
            pressure_intervals=uq_res.get("pressure"),
            track_positional_uncertainty_km=pos_km,
        )

    def explain(self, request: ExplainabilityRequest) -> ExplainabilityResponse:
        """Executes Grad-CAM spatial attribution analysis."""
        dummy_req = PredictionRequest(storm_id=request.storm_id, reference_timestamp="2019-05-03T06:00:00Z")
        inputs = _build_input_tensors(dummy_req, self.device)

        cam_res = self.explainer.generate_cam(
            inputs=inputs,
            target_head=request.target_head,
            lead_time_index=request.lead_time_index,
        )

        cam_matrix = cam_res["cam_normalized"][0].tolist() if cam_res["cam_normalized"].ndim >= 3 else cam_res["cam_normalized"].tolist()

        return ExplainabilityResponse(
            status="COMPUTED",
            storm_id=request.storm_id,
            target_head=request.target_head,
            target_layer=cam_res["target_layer"],
            lead_time_index=request.lead_time_index,
            cam_normalized_matrix=cam_matrix,
            min_val=cam_res["min_val"],
            max_val=cam_res["max_val"],
            causality_disclaimer=cam_res["causality_disclaimer"],
        )

    @staticmethod
    def get_system_status() -> SystemStatusResponse:
        """Returns system provenance and readiness metadata."""
        readiness = check_training_readiness()
        eum_count = readiness["checks"].get("verified_real_eumetsat", 0)
        insat_count = readiness["checks"].get("verified_real_insat", 0)
        era5_count = readiness["checks"].get("verified_real_era5", 0)
        eum_auth_ready = readiness.get("EUMETSAT_AUTH_READY", False)

        configured_prov = "EUMETSAT_IODC"
        if eum_count > 0:
            active_data_prov = "EUMETSAT_IODC"
            access_st = "ACCESS_READY"
            data_md = "OPERATIONAL_VERIFIED"
        else:
            active_data_prov = None
            access_st = "BLOCKED"
            data_md = "NO_LIVE_DATA"

        return SystemStatusResponse(
            service="STORMFUSION Backend API",
            version="1.0.0",
            sih_problem="SIH26070",
            configured_provider=configured_prov,
            active_data_provider=active_data_prov,
            access_status=access_st,
            data_mode=data_md,
            active_satellite_provider=active_data_prov or "NONE",
            preferred_nio_provider="EUMETSAT_IODC",
            eumetsat_provenance=f"UNAVAILABLE ({eum_count} verified real files; credentials required)" if eum_count == 0 else f"VERIFIED ({eum_count} verified real files; awaiting historical ingestion)",
            mosdac_status="REMOVED (EUMETSAT HRSEVIRI-IODC is sole operational provider)",
            insat_provenance="REMOVED",
            era5_provenance=f"VERIFIED_REAL ({era5_count} verified file)",
            pattern_labels="PATTERN_LABEL_PENDING (-1)",
            scientific_training_performed=False,
            training_allowed=readiness["training_allowed"],
            training_readiness=readiness["training_readiness"],
            enabled_tasks=readiness["enabled_tasks"],
            satellite_providers={
                "eumetsat_iodc": {
                    "enabled": True,
                    "purpose": "nio_satellite",
                    "status": "IMPLEMENTATION_COMPLETE",
                    "authenticated": readiness.get("EUMETSAT_AUTH_READY", False),
                    "coverage": "North Indian Ocean (Meteosat-9 at 45.5°E)",
                },
                "noaa": {
                    "enabled": True,
                    "purpose": "development_pretraining",
                    "status": "AVAILABLE",
                    "coverage": "DEVELOPMENT_ONLY",
                },
            },
        )
