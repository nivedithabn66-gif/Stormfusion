"""Pydantic API Validation Schemas for STORMFUSION Backend (Step 22)."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input payload for multi-modal cyclone prediction."""
    storm_id: str = Field(..., example="TC_2026_02B")
    reference_timestamp: str = Field(..., example="2019-05-03T06:00:00Z")
    satellite_available: bool = Field(True, description="Whether satellite imagery (EUMETSAT IODC / NOAA) is available for this sequence")
    satellite_provider: str = Field("EUMETSAT_IODC", description="Satellite provider: 'EUMETSAT_IODC' or 'NOAA'")
    track_available: bool = Field(True, description="Whether IBTrACS track history is available for this sequence")
    era5_available: bool = Field(True, description="Whether ERA5 reanalysis atmospheric fields are available for this sequence")
    # Optional sequence feature arrays
    track_features: Optional[List[List[float]]] = Field(None, description="8x9 history track feature matrix")
    era5_tensor: Optional[List[List[List[List[float]]]]] = Field(None, description="8x4x32x32 ERA5 spatial grid tensor")
    satellite_tensor: Optional[List[List[List[List[float]]]]] = Field(None, description="8x3x64x64 satellite spatial tensor")


class MultiHorizonForecast(BaseModel):
    h3: float = Field(..., description="+3h forecast")
    h6: float = Field(..., description="+6h forecast")
    h12: float = Field(..., description="+12h forecast")
    h24: float = Field(..., description="+24h forecast")


class TrackHorizonDelta(BaseModel):
    h3: List[float] = Field(..., description="+3h delta lat/lon")
    h6: List[float] = Field(..., description="+6h delta lat/lon")
    h12: List[float] = Field(..., description="+12h delta lat/lon")
    h24: List[float] = Field(..., description="+24h delta lat/lon")


class PredictionResponse(BaseModel):
    """Standard STORMFUSION multi-task prediction output schema."""
    status: str = Field(..., example="COMPUTED")
    storm_id: str
    reference_timestamp: str
    configured_provider: str = Field("EUMETSAT_IODC", description="Configured satellite provider")
    active_data_provider: Optional[str] = Field(None, description="Actual source provider providing data ('NOAA', 'EUMETSAT_IODC', or None)")
    access_status: str = Field("BLOCKED", description="Access status ('ACCESS_READY', 'BLOCKED', 'EUMETSAT_UNAVAILABLE')")
    data_mode: str = Field("NO_LIVE_DATA", description="Data mode ('OPERATIONAL_VERIFIED', 'DEVELOPMENT_FALLBACK', 'NO_LIVE_DATA', 'SIMULATED')")
    active_satellite_provider: Optional[str] = Field(None, description="Backwards-compatible provider label")
    insufficient_input: bool = False
    modality_availability: Dict[str, bool]
    detection_probability: Optional[float] = None
    pattern_status: str = Field("PATTERN_LABEL_PENDING", description="Pattern classification label status")
    intensity_knots: Optional[MultiHorizonForecast] = None
    pressure_hpa: Optional[MultiHorizonForecast] = None
    track_delta_deg: Optional[TrackHorizonDelta] = None
    scientific_training_performed: bool = False


class UQPredictionResponse(PredictionResponse):
    """Prediction output schema enhanced with Uncertainty Quantification metrics."""
    num_mc_samples: int = 20
    detection_variance: Optional[float] = None
    detection_entropy: Optional[float] = None
    intensity_intervals: Optional[Dict[str, Any]] = None
    pressure_intervals: Optional[Dict[str, Any]] = None
    track_positional_uncertainty_km: Optional[Dict[str, float]] = None


class ExplainabilityRequest(BaseModel):
    """Payload requesting Grad-CAM spatial feature attribution."""
    storm_id: str = Field(..., example="TC_2026_02B")
    reference_timestamp: str = Field(..., example="2019-05-03T06:00:00Z")
    target_head: str = Field("intensity", description="Target prediction head ('detection', 'intensity', 'pressure', 'track_delta')")
    lead_time_index: int = Field(0, description="Forecast horizon index (0: 3h, 1: 6h, 2: 12h, 3: 24h)")
    target_layer: Optional[str] = Field(None, description="Target conv layer for Grad-CAM (auto-discovered if None)")


class ExplainabilityResponse(BaseModel):
    """Grad-CAM spatial attribution response schema."""
    status: str = Field(..., example="COMPUTED")
    storm_id: str
    target_head: str
    target_layer: str
    lead_time_index: int
    cam_normalized_matrix: List[List[float]] = Field(..., description="2D normalized CAM heatmap matrix [0..1]")
    min_val: float
    max_val: float
    causality_disclaimer: str


class SystemStatusResponse(BaseModel):
    """System health and scientific training status metadata."""
    service: str = "STORMFUSION Backend API"
    version: str = "1.0.0"
    sih_problem: str = "SIH26070"
    configured_provider: str = "EUMETSAT_IODC"
    active_data_provider: Optional[str] = None
    access_status: str = "BLOCKED"
    data_mode: str = "NO_LIVE_DATA"
    active_satellite_provider: Optional[str] = None
    preferred_nio_provider: str = "EUMETSAT_IODC"
    eumetsat_provenance: str = "UNAVAILABLE (0 verified real files; credentials required)"
    mosdac_status: str = "REMOVED"
    insat_provenance: str = "REMOVED"
    era5_provenance: str = "VERIFIED_REAL (1 verified file)"
    pattern_labels: str = "PATTERN_LABEL_PENDING (-1)"
    scientific_training_performed: bool = False
    training_allowed: bool = False
    training_readiness: str = "NOT_READY"
    enabled_tasks: List[str]
    satellite_providers: Dict[str, Any] = Field(default_factory=dict)
