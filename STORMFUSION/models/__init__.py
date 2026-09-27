"""STORMFUSION Models Package."""

from models.convlstm import ConvLSTMCell, ConvLSTM
from models.satellite_encoder import ResNet18SpatialEncoder
from models.baseline_satellite_model import BaselineSatelliteModel
from models.track_encoder import TrackGRUEncoder
from models.era5_encoder import ERA5Encoder
from models.gated_fusion import AvailabilityAwareGatedFusion
from models.multitask_model import StormFusionMultiTaskModel
from models.stormfusion import StormFusionModel
from models.heads import (
    DetectionHead,
    PatternClassificationHead,
    IntensityHead,
    PressureHead,
    TrackPredictionHead,
)
from models.losses import (
    detection_loss,
    pattern_loss,
    masked_huber_loss,
    track_loss,
    total_multi_task_loss,
)

__all__ = [
    "ConvLSTMCell",
    "ConvLSTM",
    "ResNet18SpatialEncoder",
    "BaselineSatelliteModel",
    "TrackGRUEncoder",
    "ERA5Encoder",
    "AvailabilityAwareGatedFusion",
    "StormFusionMultiTaskModel",
    "StormFusionModel",
    "DetectionHead",
    "PatternClassificationHead",
    "IntensityHead",
    "PressureHead",
    "TrackPredictionHead",
    "detection_loss",
    "pattern_loss",
    "masked_huber_loss",
    "track_loss",
    "total_multi_task_loss",
]


