"""Sensor-Agnostic Satellite Data Ingestion Architecture for STORMFUSION.

Defines:
1. `SatelliteSample`: Standardized data container for multi-spectral satellite imagery sequences.
2. `SatelliteDataProvider`: Abstract base class for sensor-agnostic satellite data providers.
3. `SatelliteProviderFactory`: Factory for instantiating EUMETSAT IODC or NOAA satellite providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False



@dataclass
class SatelliteSample:
    """Standardized representation of a multi-spectral satellite imagery sequence.

    Shape: [T, C, H, W] where T=sequence_length (e.g. 8), C=channels (e.g. 3), H,W=spatial crop dims.
    """
    image: Any  # torch.Tensor or np.ndarray [T, C, H, W]
    timestamp: List[str]
    latitude: float
    longitude: float
    channels: List[str]
    source: str  # 'EUMETSAT_IODC' or 'NOAA'
    validity_mask: Any  # torch.Tensor or np.ndarray [T]
    modality_mask: float  # 1.0 if modality available, 0.0 otherwise
    quality_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts sample to standard dictionary output for DataLoader batching."""
        return {
            "satellite_tensor": self.image,
            "satellite_valid_mask": self.validity_mask,
            "satellite_modality_mask": (
                torch.tensor(self.modality_mask, dtype=torch.float32)
                if TORCH_AVAILABLE and not isinstance(self.modality_mask, torch.Tensor)
                else self.modality_mask
            ),
            "satellite_real_data_available": self.modality_mask > 0.0,
            "satellite_source": self.source,
            "satellite_channels": self.channels,
            "satellite_status": (
                f"{self.source}_VERIFIED_REAL_ACTIVE"
                if self.modality_mask > 0.0
                else f"UNAVAILABLE_NO_VERIFIED_{self.source}_REAL_DATA"
            ),
            "quality_metadata": self.quality_metadata,
        }


class SatelliteDataProvider(ABC):
    """Abstract Base Class for sensor-agnostic satellite data providers."""

    def __init__(self, project_root: Optional[Path] = None, crop_size: Tuple[int, int] = (500, 500)):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root)
        self.crop_size = crop_size

    @abstractmethod
    def get_source_name(self) -> str:
        """Returns provider source identifier ('EUMETSAT_IODC' or 'NOAA')."""
        pass

    @abstractmethod
    def is_real_data_available(self) -> bool:
        """Returns True if authentic verified real satellite data exists for training."""
        pass

    @abstractmethod
    def load_sequence_frames(self, sequence_row: Any) -> Dict[str, Any]:
        """Loads 8-timestep satellite frame sequence given a sequence manifest row.

        Returns standard dictionary with 'satellite_tensor', 'satellite_valid_mask',
        'satellite_modality_mask', and metadata.
        """
        pass


class SatelliteProviderFactory:
    """Factory for instantiating sensor-agnostic satellite data providers.
    
    Supported provider keys:
      - 'eumetsat_iodc' (default): Primary operational satellite provider for North Indian Ocean (Meteosat-9/8 HRSEVIRI-IODC).
      - 'noaa': Development and pretraining provider (GOES open data).
    """

    @staticmethod
    def create(
        provider_name: str = "eumetsat_iodc",
        project_root: Optional[Path] = None,
        crop_size: Tuple[int, int] = (500, 500),
        allow_fallback: bool = False,
        allow_mosdac: bool = False,
    ) -> SatelliteDataProvider:
        """Instantiates and returns the requested SatelliteDataProvider.
        
        Args:
            provider_name: 'eumetsat_iodc', 'noaa', or 'mosdac_provider'
            project_root: Repository root path
            crop_size: Target spatial crop size (H, W)
            allow_fallback: If True and requested provider has no real data, falls back to NOAA
            allow_mosdac: If True, enables MOSDAC instantiation under legacy 'mosdac'/'insat' keys
        """
        p_name = provider_name.strip().lower()

        if p_name in ["eumetsat", "eumetsat_iodc", "eumetsat-iodc", "iodc"]:
            from preprocessing.eumetsat_iodc_provider import EUMETSATIODCProvider
            provider = EUMETSATIODCProvider(project_root=project_root, crop_size=crop_size)
            if not provider.is_real_data_available() and allow_fallback:
                from preprocessing.noaa_satellite_provider import NOAAProvider
                print("[PROVIDER FACTORY] EUMETSAT real data unavailable; falling back to NOAA (DEVELOPMENT MODE).")
                return NOAAProvider(project_root=project_root, crop_size=crop_size)
            return provider

        elif p_name in ["noaa", "noaa_goes", "goes"]:
            from preprocessing.noaa_satellite_provider import NOAAProvider
            return NOAAProvider(project_root=project_root, crop_size=crop_size)

        elif p_name in ["mosdac_provider", "isro_mosdac", "mosdac_insat"] or (p_name in ["insat", "mosdac"] and allow_mosdac):
            from preprocessing.mosdac_provider import MOSDACProvider
            return MOSDACProvider(project_root=project_root, crop_size=crop_size)

        elif p_name in ["insat", "mosdac"]:
            raise ValueError(
                "MOSDAC / INSAT provider is disabled by default in operational mode to prevent pipeline regression. "
                "Use provider_name='mosdac_provider' or pass allow_mosdac=True to access the additional MOSDAC provider."
            )

        else:
            raise ValueError(
                f"Unknown satellite provider: '{provider_name}'. "
                f"Supported providers: ['eumetsat_iodc', 'noaa', 'mosdac_provider']."
            )
