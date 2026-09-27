"""PyTorch Dataset & Custom DataLoader Collation Module for STORMFUSION (Step 7).

Implements StormSequenceDataset and storm_collate_fn for sequence-level multimodal cyclone data loading.

Integrates:
  1. Satellite Provider (SatelliteProviderFactory): EUMETSAT IODC / NOAA satellite data.
  2. ERA5 Loader (ERA5Loader): NetCDF reanalysis loader.
  3. Track Loader (TrackAndTargetLoader): 8-timestep history features strictly without future leakage.
  4. Target Extractor: Future lead targets (+3h, +6h, +12h, +24h) for track, intensity, pressure, detection, pattern.

All missing modalities return explicit modality and validity masks with real_data_available flags in metadata.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import yaml

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset:
        pass

from preprocessing.satellite_provider import SatelliteProviderFactory
from preprocessing.noaa_satellite_provider import NOAAProvider
from preprocessing.era5_loader import ERA5Loader
from preprocessing.track_loader import TrackAndTargetLoader


class StormSequenceDataset(Dataset):
    """PyTorch Dataset for STORMFUSION Cyclone Sequences."""

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        split: Optional[str] = None,
        config_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for StormSequenceDataset.")

        self.project_root = Path(__file__).resolve().parent.parent

        if config is not None:
            self.config = config
        else:
            if config_path is None:
                config_path = self.project_root / "configs/dataset_config.yaml"
            
            self.config = {}
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)

        if manifest_path is None:
            rel_manifest = self.config.get("paths", {}).get("manifest_path", "data/processed/sequences/sequence_manifest.csv")
            manifest_path = self.project_root / rel_manifest

        if not manifest_path.exists():
            raise FileNotFoundError(f"Sequence manifest not found at: {manifest_path}")

        self.df_manifest = pd.read_csv(manifest_path)
        
        if split is not None:
            valid_splits = ["train", "val", "test"]
            if split not in valid_splits:
                raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}.")
            self.df_manifest = self.df_manifest[self.df_manifest["split"] == split].reset_index(drop=True)

        self.split = split if split else "all"

        # Initialize sub-loaders with configurable provider
        crop_size = tuple(self.config.get("satellite", {}).get("crop_size", [500, 500]))
        provider_name = str(self.config.get("satellite", {}).get("provider", self.config.get("satellite", {}).get("source", "eumetsat_iodc")))

        self.satellite_loader = SatelliteProviderFactory.create(
            provider_name=provider_name,
            project_root=self.project_root,
            crop_size=crop_size,
        )

        self.era5_loader = ERA5Loader(project_root=self.project_root)
        self.track_loader = TrackAndTargetLoader(project_root=self.project_root)


    def __len__(self) -> int:
        return len(self.df_manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df_manifest.iloc[idx]

        sample_id = str(row["sequence_id"])
        storm_id = str(row["storm_id"])
        timestamp = str(row["reference_time"])

        # 1. Satellite modality
        sat_data = self.satellite_loader.load_sequence_frames(row)

        # 2. Track modality & features (t-21h to t: 8 timesteps)
        track_data = self.track_loader.extract_sequence_track_features(storm_id, timestamp)

        # 3. Environment modality (ERA5)
        era5_data = self.era5_loader.load_sequence_environment(storm_id, timestamp)

        # 4. Target extraction
        targets_data = self.track_loader.extract_targets(row)

        # 5. Metadata
        metadata = {
            "sample_id": sample_id,
            "storm_id": storm_id,
            "storm_name": str(row.get("storm_name", "UNNAMED")),
            "season": int(row.get("season", 0)),
            "timestamp": timestamp,
            "split": str(row.get("split", self.split)),
            "center_lat": float(row["center_lat"]),
            "center_lon": float(row["center_lon"]),
            "crop_lat_min": float(row["crop_lat_min"]),
            "crop_lat_max": float(row["crop_lat_max"]),
            "crop_lon_min": float(row["crop_lon_min"]),
            "crop_lon_max": float(row["crop_lon_max"]),
            "satellite_real_data_available": sat_data["satellite_real_data_available"],
            "era5_real_data_available": era5_data["era5_real_data_available"],
            "satellite_provider": sat_data.get("satellite_source", self.satellite_loader.get_source_name()),
            "satellite_status": sat_data["satellite_status"],
            "era5_status": era5_data["era5_status"],
        }

        return {
            "sample_id": sample_id,
            "storm_id": storm_id,
            "timestamp": timestamp,
            "satellite": {
                "satellite_tensor": sat_data["satellite_tensor"],           # [8, 3, 500, 500]
                "satellite_valid_mask": sat_data["satellite_valid_mask"],   # [8]
                "satellite_modality_mask": sat_data["satellite_modality_mask"], # [1] (0.0 for unavailable)
            },
            "track": {
                "track_features": track_data["track_features"],           # [8, 9]
                "track_valid_mask": track_data["track_valid_mask"],       # [8]
            },
            "environment": {
                "era5_features": era5_data["era5_tensor"],               # [8, 4, 32, 32]
                "era5_valid_mask": era5_data["era5_valid_mask"],         # [8]
                "era5_modality_mask": era5_data["era5_modality_mask"],   # [1] (0.0 for unavailable)
            },
            "targets": targets_data,
            "metadata": metadata,
        }


def storm_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom batch collation function preserving metadata, modality masks, validity masks, and target tensors."""
    sample_ids = [item["sample_id"] for item in batch]
    storm_ids = [item["storm_id"] for item in batch]
    timestamps = [item["timestamp"] for item in batch]

    # Satellite batching
    sat_tensors = torch.stack([item["satellite"]["satellite_tensor"] for item in batch], dim=0) # [B, 8, 3, H, W]
    sat_valid_masks = torch.stack([item["satellite"]["satellite_valid_mask"] for item in batch], dim=0) # [B, 8]
    sat_modality_masks = torch.stack([item["satellite"]["satellite_modality_mask"] for item in batch], dim=0).unsqueeze(-1) if batch[0]["satellite"]["satellite_modality_mask"].dim() == 0 else torch.stack([item["satellite"]["satellite_modality_mask"] for item in batch], dim=0)

    # Track batching
    track_features = torch.stack([item["track"]["track_features"] for item in batch], dim=0) # [B, 8, 9]
    track_valid_masks = torch.stack([item["track"]["track_valid_mask"] for item in batch], dim=0) # [B, 8]

    # ERA5 batching
    era5_tensors = torch.stack([item["environment"]["era5_features"] for item in batch], dim=0) # [B, 8, 4, H, W]
    era5_valid_masks = torch.stack([item["environment"]["era5_valid_mask"] for item in batch], dim=0) # [B, 8]
    era5_modality_masks = torch.stack([item["environment"]["era5_modality_mask"] for item in batch], dim=0).unsqueeze(-1) if batch[0]["environment"]["era5_modality_mask"].dim() == 0 else torch.stack([item["environment"]["era5_modality_mask"] for item in batch], dim=0)

    # Targets batching
    detection_targets = torch.stack([item["targets"]["detection"] for item in batch], dim=0) # [B]
    pattern_targets = torch.stack([item["targets"]["pattern"] for item in batch], dim=0) # [B]
    intensity_targets = torch.stack([item["targets"]["intensity"] for item in batch], dim=0) # [B, 4]
    intensity_valid_masks = torch.stack([item["targets"]["intensity_valid_mask"] for item in batch], dim=0) # [B, 4]
    pressure_targets = torch.stack([item["targets"]["pressure"] for item in batch], dim=0) # [B, 4]
    pressure_valid_masks = torch.stack([item["targets"]["pressure_valid_mask"] for item in batch], dim=0) # [B, 4]
    future_track_targets = torch.stack([item["targets"]["future_track"] for item in batch], dim=0) # [B, 4, 2]
    future_track_valid_masks = torch.stack([item["targets"]["future_track_valid_mask"] for item in batch], dim=0) # [B, 4]

    # Metadata batch list
    metadata_list = [item["metadata"] for item in batch]

    return {
        "sample_id": sample_ids,
        "storm_id": storm_ids,
        "timestamp": timestamps,
        "satellite": {
            "satellite_tensor": sat_tensors,
            "satellite_valid_mask": sat_valid_masks,
            "satellite_modality_mask": sat_modality_masks,
        },
        "track": {
            "track_features": track_features,
            "track_valid_mask": track_valid_masks,
        },
        "environment": {
            "era5_features": era5_tensors,
            "era5_valid_mask": era5_valid_masks,
            "era5_modality_mask": era5_modality_masks,
        },
        "targets": {
            "detection": detection_targets,
            "pattern": pattern_targets,
            "intensity": intensity_targets,
            "intensity_valid_mask": intensity_valid_masks,
            "pressure": pressure_targets,
            "pressure_valid_mask": pressure_valid_masks,
            "future_track": future_track_targets,
            "future_track_valid_mask": future_track_valid_masks,
        },
        "metadata": metadata_list,
    }
