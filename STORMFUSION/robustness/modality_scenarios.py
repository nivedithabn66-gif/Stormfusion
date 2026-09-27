"""Modality Availability Scenario Generator for STORMFUSION.

Generates 8 deterministic availability mask combinations across Satellite, Track, and ERA5:
1. all_available            (sat=1, track=1, era5=1)
2. satellite_missing        (sat=0, track=1, era5=1)
3. track_missing            (sat=1, track=0, era5=1)
4. era5_missing             (sat=1, track=1, era5=0)
5. satellite_track_missing  (sat=0, track=0, era5=1)
6. satellite_era5_missing   (sat=0, track=1, era5=0)
7. track_era5_missing       (sat=1, track=0, era5=0)
8. all_missing              (sat=0, track=0, era5=0)
"""

from typing import Dict, Any, List
import torch


def get_modality_scenarios(batch_size: int = 1, device: str = "cpu") -> Dict[str, Dict[str, Any]]:
    """Generates all 8 deterministic modality availability mask combinations.

    Args:
        batch_size: Batch dimension B
        device: PyTorch device ('cpu' or 'cuda')

    Returns:
        Dict mapping scenario_name to scenario definition containing:
            - satellite_mask: Tensor [B, 1]
            - track_mask: Tensor [B, 1]
            - era5_mask: Tensor [B, 1]
            - expected_valid: bool
            - num_available: int
    """
    scenarios_def = {
        "all_available": (1.0, 1.0, 1.0),
        "satellite_missing": (0.0, 1.0, 1.0),
        "track_missing": (1.0, 0.0, 1.0),
        "era5_missing": (1.0, 1.0, 0.0),
        "satellite_track_missing": (0.0, 0.0, 1.0),
        "satellite_era5_missing": (0.0, 1.0, 0.0),
        "track_era5_missing": (1.0, 0.0, 0.0),
        "all_missing": (0.0, 0.0, 0.0),
    }

    scenarios = {}
    for name, (sat_v, trk_v, era_v) in scenarios_def.items():
        sat_mask = torch.full((batch_size, 1), sat_v, dtype=torch.float32, device=device)
        trk_mask = torch.full((batch_size, 1), trk_v, dtype=torch.float32, device=device)
        era_mask = torch.full((batch_size, 1), era_v, dtype=torch.float32, device=device)

        num_avail = int(sat_v + trk_v + era_v)
        expected_valid = num_avail > 0

        scenarios[name] = {
            "name": name,
            "satellite_mask": sat_mask,
            "track_mask": trk_mask,
            "era5_mask": era_mask,
            "expected_valid": expected_valid,
            "num_available": num_avail,
            "mask_values": (int(sat_v), int(trk_v), int(era_v)),
        }

    return scenarios
