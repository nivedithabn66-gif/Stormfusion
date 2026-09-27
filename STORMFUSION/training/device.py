"""STORMFUSION Device Management Utility.

Handles target execution device selection (CPU or CUDA) with clean error reporting.
"""

from typing import Dict, Any
import torch


def select_device(device_str: str = "auto") -> torch.device:
    """Selects execution device according to string configuration.

    Args:
        device_str: 'auto', 'cpu', or 'cuda'

    Returns:
        torch.device instance

    Raises:
        RuntimeError: If 'cuda' requested explicitly but CUDA is unavailable.
    """
    device_str_lower = str(device_str).lower().strip()

    if device_str_lower == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str_lower == "cpu":
        return torch.device("cpu")
    elif device_str_lower == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly requested in configuration but CUDA is unavailable on this host.")
        return torch.device("cuda")
    else:
        raise ValueError(f"Unknown device string '{device_str}'. Expected 'auto', 'cpu', or 'cuda'.")


def get_device_info(device: torch.device) -> Dict[str, Any]:
    """Returns metadata about the selected device."""
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
    }
    if torch.cuda.is_available() and device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
    return info
