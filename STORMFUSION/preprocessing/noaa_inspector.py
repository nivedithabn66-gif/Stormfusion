"""NOAA GOES Satellite Metadata Inspector & Physical Calibrator for STORMFUSION.

Inspects raw NetCDF4/HDF5 NOAA GOES satellite files, extracts product attributes,
determines physical calibration metadata (scale_factor, add_offset, valid_range, fill_value),
runs pixel quality control, and computes authentic physical channel variables.

STRICT RULE:
Determines actual calibration from official NOAA product metadata.
NEVER uses INSAT or arbitrary formulas (e.g. Tb = Count * 0.1 + 150).
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np

try:
    import netCDF4 as nc
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


class NOAAInspector:
    """Inspector and physical calibrator for NOAA GOES satellite imagery products."""

    def __init__(self):
        pass

    def inspect_file(self, file_path: Path) -> Dict[str, Any]:
        """Inspects a NOAA satellite data file and returns full metadata audit."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"NOAA file not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext in [".nc", ".nc4", ".cdf"] and NETCDF4_AVAILABLE:
            return self._inspect_netcdf(file_path)
        elif ext in [".h5", ".hdf", ".hdf5"] and H5PY_AVAILABLE:
            return self._inspect_h5(file_path)
        else:
            raise ValueError(f"Unsupported or non-scientific satellite file format: {file_path}")

    def _inspect_netcdf(self, file_path: Path) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "filename": file_path.name,
            "format": "NetCDF4",
            "file_size_bytes": file_path.stat().st_size,
            "global_attributes": {},
            "variables": {},
            "channels": [],
            "calibration_metadata": {},
        }
        with nc.Dataset(file_path, "r") as ds:
            for attr_name in ds.ncattrs():
                val = ds.getncattr(attr_name)
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="ignore")
                info["global_attributes"][attr_name] = str(val)

            for var_name, var in ds.variables.items():
                var_info = {
                    "shape": list(var.shape),
                    "dtype": str(var.dtype),
                    "attributes": {},
                }
                for attr in var.ncattrs():
                    val = var.getncattr(attr)
                    if isinstance(val, bytes):
                        val = val.decode("utf-8", errors="ignore")
                    var_info["attributes"][attr] = str(val)
                info["variables"][var_name] = var_info

                if "band" in var_name.lower() or "ch" in var_name.lower() or "rad" in var_name.lower() or "dqf" in var_name.lower():
                    info["channels"].append(var_name)

        # Extract calibration parameters
        info["calibration_metadata"] = self._extract_calibration_from_attributes(info["global_attributes"], info["variables"])
        return info

    def _inspect_h5(self, file_path: Path) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "filename": file_path.name,
            "format": "HDF5",
            "file_size_bytes": file_path.stat().st_size,
            "global_attributes": {},
            "variables": {},
            "channels": [],
            "calibration_metadata": {},
        }
        with h5py.File(file_path, "r") as f:
            for k, v in f.attrs.items():
                if isinstance(v, (bytes, np.bytes_)):
                    info["global_attributes"][k] = v.decode("utf-8", errors="ignore")
                else:
                    info["global_attributes"][k] = str(v)

            for dname in f.keys():
                if isinstance(f[dname], h5py.Dataset):
                    ds = f[dname]
                    var_info = {
                        "shape": list(ds.shape),
                        "dtype": str(ds.dtype),
                        "attributes": {
                            ak: (av.decode("utf-8", errors="ignore") if isinstance(av, (bytes, np.bytes_)) else str(av))
                            for ak, av in ds.attrs.items()
                        },
                    }
                    info["variables"][dname] = var_info
                    info["channels"].append(dname)

        info["calibration_metadata"] = self._extract_calibration_from_attributes(info["global_attributes"], info["variables"])
        return info

    def _inspect_generic(self, file_path: Path) -> Dict[str, Any]:
        return {
            "filename": file_path.name,
            "format": "GENERIC",
            "file_size_bytes": file_path.stat().st_size,
            "global_attributes": {"source": "NOAA_GOES", "coverage_status": "DEVELOPMENT_ONLY"},
            "variables": {"Rad": {"shape": [500, 500], "dtype": "float32"}},
            "channels": ["ABI_CH13_IR", "ABI_CH08_WV", "ABI_CH07_TIR"],
            "calibration_metadata": {
                "units": "mW m-2 sr-1 (cm-1)-1",
                "scale_factor": 1.0,
                "add_offset": 0.0,
                "valid_range": [0.0, 1000.0],
                "fill_value": -999.0,
            },
        }

    def _extract_calibration_from_attributes(self, global_attrs: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        calib = {
            "units": "mW m-2 sr-1 (cm-1)-1",
            "scale_factor": 1.0,
            "add_offset": 0.0,
            "valid_range": None,
            "fill_value": None,
        }

        # Look for the primary radiance/BT variable first (e.g. Rad)
        primary_var_name = None
        for v in ["Rad", "radiance", "brightness_temperature", "data"]:
            if v in variables:
                primary_var_name = v
                break

        target_vars = [variables[primary_var_name]] if primary_var_name else list(variables.values())

        for var_data in target_vars:
            attrs = var_data.get("attributes", {})
            if "scale_factor" in attrs:
                calib["scale_factor"] = float(attrs["scale_factor"])
            if "add_offset" in attrs:
                calib["add_offset"] = float(attrs["add_offset"])
            if "units" in attrs:
                calib["units"] = str(attrs["units"])
            if "_FillValue" in attrs:
                calib["fill_value"] = float(attrs["_FillValue"])
            if "valid_range" in attrs:
                calib["valid_range"] = attrs["valid_range"]
        return calib

    def calibrate_array(self, raw_array: np.ndarray, calib_meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calibrates raw integer counts to physical values using official NOAA attributes.

        Formula: physical = raw * scale_factor + add_offset (when metadata specifies).
        Returns calibrated array and QC metadata.
        """
        arr = raw_array.astype(np.float32)
        scale = calib_meta.get("scale_factor", 1.0)
        offset = calib_meta.get("add_offset", 0.0)
        fill_val = calib_meta.get("fill_value", None)

        valid_mask = np.ones_like(arr, dtype=bool)
        if fill_val is not None:
            valid_mask &= (arr != fill_val)
        valid_mask &= ~np.isnan(arr)
        valid_mask &= ~np.isinf(arr)

        if scale != 1.0 or offset != 0.0:
            arr[valid_mask] = arr[valid_mask] * scale + offset

        valid_pixel_pct = float(np.mean(valid_mask) * 100.0)
        qc_meta = {
            "total_pixels": int(arr.size),
            "valid_pixels": int(np.sum(valid_mask)),
            "valid_pixel_percentage": valid_pixel_pct,
            "min_val": float(np.min(arr[valid_mask])) if np.sum(valid_mask) > 0 else 0.0,
            "max_val": float(np.max(arr[valid_mask])) if np.sum(valid_mask) > 0 else 0.0,
            "mean_val": float(np.mean(arr[valid_mask])) if np.sum(valid_mask) > 0 else 0.0,
            "std_val": float(np.std(arr[valid_mask])) if np.sum(valid_mask) > 0 else 0.0,
            "units": calib_meta.get("units", "physical"),
        }

        # Set fill values to 0.0 for model consumption
        arr[~valid_mask] = 0.0
        return arr, qc_meta
