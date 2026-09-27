"""Strictly Deterministic Sequence Manifest Builder and Storm-Disjoint Splitter for STORMFUSION (Step 5).

Generates:
1. data/splits/train_storms.csv, val_storms.csv, test_storms.csv, split_metadata.json
2. data/processed/sequences/sequence_manifest.csv
3. data/processed/satellite/normalization_stats.json

Enforces 100% reproducible sorting, fixed seeding, explicit split sizing, and cryptographic SHA256 tracking.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def get_file_sha256(filepath: Path) -> str:
    """Computes SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def generate_storm_splits(
    catalog_path: Path,
    tracks_path: Path,
    splits_dir: Path,
    seed: int = 42,
    n_train_target: int = 1300,
    n_val_target: int = 279,
    n_test_target: int = 279,
):
    """Generates 100% deterministic storm-disjoint splits and writes train/val/test CSVs and metadata."""
    splits_dir.mkdir(parents=True, exist_ok=True)

    df_catalog = pd.read_csv(catalog_path)
    total_storms = len(df_catalog)

    # 1. Deterministic sorting of storm IDs BEFORE shuffling
    sorted_storm_ids = np.sort(df_catalog["storm_id"].unique())

    # 2. Fixed Random Seed Permutation
    np.random.seed(seed)
    shuffled_storm_ids = np.random.permutation(sorted_storm_ids)

    # 3. Slicing with exact approved partitioning (1300 Train, 279 Val, 279 Test)
    train_ids = sorted(list(shuffled_storm_ids[:n_train_target]))
    val_ids = sorted(list(shuffled_storm_ids[n_train_target : n_train_target + n_val_target]))
    test_ids = sorted(list(shuffled_storm_ids[n_train_target + n_val_target :]))

    df_train = df_catalog[df_catalog["storm_id"].isin(train_ids)].sort_values(by="storm_id").reset_index(drop=True)
    df_val = df_catalog[df_catalog["storm_id"].isin(val_ids)].sort_values(by="storm_id").reset_index(drop=True)
    df_test = df_catalog[df_catalog["storm_id"].isin(test_ids)].sort_values(by="storm_id").reset_index(drop=True)

    df_train.to_csv(splits_dir / "train_storms.csv", index=False)
    df_val.to_csv(splits_dir / "val_storms.csv", index=False)
    df_test.to_csv(splits_dir / "test_storms.csv", index=False)

    tracks_sha256 = get_file_sha256(tracks_path) if tracks_path.exists() else "N/A"
    catalog_sha256 = get_file_sha256(catalog_path) if catalog_path.exists() else "N/A"

    split_metadata = {
        "seed": seed,
        "input_dataset": str(tracks_path.name),
        "dataset_hash": tracks_sha256,
        "catalog_hash": catalog_sha256,
        "storm_count": total_storms,
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
        "train_ratio": round(len(train_ids) / total_storms, 4),
        "val_ratio": round(len(val_ids) / total_storms, 4),
        "test_ratio": round(len(test_ids) / total_storms, 4),
        "train_storms": [str(x) for x in train_ids],
        "val_storms": [str(x) for x in val_ids],
        "test_storms": [str(x) for x in test_ids],
    }

    with open(splits_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(split_metadata, f, indent=2)

    print(f"[SPLITS] Storm-disjoint splits generated (Seed={seed}):")
    print(f"  • Train storms : {len(train_ids)} ({len(train_ids)/total_storms*100:.1f}%)")
    print(f"  • Val storms   : {len(val_ids)} ({len(val_ids)/total_storms*100:.1f}%)")
    print(f"  • Test storms  : {len(test_ids)} ({len(test_ids)/total_storms*100:.1f}%)")

    # Map storm_id -> split name for fast lookup
    split_map = {}
    for sid in train_ids:
        split_map[sid] = "train"
    for sid in val_ids:
        split_map[sid] = "val"
    for sid in test_ids:
        split_map[sid] = "test"

    return split_map


def build_sequences(
    tracks_path: Path,
    split_map: dict,
    sequences_dir: Path,
    sat_norm_file: Path,
):
    """Builds candidate 8-timestep sequences from IBTrACS master timeline with deterministic ordering."""
    sequences_dir.mkdir(parents=True, exist_ok=True)
    sat_norm_file.parent.mkdir(parents=True, exist_ok=True)

    df_tracks = pd.read_csv(tracks_path)
    df_tracks["timestamp_dt"] = pd.to_datetime(df_tracks["timestamp"])
    df_tracks = df_tracks.sort_values(by=["storm_id", "timestamp_dt"]).reset_index(drop=True)

    grouped = df_tracks.groupby("storm_id", sort=True)

    sequence_rows = []

    # History window: 8 timesteps (t-21h to t, spaced by 3 hours)
    HISTORY_STEPS = 8
    STEP_HOURS = 3

    for storm_id, df_storm in grouped:
        df_storm = df_storm.reset_index(drop=True)
        num_obs = len(df_storm)

        if num_obs < HISTORY_STEPS:
            continue

        storm_name = df_storm["storm_name"].iloc[0]
        season = df_storm["season"].iloc[0]
        split = split_map.get(storm_id, "train")

        # Map timestamp_dt to row index within storm
        time_to_idx = {row["timestamp_dt"]: idx for idx, row in df_storm.iterrows()}

        for i in range(num_obs):
            ref_row = df_storm.iloc[i]
            ref_time = ref_row["timestamp_dt"]

            # Check if all 8 history timesteps exist at exact 3-hour intervals
            history_times = [ref_time - pd.Timedelta(hours=(HISTORY_STEPS - 1 - k) * STEP_HOURS) for k in range(HISTORY_STEPS)]
            
            has_full_history = all(t in time_to_idx for t in history_times)
            if not has_full_history:
                continue

            seq_id = f"SEQ_{storm_id}_{ref_time.strftime('%Y%m%dT%H%M%S')}"

            # Cyclone centered crop coordinates (5 deg x 5 deg)
            center_lat = float(ref_row["latitude"])
            center_lon = float(ref_row["longitude"])
            crop_lat_min = round(center_lat - 2.5, 4)
            crop_lat_max = round(center_lat + 2.5, 4)
            crop_lon_min = round(center_lon - 2.5, 4)
            crop_lon_max = round(center_lon + 2.5, 4)

            # Future target timestamps (t+3h, t+6h, t+12h, t+24h)
            target_times = {
                "t3": ref_time + pd.Timedelta(hours=3),
                "t6": ref_time + pd.Timedelta(hours=6),
                "t12": ref_time + pd.Timedelta(hours=12),
                "t24": ref_time + pd.Timedelta(hours=24),
            }

            targets = {}
            for t_key, t_val in target_times.items():
                if t_val in time_to_idx:
                    t_row = df_storm.iloc[time_to_idx[t_val]]
                    targets[f"target_lat_{t_key}"] = float(t_row["latitude"])
                    targets[f"target_lon_{t_key}"] = float(t_row["longitude"])
                    targets[f"target_wind_{t_key}"] = float(t_row["wind"]) if pd.notna(t_row["wind"]) else np.nan
                    targets[f"target_pressure_{t_key}"] = float(t_row["pressure"]) if pd.notna(t_row["pressure"]) else np.nan
                else:
                    targets[f"target_lat_{t_key}"] = np.nan
                    targets[f"target_lon_{t_key}"] = np.nan
                    targets[f"target_wind_{t_key}"] = np.nan
                    targets[f"target_pressure_{t_key}"] = np.nan

            row_dict = {
                "sequence_id": seq_id,
                "storm_id": storm_id,
                "storm_name": storm_name,
                "season": season,
                "reference_time": ref_time.strftime("%Y-%m-%d %H:%M:%S"),
                "history_start": history_times[0].strftime("%Y-%m-%d %H:%M:%S"),
                "history_end": history_times[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "center_lat": center_lat,
                "center_lon": center_lon,
                "crop_lat_min": crop_lat_min,
                "crop_lat_max": crop_lat_max,
                "crop_lon_min": crop_lon_min,
                "crop_lon_max": crop_lon_max,
                "wind_t": float(ref_row["wind"]) if pd.notna(ref_row["wind"]) else np.nan,
                "pressure_t": float(ref_row["pressure"]) if pd.notna(ref_row["pressure"]) else np.nan,
                "split": split,
            }

            # Add satellite frame metadata (All marked MISSING / valid=0 due to unverified INSAT provenance)
            for k in range(HISTORY_STEPS):
                row_dict[f"sat_t{k}"] = "MISSING"
                row_dict[f"sat_valid_t{k}"] = 0

            # Add target columns
            row_dict.update(targets)

            sequence_rows.append(row_dict)

    df_seq = pd.DataFrame(sequence_rows)

    # 4. Deterministic Manifest Sorting by ['storm_id', 'reference_time']
    df_seq = df_seq.sort_values(by=["storm_id", "reference_time"]).reset_index(drop=True)

    manifest_path = sequences_dir / "sequence_manifest.csv"
    df_seq.to_csv(manifest_path, index=False)
    print(f"[MANIFEST] Generated sequence manifest at: {manifest_path}")
    print(f"  • Total Candidate Sequences : {len(df_seq)}")
    print(f"  • Train Split Sequences     : {len(df_seq[df_seq['split']=='train'])}")
    print(f"  • Val Split Sequences       : {len(df_seq[df_seq['split']=='val'])}")
    print(f"  • Test Split Sequences      : {len(df_seq[df_seq['split']=='test'])}")

    # ----------------------------------------------------
    # Calculate Normalization Statistics ONLY from Train Split
    # ----------------------------------------------------
    df_train_seq = df_seq[df_seq["split"] == "train"]

    train_winds = df_train_seq["wind_t"].dropna().values
    train_pressures = df_train_seq["pressure_t"].dropna().values
    train_lats = df_train_seq["center_lat"].dropna().values
    train_lons = df_train_seq["center_lon"].dropna().values

    norm_stats = {
        "status": "STATUS: PLACEHOLDER / NOT FOR TRAINING",
        "track_features": {
            "latitude": {
                "mean": float(np.mean(train_lats)) if len(train_lats) > 0 else 0.0,
                "std": float(np.std(train_lats)) if len(train_lats) > 0 else 1.0,
            },
            "longitude": {
                "mean": float(np.mean(train_lons)) if len(train_lons) > 0 else 0.0,
                "std": float(np.std(train_lons)) if len(train_lons) > 0 else 1.0,
            },
            "wind": {
                "mean": float(np.mean(train_winds)) if len(train_winds) > 0 else 0.0,
                "std": float(np.std(train_winds)) if len(train_winds) > 0 else 1.0,
            },
            "pressure": {
                "mean": float(np.mean(train_pressures)) if len(train_pressures) > 0 else 0.0,
                "std": float(np.std(train_pressures)) if len(train_pressures) > 0 else 1.0,
            },
        },
        "satellite": {
            "channel": "INSAT_TIR1",
            "verified_satellite_pixels": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "status": "UNVERIFIED_SATELLITE_DATA",
        },
        "calculated_on_split": "train",
        "train_sequences_count": len(df_train_seq),
    }

    with open(sat_norm_file, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"[NORMALIZATION] Saved normalization stats to: {sat_norm_file} (STATUS: PLACEHOLDER / NOT FOR TRAINING)")
    return manifest_path


def main():
    project_root = Path(__file__).resolve().parent.parent

    catalog_path = project_root / "data/processed/tracks/storm_catalog.csv"
    tracks_path = project_root / "data/processed/tracks/ibtracs_nio_clean.csv"
    splits_dir = project_root / "data/splits"
    sequences_dir = project_root / "data/processed/sequences"
    sat_norm_file = project_root / "data/processed/satellite/normalization_stats.json"

    if not catalog_path.exists() or not tracks_path.exists():
        raise FileNotFoundError("Processed IBTrACS track data not found. Run Step 2 processor first.")

    split_map = generate_storm_splits(catalog_path, tracks_path, splits_dir, seed=42)
    build_sequences(tracks_path, split_map, sequences_dir, sat_norm_file)


if __name__ == "__main__":
    main()
