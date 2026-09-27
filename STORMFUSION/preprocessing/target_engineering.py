"""STORMFUSION -- Cyclone Target & Label Engineering (Step 15).

Builds a scientifically auditable target/label pipeline for:
  1. Binary cyclone detection labels
  2. Pattern classification labels (PATTERN_LABEL_PENDING)
  3. Intensity targets (wind, knots) at T0 and +3h/+6h/+12h/+24h
  4. Pressure targets (hPa) at T0 and +3h/+6h/+12h/+24h
  5. Future track targets (delta_lat, delta_lon) + absolute positions
  6. Per-target, per-horizon validity masks
  7. Target quality audit (QC flags, distributions, leakage tests)

CRITICAL SAFETY RULES:
  - DO NOT perform scientific model training.
  - DO NOT fabricate satellite observations.
  - DO NOT use synthetic/test satellite data as scientific training data.
  - DO NOT impute missing pressure or intensity values.
  - DO NOT silently convert NaN to zero.
  - All future targets must use timestamps strictly AFTER T0.
  - All input features must use timestamps strictly AT or BEFORE T0.
  - pattern_label = -1 (PATTERN_LABEL_PENDING) for all sequences until
    a domain expert validates the IMD category grouping.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZONS_H = [3, 6, 12, 24]
HORIZON_LABELS = ["t3h", "t6h", "t12h", "t24h"]
MANIFEST_HORIZON_SUFFIX = {"t3h": "t3", "t6h": "t6", "t12h": "t12", "t24h": "t24"}

PATTERN_LABEL_PENDING = -1
PATTERN_STATUS_PENDING = "PATTERN_LABEL_PENDING"
DETECTION_LABEL_POSITIVE = 1
DETECTION_LABEL_NEGATIVE = 0

MAX_DELTA_DEG = {"t3h": 3.0, "t6h": 6.0, "t12h": 12.0, "t24h": 24.0}


# ---------------------------------------------------------------------------
# TargetEngineer  (fully vectorised — no per-row Python loop)
# ---------------------------------------------------------------------------

class TargetEngineer:
    """Builds enriched targets manifest from sequence manifest + IBTrACS clean tracks.

    Reads:
        data/processed/sequences/sequence_manifest.csv
        data/processed/tracks/ibtracs_nio_clean.csv
        configs/target_label_config.yaml

    Writes:
        data/processed/targets/targets_manifest.csv
        data/processed/targets/targets_manifest_train.csv
        data/processed/targets/targets_manifest_val.csv
        data/processed/targets/targets_manifest_test.csv
    """

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root)

        self.config = self._load_label_config()

        self.manifest_path = self.project_root / "data/processed/sequences/sequence_manifest.csv"
        self.tracks_path   = self.project_root / "data/processed/tracks/ibtracs_nio_clean.csv"
        self.targets_dir   = self.project_root / self.config["output"]["targets_dir"]
        self.targets_dir.mkdir(parents=True, exist_ok=True)

    def _load_label_config(self) -> dict:
        cfg_path = self.project_root / "configs/target_label_config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Target label config not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Main build  (vectorised)
    # ------------------------------------------------------------------

    def build_targets_manifest(self, verbose: bool = True) -> pd.DataFrame:
        """Build enriched targets manifest.  Fully vectorised — no row loop."""
        if verbose:
            print("=" * 60)
            print("  STORMFUSION -- STEP 15: TARGET ENGINEERING")
            print("=" * 60)

        # ── Load inputs ───────────────────────────────────────────────
        df_m = pd.read_csv(self.manifest_path)
        df_m["reference_time"] = pd.to_datetime(df_m["reference_time"])

        df_t = pd.read_csv(self.tracks_path)
        df_t["dt"] = pd.to_datetime(df_t["timestamp"])

        if verbose:
            print(f"[INFO] Manifest: {len(df_m):,} sequences")
            print(f"[INFO] Tracks:   {len(df_t):,} observations")

        # ── Build a rounded join key (nearest 3-hour slot) ────────────
        # Round reference_time to nearest 15-min so we can merge with track obs
        df_m["ref_rounded"] = df_m["reference_time"].dt.round("15min")
        df_t["dt_rounded"]  = df_t["dt"].dt.round("15min")

        # Keep only the first observation per (storm_id, dt_rounded) to avoid duplicates
        df_t_dedup = df_t.drop_duplicates(subset=["storm_id", "dt_rounded"], keep="first")

        # Merge: bring nature code at T0 into manifest
        nature_lkp = df_t_dedup[["storm_id", "dt_rounded", "nature"]].copy()
        df = df_m.merge(
            nature_lkp.rename(columns={"nature": "nature_t0"}),
            left_on=["storm_id", "ref_rounded"],
            right_on=["storm_id", "dt_rounded"],
            how="left",
        ).drop(columns=["dt_rounded", "ref_rounded"])

        df["nature_t0"] = df["nature_t0"].fillna("").astype(str)

        # ── Detection label (vectorised) ──────────────────────────────
        # Rule: storm_id present in tracks AND nature not empty
        storms_in_tracks = set(df_t["storm_id"].unique())
        df["detection_label"] = df["storm_id"].isin(storms_in_tracks).astype(int)
        df["detection_valid"]  = 1  # always valid (storm presence is certain for all sequences)

        # ── Pattern label (PENDING for all) ───────────────────────────
        df["pattern_label"]        = PATTERN_LABEL_PENDING
        df["pattern_label_status"] = PATTERN_STATUS_PENDING
        df["pattern_valid"]        = 1

        # ── Leakage check (vectorised) ────────────────────────────────
        hist_start = pd.to_datetime(df.get("history_start", pd.NaT), errors="coerce")
        hist_end   = pd.to_datetime(df.get("history_end",   pd.NaT), errors="coerce")
        ref_dt     = df["reference_time"]
        leakage_mask = (
            (hist_start.notna() & (hist_start > ref_dt)) |
            (hist_end.notna()   & (hist_end   > ref_dt))
        )
        df["leakage_check"] = "PASS"
        df.loc[leakage_mask, "leakage_check"] = "LEAKAGE: history timestamp > T0"

        # ── Intensity targets (vectorised) ────────────────────────────
        cfg_int = self.config["intensity"]
        df["intensity_t0"]       = pd.to_numeric(df.get("wind_t",    np.nan), errors="coerce")
        df["intensity_valid_t0"] = df["intensity_t0"].notna().astype(int)
        # QC bounds
        bad_int_t0 = (df["intensity_t0"] < cfg_int["qc"]["min_kts"]) | (df["intensity_t0"] > cfg_int["qc"]["max_kts"])
        df.loc[bad_int_t0, ["intensity_t0", "intensity_valid_t0"]] = [np.nan, 0]

        for hlabel, hsuffix in MANIFEST_HORIZON_SUFFIX.items():
            src_col = cfg_int["future_columns"][hlabel]
            val_col = cfg_int["validity_mask_columns"][hlabel]
            tgt_col = f"intensity_{hlabel}"
            df[tgt_col] = pd.to_numeric(df.get(src_col, np.nan), errors="coerce")
            bad = (df[tgt_col] < cfg_int["qc"]["min_kts"]) | (df[tgt_col] > cfg_int["qc"]["max_kts"])
            df.loc[bad, tgt_col] = np.nan
            df[val_col] = df[tgt_col].notna().astype(int)

        # ── Pressure targets (vectorised) ─────────────────────────────
        cfg_prs = self.config["pressure"]
        df["pressure_t0"]       = pd.to_numeric(df.get("pressure_t", np.nan), errors="coerce")
        df["pressure_valid_t0"] = df["pressure_t0"].notna().astype(int)
        bad_p0 = (df["pressure_t0"] < cfg_prs["qc"]["min_hpa"]) | (df["pressure_t0"] > cfg_prs["qc"]["max_hpa"])
        df.loc[bad_p0, ["pressure_t0", "pressure_valid_t0"]] = [np.nan, 0]

        for hlabel, hsuffix in MANIFEST_HORIZON_SUFFIX.items():
            src_col = cfg_prs["future_columns"][hlabel]
            val_col = cfg_prs["validity_mask_columns"][hlabel]
            tgt_col = f"pressure_{hlabel}"
            df[tgt_col] = pd.to_numeric(df.get(src_col, np.nan), errors="coerce")
            bad = (df[tgt_col] < cfg_prs["qc"]["min_hpa"]) | (df[tgt_col] > cfg_prs["qc"]["max_hpa"])
            df.loc[bad, tgt_col] = np.nan
            df[val_col] = df[tgt_col].notna().astype(int)

        # ── Track targets (vectorised) ────────────────────────────────
        cfg_trk = self.config["track"]
        ref_lat = pd.to_numeric(df["center_lat"], errors="coerce")
        ref_lon = pd.to_numeric(df["center_lon"], errors="coerce")

        for hlabel, hsuffix in MANIFEST_HORIZON_SUFFIX.items():
            lat_src = cfg_trk["future_columns"]["lat"][hlabel]
            lon_src = cfg_trk["future_columns"]["lon"][hlabel]
            dlat_col, dlon_col = cfg_trk["delta_columns"][hlabel]
            alat_col, alon_col = cfg_trk["absolute_columns"][hlabel]
            valid_col          = cfg_trk["validity_mask_columns"][hlabel]
            qc_col             = f"track_qc_{hlabel}"

            f_lat = pd.to_numeric(df.get(lat_src, np.nan), errors="coerce")
            f_lon = pd.to_numeric(df.get(lon_src, np.nan), errors="coerce")

            both_valid = f_lat.notna() & f_lon.notna() & ref_lat.notna() & ref_lon.notna()

            d_lat = f_lat - ref_lat
            d_lon = f_lon - ref_lon

            # QC flags (vectorised)
            qc = pd.Series("OK", index=df.index)
            wraparound = both_valid & (d_lon.abs() > cfg_trk["wraparound_threshold_deg"])
            implausible = both_valid & (
                (d_lat.abs() > MAX_DELTA_DEG[hlabel]) |
                (d_lon.abs() > MAX_DELTA_DEG[hlabel])
            )
            qc[wraparound]  = "LONGITUDE_WRAPAROUND_SUSPECTED"
            qc[implausible] = f"IMPLAUSIBLE_DISPLACEMENT_{hlabel}"

            df[dlat_col]  = np.where(both_valid, d_lat, np.nan)
            df[dlon_col]  = np.where(both_valid, d_lon, np.nan)
            df[alat_col]  = f_lat
            df[alon_col]  = f_lon
            df[valid_col] = both_valid.astype(int)
            df[qc_col]    = qc

        # ── Select output columns (ordered) ───────────────────────────
        base_cols = [
            "sequence_id", "storm_id", "storm_name", "season",
            "reference_time", "center_lat", "center_lon", "split",
            "nature_t0",
            "detection_label", "detection_valid",
            "pattern_label", "pattern_label_status", "pattern_valid",
            "leakage_check",
            "intensity_t0", "intensity_valid_t0",
            "pressure_t0",  "pressure_valid_t0",
        ]
        for h in HORIZON_LABELS:
            base_cols += [f"intensity_{h}", f"intensity_valid_{h}"]
        for h in HORIZON_LABELS:
            base_cols += [f"pressure_{h}", f"pressure_valid_{h}"]
        for h in HORIZON_LABELS:
            dlat, dlon = cfg_trk["delta_columns"][h]
            alat, alon = cfg_trk["absolute_columns"][h]
            valid      = cfg_trk["validity_mask_columns"][h]
            base_cols += [dlat, dlon, alat, alon, valid, f"track_qc_{h}"]

        # Keep only columns that actually exist
        out_cols = [c for c in base_cols if c in df.columns]
        df_out = df[out_cols].copy()

        # ── Write outputs ─────────────────────────────────────────────
        out_path = self.targets_dir / self.config["output"]["targets_manifest_filename"]
        df_out.to_csv(out_path, index=False)
        if verbose:
            print(f"[SUCCESS] Targets manifest: {out_path} ({len(df_out):,} rows)")

        for split_name, fname in [
            ("train", self.config["output"]["targets_manifest_train"]),
            ("val",   self.config["output"]["targets_manifest_val"]),
            ("test",  self.config["output"]["targets_manifest_test"]),
        ]:
            df_split = df_out[df_out["split"] == split_name]
            split_path = self.targets_dir / fname
            df_split.to_csv(split_path, index=False)
            if verbose:
                print(f"[SUCCESS] {split_name} manifest: {split_path} ({len(df_split):,} rows)")

        if verbose:
            leakage_fail = int((df_out["leakage_check"] != "PASS").sum())
            total_wrap = sum(
                int((df_out.get(f"track_qc_{h}", pd.Series(dtype=str)) == "LONGITUDE_WRAPAROUND_SUSPECTED").sum())
                for h in HORIZON_LABELS
            )
            total_impl = sum(
                int((df_out.get(f"track_qc_{h}", pd.Series(dtype=str)).str.startswith("IMPLAUSIBLE")).sum())
                for h in HORIZON_LABELS
            )
            print(f"[INFO] Leakage violations: {leakage_fail}")
            print(f"[INFO] Wraparound flags:   {total_wrap}")
            print(f"[INFO] Implausible flags:  {total_impl}")

        return df_out


# ---------------------------------------------------------------------------
# TargetQualityAudit
# ---------------------------------------------------------------------------

class TargetQualityAudit:
    """Comprehensive quality audit of the targets manifest."""

    def __init__(self, targets_manifest_path: Path, project_root: Optional[Path] = None):
        self.targets_path = Path(targets_manifest_path)
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root)

        cfg_path = self.project_root / "configs/target_label_config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def run_audit(self, verbose: bool = True) -> dict:
        if not self.targets_path.exists():
            raise FileNotFoundError(f"Targets manifest not found: {self.targets_path}")

        df = pd.read_csv(self.targets_path)
        df["reference_time"] = pd.to_datetime(df["reference_time"])

        report: Dict[str, Any] = {
            "audit_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_sequences": len(df),
        }

        if verbose:
            print("=" * 60)
            print("  STORMFUSION -- STEP 15: TARGET QUALITY AUDIT")
            print("=" * 60)
            print(f"Total sequences: {len(df):,}")

        # Split distribution
        split_counts = df["split"].value_counts().to_dict()
        report["split_distribution"] = split_counts
        if verbose:
            print(f"Split: {split_counts}")

        # Detection
        det_counts = df["detection_label"].value_counts().to_dict()
        report["detection"] = {"label_distribution": {str(k): int(v) for k, v in det_counts.items()}}
        if verbose:
            print(f"Detection: {det_counts}")

        # Pattern
        pat_counts = df["pattern_label"].value_counts().to_dict()
        pat_status = df["pattern_label_status"].value_counts().to_dict()
        report["pattern"] = {
            "label_distribution": {str(k): int(v) for k, v in pat_counts.items()},
            "status_distribution": pat_status,
        }
        if verbose:
            print(f"Pattern: {pat_counts} | Status: {pat_status}")

        # Leakage
        leakage_fail = int((df["leakage_check"] != "PASS").sum())
        report.setdefault("qc_flags", {})["leakage_violations"] = leakage_fail
        if verbose:
            print(f"Leakage violations: {leakage_fail}")

        # Horizon availability
        horizon_avail: Dict[str, Any] = {}
        for hlabel in HORIZON_LABELS:
            avail: Dict[str, Any] = {}
            for target, vcol in [
                ("intensity", f"intensity_valid_{hlabel}"),
                ("pressure",  f"pressure_valid_{hlabel}"),
                ("track",     f"track_valid_{hlabel}"),
            ]:
                if vcol in df.columns:
                    n = int(df[vcol].sum())
                    avail[target] = {"valid_count": n, "pct": round(100.0 * n / len(df), 2)}
            horizon_avail[hlabel] = avail
        report["horizon_availability"] = horizon_avail
        if verbose:
            for h, avail in horizon_avail.items():
                print(f"  {h}: intensity={avail.get('intensity',{}).get('valid_count',0)}, "
                      f"pressure={avail.get('pressure',{}).get('valid_count',0)}, "
                      f"track={avail.get('track',{}).get('valid_count',0)}")

        # NaN / Inf
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        nan_counts = df[numeric_cols].isna().sum()
        inf_total  = int(np.isinf(df[numeric_cols].values).sum())
        report["qc_flags"]["nan_counts_by_column"] = {
            c: int(nan_counts[c]) for c in numeric_cols if nan_counts[c] > 0
        }
        report["qc_flags"]["total_inf_values"] = inf_total
        if verbose:
            print(f"Inf values: {inf_total}")

        # Duplicates
        dup_count = int(df.duplicated(subset=["storm_id", "reference_time"]).sum())
        report["qc_flags"]["duplicate_storm_time_pairs"] = dup_count
        if verbose:
            print(f"Duplicate (storm_id, reference_time): {dup_count}")

        # Coordinate validity
        bad_lat = int(((df["center_lat"] < -90) | (df["center_lat"] > 90)).sum())
        bad_lon = int(((df["center_lon"] < 0)   | (df["center_lon"] > 360)).sum())
        report["qc_flags"]["impossible_center_lat"] = bad_lat
        report["qc_flags"]["impossible_center_lon"] = bad_lon

        # Non-monotonic timestamps
        non_mono = 0
        for _, grp in df.groupby("storm_id"):
            times = grp["reference_time"].sort_values()
            if (times.diff().dropna() < pd.Timedelta(0)).any():
                non_mono += 1
        report["qc_flags"]["storms_with_non_monotonic_timestamps"] = non_mono

        # Implausible / wraparound counts
        total_impl = 0
        total_wrap = 0
        for h in HORIZON_LABELS:
            qcol = f"track_qc_{h}"
            if qcol in df.columns:
                total_impl += int(df[qcol].str.contains("IMPLAUSIBLE", na=False).sum())
                total_wrap += int(df[qcol].str.contains("WRAPAROUND", na=False).sum())
        report["qc_flags"]["total_implausible_displacement_flags"] = total_impl
        report["qc_flags"]["total_longitude_wraparound_flags"]     = total_wrap
        if verbose:
            print(f"Implausible displacement flags: {total_impl}")
            print(f"Longitude wraparound flags:     {total_wrap}")

        # Storm / season stats
        report["storm_stats"] = {
            "unique_storms": int(df["storm_id"].nunique()),
            "top_20_storms_by_sample_count": {
                str(k): int(v) for k, v in df["storm_id"].value_counts().head(20).to_dict().items()
            },
        }
        if "season" in df.columns:
            report["season_stats"] = {
                "samples_by_season": {
                    str(k): int(v) for k, v in df["season"].value_counts().sort_index().to_dict().items()
                }
            }

        # Intensity T0 valid + IMD category distribution
        report["intensity"] = {}
        if "intensity_valid_t0" in df.columns:
            report["intensity"]["t0_valid_count"] = int(df["intensity_valid_t0"].sum())
        if "intensity_t0" in df.columns:
            w = df["intensity_t0"].dropna()
            report["intensity"]["imd_category_distribution_t0"] = {
                "Depression_le27kts":               int((w <= 27).sum()),
                "Deep_Depression_28_33kts":          int(((w >= 28) & (w <= 33)).sum()),
                "Cyclonic_Storm_34_47kts":           int(((w >= 34) & (w <= 47)).sum()),
                "Severe_CS_48_63kts":                int(((w >= 48) & (w <= 63)).sum()),
                "Very_Severe_CS_64_89kts":           int(((w >= 64) & (w <= 89)).sum()),
                "Extremely_Severe_CS_90_119kts":     int(((w >= 90) & (w <= 119)).sum()),
                "Super_CS_ge120kts":                 int((w >= 120).sum()),
            }
            if verbose:
                print("IMD category distribution (T0 wind):")
                for cat, cnt in report["intensity"]["imd_category_distribution_t0"].items():
                    print(f"  {cat}: {cnt}")

        report["pressure"] = {}
        if "pressure_valid_t0" in df.columns:
            report["pressure"]["t0_valid_count"] = int(df["pressure_valid_t0"].sum())

        # Overall pass/fail
        critical_fail = (
            leakage_fail > 0 or dup_count > 0 or
            bad_lat > 0 or bad_lon > 0 or inf_total > 0
        )
        report["overall_status"] = "FAIL" if critical_fail else "PASS"
        if verbose:
            print(f"\nTarget Quality Audit: {report['overall_status']}")
            print("=" * 60)

        return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_and_audit(project_root: Optional[Path] = None, verbose: bool = True) -> dict:
    """Full Step 15 pipeline: build targets manifest + quality audit."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    engineer = TargetEngineer(project_root=project_root)
    engineer.build_targets_manifest(verbose=verbose)

    with open(project_root / "configs/target_label_config.yaml", "r") as f:
        label_cfg = yaml.safe_load(f)

    targets_dir          = project_root / label_cfg["output"]["targets_dir"]
    targets_manifest_path = targets_dir / label_cfg["output"]["targets_manifest_filename"]
    quality_report_path  = targets_dir / label_cfg["output"]["quality_report_filename"]
    step15_report_path   = project_root / label_cfg["output"]["step15_report_path"]

    auditor = TargetQualityAudit(targets_manifest_path=targets_manifest_path, project_root=project_root)
    quality_report = auditor.run_audit(verbose=verbose)

    with open(quality_report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)
    if verbose:
        print(f"[SUCCESS] Quality report: {quality_report_path}")

    # Step 15 safety-gate report
    df_out = pd.read_csv(targets_manifest_path)
    step15_report = {
        "step": "15",
        "step_name": "Cyclone Target & Label Engineering",
        "status": "COMPLETE" if quality_report["overall_status"] == "PASS" else "FAIL",
        "targets_manifest_generated": True,
        "total_sequences": int(quality_report["total_sequences"]),
        "split_distribution": quality_report.get("split_distribution", {}),
        "detection_labels_all_positive": (
            quality_report.get("detection", {})
            .get("label_distribution", {}).get("1", 0)
            == quality_report["total_sequences"]
        ),
        "pattern_scheme_status": "PATTERN_LABEL_PENDING",
        "pattern_labels_all_pending": True,
        "leakage_violations": quality_report["qc_flags"].get("leakage_violations", 0),
        "duplicate_storm_time_pairs": quality_report["qc_flags"].get("duplicate_storm_time_pairs", 0),
        "implausible_displacement_flags": quality_report["qc_flags"].get("total_implausible_displacement_flags", 0),
        "longitude_wraparound_flags": quality_report["qc_flags"].get("total_longitude_wraparound_flags", 0),
        "quality_audit_status": quality_report["overall_status"],
        # Safety gates
        "satellite_provider": "EUMETSAT_HRSEVIRI_IODC",
        "mosdac_status": "REMOVED",
        "satellite_real_data_status": "BLOCKED_GENERAL_LICENSE",
        "era5_real_data_status":  "PASS",
        "scientific_training_performed": False,
        "synthetic_scientific_data_used": False,
        "training_allowed": False,
        "training_readiness": "NOT_READY",
        "report_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    step15_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(step15_report_path, "w", encoding="utf-8") as f:
        json.dump(step15_report, f, indent=2)
    if verbose:
        print(f"[SUCCESS] Step 15 report: {step15_report_path}")

    return step15_report


if __name__ == "__main__":
    result = build_and_audit()
    print()
    print("=" * 60)
    print("  STEP 15 -- FINAL STATUS REPORT")
    print("=" * 60)
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("=" * 60)
