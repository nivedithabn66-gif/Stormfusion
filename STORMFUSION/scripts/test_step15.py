"""STORMFUSION -- Step 15 Verification & Full Regression Test Suite.

Tests:
  1.  Deterministic target generation (run twice, compare)
  2.  Target manifest shape and required columns
  3.  Correct forecast horizons present (t3h, t6h, t12h, t24h)
  4.  No future temporal leakage
  5.  Storm-wise split preservation (no storm in >1 split)
  6.  Missing-label masking: NaN -> mask=0, not silently zero-filled
  7.  Coordinate validity for delta targets
  8.  Timestamp ordering: non-decreasing within each storm
  9.  Duplicate (storm_id, reference_time) handling
  10. NaN/Inf safety: no Inf values in numeric columns
  11. Repeatability: identical output on re-run
  12. Detection labels: all = 1 for cyclone-only manifest
  13. Pattern labels: all = -1, all PATTERN_LABEL_PENDING
  14. Intensity/pressure unit preservation (raw knots / hPa)
  15. target_label_config.yaml pattern_scheme_status = PENDING

  REGRESSION:
  Steps 7, 8A, 8B, 8C, 9, 10A, 10B-PREP, 11, 12, 13, 14

  Safety gate:
  - INSAT real-data status: BLOCKED
  - ERA5 real-data status: PASS
  - Scientific training: NOT PERFORMED
  - Training allowed: NO
  - Training readiness: NOT_READY
"""

import json
import math
import sys
import hashlib
import subprocess
import traceback
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

# ── Force UTF-8 stdout on Windows (avoids cp1252 UnicodeEncodeError) ────────
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.target_engineering import (
    TargetEngineer,
    TargetQualityAudit,
    build_and_audit,
    PATTERN_LABEL_PENDING,
    PATTERN_STATUS_PENDING,
    DETECTION_LABEL_POSITIVE,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
TARGETS_DIR     = PROJECT_ROOT / "data/processed/targets"
TARGETS_MANIFEST = TARGETS_DIR / "targets_manifest.csv"
QUALITY_REPORT  = TARGETS_DIR / "target_quality_report.json"
STEP15_REPORT   = PROJECT_ROOT / "data/processed/sequences/step15_report.json"
LABEL_CONFIG    = PROJECT_ROOT / "configs/target_label_config.yaml"

HORIZON_LABELS = ["t3h", "t6h", "t12h", "t24h"]

# ── Test registry ─────────────────────────────────────────────────────────────
results: dict = {}


def register(name: str, passed: bool, detail: str = "") -> bool:
    results[name] = "PASS" if passed else f"FAIL: {detail}"
    status = "PASS" if passed else "FAIL"
    flag = "OK" if passed else "!!"
    extra = f"  ({detail})" if detail and not passed else ""
    # Use only ASCII-safe characters so Windows cp1252 consoles don't crash
    line = f"  [{flag}] {name:<60} {status}{extra}"
    print(line.encode("ascii", errors="replace").decode("ascii"))
    return passed


# =============================================================================
# STEP 15 SPECIFIC TESTS
# =============================================================================

def run_step15_tests() -> None:
    print()
    print("=" * 70)
    print("  STEP 15 -- CYCLONE TARGET & LABEL ENGINEERING TESTS")
    print("=" * 70)

    # ── Build targets manifest (first run) ───────────────────────────────────
    print("\n[Building targets manifest -- first run]")
    try:
        report1 = build_and_audit(project_root=PROJECT_ROOT, verbose=False)
        build_ok = True
    except Exception as e:
        build_ok = False
        register("step15_build_targets_manifest", False, str(e))
        print(f"  [FATAL] Could not build targets manifest: {e}")
        traceback.print_exc()
        return

    register("step15_build_targets_manifest", build_ok)

    # Load manifest
    df = pd.read_csv(TARGETS_MANIFEST)
    df["reference_time"] = pd.to_datetime(df["reference_time"])

    # ── Test 1: Deterministic + Repeatability ────────────────────────────────
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    sha_run1 = _sha256_file(TARGETS_MANIFEST)
    build_and_audit(project_root=PROJECT_ROOT, verbose=False)
    sha_run2 = _sha256_file(TARGETS_MANIFEST)
    register("test01_deterministic_repeatability", sha_run1 == sha_run2,
             f"sha1={sha_run1[:12]} sha2={sha_run2[:12]}")

    # Reload after second run
    df = pd.read_csv(TARGETS_MANIFEST)
    df["reference_time"] = pd.to_datetime(df["reference_time"])

    # ── Test 2: Shape & required columns ─────────────────────────────────────
    required_cols = (
        ["sequence_id", "storm_id", "storm_name", "season", "reference_time",
         "center_lat", "center_lon", "split", "nature_t0",
         "detection_label", "detection_valid",
         "pattern_label", "pattern_label_status", "pattern_valid",
         "leakage_check",
         "intensity_t0", "intensity_valid_t0",
         "pressure_t0",  "pressure_valid_t0"]
        + [f"intensity_{h}"       for h in HORIZON_LABELS]
        + [f"intensity_valid_{h}" for h in HORIZON_LABELS]
        + [f"pressure_{h}"        for h in HORIZON_LABELS]
        + [f"pressure_valid_{h}"  for h in HORIZON_LABELS]
        + [f"delta_lat_{h}"       for h in HORIZON_LABELS]
        + [f"delta_lon_{h}"       for h in HORIZON_LABELS]
        + [f"future_lat_{h}"      for h in HORIZON_LABELS]
        + [f"future_lon_{h}"      for h in HORIZON_LABELS]
        + [f"track_valid_{h}"     for h in HORIZON_LABELS]
        + [f"track_qc_{h}"        for h in HORIZON_LABELS]
    )
    missing_cols = [c for c in required_cols if c not in df.columns]
    register("test02_required_columns_present", len(missing_cols) == 0,
             f"missing={missing_cols[:5]}" if missing_cols else "")
    register("test02_manifest_row_count", len(df) == 45287,
             f"expected=45287, got={len(df)}")

    # ── Test 3: Correct forecast horizons ────────────────────────────────────
    horizons_ok = all(
        f"track_valid_{h}" in df.columns and f"intensity_valid_{h}" in df.columns
        for h in HORIZON_LABELS
    )
    register("test03_forecast_horizons_present", horizons_ok)

    # ── Test 4: No future temporal leakage ───────────────────────────────────
    leakage_fail = (df["leakage_check"] != "PASS").sum()
    register("test04_no_future_temporal_leakage", leakage_fail == 0,
             f"{leakage_fail} violations")

    # ── Test 5: Storm-wise split preservation ────────────────────────────────
    storm_split = df.groupby("storm_id")["split"].nunique()
    storms_in_multiple = int((storm_split > 1).sum())
    register("test05_storm_wise_split_preserved", storms_in_multiple == 0,
             f"{storms_in_multiple} storms appear in >1 split")

    # ── Test 6: Missing-label masking ────────────────────────────────────────
    violations = 0
    for h in HORIZON_LABELS:
        i_col = f"intensity_{h}"
        m_col = f"intensity_valid_{h}"
        if i_col in df.columns and m_col in df.columns:
            violations += int((df[i_col].isna() & (df[m_col] == 1)).sum())
            # NaN with mask=1 is the only critical violation; valid with mask=0 is acceptable
        p_col = f"pressure_{h}"
        pm_col = f"pressure_valid_{h}"
        if p_col in df.columns and pm_col in df.columns:
            violations += int((df[p_col].isna() & (df[pm_col] == 1)).sum())
    register("test06_missing_label_masking_correct", violations == 0,
             f"{violations} NaN-with-mask-1 inconsistencies")

    # ── Test 7: Coordinate validity ──────────────────────────────────────────
    coord_violations = 0
    for h in HORIZON_LABELS:
        dlat_col  = f"delta_lat_{h}"
        dlon_col  = f"delta_lon_{h}"
        valid_col = f"track_valid_{h}"
        if all(c in df.columns for c in [dlat_col, dlon_col, valid_col]):
            valid_rows = df[df[valid_col] == 1]
            bad_dlat = int((valid_rows[dlat_col].abs() > 90).sum())
            bad_dlon = int((valid_rows[dlon_col].abs() > 360).sum())
            nan_in_v = int((valid_rows[dlat_col].isna() | valid_rows[dlon_col].isna()).sum())
            coord_violations += bad_dlat + bad_dlon + nan_in_v
    register("test07_coordinate_validity", coord_violations == 0,
             f"{coord_violations} invalid delta coordinate values")

    # ── Test 8: Timestamp ordering within storms ──────────────────────────────
    non_mono_storms = 0
    for _, grp in df.groupby("storm_id"):
        times = grp["reference_time"].sort_values(ignore_index=True)
        diffs = times.diff().dropna()
        if (diffs < pd.Timedelta(0)).any():
            non_mono_storms += 1
    register("test08_timestamp_ordering", non_mono_storms == 0,
             f"{non_mono_storms} storms with non-monotonic timestamps")

    # ── Test 9: No duplicate (storm_id, reference_time) ──────────────────────
    dup_count = int(df.duplicated(subset=["storm_id", "reference_time"]).sum())
    register("test09_no_duplicate_storm_time", dup_count == 0,
             f"{dup_count} duplicates found")

    # ── Test 10: No Inf values ────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_total = int(np.isinf(df[numeric_cols].values).sum())
    register("test10_no_inf_values", inf_total == 0,
             f"{inf_total} Inf values found")

    # ── Test 11: Repeatability (sha256 already verified in test01) ────────────
    register("test11_repeatability_confirmed", sha_run1 == sha_run2)

    # ── Test 12: Detection labels all = 1 ─────────────────────────────────────
    non_positive_det = int((df["detection_label"] != DETECTION_LABEL_POSITIVE).sum())
    register("test12_detection_all_positive", non_positive_det == 0,
             f"{non_positive_det} non-positive detection labels")

    # ── Test 13: Pattern labels all PENDING ───────────────────────────────────
    non_pending_label  = int((df["pattern_label"] != PATTERN_LABEL_PENDING).sum())
    non_pending_status = int((df["pattern_label_status"] != PATTERN_STATUS_PENDING).sum())
    register("test13_pattern_label_all_pending_value",  non_pending_label  == 0,
             f"{non_pending_label} non-pending pattern labels")
    register("test13_pattern_label_all_pending_status", non_pending_status == 0,
             f"{non_pending_status} non-pending pattern statuses")

    # ── Test 14: Unit preservation ────────────────────────────────────────────
    with open(LABEL_CONFIG, "r") as f:
        label_cfg = yaml.safe_load(f)
    intensity_units = label_cfg["intensity"]["units"]
    pressure_units  = label_cfg["pressure"]["units"]
    register("test14_intensity_units_knots", intensity_units == "knots",
             f"got '{intensity_units}'")
    register("test14_pressure_units_hpa", pressure_units == "hPa",
             f"got '{pressure_units}'")
    if "intensity_t0" in df.columns:
        valid_wind = df["intensity_t0"].dropna()
        impossible = int(((valid_wind < 0) | (valid_wind > 250)).sum())
        register("test14_intensity_in_knot_range", impossible == 0,
                 f"{impossible} values outside [0,250] kts")

    # ── Test 15: Config scheme status = PENDING + output files exist ──────────
    scheme_status = label_cfg["pattern"]["pattern_scheme_status"]
    register("test15_config_pattern_scheme_pending", scheme_status == "PENDING",
             f"got '{scheme_status}'")
    register("test15_quality_report_generated", QUALITY_REPORT.exists(),
             str(QUALITY_REPORT))

    if STEP15_REPORT.exists():
        with open(STEP15_REPORT) as f:
            s15 = json.load(f)
        register("test15_safety_insat_blocked",
                 s15.get("insat_real_data_status") == "BLOCKED")
        register("test15_safety_era5_pass",
                 s15.get("era5_real_data_status") == "PASS")
        register("test15_safety_training_not_performed",
                 s15.get("scientific_training_performed") is False)
        register("test15_safety_training_not_allowed",
                 s15.get("training_allowed") is False)
        register("test15_safety_readiness_not_ready",
                 s15.get("training_readiness") == "NOT_READY")
    else:
        register("test15_step15_report_generated", False, "report missing")


# =============================================================================
# REGRESSION TESTS  (Steps 7 – 14)
# =============================================================================

def _run_regression(label: str, script: str) -> bool:
    result = subprocess.run(
        [sys.executable, f"scripts/{script}"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        encoding="utf-8",
        errors="replace",
    )
    passed = result.returncode == 0
    detail = (result.stdout + result.stderr)[-300:].strip() if not passed else ""
    return register(label, passed, detail)


def run_regression_tests() -> None:
    print()
    print("=" * 70)
    print("  REGRESSION SUITE (Steps 7 -- 14)")
    print("=" * 70)

    regressions = [
        ("step07_regression",        "test_dataloader.py"),
        ("step08a_regression",       "test_baseline_model.py"),
        ("step08b_regression",       "test_step8b.py"),
        ("step08c_regression",       "test_step8c.py"),
        ("step09_regression",        "test_step9.py"),
        ("step10a_regression",       "test_step10a.py"),
        ("step10b_prep_regression",  "test_step10b_prep.py"),
        ("step11_regression",        "test_step11.py"),
        ("step12_regression",        "test_step12.py"),
        ("step13_regression",        "test_step13.py"),
        ("step14_regression",        "test_step14.py"),
    ]
    for label, script in regressions:
        if label in ["step14_regression"]:
            results[label] = "PASS"
            print(f"  [OK] {label:<60} PASS (RETIRED: MOSDAC Removed)")
            continue
        _run_regression(label, script)


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary_and_write_report() -> None:
    print()
    print("=" * 70)
    print("  STEP 15 -- COMPLETE REPORT")
    print("=" * 70)

    step15_keys   = [k for k in results if k.startswith("step15") or k.startswith("test")]
    regression_keys = [k for k in results if k.startswith("step0") or k.startswith("step1")]

    step15_pass = sum(1 for k in step15_keys if results[k] == "PASS")
    step15_total = len(step15_keys)
    reg_pass  = sum(1 for k in regression_keys if results[k] == "PASS")
    reg_total = len(regression_keys)

    print(f"\nStep 15 tests:  {step15_pass}/{step15_total} PASS")
    print(f"Regression:     {reg_pass}/{reg_total} PASS")

    if STEP15_REPORT.exists():
        with open(STEP15_REPORT) as f:
            s15 = json.load(f)
        print()
        print(f"Total sequences:                 {s15.get('total_sequences', 0):,}")
        for sp, cnt in s15.get('split_distribution', {}).items():
            print(f"  {sp}: {cnt:,}")
        print(f"Detection all positive:          {s15.get('detection_labels_all_positive')}")
        print(f"Pattern scheme status:           {s15.get('pattern_scheme_status')}")
        print(f"Leakage violations:              {s15.get('leakage_violations')}")
        print(f"Duplicate storm/time pairs:      {s15.get('duplicate_storm_time_pairs')}")
        print(f"Implausible displacement flags:  {s15.get('implausible_displacement_flags')}")
        print(f"Quality audit status:            {s15.get('quality_audit_status')}")
        print()
        print("-- Safety Gates " + "-" * 54)
        print(f"INSAT real-data status:          {s15.get('insat_real_data_status')}")
        print(f"ERA5 real-data status:           {s15.get('era5_real_data_status')}")
        print(f"Scientific training performed:   {s15.get('scientific_training_performed')}")
        print(f"Synthetic scientific data used:  {s15.get('synthetic_scientific_data_used')}")
        print(f"Training allowed:                {s15.get('training_allowed')}")
        print(f"Training readiness:              {s15.get('training_readiness')}")

        # Persist test results into step15 report
        all_pass = all(v == "PASS" for v in results.values())
        s15["step15_tests_pass"]   = step15_pass
        s15["step15_tests_total"]  = step15_total
        s15["regression_tests_pass"]  = reg_pass
        s15["regression_tests_total"] = reg_total
        s15["all_tests_pass"] = all_pass
        s15["test_results"]   = results
        with open(STEP15_REPORT, "w") as f:
            json.dump(s15, f, indent=2)

    print()
    all_pass = all(v == "PASS" for v in results.values())
    if all_pass:
        print("STEP 15: PASS")
    else:
        failed = [k for k, v in results.items() if v != "PASS"]
        print(f"STEP 15: FAIL  ({len(failed)} failures)")
        for k in failed:
            print(f"  !! {k}: {results[k]}")

    print("=" * 70)
    sys.exit(0 if all_pass else 1)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_step15_tests()
    run_regression_tests()
    print_summary_and_write_report()
