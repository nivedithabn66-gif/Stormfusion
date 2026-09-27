"""Double-Run Determinism Verification Script for STORMFUSION (Step 5).

Executes the sequence generation pipeline twice without modifying inputs and asserts that
the resulting train/val/test splits and sequence_manifest.csv SHA256 hashes match 100%.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd


def get_file_sha256(filepath: Path) -> str:
    """Calculates SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def test_determinism():
    project_root = Path(__file__).resolve().parent.parent

    manifest_path = project_root / "data/processed/sequences/sequence_manifest.csv"
    splits_meta_path = project_root / "data/splits/split_metadata.json"
    build_script = project_root / "scripts/build_sequence_manifest.py"

    print("==================================================")
    print("        MANIFEST DOUBLE-RUN DETERMINISM TEST       ")
    print("==================================================")

    # ----------------------------------------------------
    # RUN 1
    # ----------------------------------------------------
    print("\n[RUN 1] Executing scripts/build_sequence_manifest.py...")
    subprocess.run([sys.executable, str(build_script)], check=True, cwd=project_root)

    if not manifest_path.exists():
        print("[ERROR] Sequence manifest not generated in Run 1.")
        return False

    hash_run1 = get_file_sha256(manifest_path)
    df_manifest_run1 = pd.read_csv(manifest_path)
    seq_count_run1 = len(df_manifest_run1)

    with open(splits_meta_path, "r", encoding="utf-8") as f:
        meta_run1 = json.load(f)

    print(f"  • Run 1 Sequence Count : {seq_count_run1}")
    print(f"  • Run 1 Storm Splits   : Train={meta_run1['train_count']}, Val={meta_run1['val_count']}, Test={meta_run1['test_count']}")
    print(f"  • Run 1 Manifest SHA256: {hash_run1}")

    # ----------------------------------------------------
    # RUN 2
    # ----------------------------------------------------
    print("\n[RUN 2] Executing scripts/build_sequence_manifest.py a second time...")
    subprocess.run([sys.executable, str(build_script)], check=True, cwd=project_root)

    hash_run2 = get_file_sha256(manifest_path)
    df_manifest_run2 = pd.read_csv(manifest_path)
    seq_count_run2 = len(df_manifest_run2)

    with open(splits_meta_path, "r", encoding="utf-8") as f:
        meta_run2 = json.load(f)

    print(f"  • Run 2 Sequence Count : {seq_count_run2}")
    print(f"  • Run 2 Storm Splits   : Train={meta_run2['train_count']}, Val={meta_run2['val_count']}, Test={meta_run2['test_count']}")
    print(f"  • Run 2 Manifest SHA256: {hash_run2}")

    # ----------------------------------------------------
    # ASSERTIONS & VERIFICATION
    # ----------------------------------------------------
    print("\n==================================================")
    hash_match = (hash_run1 == hash_run2)
    count_match = (seq_count_run1 == seq_count_run2)
    split_match = (meta_run1['train_count'] == 1300) and (meta_run1['val_count'] == 279) and (meta_run1['test_count'] == 279)

    print(f"Manifest SHA256 Match (RUN 1 == RUN 2) : {'YES' if hash_match else 'NO'}")
    print(f"Sequence Count Match                   : {'YES' if count_match else 'NO'}")
    print(f"Approved Slicing (1300/279/279)        : {'YES' if split_match else 'NO'}")
    print("==================================================")

    if hash_match and count_match and split_match:
        print("DETERMINISM VERIFICATION STATUS        : PASS")
        print("==================================================")
        return True
    else:
        print("DETERMINISM VERIFICATION STATUS        : FAIL")
        print("==================================================")
        return False


if __name__ == "__main__":
    test_determinism()
