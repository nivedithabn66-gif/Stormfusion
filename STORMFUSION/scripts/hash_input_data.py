"""SHA256 Input Data Hasher for STORMFUSION.

Computes cryptographic SHA256 hashes of input track datasets and raw files
to establish verifiable data provenance and detect underlying data mutation.
"""

import hashlib
import json
from pathlib import Path


def get_file_sha256(filepath: Path) -> str:
    """Calculates SHA256 checksum of a file reading in 64KB chunks."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def hash_inputs():
    project_root = Path(__file__).resolve().parent.parent
    splits_dir = project_root / "data/splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    input_files = {
        "ibtracs_nio_clean.csv": project_root / "data/processed/tracks/ibtracs_nio_clean.csv",
        "storm_catalog.csv": project_root / "data/processed/tracks/storm_catalog.csv",
        "IBTrACS.ALL.v04r01.nc": project_root / "data/raw/ibtracs/IBTrACS.ALL.v04r01.nc",
    }

    hashes = {}

    print("==================================================")
    print("        INPUT DATASET SHA256 PROVENANCE          ")
    print("==================================================")

    for name, path in input_files.items():
        if path.exists():
            h_val = get_file_sha256(path)
            size_bytes = path.stat().st_size
            hashes[name] = {
                "rel_path": str(path.relative_to(project_root)),
                "sha256": h_val,
                "size_bytes": size_bytes,
            }
            print(f"• File: {name:<25} | Size: {size_bytes / 1024 / 1024:.2f} MB")
            print(f"  SHA256: {h_val}")
        else:
            print(f"• File: {name:<25} | NOT FOUND")

    output_json = splits_dir / "input_hashes.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)

    print(f"\n[SUCCESS] Saved input hashes to: {output_json}")
    print("==================================================")
    return hashes


if __name__ == "__main__":
    hash_inputs()
