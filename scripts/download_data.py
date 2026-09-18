#!/usr/bin/env python3
"""Download the FlyWire v783 dataset and verify its SHA-256 checksums.

Fetches the aggregated v783 snapshot hosted at
https://github.com/eonsystemspbc/fly-brain/tree/main/data
(the official FlyWire v783 release is https://doi.org/10.5281/zenodo.10676866)
and stores it under data/ with the file names used by the simulator:

  - data/flywire_connectivity_v783.parquet  (upstream: 2025_Connectivity_783.parquet)
  - data/flywire_neurons_v783.csv           (upstream: 2025_Completeness_783.csv)

Files that already exist with a matching checksum are skipped. The script
downloads to a temporary file first and verifies before moving into place,
so a partial download never looks like a valid dataset. It performs no
other network calls and uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

FILES = {
    "flywire_connectivity_v783.parquet": (
        "https://raw.githubusercontent.com/eonsystemspbc/fly-brain/main/data/2025_Connectivity_783.parquet",
        "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
    ),
    "flywire_neurons_v783.csv": (
        "https://raw.githubusercontent.com/eonsystemspbc/fly-brain/main/data/2025_Completeness_783.csv",
        "52b0ac6094cd32c546f8d4c341e094376f48f4e791f8db9b166de5dff8199ea4",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".download-", suffix=".tmp", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    try:
        tmp_path.replace(destination)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="target directory (default: data/ in the repository)",
    )
    args = parser.parse_args()

    failures = []
    for name, (url, expected) in FILES.items():
        target = args.data_dir / name
        if target.exists() and sha256_file(target) == expected:
            print(f"OK        {name} (already present, checksum verified)")
            continue
        print(f"fetching  {name} ...")
        try:
            download(url, target)
        except Exception as error:  # noqa: BLE001 - report and continue
            print(f"ERROR     {name}: {error}", file=sys.stderr)
            failures.append(name)
            continue
        actual = sha256_file(target)
        if actual != expected:
            print(
                f"ERROR     {name}: checksum mismatch (got {actual}, expected {expected})",
                file=sys.stderr,
            )
            target.unlink(missing_ok=True)
            failures.append(name)
            continue
        print(f"OK        {name} (checksum verified)")

    if failures:
        print(f"\nfailed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nAll data files verified. The simulator is ready to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
