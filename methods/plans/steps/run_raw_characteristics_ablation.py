#!/usr/bin/env python3
"""
Run raw point-cloud characterisation (non-GT) for the reference sample and all subset tunnels.

Uses sam4tun.plugins.raw_characteristics.analyze_point_cloud and writes:
  - tunnel_id ``sample`` -> data/sample/characteristics/raw_characteristics.json
  - any other stem       -> data/ablation/memory/{tunnel_id}/characteristics/raw_characteristics.json
    (see sam4tun.plugins.paths.ABLATION_TUNNEL_SUBROOT to change layout)

Default inputs:
  - Sample:  data/sample.txt   -> tunnel_id "sample"
  - Subsets: data/subsets/*.txt -> tunnel_id = filename stem

Run from repository root:
  python methods/plans/steps/run_raw_characteristics_ablation.py
  python methods/plans/steps/run_raw_characteristics_ablation.py --sample path/to/sample.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Repo root = parents[3] from methods/plans/steps/this_file.py
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sam4tun.plugins.paths import tunnel_characteristics_dir  # noqa: E402
from sam4tun.plugins.raw_characteristics import (  # noqa: E402
    NumpyEncoder,
    analyze_point_cloud,
)


def save_raw_characteristics(file_path: str, tunnel_id: str) -> str:
    results = analyze_point_cloud(file_path, tunnel_id)
    out_dir = tunnel_characteristics_dir(tunnel_id)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "raw_characteristics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    return out_file


def main() -> int:
    os.chdir(_REPO_ROOT)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sample",
        default="data/sample.txt",
        help="Path to reference sample point cloud (.txt), default: data/sample.txt",
    )
    p.add_argument(
        "--subsets_dir",
        default="data/subsets",
        help="Directory containing subset .txt point clouds (default: data/subsets)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="List targets only, do not write files",
    )
    args = p.parse_args()

    jobs: list[tuple[str, str]] = []
    sample_path = Path(args.sample)
    if sample_path.is_file():
        jobs.append(("sample", str(sample_path.resolve())))
    else:
        print(f"⚠️  Skip sample (not found): {sample_path}")

    subsets_dir = Path(args.subsets_dir)
    if subsets_dir.is_dir():
        for txt in sorted(subsets_dir.glob("*.txt")):
            stem = txt.stem
            jobs.append((stem, str(txt.resolve())))
    else:
        print(f"⚠️  Subsets directory missing (skip): {subsets_dir}")

    if not jobs:
        print("No input files to process.")
        return 1

    if args.dry_run:
        for tid, fp in jobs:
            print(f"Would process tunnel_id={tid!r} <- {fp}")
        return 0

    for tunnel_id, file_path in jobs:
        print(f"Processing tunnel_id={tunnel_id!r} <- {file_path}")
        try:
            out = save_raw_characteristics(file_path, tunnel_id)
            print(f"  ✓ {out}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
