#!/usr/bin/env python3
"""
Filter tunnel point clouds (6 cols: x y z intensity segment ring) to a contiguous
block of N rings using ground-truth ring IDs (column index 5).

By default picks the **center** N rings among the sorted unique ring values present
in the file (stable for experiments; avoids tunnel ends).

Example:
  ./venv/bin/python3 skills/extract_subset_n_rings.py --n-rings 10 \\
      --ids 1-1 1-2 1-3 1-4 1-5 2-1 2-2 2-3 2-4 2-5

All tunnels in data/subsets:
  ids=$(ls data/subsets/*.txt | xargs -n1 basename | sed 's/\\.txt$//' | sort -V)
  ./venv/bin/python3 skills/extract_subset_n_rings.py --n-rings 10 --ids $ids
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def select_ring_ids(unique_sorted: list[int], n_target: int) -> list[int]:
    u = unique_sorted
    if len(u) <= n_target:
        return u
    excess = len(u) - n_target
    start = excess // 2
    return u[start : start + n_target]


def process_one(
    tunnel_id: str,
    subset_dir: str,
    n_rings: int,
    dry_run: bool,
) -> dict:
    import numpy as np

    path = os.path.join(subset_dir, f"{tunnel_id}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        raise ValueError(f"{path}: expected >= 6 columns, got {data.shape[1]}")

    rings = data[:, 5].astype(np.int64)
    uniq = sorted(np.unique(rings).tolist())
    keep = select_ring_ids(uniq, n_rings)
    keep_arr = np.array(keep, dtype=np.int64)
    mask = np.isin(rings, keep_arr)
    out = data[mask]

    meta = {
        "tunnel_id": tunnel_id,
        "n_rings_requested": n_rings,
        "ring_ids_kept": keep,
        "n_rings_kept": len(keep),
        "n_points_in": int(data.shape[0]),
        "n_points_out": int(out.shape[0]),
        "unique_rings_before": uniq,
    }

    if dry_run:
        return meta

    tmp = path + ".tmp"
    fmt = "%.8f %.8f %.8f %.5f %d %d"
    np.savetxt(tmp, out, fmt=fmt)
    os.replace(tmp, path)

    meta_path = os.path.join(subset_dir, f"{tunnel_id}_rings_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Subset point clouds by GT ring column.")
    ap.add_argument("--n-rings", type=int, default=10, help="Target number of rings (default 10)")
    ap.add_argument(
        "--subset-dir",
        default="data/subsets",
        help="Directory containing <tunnel_id>.txt",
    )
    ap.add_argument("--ids", nargs="+", required=True, help="Tunnel ids, e.g. 1-4 2-1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)
    subset_dir = args.subset_dir if os.path.isabs(args.subset_dir) else os.path.join(root, args.subset_dir)

    try:
        import numpy as np  # noqa: F401
    except ImportError:
        print("numpy required", file=sys.stderr)
        return 1

    for tid in args.ids:
        meta = process_one(tid, subset_dir, args.n_rings, args.dry_run)
        print(
            f"{tid}: rings {meta['ring_ids_kept']} "
            f"({meta['n_rings_kept']} rings, {meta['n_points_in']} -> {meta['n_points_out']} pts)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
