#!/usr/bin/env python3
"""Repair broken 5-1/r110 and 5-1/r111 source preprocessing.

Issue: ``logs/proxy_validation_v1/heldout_reflection_test/5-1/r{110,111}/A0_no_reflection/depth_map.npy``
is ``(1, 1)`` — context-preprocessing failed silently for these rings.
The point cloud and ``context_unwrapped.csv`` are fine; the bug was in
the depth-map projection step (now fixed in ``_ring_enhancing.py`` to
handle sparse data).

Fix: build a repaired A0 sandbox by combining
  - ``depth_map.npy``, ``depth_map_outlier.npy``, ``pixel_to_point.pkl``,
    ``unwrapped.csv``, ``denoised.csv``, ``enhanced.csv`` from
    ``data/ablation/baseline/5-1/r<id>/`` (these are correct)
  - ``context_unwrapped.csv``, ``parameters_detection.json``, ``ring_count.txt``
    from the proxy_validation_v1 A0 dir (these are fine)

Output: ``logs/gravity_v1/heldout_data_repair/5-1/r<id>/`` mirroring the
A0 schema. Then re-run gravity-promote / iterative-reflection from this
sandbox.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ABLATION_BASE = REPO_ROOT / "data" / "ablation" / "baseline"
PROXY_BASE = REPO_ROOT / "logs" / "proxy_validation_v1" / "heldout_reflection_test"
REPAIR_ROOT = REPO_ROOT / "logs" / "gravity_v1" / "heldout_data_repair"

# Originally just 5-1; extended to all calibrated tunnels because
# proxy_validation_v1's preprocessing produced sparse depth maps for
# many 4-4/5-6 rings.
RINGS = [
    ("4-3", "r170"), ("4-3", "r171"),
    ("4-4", "r212"), ("4-4", "r217"),
    ("4-5", "r244"),
    ("4-6", "r275"), ("4-6", "r276"),
    ("5-1", "r110"), ("5-1", "r111"),
    ("5-6", "r284"),
    ("5-7", "r316"), ("5-7", "r322"),
]


def repair(tunnel: str, ring: str) -> None:
    src_ablation = ABLATION_BASE / tunnel / ring
    src_proxy = PROXY_BASE / tunnel / ring / "A0_no_reflection"
    dst = REPAIR_ROOT / tunnel / ring
    dst.mkdir(parents=True, exist_ok=True)

    # From ablation/baseline (working depth + unwrap)
    for name in ("depth_map.npy", "depth_map_outlier.npy", "pixel_to_point.pkl",
                 "unwrapped.csv", "denoised.csv", "enhanced.csv", "depth_map.png", "ring_count.txt"):
        src = src_ablation / name
        if src.exists():
            shutil.copy2(src, dst / name)

    # From proxy A0 (context csv, detection params, etc.)
    for name in ("context_unwrapped.csv", "context_denoised.csv", "context_enhanced.csv",
                 "context_depth_map.npy", "context_pixel_to_point.pkl",
                 "parameters_detection.json", "all_segments.csv",
                 "boundaries_per_ring.json"):
        src = src_proxy / name
        if src.exists():
            shutil.copy2(src, dst / name)

    print(f"[ok] {tunnel}/{ring} repaired -> {dst}")


def main() -> int:
    REPAIR_ROOT.mkdir(parents=True, exist_ok=True)
    for tunnel, ring in RINGS:
        repair(tunnel, ring)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
