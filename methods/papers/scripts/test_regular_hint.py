#!/usr/bin/env python3
"""Offline tests for regular_hint_lib (no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from regular_hint_lib import (  # noqa: E402
    apply_hint_mode,
    gt_k_pixel_positions,
    hint_level_to_mode,
    infer_zigzag_levels,
    propagate_zigzag_y,
    uniform_ring_x_positions,
)

MSK = REPO_ROOT / "data" / "ablation_anthropic" / "memory+state+knowledge"


def test_uniform_x():
    xs = uniform_ring_x_positions(2420, 10)
    assert len(xs) == 10
    assert abs(xs[1] - xs[0] - 242) < 1


def test_zigzag_propagate():
    ys = propagate_zigzag_y(6, 1200, 1640, low_parity=0)
    assert ys[0] == 1200 and ys[1] == 1640


def test_gt_calibration_2_2():
    tunnel_dir = MSK / "2-2"
    det = __import__("pandas").read_csv(tunnel_dir / "detected.csv")
    xs, ys, _ = gt_k_pixel_positions(
        tunnel_dir,
        detected_x=det["X"].tolist(),
        detected_y=det["Y"].tolist(),
    )
    rmse_x = float(np.sqrt(np.mean((np.array(xs) - det["X"].to_numpy()) ** 2)))
    rmse_y = float(np.sqrt(np.mean((np.array(ys) - det["Y"].to_numpy()) ** 2)))
    assert rmse_x < 15
    assert rmse_y < 15


def test_hint_modes_offline():
    tunnel_dir = MSK / "2-2"
    det = __import__("pandas").read_csv(tunnel_dir / "detected.csv")
    hough = [(str(r.Type), (float(r.X), float(r.Y))) for r in det.itertuples()]
    for level in ("L1", "L5", "L6"):
        mode = hint_level_to_mode(level)
        out = apply_hint_mode(
            hough,
            "2-2",
            tunnel_dir,
            hint_mode=mode,
            ring_count=len(hough),
            image_width=2420,
            image_height=2580,
        )
        assert len(out) == len(hough)


def main() -> None:
    test_uniform_x()
    test_zigzag_propagate()
    test_gt_calibration_2_2()
    test_hint_modes_offline()
    print("All offline tests passed.")


if __name__ == "__main__":
    main()
