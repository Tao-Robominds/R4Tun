#!/usr/bin/env python3
"""Offline validation: v1 vs v2 K-pattern consensus on existing detected.csv files."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from k_consensus_lib import (  # noqa: E402
    correct_fallback_y_positions_v1,
    correct_k_pattern_y_positions,
)

REGULAR_TUNNELS = [
    "1-1", "1-2", "1-3", "1-4", "1-5",
    "2-1", "2-2", "2-3", "2-4", "2-5",
    "3-1-1", "3-1-2", "3-1-3",
]


def find_detected_csv(tunnel: str) -> Path | None:
    candidates = [
        REPO_ROOT / "logs" / tunnel / "regular_hint" / "opus4.6" / "data" / "detected.csv",
        REPO_ROOT / "data" / "ablation_anthropic" / "memory+state+knowledge" / tunnel / "detected.csv",
        REPO_ROOT / "logs" / tunnel / "repeatability" / "run1" / "opus4.6" / "data" / "detected.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_points(path: Path) -> list[tuple[str, tuple[float, float]]]:
    df = pd.read_csv(path)
    return [(str(row.Type), (float(row.X), float(row.Y))) for row in df.itertuples()]


def ys_from_points(pts) -> list[float]:
    return [xy[1] for _, xy in pts]


def max_y_delta(a, b) -> float:
    return max(abs(ya - yb) for (_, (_, ya)), (_, (_, yb)) in zip(a, b))


def main() -> None:
    print("K-pattern consensus offline test\n" + "=" * 60)
    for tunnel in REGULAR_TUNNELS:
        path = find_detected_csv(tunnel)
        if path is None:
            print(f"{tunnel}: SKIP (no detected.csv)")
            continue

        raw = load_points(path)
        v1 = correct_fallback_y_positions_v1(raw, tunnel)
        v2, snapped = correct_k_pattern_y_positions(raw, tunnel, k_pattern_correction=True, k_pattern_outlier_tol_px=215.0)

        d_v1 = max_y_delta(raw, v1)
        d_v2 = max_y_delta(raw, v2)
        print(f"\n{tunnel} ({path.parent.parent.name if 'logs' in str(path) else 'data'})")
        print(f"  v1 max |ΔY|: {d_v1:.1f}px  v2 max |ΔY|: {d_v2:.1f}px  snapped: {snapped}")
        if snapped:
            for i, ((t0, (_, y0)), (t1, (_, y1))) in enumerate(zip(raw, v2)):
                if abs(y0 - y1) > 0.5:
                    print(f"    ring {i}: {t0} Y {y0:.0f} -> {y1:.0f}")

    print("\n" + "=" * 60)
    print("Acceptance checks:")
    for tunnel, ring, y_before_approx, y_after_approx in [
        ("3-1-1", 7, 296, 1778),
        ("1-4", None, None, None),
        ("2-1", None, None, None),
    ]:
        path = find_detected_csv(tunnel)
        if not path:
            continue
        raw = load_points(path)
        v2, _ = correct_k_pattern_y_positions(raw, tunnel)
        if ring is not None:
            y0, y1 = raw[ring][1][1], v2[ring][1][1]
            ok = abs(y0 - y_before_approx) < 50 and abs(y1 - y_after_approx) < 200
            print(f"  {tunnel} ring {ring}: {y0:.0f}->{y1:.0f} {'OK' if ok else 'CHECK'}")
        elif tunnel == "1-4":
            ys = [v2[i][1][1] for i in range(len(v2))]
            outliers = [y for y in ys if y > 1850 or y < 1100]
            print(f"  1-4 v2 Y range [{min(ys):.0f}, {max(ys):.0f}] outliers>1850: {len([y for y in ys if y>1850])}")
        elif tunnel == "2-1":
            d = max_y_delta(raw, v2)
            print(f"  2-1 unchanged: {'OK' if d < 1 else f'CHANGED {d:.1f}px'}")


if __name__ == "__main__":
    main()
