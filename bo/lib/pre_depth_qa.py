"""Ring-constant preprocessing QA (3a + PRE7) from frozen depth maps."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_depth_3a(ring_dir: Path) -> dict[str, float | bool]:
    """Compute 3a guardrail metrics from depth_map.npy (read-only)."""
    dm = np.load(ring_dir / "depth_map.npy")
    finite = np.isfinite(dm)
    valid = finite & (dm > 0.0)
    h = int(dm.shape[0])
    row_ok = np.mean(valid, axis=1) > 0.01
    row_nonempty_ratio = float(np.count_nonzero(row_ok) / max(h, 1))
    finite_ratio = float(np.count_nonzero(finite) / max(int(dm.size), 1))
    empty = ~row_ok
    max_gap = cur = 0
    for v in empty:
        if v:
            cur += 1
            max_gap = max(max_gap, cur)
        else:
            cur = 0
    gap_frac = float(max_gap / max(h, 1))
    passed = finite_ratio >= 0.60 and row_nonempty_ratio >= 0.90 and gap_frac <= 0.08
    return {
        "finite_ratio": finite_ratio,
        "row_nonempty_ratio": row_nonempty_ratio,
        "largest_empty_vertical_gap_frac": gap_frac,
        "passed_3a": passed,
        "shape_h": h,
    }
