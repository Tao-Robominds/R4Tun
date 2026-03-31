"""
GT-free intrinsic quality metrics from m_s_k (or any) pipeline outputs.

Used by reflection ablation analysts to tune detecting + SAM parameters without mIoU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[4]

_GOOD_DETECTION_TYPES = frozenset({"midpoint", "positive_slope", "negative_slope", "horizontal"})
_FALLBACK_DETECTION_TYPES = frozenset({"assume", "default"})

_DEFAULT_MSK_PREFIX = "data/ablation/memory+state+knowledge"


def msk_tunnel_dir(tunnel_id: str, out_prefix: str | None = None) -> Path:
    pfx = (out_prefix or _DEFAULT_MSK_PREFIX).strip().rstrip("/")
    return _REPO_ROOT / pfx / tunnel_id


def compute_detection_quality(
    detected_csv: Path,
    ring_count_path: Path | None = None,
) -> dict[str, Any]:
    """Metrics from detected.csv + optional ring_count.txt (GT-free)."""
    out: dict[str, Any] = {"detected_csv": str(detected_csv), "error": None}
    if not detected_csv.is_file():
        out["error"] = f"missing {detected_csv}"
        return out

    df = pd.read_csv(detected_csv)
    if "Type" not in df.columns or "X" not in df.columns:
        out["error"] = "detected.csv missing Type or X column"
        return out

    types = df["Type"].astype(str).str.strip()
    n = int(len(types))
    if n == 0:
        out.update(
            {
                "total_points": 0,
                "good_detection_ratio": 0.0,
                "fallback_ratio": 0.0,
                "type_counts": {},
            }
        )
        return out

    good = int(types.isin(_GOOD_DETECTION_TYPES).sum())
    fallback = int(types.isin(_FALLBACK_DETECTION_TYPES).sum())
    type_counts = types.value_counts().to_dict()
    type_counts = {str(k): int(v) for k, v in type_counts.items()}

    xs = df.sort_values("X")["X"].astype(float).values
    if len(xs) >= 2:
        dx = np.diff(xs)
        mean_dx = float(np.mean(dx))
        cv_x = float(np.std(dx) / mean_dx) if mean_dx > 0 else 0.0
    else:
        cv_x = 0.0

    ring_expected = None
    if ring_count_path and ring_count_path.is_file():
        try:
            ring_expected = int(ring_count_path.read_text().strip())
        except (ValueError, OSError):
            ring_expected = None

    out.update(
        {
            "total_points": n,
            "good_detection_ratio": good / n,
            "fallback_ratio": fallback / n,
            "type_counts": type_counts,
            "x_spacing_cv": cv_x,
            "num_detected_rings": n,
            "ring_count_expected": ring_expected,
            "ring_count_match": (
                n == ring_expected if ring_expected is not None else None
            ),
        }
    )
    return out


def compute_coverage_balance(final_csv: Path) -> dict[str, Any]:
    """Per-block pred counts, CV, critical blocks — uses ``pred`` only (GT-free)."""
    out: dict[str, Any] = {"final_csv": str(final_csv), "error": None}
    if not final_csv.is_file():
        out["error"] = f"missing {final_csv}"
        return out

    df = pd.read_csv(final_csv)
    if "pred" not in df.columns:
        out["error"] = "final.csv missing pred column"
        return out

    pred = pd.to_numeric(df["pred"], errors="coerce").fillna(-1).astype(int)
    total = int(len(pred))
    bg = int((pred == 0).sum())
    non_bg = total - bg
    non_bg_ratio = float(non_bg / total) if total else 0.0

    # Semantic block labels only (SAM schema); ignore stray high IDs for balance stats
    _in_block = (pred >= 1) & (pred <= 7)
    labels = [c for c in sorted(pred[_in_block].unique()) if c > 0]
    block_counts: dict[str, int] = {}
    for c in labels:
        block_counts[f"class_{c}"] = int((pred == c).sum())

    counts = list(block_counts.values())
    if counts:
        arr = np.array(counts, dtype=float)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        cv = float(100.0 * sigma / mu) if mu > 0 else 0.0
        critical_threshold = 0.3 * mu
        critical = [k for k, v in block_counts.items() if v < critical_threshold]
        weakest = min(block_counts, key=lambda k: block_counts[k])
    else:
        mu = sigma = cv = 0.0
        critical = []
        weakest = None

    per_ring: dict[str, Any] = {}
    ring_col = "pred_ring" if "pred_ring" in df.columns else None
    if ring_col:
        for ring_id, sub in df.groupby(ring_col):
            pr = pd.to_numeric(sub["pred"], errors="coerce").fillna(-1).astype(int)
            counts_r = pr.value_counts()
            non_bg_r = int((pr != 0).sum())
            min_cls = None
            min_n = None
            for c in counts_r.index:
                if c == 0:
                    continue
                n = int(counts_r[c])
                if min_n is None or n < min_n:
                    min_n = n
                    min_cls = int(c)
            per_ring[str(ring_id)] = {
                "non_background_points": non_bg_r,
                "sparsest_pred_class": min_cls,
                "sparsest_pred_count": min_n,
            }

    out.update(
        {
            "total_points": total,
            "non_background_ratio": non_bg_ratio,
            "per_block_counts": block_counts,
            "average_points_per_block": mu,
            "coefficient_of_variation_pct": cv,
            "critical_blocks": critical,
            "weakest_block": weakest,
            "per_ring_summary": per_ring,
        }
    )
    return out


def compute_depth_map_context(depth_npy: Path) -> dict[str, Any]:
    """NaN ratio and column-wise NaN fraction (proxy for ring-wise gaps)."""
    out: dict[str, Any] = {"depth_map": str(depth_npy), "error": None}
    if not depth_npy.is_file():
        out["error"] = f"missing {depth_npy}"
        return out

    arr = np.load(depth_npy)
    if arr.ndim != 2:
        out["error"] = f"expected 2D array, got shape {arr.shape}"
        return out

    nan_mask = np.isnan(arr)
    total = int(arr.size)
    nan_count = int(nan_mask.sum())
    nan_ratio = float(nan_count / total) if total else 0.0

    # Per-column (x / ring axis) NaN fraction
    col_nan_frac = nan_mask.mean(axis=0)
    worst_col_idx = int(np.argmax(col_nan_frac)) if col_nan_frac.size else -1
    worst_col_frac = float(np.max(col_nan_frac)) if col_nan_frac.size else 0.0

    out.update(
        {
            "shape": list(arr.shape),
            "nan_ratio": nan_ratio,
            "nan_pixel_count": nan_count,
            "worst_column_nan_fraction": worst_col_frac,
            "worst_column_index": worst_col_idx,
        }
    )
    return out


def build_intrinsic_quality_report(
    tunnel_id: str,
    msk_out_prefix: str | None = None,
) -> dict[str, Any]:
    """Aggregate all intrinsic signals for one tunnel (m_s_k output tree)."""
    base = msk_tunnel_dir(tunnel_id, msk_out_prefix)
    ring_count = base / "ring_count.txt"
    detected = base / "detected.csv"
    final_csv = base / "final.csv"
    depth_npy = base / "depth_map_outlier.npy"

    return {
        "tunnel_id": tunnel_id,
        "source_output_dir": str(base),
        "detection_quality": compute_detection_quality(detected, ring_count),
        "coverage_balance": compute_coverage_balance(final_csv),
        "depth_map_context": compute_depth_map_context(depth_npy),
    }


def intrinsic_report_json(
    tunnel_id: str,
    msk_out_prefix: str | None = None,
    indent: int = 2,
) -> str:
    return json.dumps(
        build_intrinsic_quality_report(tunnel_id, msk_out_prefix),
        indent=indent,
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Print intrinsic quality JSON for a tunnel.")
    p.add_argument("tunnel_id", help="e.g. 1-1")
    p.add_argument(
        "--msk-prefix",
        default=_DEFAULT_MSK_PREFIX,
        help="Pipeline output prefix for the reference run (default: m_s_k)",
    )
    args = p.parse_args()
    print(intrinsic_report_json(args.tunnel_id, args.msk_prefix))


if __name__ == "__main__":
    main()
