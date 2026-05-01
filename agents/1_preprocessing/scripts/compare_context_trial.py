#!/usr/bin/env python3
"""Compare fixed baseline trial vs existing baselines for one ring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def _largest_empty_row_band(valid_mask: np.ndarray) -> int:
    row_valid = valid_mask.sum(axis=1)
    largest = 0
    cur = 0
    for x in row_valid == 0:
        if x:
            cur += 1
            largest = max(largest, cur)
        else:
            cur = 0
    return int(largest)


def _stats_for_dir(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": str(path)}
    for name in ("depth_map.npy", "depth_map_outlier.npy"):
        f = path / name
        if not f.exists():
            out[name] = None
            continue
        arr = np.load(f)
        valid = np.isfinite(arr) & (arr > 0)
        nonempty_rows = int(np.count_nonzero(valid.sum(axis=1) > 0))
        out[name] = {
            "shape_h": int(arr.shape[0]),
            "shape_w": int(arr.shape[1]),
            "valid_ratio": float(valid.mean()) if valid.size else 0.0,
            "largest_empty_row_band": _largest_empty_row_band(valid),
            "rows_with_data": nonempty_rows,
        }
    return out


def _try_iou(path: Path) -> Dict[str, Any] | None:
    try:
        import sys

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from bo.preprocessing_iou_metrics import compute_foreground_mask_iou_metrics

        return compute_foreground_mask_iou_metrics(path)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tunnel-id", required=True)
    p.add_argument("--ring-id", required=True, type=int)
    p.add_argument("--trial-root", default="logs/context_preprocessing_v1")
    p.add_argument("--baseline-root", default="data/ablation/baseline")
    p.add_argument("--working-root", default="data")
    args = p.parse_args()

    tunnel_id = str(args.tunnel_id)
    ring_id = int(args.ring_id)
    ring_key = f"r{ring_id}"

    trial = (REPO_ROOT / args.trial_root / tunnel_id / ring_key).resolve()
    baseline = (REPO_ROOT / args.baseline_root / tunnel_id / ring_key).resolve()
    working = (REPO_ROOT / args.working_root / tunnel_id / ring_key).resolve()

    report = {
        "tunnel_id": tunnel_id,
        "ring_id": ring_id,
        "dirs": {
            "trial_fixed_bcd": _stats_for_dir(trial),
            "baseline_ablation": _stats_for_dir(baseline),
            "working_data": _stats_for_dir(working),
        },
        "iou_diagnostic": {
            "trial_fixed_bcd": _try_iou(trial),
            "baseline_ablation": _try_iou(baseline),
            "working_data": _try_iou(working),
        },
    }

    # Simple "improved" decision for orchestration:
    # - trial depth_map valid ratio not less than 70% of baseline
    # - trial largest empty row band is strictly smaller than baseline
    b = report["dirs"]["baseline_ablation"]["depth_map.npy"]
    t = report["dirs"]["trial_fixed_bcd"]["depth_map.npy"]
    improved = False
    if b is not None and t is not None:
        cov_ok = t["valid_ratio"] >= 0.70 * b["valid_ratio"]
        empty_better = t["largest_empty_row_band"] < b["largest_empty_row_band"]
        improved = bool(cov_ok and empty_better)
    report["improved_vs_baseline"] = improved

    out_json = trial / "comparison_report.json"
    out_md = trial / "comparison_report.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        f"# Context Trial Comparison — {tunnel_id}/{ring_key}",
        "",
        f"- improved_vs_baseline: **{improved}**",
        "",
        "## depth_map.npy",
    ]
    for k in ("trial_fixed_bcd", "baseline_ablation", "working_data"):
        d = report["dirs"][k]["depth_map.npy"]
        if d is None:
            lines.append(f"- {k}: missing")
            continue
        lines.append(
            f"- {k}: valid_ratio={d['valid_ratio']:.6f}, "
            f"largest_empty_row_band={d['largest_empty_row_band']}, "
            f"shape={d['shape_h']}x{d['shape_w']}"
        )
    lines += ["", "## depth_map_outlier.npy"]
    for k in ("trial_fixed_bcd", "baseline_ablation", "working_data"):
        d = report["dirs"][k]["depth_map_outlier.npy"]
        if d is None:
            lines.append(f"- {k}: missing")
            continue
        lines.append(
            f"- {k}: valid_ratio={d['valid_ratio']:.6f}, "
            f"largest_empty_row_band={d['largest_empty_row_band']}, "
            f"shape={d['shape_h']}x{d['shape_w']}"
        )
    out_md.write_text("\n".join(lines) + "\n")

    print(json.dumps({"improved_vs_baseline": improved, "report_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
