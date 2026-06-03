#!/usr/bin/env python3
"""E2E plus vs minus on a GT-plus ring: det → direction select → seg → GT mIoU."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BO = Path(__file__).resolve().parent
sys.path.insert(0, str(_BO))

from lib.ceiling_gate import REPO_ROOT
from lib.layout_bo import _compute_miou, build_ring_context, evaluate_trial
from lib.search_space import decode_layout_params, default_layout_fracs


def _load_gt(src_ring: Path) -> tuple[float, dict[str, float], float, int]:
    gt = json.loads((src_ring / "gt_layout.json").read_text(encoding="utf-8"))
    ceiling = json.loads((src_ring / "ceiling.json").read_text(encoding="utf-8"))
    seg_n = len(gt["offsets"])
    r_surf = float(ceiling["r_surface_min_selected"])
    return float(gt["k_y"]), {k: float(v) for k, v in gt["offsets"].items()}, r_surf, seg_n


def _miou_on_path(path: Path, max_class: int) -> float | None:
    if not path.is_file():
        return None
    return _compute_miou(pd.read_csv(path), max_class=max_class)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tunnel-id", default="1-1")
    ap.add_argument("--ring-id", type=int, default=20)
    ap.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "data" / "bo_calibration",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "direction_e2e_verify_v1",
    )
    args = ap.parse_args()

    src_ring = args.source_dir / args.tunnel_id / f"r{args.ring_id}"
    k_y, offsets, r_surf, seg_n = _load_gt(src_ring)
    layout = decode_layout_params(
        np.concatenate([[0.0], np.zeros(seg_n), default_layout_fracs()]),
        seg_n,
    )

    rows = []
    branch_miou: dict[str, float | None] = {}
    for branch in ("plus", "minus"):
        run_root = args.out_dir / branch
        ctx = build_ring_context(
            args.tunnel_id,
            args.ring_id,
            source_root=args.source_dir,
            run_root=run_root,
            segment_count=seg_n,
        )
        metrics = evaluate_trial(
            ctx,
            k_y,
            offsets,
            layout,
            r_surf,
            tag=f"e2e_{branch}",
            order_branch=branch,
        )
        ring_dir = ctx.sandbox_ring
        committed = _miou_on_path(ring_dir / "final.csv", seg_n)
        branch_miou[branch] = committed
        rows.append(
            {
                "prefer_branch": branch,
                "selected_branch": metrics.get("order_branch"),
                "committed_gt_miou": committed,
                "artifact_plus_gt_miou": _miou_on_path(ring_dir / "final_direction_plus.csv", seg_n),
                "artifact_minus_gt_miou": _miou_on_path(ring_dir / "final_direction_minus.csv", seg_n),
                "direction_score_plus": metrics.get("direction_score_plus"),
                "direction_score_minus": metrics.get("direction_score_minus"),
                "direction_margin": metrics.get("direction_margin"),
                "agent_error": metrics.get("agent_error"),
            }
        )

    gt = json.loads((src_ring / "gt_layout.json").read_text(encoding="utf-8"))
    p = branch_miou.get("plus")
    m = branch_miou.get("minus")
    gate = {
        "case_id": f"{args.tunnel_id}/r{args.ring_id}",
        "direction_tier_gt": "plus",
        "spatial_order_by_label": gt.get("spatial_order_by_label"),
        "ceiling_miou_reference": json.loads((src_ring / "ceiling.json").read_text())[
            "agents_gt_ceiling_miou"
        ],
        "runs": rows,
        "delta_committed_minus_vs_plus": round(m - p, 4) if p is not None and m is not None else None,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "single_instance_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
