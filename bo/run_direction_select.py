#!/usr/bin/env python3
"""Deploy-time direction (plus/minus) selection for one ring.

Runs detection once, scores both direction hypotheses with label-free metrics,
runs segmentation on each branch, and commits the higher-scoring branch.

All writes go to logs/<run_id>/; corpora under data/ are read-only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_BO_DIR = Path(__file__).resolve().parent
REPO_ROOT = _BO_DIR.parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))

from lib.ceiling_gate import DET_CLI, DET_DEFAULT, SEG_DEFAULT, VENV_PY  # noqa: E402
from lib.direction_select import (  # noqa: E402
    SELECTION_FILENAME,
    select_direction_and_segment,
    write_selection_to_out_dir,
)
from lib.layout_bo import build_ring_context, _compute_miou  # noqa: E402


def _write_agent_params(
    ctx,
    *,
    r_surface_min: float | None,
    slot_inset_y: float,
    k_y: float | None = None,
    offsets: dict[str, float] | None = None,
    layout_params: dict | None = None,
) -> None:
    det = json.loads(DET_DEFAULT.read_text(encoding="utf-8"))
    seg = json.loads(SEG_DEFAULT.read_text(encoding="utf-8"))
    det["segment_count"] = ctx.segment_count
    det["enabled_blocks"] = ctx.blocks
    seg["segment_count"] = ctx.segment_count
    seg["slot_inset_y"] = float(slot_inset_y)
    if r_surface_min is not None:
        seg["r_surface_min"] = float(r_surface_min)
    if k_y is not None and offsets is not None:
        H = ctx.H
        det["k_anchor_semantics"] = "boundary_start"
        det["k_y_positions"] = [float(k_y) % H]
        det["per_ring_offsets"] = {"0": {b: float(offsets[b]) for b in ctx.blocks}}
    if layout_params:
        for key, val in layout_params.items():
            if key == "slot_inset_y":
                continue
            if key in ("hough_threshold", "hough_horizontal_threshold"):
                det[key] = int(round(val))
            else:
                det[key] = val
    ctx.sandbox_ring.mkdir(parents=True, exist_ok=True)
    (ctx.sandbox_ring / "parameters_detection.json").write_text(json.dumps(det, indent=2) + "\n", encoding="utf-8")
    (ctx.sandbox_ring / "parameters_segmentation.json").write_text(json.dumps(seg, indent=2) + "\n", encoding="utf-8")


def _load_layout_from_json(path: Path) -> tuple[float, dict[str, float], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "best_k_y" in data:
        k_y = float(data["best_k_y"])
        offsets = {k: float(v) for k, v in data["best_offsets"].items()}
        layout = dict(data.get("best_layout_params") or {})
        return k_y, offsets, layout
    k_y = float(data["k_y"])
    offsets = {k: float(v) for k, v in data["offsets"].items()}
    return k_y, offsets, {}


def _run_detection(ctx) -> bool:
    import os

    log = ctx.sandbox_ring / "logs" / "direction_select_2_detection.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["INTRINSIC_PARAMS_BASE_DIR_ONLY"] = "1"
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            [str(VENV_PY), str(DET_CLI), ctx.tunnel_id, str(ctx.ring_id), "--data-dir", str(ctx.sandbox_data)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=900,
            check=False,
        )
    return proc.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description="GT-free plus/minus direction selection")
    ap.add_argument("tunnel_id")
    ap.add_argument("ring_id", type=int)
    ap.add_argument("--source-dir", required=True, help="Read-only preprocessing root")
    ap.add_argument("--run-root", required=True, help="Output root under logs/")
    ap.add_argument("--prefer-branch", default="plus", choices=("plus", "minus"))
    ap.add_argument("--r-surface-min", type=float, default=None)
    ap.add_argument("--slot-inset-y", type=float, default=0.0)
    ap.add_argument("--layout-json", default=None, help="gt_layout.json or best_bo_trial.json with k_y/offsets")
    ap.add_argument("--skip-detection", action="store_true", help="Reuse existing detection in sandbox")
    args = ap.parse_args()

    source = Path(args.source_dir).resolve()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    ctx = build_ring_context(args.tunnel_id, args.ring_id, source_root=source, run_root=run_root)

    k_y, offsets, layout_params = None, None, {}
    r_surface = args.r_surface_min
    slot_inset = args.slot_inset_y
    if args.layout_json:
        k_y, offsets, layout_params = _load_layout_from_json(Path(args.layout_json))
        if r_surface is None and "best_r_surface_min" in json.loads(Path(args.layout_json).read_text()):
            r_surface = float(json.loads(Path(args.layout_json).read_text())["best_r_surface_min"])
        if not slot_inset and layout_params.get("slot_inset_y"):
            slot_inset = float(layout_params["slot_inset_y"])

    _write_agent_params(
        ctx,
        r_surface_min=r_surface,
        slot_inset_y=slot_inset,
        k_y=k_y,
        offsets=offsets,
        layout_params=layout_params or None,
    )

    if not args.skip_detection:
        if not _run_detection(ctx):
            print("Detection failed", file=sys.stderr)
            return 1

    selection = select_direction_and_segment(
        tunnel_id=ctx.tunnel_id,
        ring_id=ctx.ring_id,
        sandbox_data=ctx.sandbox_data,
        ring_dir=ctx.sandbox_ring,
        tag="direction_select",
        prefer_branch=args.prefer_branch,
        segment_count=ctx.segment_count,
    )
    write_selection_to_out_dir(ctx.sandbox_ring, ctx.out_dir)
    (ctx.out_dir / SELECTION_FILENAME).write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )

    gt_miou = None
    final_path = ctx.sandbox_ring / "final.csv"
    if final_path.is_file():
        import pandas as pd

        gt_miou = _compute_miou(pd.read_csv(final_path), max_class=ctx.segment_count)

    print(f"== direction select: {ctx.case_id} ==")
    print(f"  selected: {selection.get('selected_branch')}")
    print(f"  score_plus: {selection.get('score_plus')}")
    print(f"  score_minus: {selection.get('score_minus')}")
    print(f"  margin: {selection.get('margin')}")
    if gt_miou is not None:
        print(f"  gt_miou (audit): {gt_miou:.4f}")
    print(f"  output: {ctx.out_dir / SELECTION_FILENAME}")
    return 0 if not selection.get("agent_error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
