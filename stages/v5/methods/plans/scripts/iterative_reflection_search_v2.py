#!/usr/bin/env python3
"""Aggressive iterative intrinsic reflection search (tunnels 4/5 focus).

Outputs:
  logs/iterative_reflection_proof_v2/panel/r0/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "logs" / "iterative_reflection_proof_v2"
PANEL_ROOT = OUT_ROOT / "panel" / "r0"
RINGS_ROOT = OUT_ROOT / "heldout_iterative_reflection"

PAIRS_STEP7 = REPO_ROOT / "logs" / "reflection_proof_v1" / "panel" / "r0" / "reflection_proof_pairs.csv"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _import_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _guarded_j(det: dict[str, Any], pre: dict[str, Any], base_det: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    s_boundary = _safe_float(det.get("S_boundary")) or 0.0
    g_pre = float(
        np.clip(
            min(
                _safe_float(pre.get("coverage_factor")) or 0.0,
                _safe_float(pre.get("empty_factor")) or 0.0,
                _safe_float(pre.get("shape_factor")) or 0.0,
            ),
            0.0,
            1.0,
        )
    )
    s_cont = _safe_float(det.get("S_continuity")) or 0.0
    s_k = _safe_float(det.get("S_K")) or 0.0
    s_spacing = _safe_float(det.get("S_spacing")) or 0.0
    s_cov = _safe_float(det.get("S_layout_coverage")) or 0.0
    g_layout = float(
        np.clip(
            s_cont
            * max(0.1, min(1.0, s_k / 0.25))
            * max(0.1, min(1.0, s_spacing / 0.3))
            * max(0.1, min(1.0, s_cov / 0.001)),
            0.0,
            1.0,
        )
    )
    base_s = _safe_float(base_det.get("S_boundary")) or 0.0
    g_stability = float(np.clip((s_boundary / base_s), 0.0, 1.0)) if base_s > 0 else 1.0
    guard_pass = bool(g_pre >= 0.25 and g_layout >= 0.05 and g_stability >= 0.2)
    j = float(s_boundary * g_pre * g_layout * g_stability)
    return j, {"G_pre": g_pre, "G_layout": g_layout, "G_stability": g_stability, "guardrail_pass": guard_pass}


def _candidate_params(params: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    base = dict(params)
    min_score = _safe_float(base.get("single_ring_visual_slot_min_score"))
    snap_px = _safe_float(base.get("single_ring_visual_slot_snap_px"))
    bthr = _safe_float(base.get("binary_threshold"))
    min_len = _safe_float(base.get("hough_min_length"))
    max_gap = _safe_float(base.get("hough_max_gap"))

    score_factors = [0.75, 0.9, 1.1]
    snap_deltas = [-20, -10, 10, 20]
    thr_deltas = [-20, -10, 10, 20]

    for sf in score_factors:
        cand = dict(base)
        if min_score is not None:
            cand["single_ring_visual_slot_min_score"] = float(np.clip(min_score * sf, 0.01, 0.95))
        out.append(cand)
    for ds in snap_deltas:
        cand = dict(base)
        if snap_px is not None:
            cand["single_ring_visual_slot_snap_px"] = int(max(1, round(snap_px + ds)))
        out.append(cand)
    for dt in thr_deltas:
        cand = dict(base)
        if bthr is not None:
            cand["binary_threshold"] = int(np.clip(round(bthr + dt), 10, 250))
        if min_len is not None:
            cand["hough_min_length"] = int(max(1, round(min_len + dt)))
        if max_gap is not None:
            cand["hough_max_gap"] = int(max(1, round(max_gap + dt)))
        out.append(cand)
    # Deduplicate by json fingerprint.
    uniq = {}
    for c in out:
        uniq[json.dumps(c, sort_keys=True)] = c
    return list(uniq.values())


def _run_one_ring(
    *,
    tunnel_id: str,
    ring_id: int,
    a0_output_dir: Path,
    max_rounds: int,
    patience: int,
    min_delta_proxy: float,
    max_candidates: int,
) -> dict[str, Any]:
    ring_key = f"{tunnel_id}/r{ring_id}"
    ring_root = RINGS_ROOT / tunnel_id / f"r{ring_id}" / "A2_iterative_intrinsic_reflection"
    work_base = RINGS_ROOT / "_work"
    work_ring = work_base / tunnel_id / f"r{ring_id}"
    if ring_root.exists():
        shutil.rmtree(ring_root)
    if work_ring.exists():
        shutil.rmtree(work_ring)
    work_ring.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(a0_output_dir, work_ring)

    detection = _import_mod(f"det_{tunnel_id}_{ring_id}", REPO_ROOT / "agents" / "2_detection" / "2_detection.py")
    segmentation = _import_mod(f"seg_{tunnel_id}_{ring_id}", REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py")
    evaluation = _import_mod(f"eval_{tunnel_id}_{ring_id}", REPO_ROOT / "agents" / "evaluation.py")
    pre_metrics_mod = _import_mod(f"prem_{tunnel_id}_{ring_id}", REPO_ROOT / "bo" / "preprocessing_iou_metrics.py")
    det_metrics_mod = _import_mod(f"detm_{tunnel_id}_{ring_id}", REPO_ROOT / "bo" / "detection_boundary_metrics.py")

    param_path = work_ring / "parameters_detection.json"
    cur_params = _load_json(param_path) if param_path.exists() else {}

    # Baseline intrinsic.
    base_pre = pre_metrics_mod.compute_target_guarded_metrics(work_ring)
    base_det = det_metrics_mod.compute_detection_boundary_metrics(work_ring)
    best_j, best_g = _guarded_j(base_det, base_pre, base_det)
    best_params = dict(cur_params)
    best_round = {
        "round_id": 0,
        "accepted": True,
        "selected": True,
        "params": best_params,
        "J_reflect": best_j,
        **best_g,
        "S_boundary": _safe_float(base_det.get("S_boundary")),
    }
    trace = [best_round]

    no_improve = 0
    for rid in range(1, max_rounds + 1):
        improved = False
        for cand in _candidate_params(best_params)[:max_candidates]:
            param_path.write_text(json.dumps(cand, indent=2, sort_keys=True) + "\n")
            try:
                detection.run_detection(tunnel_id, ring_id, base_dir=str(work_base))
                segmentation.run_segmentation(tunnel_id, ring_id, base_dir=str(work_base))
                det_m = det_metrics_mod.compute_detection_boundary_metrics(work_ring)
                pre_m = pre_metrics_mod.compute_target_guarded_metrics(work_ring)
                j, g = _guarded_j(det_m, pre_m, base_det)
            except Exception as exc:  # noqa: BLE001
                trace.append(
                    {
                        "round_id": rid,
                        "accepted": False,
                        "selected": False,
                        "params": cand,
                        "error": str(exc),
                    }
                )
                continue

            accept = bool(g["guardrail_pass"] and j >= (best_j + min_delta_proxy))
            trace.append(
                {
                    "round_id": rid,
                    "accepted": accept,
                    "selected": False,
                    "params": cand,
                    "J_reflect": j,
                    **g,
                    "S_boundary": _safe_float(det_m.get("S_boundary")),
                }
            )
            if accept:
                best_j = float(j)
                best_params = dict(cand)
                improved = True
        # rollback or keep best params.
        param_path.write_text(json.dumps(best_params, indent=2, sort_keys=True) + "\n")
        if improved:
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    # Final run with best params and evaluate.
    detection.run_detection(tunnel_id, ring_id, base_dir=str(work_base))
    segmentation.run_segmentation(tunnel_id, ring_id, base_dir=str(work_base))
    eval_res = evaluation.evaluate(tunnel_id, ring_id, base_dir=str(work_base))
    final_det = det_metrics_mod.compute_detection_boundary_metrics(work_ring)
    final_pre = pre_metrics_mod.compute_target_guarded_metrics(work_ring)
    final_j, final_g = _guarded_j(final_det, final_pre, base_det)
    ring_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(work_ring, ring_root)
    if work_ring.exists():
        shutil.rmtree(work_ring)
    out = {
        "ring_key": ring_key,
        "tunnel_id": tunnel_id,
        "ring_id": ring_id,
        "best_J_reflect": final_j,
        "best_params": best_params,
        "trace_len": len(trace),
        "trace": trace,
        "final_mIoU": _safe_float(eval_res.get("mIoU")),
        "final_OA": _safe_float(eval_res.get("OA")),
        "S_boundary": _safe_float(final_det.get("S_boundary")),
        **final_g,
        "output_dir": str(ring_root),
    }
    _write_json(ring_root / "iterative_trace.json", out)
    return out


def _main(args: argparse.Namespace) -> int:
    PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    RINGS_ROOT.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(PAIRS_STEP7)
    a1 = pairs[pairs["variant"] == "A1_proxy_reflection"].copy()
    # focus tunnel 4/5 as requested
    target = a1[a1["tunnel_id"].astype(str).str.startswith(("4-", "5-"))].copy().reset_index(drop=True)
    if args.max_rings is not None:
        target = target.head(int(args.max_rings)).copy().reset_index(drop=True)
    rows = []
    for _, row in target.iterrows():
        t = str(row["tunnel_id"])
        r = int(row["ring_id"])
        a0_dir = Path(str(row["A0_output_dir"]))
        result = _run_one_ring(
            tunnel_id=t,
            ring_id=r,
            a0_output_dir=a0_dir,
            max_rounds=int(args.max_rounds),
            patience=int(args.patience),
            min_delta_proxy=float(args.min_delta_proxy),
            max_candidates=int(args.max_candidates),
        )
        rows.append(
            {
                "ring_key": result["ring_key"],
                "tunnel_id": t,
                "ring_id": r,
                "mIoU_no_reflection": _safe_float(row["mIoU_no_reflection"]),
                "mIoU_A1_single_pass": _safe_float(row["mIoU_reflection"]),
                "mIoU_A2_iterative": result["final_mIoU"],
                "delta_mIoU_A2_vs_A0": None
                if _safe_float(row["mIoU_no_reflection"]) is None or result["final_mIoU"] is None
                else float(result["final_mIoU"] - float(row["mIoU_no_reflection"])),
                "OA_no_reflection": _safe_float(row["OA_no_reflection"]),
                "OA_A1_single_pass": _safe_float(row["OA_reflection"]),
                "OA_A2_iterative": result["final_OA"],
                "delta_OA_A2_vs_A0": None
                if _safe_float(row["OA_no_reflection"]) is None or result["final_OA"] is None
                else float(result["final_OA"] - float(row["OA_no_reflection"])),
                "best_J_reflect": result["best_J_reflect"],
                "S_boundary_A2": result["S_boundary"],
                "output_dir": result["output_dir"],
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(PANEL_ROOT / "t45_iterative_results.csv", index=False)

    valid = out_df.dropna(subset=["delta_mIoU_A2_vs_A0", "delta_OA_A2_vs_A0"])
    dmiou = valid["delta_mIoU_A2_vs_A0"].to_numpy(dtype=float) if not valid.empty else np.array([])
    doa = valid["delta_OA_A2_vs_A0"].to_numpy(dtype=float) if not valid.empty else np.array([])
    t_p = _safe_float(ttest_rel(valid["mIoU_A2_iterative"], valid["mIoU_no_reflection"], nan_policy="omit").pvalue) if len(valid) >= 2 else None
    try:
        w_p = _safe_float(wilcoxon(dmiou).pvalue) if len(valid) >= 2 else None
    except ValueError:
        w_p = None
    summary = {
        "timestamp_utc": _now(),
        "focus": "tunnel prefix 4 and 5",
        "n_rows": int(len(valid)),
        "mean_mIoU_A0": _safe_float(valid["mIoU_no_reflection"].mean()) if not valid.empty else None,
        "mean_mIoU_A1": _safe_float(valid["mIoU_A1_single_pass"].mean()) if not valid.empty else None,
        "mean_mIoU_A2": _safe_float(valid["mIoU_A2_iterative"].mean()) if not valid.empty else None,
        "median_mIoU_A2": _safe_float(valid["mIoU_A2_iterative"].median()) if not valid.empty else None,
        "share_A2_ge_0_4": _safe_float((valid["mIoU_A2_iterative"] >= 0.4).mean()) if not valid.empty else None,
        "share_A2_ge_0_5": _safe_float((valid["mIoU_A2_iterative"] >= 0.5).mean()) if not valid.empty else None,
        "mean_delta_mIoU_A2_vs_A0": _safe_float(np.mean(dmiou)) if dmiou.size else None,
        "mean_delta_OA_A2_vs_A0": _safe_float(np.mean(doa)) if doa.size else None,
        "paired_ttest_p_mIoU": t_p,
        "wilcoxon_p_mIoU": w_p,
        "max_rounds": int(args.max_rounds),
        "patience": int(args.patience),
        "min_delta_proxy": float(args.min_delta_proxy),
    }
    _write_json(PANEL_ROOT / "t45_iterative_summary.json", summary)
    report = [
        "# Tunnel 4/5 Iterative Reflection Report (v2)",
        "",
        f"- rows: `{summary['n_rows']}`",
        f"- mean mIoU A0: `{summary['mean_mIoU_A0']}`",
        f"- mean mIoU A1 single-pass: `{summary['mean_mIoU_A1']}`",
        f"- mean mIoU A2 iterative: `{summary['mean_mIoU_A2']}`",
        f"- median mIoU A2 iterative: `{summary['median_mIoU_A2']}`",
        f"- A2 share >=0.4: `{summary['share_A2_ge_0_4']}`",
        f"- A2 share >=0.5: `{summary['share_A2_ge_0_5']}`",
        f"- mean delta mIoU A2 vs A0: `{summary['mean_delta_mIoU_A2_vs_A0']}`",
        f"- paired t-test p mIoU: `{summary['paired_ttest_p_mIoU']}`",
        f"- Wilcoxon p mIoU: `{summary['wilcoxon_p_mIoU']}`",
    ]
    (PANEL_ROOT / "t45_iterative_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--min-delta-proxy", type=float, default=0.0005)
    p.add_argument("--max-candidates", type=int, default=6)
    p.add_argument("--max-rings", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_main(parse_args()))
