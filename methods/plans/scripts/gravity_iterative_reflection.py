#!/usr/bin/env python3
"""Iterative reflection on gravity-aligned data (v5).

Why this is different from v3
-----------------------------
With gravity-aligned unwrap, the rotation degree of freedom is collapsed
at preprocessing time. So the highest-leverage candidate axis from v3
(template rotations + ring_offset jitter) is no longer needed. The
iterative search now only needs to perturb scalar detection-head knobs:

  * ``single_ring_visual_slot_min_score``
  * ``single_ring_visual_slot_snap_px``
  * ``binary_threshold`` / ``hough_min_length`` / ``hough_max_gap``
  * fine ``y_frac`` jitter (small ±0.005 .. ±0.02 to compensate for
    minor gravity-shift residuals)

Inputs
------
A0 baseline per ring:    ``logs/gravity_v1/heldout/<tunnel>/<ring>/``
Persistent calibration:  ``logs/gravity_v1/calibration/<tunnel>/``

Outputs
-------
``logs/gravity_v1/iterative/<tunnel>/<ring>/A2_iterative_intrinsic_reflection/``
``logs/gravity_v1/iterative_summary.csv``
``logs/gravity_v1/iterative_report.md``
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

GRAVITY_ROOT = REPO_ROOT / "logs" / "gravity_v1"
HELDOUT_GRAVITY_ROOT = GRAVITY_ROOT / "heldout"
ITER_ROOT = GRAVITY_ROOT / "iterative"
WORK_ROOT = GRAVITY_ROOT / "_iter_work"

CANONICAL_RELABEL_ROOT = REPO_ROOT / "logs" / "canonical_relabel"

from canonical_eval import canonical_miou_from_final_csv  # noqa: E402


def _import_mod(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _infer_segment_count(a0_dir: Path, default: int = 7) -> int:
    """Infer segment count from A0 final.csv GT labels when available."""
    final_csv = a0_dir / "final.csv"
    if not final_csv.exists():
        return int(default)
    try:
        df = pd.read_csv(final_csv, usecols=["segment"])
        vals = sorted(set(int(v) for v in df["segment"].fillna(0).astype(int).tolist() if int(v) > 0))
        if vals:
            return int(max(vals))
    except Exception:  # noqa: BLE001
        pass
    return int(default)


# ---------------------------------------------------------------------------
# J_reflect (same shape as v3/v4 but no structural axis since we now have
# gravity-anchored y_frac which IS the structural anchor)

def _guarded_j(det: dict[str, Any], pre: dict[str, Any], base_det: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    s_boundary = _safe_float(det.get("S_boundary")) or 0.0
    g_pre = float(np.clip(min(
        _safe_float(pre.get("coverage_factor")) or 0.0,
        _safe_float(pre.get("empty_factor")) or 0.0,
        _safe_float(pre.get("shape_factor")) or 0.0,
    ), 0.0, 1.0))
    s_cont = _safe_float(det.get("S_continuity")) or 0.0
    s_k = _safe_float(det.get("S_K")) or 0.0
    s_spacing = _safe_float(det.get("S_spacing")) or 0.0
    s_cov = _safe_float(det.get("S_layout_coverage")) or 0.0
    g_layout = float(np.clip(
        s_cont
        * max(0.1, min(1.0, s_k / 0.25))
        * max(0.1, min(1.0, s_spacing / 0.3))
        * max(0.1, min(1.0, s_cov / 0.001)),
        0.0, 1.0
    ))
    base_s = _safe_float(base_det.get("S_boundary")) or 0.0
    g_stability = float(np.clip((s_boundary / base_s), 0.0, 1.0)) if base_s > 0 else 1.0
    j = float(s_boundary * g_pre * g_layout * g_stability)
    return j, {
        "G_pre": g_pre,
        "G_layout": g_layout,
        "G_stability": g_stability,
        "guardrail_pass": bool(g_pre >= 0.25 and g_layout >= 0.05 and g_stability >= 0.2),
    }


# ---------------------------------------------------------------------------
# Candidate generation: gravity-aware (no rotation, scalar-only + fine y_frac)

def _candidate_params(base_params: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    template = base_params.get("single_ring_visual_slot_template")
    has_tpl = isinstance(template, list) and len(template) >= 4

    # 1) Fine y_frac jitter (small drift to absorb residual gravity-shift error)
    if has_tpl:
        for off in (-0.020, -0.010, -0.005, 0.005, 0.010, 0.020):
            cand = dict(base_params)
            new_tpl = []
            for entry in template:
                e = dict(entry)
                yf = float(entry.get("y_frac", 0.0))
                yn = (yf + float(off)) % 1.0
                e["y_frac"] = float(np.clip(yn, 0.0, 1.0 - 1e-6))
                new_tpl.append(e)
            new_tpl.sort(key=lambda r: float(r.get("y_frac", 0.0)))
            cand["single_ring_visual_slot_template"] = new_tpl
            cand["__candidate_kind"] = f"yfrac_drift{off:+.3f}"
            out.append(cand)

    # 2) Scalar detection-head jitter
    ms = _safe_float(base_params.get("single_ring_visual_slot_min_score"))
    sp = _safe_float(base_params.get("single_ring_visual_slot_snap_px"))
    bt = _safe_float(base_params.get("binary_threshold"))
    ml = _safe_float(base_params.get("hough_min_length"))
    mg = _safe_float(base_params.get("hough_max_gap"))

    if ms is not None:
        for sf in (0.5, 0.7, 1.4, 2.0):
            cand = dict(base_params)
            cand["single_ring_visual_slot_min_score"] = float(np.clip(ms * sf, 0.005, 0.95))
            cand["__candidate_kind"] = f"min_score*{sf}"
            out.append(cand)
    if sp is not None:
        for ds in (-15, -8, 8, 15, 30):
            cand = dict(base_params)
            cand["single_ring_visual_slot_snap_px"] = int(max(1, round(sp + ds)))
            cand["__candidate_kind"] = f"snap_px{ds:+d}"
            out.append(cand)

    if bt is not None or ml is not None or mg is not None:
        for dt in (-25, -10, 10, 25):
            cand = dict(base_params)
            if bt is not None:
                cand["binary_threshold"] = int(np.clip(round(bt + dt), 10, 250))
            if ml is not None:
                cand["hough_min_length"] = int(max(1, round(ml + dt)))
            if mg is not None:
                cand["hough_max_gap"] = int(max(1, round(mg + dt)))
            cand["__candidate_kind"] = f"hough{dt:+d}"
            out.append(cand)

    # dedupe
    seen: dict[str, dict[str, Any]] = {}
    for c in out:
        key_obj = {k: v for k, v in c.items() if k != "__candidate_kind"}
        key = json.dumps(key_obj, sort_keys=True)
        if key not in seen:
            seen[key] = c
    return list(seen.values())


# ---------------------------------------------------------------------------
# Per-ring iterative loop

def _evaluate_candidate(
    work_base: Path,
    work_ring: Path,
    detection_mod, segmentation_mod, evaluation_mod, pre_metrics_mod, det_metrics_mod,
    base_det: dict[str, Any],
    cand: dict[str, Any],
    tunnel: str,
    ring_id: int,
    segment_count: int,
) -> dict[str, Any]:
    cand_clean = {k: v for k, v in cand.items() if not k.startswith("__")}
    (work_ring / "parameters_detection.json").write_text(
        json.dumps(cand_clean, indent=2, sort_keys=True) + "\n"
    )
    out: dict[str, Any] = {"candidate_kind": cand.get("__candidate_kind", "unknown"), "params": cand_clean}
    try:
        detection_mod.run_detection(tunnel, ring_id, base_dir=str(work_base))
        segmentation_mod.run_segmentation(tunnel, ring_id, base_dir=str(work_base))
        det_m = det_metrics_mod.compute_detection_boundary_metrics(work_ring)
        pre_m = pre_metrics_mod.compute_target_guarded_metrics(work_ring)
        j, g = _guarded_j(det_m, pre_m, base_det)
        eval_res = evaluation_mod.evaluate(tunnel, ring_id, base_dir=str(work_base), segment_count=int(segment_count))
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        out.update({"J_reflect": None, "miou": None, "oa": None,
                    "G_pre": None, "G_layout": None, "G_stability": None,
                    "guardrail_pass": False, "S_boundary": None})
        return out
    out["J_reflect"] = float(j)
    out["S_boundary"] = _safe_float(det_m.get("S_boundary"))
    out["miou"] = _safe_float(eval_res.get("mIoU"))
    out["oa"] = _safe_float(eval_res.get("OA"))
    out.update(g)
    return out


def _run_one_ring(
    tunnel: str,
    ring_id: int,
    a0_dir: Path,
    *,
    max_rounds: int = 4,
    patience: int = 2,
    min_delta_proxy: float = 0.005,
    max_candidates_per_round: int = 18,
    segment_count: int = 7,
) -> dict[str, Any]:
    ring_key = f"{tunnel}/r{ring_id}"
    out_dir = ITER_ROOT / tunnel / f"r{ring_id}" / "A2_iterative_intrinsic_reflection"
    work_ring = WORK_ROOT / tunnel / f"r{ring_id}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if work_ring.exists():
        shutil.rmtree(work_ring)
    work_ring.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(a0_dir, work_ring)

    detection = _import_mod(f"_g_det_{tunnel}_{ring_id}", REPO_ROOT / "agents" / "2_detection" / "2_detection.py")
    segmentation = _import_mod(f"_g_seg_{tunnel}_{ring_id}", REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py")
    evaluation = _import_mod(f"_g_eval_{tunnel}_{ring_id}", REPO_ROOT / "agents" / "evaluation.py")
    pre_metrics = _import_mod(f"_g_prem_{tunnel}_{ring_id}", REPO_ROOT / "bo" / "preprocessing_iou_metrics.py")
    det_metrics = _import_mod(f"_g_detm_{tunnel}_{ring_id}", REPO_ROOT / "bo" / "detection_boundary_metrics.py")

    cur_params = json.loads((work_ring / "parameters_detection.json").read_text())

    # Baseline metrics
    base_pre = pre_metrics.compute_target_guarded_metrics(work_ring)
    base_det = det_metrics.compute_detection_boundary_metrics(work_ring)
    base_eval = evaluation.evaluate(tunnel, ring_id, base_dir=str(WORK_ROOT), segment_count=int(segment_count))
    base_j, base_g = _guarded_j(base_det, base_pre, base_det)

    best = {
        "candidate_kind": "baseline_A0",
        "params": dict(cur_params),
        "J_reflect": base_j,
        "miou": _safe_float(base_eval.get("mIoU")),
        "oa": _safe_float(base_eval.get("OA")),
        "S_boundary": _safe_float(base_det.get("S_boundary")),
        **base_g,
    }
    rounds_log = [{"round_id": 0, "selected": dict(best)}]

    no_improve = 0
    oracle_best = dict(best)
    rng = np.random.default_rng(seed=int(ring_id))

    for rid in range(1, max_rounds + 1):
        cand_pool = _candidate_params(best["params"])
        rng.shuffle(cand_pool)
        batch = cand_pool[: int(max_candidates_per_round)]

        round_results: list[dict[str, Any]] = []
        for cand in batch:
            res = _evaluate_candidate(
                WORK_ROOT, work_ring, detection, segmentation, evaluation,
                pre_metrics, det_metrics, base_det, cand, tunnel, ring_id, int(segment_count),
            )
            round_results.append(res)
            m_new = res.get("miou")
            m_cur = oracle_best.get("miou")
            if m_new is not None and (m_cur is None or float(m_new) > float(m_cur)):
                oracle_best = dict(res)

        passing = [r for r in round_results if r.get("guardrail_pass") and r.get("J_reflect") is not None]
        if passing:
            top = max(passing, key=lambda r: float(r["J_reflect"]))
            if float(top["J_reflect"]) >= float(best["J_reflect"]) + float(min_delta_proxy):
                best = dict(top)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        rounds_log.append({"round_id": rid, "round_results": round_results, "selected": dict(best)})
        if no_improve >= patience:
            break

    # Apply best params and re-run pipeline once
    best_clean = {k: v for k, v in best["params"].items() if not k.startswith("__")}
    (work_ring / "parameters_detection.json").write_text(
        json.dumps(best_clean, indent=2, sort_keys=True) + "\n"
    )
    detection.run_detection(tunnel, ring_id, base_dir=str(WORK_ROOT))
    segmentation.run_segmentation(tunnel, ring_id, base_dir=str(WORK_ROOT))
    final_eval = evaluation.evaluate(tunnel, ring_id, base_dir=str(WORK_ROOT), segment_count=int(segment_count))
    final_det = det_metrics.compute_detection_boundary_metrics(work_ring)
    final_pre = pre_metrics.compute_target_guarded_metrics(work_ring)
    final_j, final_g = _guarded_j(final_det, final_pre, base_det)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(work_ring, out_dir)
    if work_ring.exists():
        shutil.rmtree(work_ring)

    # Canonical mIoU
    mapping_path = CANONICAL_RELABEL_ROOT / f"{tunnel}.json"
    canon_a0 = canonical_miou_from_final_csv(a0_dir / "final.csv",
                                              rank_to_class=json.loads(mapping_path.read_text())["rank_to_class"]) if mapping_path.exists() else None
    canon_iter = canonical_miou_from_final_csv(out_dir / "final.csv",
                                                rank_to_class=json.loads(mapping_path.read_text())["rank_to_class"]) if mapping_path.exists() else None

    result = {
        "ring_key": ring_key,
        "tunnel": tunnel,
        "ring_id": ring_id,
        "segment_count": int(segment_count),
        "baseline_A0": {
            "J_reflect": base_j,
            "naive_mIoU": _safe_float(base_eval.get("mIoU")),
            "naive_OA": _safe_float(base_eval.get("OA")),
            "canonical_mIoU": canon_a0["canonical_mIoU"] if canon_a0 else None,
            **base_g,
        },
        "iterative_best": {
            "candidate_kind": best.get("candidate_kind"),
            "J_reflect": float(final_j),
            "naive_mIoU": _safe_float(final_eval.get("mIoU")),
            "naive_OA": _safe_float(final_eval.get("OA")),
            "canonical_mIoU": canon_iter["canonical_mIoU"] if canon_iter else None,
            **final_g,
        },
        "oracle_in_pool": {
            "candidate_kind": oracle_best.get("candidate_kind"),
            "naive_mIoU": _safe_float(oracle_best.get("miou")),
        },
        "rounds_explored": len(rounds_log) - 1,
        "best_params": best_clean,
        "output_dir": str(out_dir),
    }
    (out_dir / "gravity_iter_trace.json").write_text(json.dumps({**result, "rounds": rounds_log}, indent=2, sort_keys=True) + "\n")
    return result


# ---------------------------------------------------------------------------
# Driver

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rings", type=str, default=None,
                   help="csv of tunnel/ring (e.g. 4-3/r170,4-3/r171). Default: all heldout dirs.")
    p.add_argument("--max-rounds", type=int, default=4)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--min-delta-proxy", type=float, default=0.005)
    p.add_argument("--max-candidates", type=int, default=18)
    args = p.parse_args()

    ITER_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    # Discover rings
    if args.rings:
        wanted = [s.strip() for s in args.rings.split(",") if s.strip()]
        targets = []
        for w in wanted:
            tunnel, ring = w.split("/", 1)
            targets.append((tunnel, ring))
    else:
        targets = []
        for tunnel_dir in sorted(HELDOUT_GRAVITY_ROOT.iterdir()):
            if not tunnel_dir.is_dir():
                continue
            for ring_dir in sorted(tunnel_dir.iterdir()):
                if not ring_dir.is_dir():
                    continue
                if (ring_dir / "final.csv").exists():
                    targets.append((tunnel_dir.name, ring_dir.name))

    rows: list[dict[str, Any]] = []
    for tunnel, ring in targets:
        ring_id = int(ring.lstrip("r"))
        a0_dir = HELDOUT_GRAVITY_ROOT / tunnel / ring
        if not a0_dir.exists() or not (a0_dir / "final.csv").exists():
            print(f"SKIP {tunnel}/{ring}: missing gravity A0 baseline at {a0_dir}")
            continue
        try:
            t0 = time.time()
            segment_count = _infer_segment_count(a0_dir, default=7)
            res = _run_one_ring(
                tunnel=tunnel,
                ring_id=ring_id,
                a0_dir=a0_dir,
                max_rounds=args.max_rounds,
                patience=args.patience,
                min_delta_proxy=args.min_delta_proxy,
                max_candidates_per_round=args.max_candidates,
                segment_count=segment_count,
            )
            elapsed = time.time() - t0
            print(f"{tunnel}/{ring}: A0_canon={res['baseline_A0']['canonical_mIoU']} -> "
                  f"iter_canon={res['iterative_best']['canonical_mIoU']} "
                  f"oracle_naive={res['oracle_in_pool']['naive_mIoU']} "
                  f"rounds={res['rounds_explored']} ({elapsed:.1f}s)")
            rows.append({
                "ring": res["ring_key"],
                "tunnel": tunnel,
                "ring_id": ring_id,
                "A0_canon_mIoU": res["baseline_A0"]["canonical_mIoU"],
                "iter_canon_mIoU": res["iterative_best"]["canonical_mIoU"],
                "iter_naive_mIoU": res["iterative_best"]["naive_mIoU"],
                "oracle_naive_mIoU": res["oracle_in_pool"]["naive_mIoU"],
                "rounds_explored": res["rounds_explored"],
                "best_kind": res["iterative_best"]["candidate_kind"],
                "elapsed_sec": round(elapsed, 1),
            })
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {tunnel}/{ring}: {exc}")
            traceback.print_exc()
            rows.append({
                "ring": f"{tunnel}/{ring}",
                "error": str(exc),
            })

    df = pd.DataFrame(rows)
    df.to_csv(GRAVITY_ROOT / "iterative_summary.csv", index=False)
    if not df.empty and "A0_canon_mIoU" in df.columns and "iter_canon_mIoU" in df.columns:
        valid = df.dropna(subset=["A0_canon_mIoU", "iter_canon_mIoU"]).copy()
        valid["delta"] = valid["iter_canon_mIoU"] - valid["A0_canon_mIoU"]
        md = []
        md.append("# Iterative reflection on gravity-aligned data\n")
        md.append(f"Total rings: {len(df)}\n\n")
        md.append(f"**Mean A0 canon_mIoU**: {valid['A0_canon_mIoU'].mean():.3f}")
        md.append(f"\n**Mean iterative canon_mIoU**: {valid['iter_canon_mIoU'].mean():.3f}")
        md.append(f"\n**Δ canon_mIoU**: {valid['delta'].mean():+.3f}")
        md.append(f"\n**Mean oracle naive_mIoU**: {valid['oracle_naive_mIoU'].mean():.3f}")
        md.append("\n## Per-ring\n")
        md.append("| ring | A0 canon | iter canon | Δ | oracle naive | rounds | best_kind |")
        md.append("|------|---------|----------|---|---|---|---|")
        for _, r in valid.iterrows():
            on = r.get('oracle_naive_mIoU')
            on_s = f"{float(on):.3f}" if on is not None and np.isfinite(on) else "na"
            md.append(f"| {r['ring']} | {r['A0_canon_mIoU']:.3f} | {r['iter_canon_mIoU']:.3f} | {r['delta']:+.3f} | {on_s} | {int(r['rounds_explored'])} | {r['best_kind']} |")
        (GRAVITY_ROOT / "iterative_report.md").write_text("\n".join(md) + "\n")
        print(f"\nReport: {GRAVITY_ROOT / 'iterative_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
