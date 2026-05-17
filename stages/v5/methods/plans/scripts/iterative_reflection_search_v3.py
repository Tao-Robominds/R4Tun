#!/usr/bin/env python3
"""Iterative intrinsic reflection search v3 (tunnel 4/5 focus).

Key additions over v2 (which only perturbed scalar detection knobs):

  1) **Template-rotation candidates**: held-out 4/5 rings reuse a 7-block
     `single_ring_visual_slot_template` borrowed from a BO-calibrated ring
     of the same tunnel. Held-out unwraps have a different rotational
     offset, so the template lands in the wrong place. We add cyclic
     rotations of `y_frac` (and corresponding `block` permutations are
     unchanged) as new candidates. This is the highest-leverage axis and
     was missing in v2.

  2) **Multi-armed candidate batches per round**: each round generates a
     mix of (a) template rotations, (b) `min_score`/`snap_px` jitter,
     (c) Hough threshold jitter, all evaluated together; intrinsic
     `J_reflect` picks the winner. This avoids a sequential local search
     getting stuck on the wrong axis.

  3) **Per-ring failure-mode branching**: at round 0 we triage which
     guardrail is weakest (`G_pre`, `G_layout`, or `G_stability`) and
     bias the candidate batch toward the corresponding axis (preprocessing
     thresholds, layout rotations, or detection-head jitter).

We also record, for each ring, **two** statistics:

  - *intrinsic-best*: the candidate the intrinsic policy actually selects,
    using only `J_reflect` + guardrails (no GT, deployable at inference).
  - *mIoU-best (oracle)*: the candidate with the highest mIoU among the
    same batch, evaluated only for diagnosis. This is the upper bound of
    what the candidate space allows; if intrinsic-best lags far behind it,
    the limit is the ranker; if both are similar, the limit is the
    candidate space.

Outputs:
  logs/iterative_reflection_proof_v3/panel/r0/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "logs" / "iterative_reflection_proof_v3"
PANEL_ROOT = OUT_ROOT / "panel" / "r0"
RINGS_ROOT = OUT_ROOT / "heldout_iterative_reflection"
WORK_ROOT = OUT_ROOT / "_work"

PAIRS_STEP7 = REPO_ROOT / "logs" / "reflection_proof_v1" / "panel" / "r0" / "reflection_proof_pairs.csv"


# ----------------------------------------------------------------------------
# small utils
# ----------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------
# intrinsic objective (same shape as v2; rebalanced for layout-driven gains)
# ----------------------------------------------------------------------------


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
    return j, {
        "G_pre": g_pre,
        "G_layout": g_layout,
        "G_stability": g_stability,
        "guardrail_pass": guard_pass,
    }


# ----------------------------------------------------------------------------
# candidate generation (template-rotation aware)
# ----------------------------------------------------------------------------


def _rotate_template(template: list[dict[str, Any]], offset_frac: float) -> list[dict[str, Any]]:
    """Cyclically shift the y_fracs by ``offset_frac`` (mod 1), keep block order.

    This emulates the rotation of the unwrap relative to the template's
    original calibration ring. We keep the block sequence the same and shift
    every y_frac by the same amount mod 1, then re-sort so y_fracs ascend.
    The detector treats the layout as a circular sequence so the underlying
    block ordering is preserved either way.
    """

    if not template:
        return template
    out: list[dict[str, Any]] = []
    for entry in template:
        try:
            yf = float(entry.get("y_frac", 0.0))
        except (TypeError, ValueError):
            yf = 0.0
        new_yf = yf + float(offset_frac)
        new_yf = new_yf - np.floor(new_yf)
        e = dict(entry)
        e["y_frac"] = float(np.clip(new_yf, 0.0, 1.0 - 1e-6))
        out.append(e)
    out.sort(key=lambda e: float(e.get("y_frac", 0.0)))
    return out


def _candidate_params(
    base_params: dict[str, Any],
    *,
    weakest_axis: str,
    rotation_steps: int = 12,
    include_scalar: bool = True,
) -> list[dict[str, Any]]:
    """Generate candidate parameter dicts.

    weakest_axis : one of {"G_layout", "G_pre", "G_stability"}, decides which
                   axis gets denser sampling.
    """

    template = base_params.get("single_ring_visual_slot_template")
    has_tpl = isinstance(template, list) and len(template) >= 4

    out: list[dict[str, Any]] = []

    # 1) template rotations -- the highest-leverage knob for held-out 4/5
    if has_tpl:
        # primary rotations
        for k in range(rotation_steps):
            offset = k / float(rotation_steps)
            cand = dict(base_params)
            cand["single_ring_visual_slot_template"] = _rotate_template(template, offset)
            cand["__candidate_kind"] = f"rot{k}/{rotation_steps}"
            out.append(cand)
        # finer rotations near 0 if layout was weakest
        if weakest_axis == "G_layout":
            for fine in (-0.03, -0.015, 0.015, 0.03):
                cand = dict(base_params)
                cand["single_ring_visual_slot_template"] = _rotate_template(template, fine)
                cand["__candidate_kind"] = f"rot_fine{fine:+.3f}"
                out.append(cand)

    # 1b) ring_offset jitter (axial alignment between calibration ring and held-out)
    ring_off = _safe_float(base_params.get("ring_offset"))
    if ring_off is not None:
        for d in (-200.0, -100.0, -50.0, 50.0, 100.0, 200.0):
            cand = dict(base_params)
            cand["ring_offset"] = float(ring_off + d)
            cand["__candidate_kind"] = f"ring_off{d:+.0f}"
            out.append(cand)

    # 2) scalar detection-head jitter
    if include_scalar:
        ms = _safe_float(base_params.get("single_ring_visual_slot_min_score"))
        sp = _safe_float(base_params.get("single_ring_visual_slot_snap_px"))
        bt = _safe_float(base_params.get("binary_threshold"))
        ml = _safe_float(base_params.get("hough_min_length"))
        mg = _safe_float(base_params.get("hough_max_gap"))
        for sf in (0.6, 0.8, 1.2, 1.5):
            if ms is None:
                continue
            cand = dict(base_params)
            cand["single_ring_visual_slot_min_score"] = float(np.clip(ms * sf, 0.005, 0.95))
            cand["__candidate_kind"] = f"min_score*{sf}"
            out.append(cand)
        for ds in (-15, -8, 8, 15, 30):
            if sp is None:
                continue
            cand = dict(base_params)
            cand["single_ring_visual_slot_snap_px"] = int(max(1, round(sp + ds)))
            cand["__candidate_kind"] = f"snap_px{ds:+d}"
            out.append(cand)
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


def _weakest_axis(g: dict[str, Any]) -> str:
    options = {
        "G_layout": _safe_float(g.get("G_layout")) or 0.0,
        "G_pre": _safe_float(g.get("G_pre")) or 0.0,
        "G_stability": _safe_float(g.get("G_stability")) or 0.0,
    }
    return min(options, key=lambda k: options[k])


# ----------------------------------------------------------------------------
# per-ring iterative search
# ----------------------------------------------------------------------------


def _evaluate_candidate(
    *,
    work_base: Path,
    work_ring: Path,
    detection_mod,
    segmentation_mod,
    evaluation_mod,
    pre_metrics_mod,
    det_metrics_mod,
    base_det: dict[str, Any],
    cand: dict[str, Any],
    tunnel_id: str,
    ring_id: int,
) -> dict[str, Any]:
    cand_clean = {k: v for k, v in cand.items() if not k.startswith("__")}
    param_path = work_ring / "parameters_detection.json"
    param_path.write_text(json.dumps(cand_clean, indent=2, sort_keys=True) + "\n")
    out: dict[str, Any] = {
        "candidate_kind": cand.get("__candidate_kind", "unknown"),
        "params": cand_clean,
    }
    try:
        detection_mod.run_detection(tunnel_id, ring_id, base_dir=str(work_base))
        segmentation_mod.run_segmentation(tunnel_id, ring_id, base_dir=str(work_base))
        det_m = det_metrics_mod.compute_detection_boundary_metrics(work_ring)
        pre_m = pre_metrics_mod.compute_target_guarded_metrics(work_ring)
        j, g = _guarded_j(det_m, pre_m, base_det)
        eval_res = evaluation_mod.evaluate(tunnel_id, ring_id, base_dir=str(work_base))
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        out["J_reflect"] = None
        out["miou"] = None
        out["oa"] = None
        out["G_pre"] = None
        out["G_layout"] = None
        out["G_stability"] = None
        out["guardrail_pass"] = False
        out["S_boundary"] = None
        return out
    out["J_reflect"] = float(j)
    out["S_boundary"] = _safe_float(det_m.get("S_boundary"))
    out["miou"] = _safe_float(eval_res.get("mIoU"))
    out["oa"] = _safe_float(eval_res.get("OA"))
    out.update(g)
    return out


def _run_one_ring(
    *,
    tunnel_id: str,
    ring_id: int,
    a0_output_dir: Path,
    max_rounds: int,
    patience: int,
    min_delta_proxy: float,
    max_candidates_per_round: int,
    rotation_steps: int,
) -> dict[str, Any]:
    ring_key = f"{tunnel_id}/r{ring_id}"
    ring_root = RINGS_ROOT / tunnel_id / f"r{ring_id}" / "A2_iterative_intrinsic_reflection"
    work_ring = WORK_ROOT / tunnel_id / f"r{ring_id}"
    if ring_root.exists():
        shutil.rmtree(ring_root)
    if work_ring.exists():
        shutil.rmtree(work_ring)
    work_ring.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(a0_output_dir, work_ring)

    detection_mod = _import_mod(f"det_v3_{tunnel_id}_{ring_id}", REPO_ROOT / "agents" / "2_detection" / "2_detection.py")
    segmentation_mod = _import_mod(
        f"seg_v3_{tunnel_id}_{ring_id}", REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py"
    )
    evaluation_mod = _import_mod(f"eval_v3_{tunnel_id}_{ring_id}", REPO_ROOT / "agents" / "evaluation.py")
    pre_metrics_mod = _import_mod(
        f"prem_v3_{tunnel_id}_{ring_id}", REPO_ROOT / "bo" / "preprocessing_iou_metrics.py"
    )
    det_metrics_mod = _import_mod(
        f"detm_v3_{tunnel_id}_{ring_id}", REPO_ROOT / "bo" / "detection_boundary_metrics.py"
    )

    param_path = work_ring / "parameters_detection.json"
    cur_params = _load_json(param_path) if param_path.exists() else {}

    # Baseline (before any candidate -- this is A0).
    base_pre = pre_metrics_mod.compute_target_guarded_metrics(work_ring)
    base_det = det_metrics_mod.compute_detection_boundary_metrics(work_ring)
    base_eval = evaluation_mod.evaluate(tunnel_id, ring_id, base_dir=str(WORK_ROOT))
    base_j, base_g = _guarded_j(base_det, base_pre, base_det)
    weakest_axis = _weakest_axis(base_g)

    best = {
        "candidate_kind": "baseline_A0",
        "params": dict(cur_params),
        "J_reflect": base_j,
        "miou": _safe_float(base_eval.get("mIoU")),
        "oa": _safe_float(base_eval.get("OA")),
        "S_boundary": _safe_float(base_det.get("S_boundary")),
        **base_g,
    }
    rounds_log: list[dict[str, Any]] = [{"round_id": 0, "weakest_axis": weakest_axis, "selected": best.copy()}]

    no_improve = 0
    oracle_best = dict(best)

    rng = np.random.default_rng(seed=int(ring_id))
    cand_pool = _candidate_params(best["params"], weakest_axis=weakest_axis, rotation_steps=rotation_steps)
    # Always test all rotations first (they are the dominant new axis), then
    # the rest in shuffled order. This guarantees the rotation arm is fully
    # explored even when max_candidates_per_round is small.
    rotations = [c for c in cand_pool if str(c.get("__candidate_kind", "")).startswith("rot")]
    others = [c for c in cand_pool if not str(c.get("__candidate_kind", "")).startswith("rot")]
    rng.shuffle(others)
    cand_pool = rotations + others

    for rid in range(1, max_rounds + 1):
        round_results: list[dict[str, Any]] = []
        # build the per-round batch from best's pool
        # in subsequent rounds, regenerate around current best params (so we
        # explore around the new template if it shifted).
        if rid > 1:
            cand_pool = _candidate_params(
                best["params"], weakest_axis=weakest_axis, rotation_steps=rotation_steps
            )
            rotations = [c for c in cand_pool if str(c.get("__candidate_kind", "")).startswith("rot")]
            others = [c for c in cand_pool if not str(c.get("__candidate_kind", "")).startswith("rot")]
            rng.shuffle(others)
            cand_pool = rotations + others
        batch = cand_pool[: int(max_candidates_per_round)]
        for cand in batch:
            res = _evaluate_candidate(
                work_base=WORK_ROOT,
                work_ring=work_ring,
                detection_mod=detection_mod,
                segmentation_mod=segmentation_mod,
                evaluation_mod=evaluation_mod,
                pre_metrics_mod=pre_metrics_mod,
                det_metrics_mod=det_metrics_mod,
                base_det=base_det,
                cand=cand,
                tunnel_id=tunnel_id,
                ring_id=ring_id,
            )
            round_results.append(res)
            # update mIoU oracle (diagnostic only)
            m_cur = oracle_best.get("miou")
            m_new = res.get("miou")
            if m_new is not None and (m_cur is None or float(m_new) > float(m_cur)):
                oracle_best = dict(res)

        # intrinsic accept: best by J_reflect among guardrail_pass candidates,
        # only if it improves over the current best by min_delta_proxy.
        passing = [
            r for r in round_results if r.get("guardrail_pass") and r.get("J_reflect") is not None
        ]
        if passing:
            top = max(passing, key=lambda r: float(r["J_reflect"]))
            if float(top["J_reflect"]) >= float(best["J_reflect"]) + float(min_delta_proxy):
                best = dict(top)
                weakest_axis = _weakest_axis(best)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        rounds_log.append({"round_id": rid, "weakest_axis": weakest_axis, "round_results": round_results, "selected": best.copy()})
        if no_improve >= patience:
            break

    # Persist best parameter set and re-run pipeline with the chosen params so
    # the work_ring artifacts match `best`.
    best_clean = {k: v for k, v in best.get("params", {}).items() if not k.startswith("__")}
    param_path.write_text(json.dumps(best_clean, indent=2, sort_keys=True) + "\n")
    detection_mod.run_detection(tunnel_id, ring_id, base_dir=str(WORK_ROOT))
    segmentation_mod.run_segmentation(tunnel_id, ring_id, base_dir=str(WORK_ROOT))
    final_eval = evaluation_mod.evaluate(tunnel_id, ring_id, base_dir=str(WORK_ROOT))
    final_det = det_metrics_mod.compute_detection_boundary_metrics(work_ring)
    final_pre = pre_metrics_mod.compute_target_guarded_metrics(work_ring)
    final_j, final_g = _guarded_j(final_det, final_pre, base_det)

    # Move work to ring_root for inspection.
    ring_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(work_ring, ring_root)
    if work_ring.exists():
        shutil.rmtree(work_ring)

    out = {
        "ring_key": ring_key,
        "tunnel_id": tunnel_id,
        "ring_id": ring_id,
        "weakest_axis_at_baseline": _weakest_axis(base_g),
        "baseline_A0": {
            "J_reflect": base_j,
            "miou": _safe_float(base_eval.get("mIoU")),
            "oa": _safe_float(base_eval.get("OA")),
            **base_g,
        },
        "intrinsic_best": {
            "candidate_kind": best.get("candidate_kind"),
            "J_reflect": float(final_j),
            "miou": _safe_float(final_eval.get("mIoU")),
            "oa": _safe_float(final_eval.get("OA")),
            **final_g,
        },
        "oracle_mIoU_best_in_pool": {
            "candidate_kind": oracle_best.get("candidate_kind"),
            "J_reflect": _safe_float(oracle_best.get("J_reflect")),
            "miou": _safe_float(oracle_best.get("miou")),
        },
        "rounds": rounds_log,
        "best_params": best_clean,
        "output_dir": str(ring_root),
    }
    _write_json(ring_root / "iterative_trace_v3.json", out)
    return out


# ----------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------


def _main(args: argparse.Namespace) -> int:
    PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    RINGS_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(PAIRS_STEP7)
    a1 = pairs[pairs["variant"] == "A1_proxy_reflection"].copy()
    target = a1[a1["tunnel_id"].astype(str).str.startswith(("4-", "5-"))].copy().reset_index(drop=True)
    if args.only_rings:
        wanted = set(s.strip() for s in args.only_rings.split(",") if s.strip())
        target = target[target.apply(lambda r: f"{r['tunnel_id']}/r{int(r['ring_id'])}" in wanted, axis=1)].reset_index(drop=True)
    if args.max_rings is not None:
        target = target.head(int(args.max_rings)).copy().reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for _, row in target.iterrows():
        t = str(row["tunnel_id"])
        r = int(row["ring_id"])
        a0_dir = Path(str(row["A0_output_dir"]))
        try:
            result = _run_one_ring(
                tunnel_id=t,
                ring_id=r,
                a0_output_dir=a0_dir,
                max_rounds=int(args.max_rounds),
                patience=int(args.patience),
                min_delta_proxy=float(args.min_delta_proxy),
                max_candidates_per_round=int(args.max_candidates),
                rotation_steps=int(args.rotation_steps),
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "ring_key": f"{t}/r{r}",
                    "tunnel_id": t,
                    "ring_id": r,
                    "error": str(exc),
                    "mIoU_no_reflection": _safe_float(row["mIoU_no_reflection"]),
                    "mIoU_A1_single_pass": _safe_float(row["mIoU_reflection"]),
                    "mIoU_A2_v3_intrinsic": None,
                    "mIoU_oracle_best": None,
                    "delta_mIoU_v3_vs_A0": None,
                    "delta_mIoU_oracle_vs_A0": None,
                    "weakest_axis_at_baseline": None,
                    "intrinsic_best_kind": None,
                }
            )
            continue
        ib = result["intrinsic_best"]
        ob = result["oracle_mIoU_best_in_pool"]
        rows.append(
            {
                "ring_key": result["ring_key"],
                "tunnel_id": t,
                "ring_id": r,
                "mIoU_no_reflection": _safe_float(row["mIoU_no_reflection"]),
                "mIoU_A1_single_pass": _safe_float(row["mIoU_reflection"]),
                "mIoU_A2_v3_intrinsic": ib.get("miou"),
                "mIoU_oracle_best": ob.get("miou"),
                "delta_mIoU_v3_vs_A0": None
                if _safe_float(row["mIoU_no_reflection"]) is None or ib.get("miou") is None
                else float(ib["miou"] - float(row["mIoU_no_reflection"])),
                "delta_mIoU_oracle_vs_A0": None
                if _safe_float(row["mIoU_no_reflection"]) is None or ob.get("miou") is None
                else float(ob["miou"] - float(row["mIoU_no_reflection"])),
                "weakest_axis_at_baseline": result.get("weakest_axis_at_baseline"),
                "intrinsic_best_kind": ib.get("candidate_kind"),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(PANEL_ROOT / "t45_iterative_v3_results.csv", index=False)

    valid = out_df.dropna(subset=["mIoU_A2_v3_intrinsic", "mIoU_no_reflection"]).copy()
    if not valid.empty:
        dmiou = valid["mIoU_A2_v3_intrinsic"] - valid["mIoU_no_reflection"]
        try:
            t_p = _safe_float(ttest_rel(valid["mIoU_A2_v3_intrinsic"], valid["mIoU_no_reflection"]).pvalue) if len(valid) >= 2 else None
        except Exception:  # noqa: BLE001
            t_p = None
        try:
            w_p = _safe_float(wilcoxon(dmiou.to_numpy(dtype=float)).pvalue) if len(valid) >= 2 else None
        except Exception:  # noqa: BLE001
            w_p = None
        oracle_valid = out_df.dropna(subset=["mIoU_oracle_best", "mIoU_no_reflection"])
        share_intrinsic_ge_04 = float((valid["mIoU_A2_v3_intrinsic"] >= 0.4).mean())
        share_intrinsic_ge_05 = float((valid["mIoU_A2_v3_intrinsic"] >= 0.5).mean())
        share_oracle_ge_04 = float((oracle_valid["mIoU_oracle_best"] >= 0.4).mean()) if not oracle_valid.empty else None
        share_oracle_ge_05 = float((oracle_valid["mIoU_oracle_best"] >= 0.5).mean()) if not oracle_valid.empty else None
    else:
        t_p = None
        w_p = None
        share_intrinsic_ge_04 = None
        share_intrinsic_ge_05 = None
        share_oracle_ge_04 = None
        share_oracle_ge_05 = None

    summary = {
        "timestamp_utc": _now(),
        "focus": "tunnel prefix 4 and 5 (held-out)",
        "n_rows_total": int(len(out_df)),
        "n_rows_evaluated": int(len(valid)),
        "max_rounds": int(args.max_rounds),
        "patience": int(args.patience),
        "min_delta_proxy": float(args.min_delta_proxy),
        "max_candidates_per_round": int(args.max_candidates),
        "rotation_steps": int(args.rotation_steps),
        "mean_mIoU_A0": _safe_float(valid["mIoU_no_reflection"].mean()) if not valid.empty else None,
        "mean_mIoU_A1": _safe_float(valid["mIoU_A1_single_pass"].mean()) if not valid.empty else None,
        "mean_mIoU_v3_intrinsic": _safe_float(valid["mIoU_A2_v3_intrinsic"].mean()) if not valid.empty else None,
        "mean_mIoU_v3_oracle_best": _safe_float(out_df["mIoU_oracle_best"].dropna().mean()) if not out_df.empty else None,
        "median_mIoU_v3_intrinsic": _safe_float(valid["mIoU_A2_v3_intrinsic"].median()) if not valid.empty else None,
        "share_intrinsic_ge_04": share_intrinsic_ge_04,
        "share_intrinsic_ge_05": share_intrinsic_ge_05,
        "share_oracle_ge_04": share_oracle_ge_04,
        "share_oracle_ge_05": share_oracle_ge_05,
        "mean_delta_mIoU_v3_vs_A0": _safe_float((valid["mIoU_A2_v3_intrinsic"] - valid["mIoU_no_reflection"]).mean()) if not valid.empty else None,
        "paired_ttest_p_mIoU": t_p,
        "wilcoxon_p_mIoU": w_p,
    }
    _write_json(PANEL_ROOT / "t45_iterative_v3_summary.json", summary)

    report = [
        "# Tunnel 4/5 Iterative Reflection Report (v3)",
        "",
        "## Setup",
        "",
        f"- Held-out subset: rings with tunnel prefix `4-`/`5-` ({summary['n_rows_total']} rings).",
        f"- Per-round candidate budget: `{summary['max_candidates_per_round']}` mixed (template rotations + scalar jitter).",
        f"- Rotation grid: `{summary['rotation_steps']}` cyclic offsets of the visual-layout template.",
        f"- Rounds <= `{summary['max_rounds']}`, patience `{summary['patience']}`, min_delta_proxy `{summary['min_delta_proxy']}`.",
        "",
        "## Aggregate mIoU",
        "",
        f"- mean mIoU A0: `{summary['mean_mIoU_A0']}`",
        f"- mean mIoU A1 (single-pass): `{summary['mean_mIoU_A1']}`",
        f"- mean mIoU v3 (intrinsic-best): `{summary['mean_mIoU_v3_intrinsic']}`",
        f"- mean mIoU v3 (oracle mIoU-best in same candidate pool, **diagnostic only**): `{summary['mean_mIoU_v3_oracle_best']}`",
        f"- median mIoU v3 intrinsic: `{summary['median_mIoU_v3_intrinsic']}`",
        f"- share intrinsic >= 0.4: `{summary['share_intrinsic_ge_04']}`",
        f"- share intrinsic >= 0.5: `{summary['share_intrinsic_ge_05']}`",
        f"- share oracle >= 0.4: `{summary['share_oracle_ge_04']}`",
        f"- share oracle >= 0.5: `{summary['share_oracle_ge_05']}`",
        f"- mean delta mIoU v3 vs A0: `{summary['mean_delta_mIoU_v3_vs_A0']}`",
        f"- paired t-test p (v3 vs A0): `{summary['paired_ttest_p_mIoU']}`",
        f"- Wilcoxon p (v3 vs A0): `{summary['wilcoxon_p_mIoU']}`",
        "",
        "## Per-ring",
        "",
        "| ring_key | weakest_axis@A0 | mIoU A0 | mIoU A1 | mIoU v3 intrinsic | mIoU oracle-best | delta v3 vs A0 | delta oracle vs A0 | intrinsic_best_kind |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in out_df.iterrows():
        report.append(
            "| {ring_key} | {axis} | {a0:.4f} | {a1:.4f} | {v3:s} | {oracle:s} | {dv:s} | {do:s} | {kind} |".format(
                ring_key=r["ring_key"],
                axis=r.get("weakest_axis_at_baseline") or "?",
                a0=float(r["mIoU_no_reflection"]) if r["mIoU_no_reflection"] is not None and not pd.isna(r["mIoU_no_reflection"]) else float("nan"),
                a1=float(r["mIoU_A1_single_pass"]) if r["mIoU_A1_single_pass"] is not None and not pd.isna(r["mIoU_A1_single_pass"]) else float("nan"),
                v3=("{:.4f}".format(r["mIoU_A2_v3_intrinsic"]) if pd.notna(r["mIoU_A2_v3_intrinsic"]) else "nan"),
                oracle=("{:.4f}".format(r["mIoU_oracle_best"]) if pd.notna(r["mIoU_oracle_best"]) else "nan"),
                dv=("{:+.4f}".format(r["delta_mIoU_v3_vs_A0"]) if pd.notna(r["delta_mIoU_v3_vs_A0"]) else "nan"),
                do=("{:+.4f}".format(r["delta_mIoU_oracle_vs_A0"]) if pd.notna(r["delta_mIoU_oracle_vs_A0"]) else "nan"),
                kind=r.get("intrinsic_best_kind") or "-",
            )
        )
    (PANEL_ROOT / "t45_iterative_v3_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--min-delta-proxy", type=float, default=1e-6)
    p.add_argument("--max-candidates", type=int, default=14, help="candidates per round")
    p.add_argument("--rotation-steps", type=int, default=12)
    p.add_argument("--max-rings", type=int, default=None)
    p.add_argument("--only-rings", type=str, default=None, help="csv of ring_keys e.g. '4-3/r170,4-4/r212'")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_main(parse_args()))
