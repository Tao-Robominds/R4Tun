#!/usr/bin/env python3
"""Iterative intrinsic reflection search v4 (tunnel 4/5 focus).

Adds the structural-alignment intrinsic ``G_structural`` from
``bo/structural_alignment_metrics.py`` into the objective:

    J_reflect_v4 = S_boundary * G_pre * G_layout * G_stability
                   * (floor + (1 - floor) * G_structural)

with ``floor = 0.05`` so candidates that fail the structural check still
have some signal but are heavily down-weighted compared with structurally
clean candidates. The structural metric is computed on the candidate's
``detection/labelmap.npy``, which is saved per candidate during the
search (we copy ``labelmap.npy`` and ``labelmap_meta.json`` to a per-
candidate cache so we can also re-rank later).

We otherwise reuse the same candidate axes as v3:

  - 12 cyclic offsets of the visual-layout template
  - 4 fine offsets near current best
  - 6 ``ring_offset`` perturbations
  - scalar jitter on detection-head knobs

Outputs
-------
``logs/iterative_reflection_proof_v4/panel/r0/``
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
OUT_ROOT = REPO_ROOT / "logs" / "iterative_reflection_proof_v4"
PANEL_ROOT = OUT_ROOT / "panel" / "r0"
RINGS_ROOT = OUT_ROOT / "heldout_iterative_reflection"
WORK_ROOT = OUT_ROOT / "_work"
CAND_LABELMAP_ROOT = OUT_ROOT / "candidate_labelmaps"

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


# ----------------------------------------------------------------------------
# objective
# ----------------------------------------------------------------------------


STRUCT_FLOOR = 0.05


def _guarded_j(
    det: dict[str, Any],
    pre: dict[str, Any],
    base_det: dict[str, Any],
    structural: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
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
    g_struct = _safe_float((structural or {}).get("G_structural")) or 0.0
    struct_factor = float(STRUCT_FLOOR + (1.0 - STRUCT_FLOOR) * float(np.clip(g_struct, 0.0, 1.0)))
    guard_pass = bool(g_pre >= 0.25 and g_layout >= 0.05 and g_stability >= 0.2)
    j = float(s_boundary * g_pre * g_layout * g_stability * struct_factor)
    return j, {
        "G_pre": g_pre,
        "G_layout": g_layout,
        "G_stability": g_stability,
        "G_structural": float(g_struct),
        "struct_factor": struct_factor,
        "guardrail_pass": guard_pass,
    }


# ----------------------------------------------------------------------------
# candidate generation (same as v3)
# ----------------------------------------------------------------------------


def _rotate_template(template: list[dict[str, Any]], offset_frac: float) -> list[dict[str, Any]]:
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
) -> list[dict[str, Any]]:
    template = base_params.get("single_ring_visual_slot_template")
    has_tpl = isinstance(template, list) and len(template) >= 4

    out: list[dict[str, Any]] = []
    if has_tpl:
        for k in range(rotation_steps):
            offset = k / float(rotation_steps)
            cand = dict(base_params)
            cand["single_ring_visual_slot_template"] = _rotate_template(template, offset)
            cand["__candidate_kind"] = f"rot{k}/{rotation_steps}"
            out.append(cand)
        if weakest_axis in ("G_layout", "G_structural"):
            for fine in (-0.03, -0.015, 0.015, 0.03):
                cand = dict(base_params)
                cand["single_ring_visual_slot_template"] = _rotate_template(template, fine)
                cand["__candidate_kind"] = f"rot_fine{fine:+.3f}"
                out.append(cand)

    ring_off = _safe_float(base_params.get("ring_offset"))
    if ring_off is not None:
        for d in (-200.0, -100.0, -50.0, 50.0, 100.0, 200.0):
            cand = dict(base_params)
            cand["ring_offset"] = float(ring_off + d)
            cand["__candidate_kind"] = f"ring_off{d:+.0f}"
            out.append(cand)

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
        "G_structural": _safe_float(g.get("G_structural")) or 0.0,
    }
    return min(options, key=lambda k: options[k])


# ----------------------------------------------------------------------------
# per-ring search
# ----------------------------------------------------------------------------


def _evaluate_candidate(
    *,
    work_base: Path,
    work_ring: Path,
    cand_cache_dir: Path,
    detection_mod,
    segmentation_mod,
    evaluation_mod,
    pre_metrics_mod,
    det_metrics_mod,
    structural_mod,
    base_det: dict[str, Any],
    cand: dict[str, Any],
    cand_idx: int,
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
        struct_m = structural_mod.compute_structural_alignment(work_ring)
        j, g = _guarded_j(det_m, pre_m, base_det, structural=struct_m)
        eval_res = evaluation_mod.evaluate(tunnel_id, ring_id, base_dir=str(work_base))
        # cache labelmap snapshot for this candidate
        labelmap_src = work_ring / "detection" / "labelmap.npy"
        labelmap_meta_src = work_ring / "detection" / "labelmap_meta.json"
        if labelmap_src.exists():
            cand_dir = cand_cache_dir / f"cand_{cand_idx:04d}"
            cand_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(labelmap_src, cand_dir / "labelmap.npy")
            if labelmap_meta_src.exists():
                shutil.copy2(labelmap_meta_src, cand_dir / "labelmap_meta.json")
            (cand_dir / "params.json").write_text(json.dumps(cand_clean, indent=2, sort_keys=True))
            (cand_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "candidate_kind": out["candidate_kind"],
                        "miou": _safe_float(eval_res.get("mIoU")),
                        "oa": _safe_float(eval_res.get("OA")),
                        "J_reflect": float(j),
                        "S_boundary": _safe_float(det_m.get("S_boundary")),
                        "structural": struct_m,
                        **g,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        out["J_reflect"] = None
        out["miou"] = None
        out["oa"] = None
        out["G_structural"] = None
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
    out["structural"] = {
        "G_structural": _safe_float(struct_m.get("G_structural")),
        "block_count_match": _safe_float(struct_m.get("block_count_match")),
        "k_height_match": _safe_float(struct_m.get("k_height_match")),
        "k_uniqueness": _safe_float(struct_m.get("k_uniqueness")),
        "k_centrality": _safe_float(struct_m.get("k_centrality")),
        "block_height_uniformity": _safe_float(struct_m.get("block_height_uniformity")),
        "per_label_fragmentation": _safe_float(struct_m.get("per_label_fragmentation")),
    }
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
    cand_cache_dir = CAND_LABELMAP_ROOT / tunnel_id / f"r{ring_id}"
    if ring_root.exists():
        shutil.rmtree(ring_root)
    if work_ring.exists():
        shutil.rmtree(work_ring)
    if cand_cache_dir.exists():
        shutil.rmtree(cand_cache_dir)
    work_ring.parent.mkdir(parents=True, exist_ok=True)
    cand_cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(a0_output_dir, work_ring)

    detection_mod = _import_mod(
        f"det_v4_{tunnel_id}_{ring_id}", REPO_ROOT / "agents" / "2_detection" / "2_detection.py"
    )
    segmentation_mod = _import_mod(
        f"seg_v4_{tunnel_id}_{ring_id}", REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py"
    )
    evaluation_mod = _import_mod(f"eval_v4_{tunnel_id}_{ring_id}", REPO_ROOT / "agents" / "evaluation.py")
    pre_metrics_mod = _import_mod(
        f"prem_v4_{tunnel_id}_{ring_id}", REPO_ROOT / "bo" / "preprocessing_iou_metrics.py"
    )
    det_metrics_mod = _import_mod(
        f"detm_v4_{tunnel_id}_{ring_id}", REPO_ROOT / "bo" / "detection_boundary_metrics.py"
    )
    structural_mod = _import_mod(
        f"struct_v4_{tunnel_id}_{ring_id}", REPO_ROOT / "bo" / "structural_alignment_metrics.py"
    )

    param_path = work_ring / "parameters_detection.json"
    cur_params = _load_json(param_path) if param_path.exists() else {}

    base_pre = pre_metrics_mod.compute_target_guarded_metrics(work_ring)
    base_det = det_metrics_mod.compute_detection_boundary_metrics(work_ring)
    base_struct = structural_mod.compute_structural_alignment(work_ring)
    base_eval = evaluation_mod.evaluate(tunnel_id, ring_id, base_dir=str(WORK_ROOT))
    base_j, base_g = _guarded_j(base_det, base_pre, base_det, structural=base_struct)
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
    rounds_log: list[dict[str, Any]] = [{"round_id": 0, "weakest_axis": weakest_axis, "selected": dict(best)}]

    no_improve = 0
    oracle_best = dict(best)
    oracle_best["miou"] = _safe_float(base_eval.get("mIoU"))

    rng = np.random.default_rng(seed=int(ring_id))
    cand_idx = 0
    for rid in range(1, max_rounds + 1):
        cand_pool = _candidate_params(best["params"], weakest_axis=weakest_axis, rotation_steps=rotation_steps)
        rotations = [c for c in cand_pool if str(c.get("__candidate_kind", "")).startswith("rot")]
        others = [c for c in cand_pool if not str(c.get("__candidate_kind", "")).startswith("rot")]
        rng.shuffle(others)
        cand_pool = rotations + others
        batch = cand_pool[: int(max_candidates_per_round)]

        round_results: list[dict[str, Any]] = []
        for cand in batch:
            cand_idx += 1
            res = _evaluate_candidate(
                work_base=WORK_ROOT,
                work_ring=work_ring,
                cand_cache_dir=cand_cache_dir,
                detection_mod=detection_mod,
                segmentation_mod=segmentation_mod,
                evaluation_mod=evaluation_mod,
                pre_metrics_mod=pre_metrics_mod,
                det_metrics_mod=det_metrics_mod,
                structural_mod=structural_mod,
                base_det=base_det,
                cand=cand,
                cand_idx=cand_idx,
                tunnel_id=tunnel_id,
                ring_id=ring_id,
            )
            round_results.append(res)
            m_cur = oracle_best.get("miou")
            m_new = res.get("miou")
            if m_new is not None and (m_cur is None or float(m_new) > float(m_cur)):
                oracle_best = dict(res)

        passing = [r for r in round_results if r.get("guardrail_pass") and r.get("J_reflect") is not None]
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

        rounds_log.append(
            {
                "round_id": rid,
                "weakest_axis": weakest_axis,
                "round_results": round_results,
                "selected": dict(best),
            }
        )
        if no_improve >= patience:
            break

    # Re-run final pipeline with best params and persist artifacts.
    best_clean = {k: v for k, v in best.get("params", {}).items() if not k.startswith("__")}
    param_path.write_text(json.dumps(best_clean, indent=2, sort_keys=True) + "\n")
    detection_mod.run_detection(tunnel_id, ring_id, base_dir=str(WORK_ROOT))
    segmentation_mod.run_segmentation(tunnel_id, ring_id, base_dir=str(WORK_ROOT))
    final_eval = evaluation_mod.evaluate(tunnel_id, ring_id, base_dir=str(WORK_ROOT))
    final_det = det_metrics_mod.compute_detection_boundary_metrics(work_ring)
    final_pre = pre_metrics_mod.compute_target_guarded_metrics(work_ring)
    final_struct = structural_mod.compute_structural_alignment(work_ring)
    final_j, final_g = _guarded_j(final_det, final_pre, base_det, structural=final_struct)
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
    _write_json(ring_root / "iterative_trace_v4.json", out)
    return out


# ----------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------


def _main(args: argparse.Namespace) -> int:
    PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    RINGS_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    CAND_LABELMAP_ROOT.mkdir(parents=True, exist_ok=True)

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
                    "mIoU_v4_intrinsic": None,
                    "mIoU_oracle_best": None,
                    "delta_mIoU_v4_vs_A0": None,
                    "delta_mIoU_oracle_vs_A0": None,
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
                "mIoU_v4_intrinsic": ib.get("miou"),
                "mIoU_oracle_best": ob.get("miou"),
                "delta_mIoU_v4_vs_A0": None
                if _safe_float(row["mIoU_no_reflection"]) is None or ib.get("miou") is None
                else float(ib["miou"] - float(row["mIoU_no_reflection"])),
                "delta_mIoU_oracle_vs_A0": None
                if _safe_float(row["mIoU_no_reflection"]) is None or ob.get("miou") is None
                else float(ob["miou"] - float(row["mIoU_no_reflection"])),
                "weakest_axis_at_baseline": result.get("weakest_axis_at_baseline"),
                "intrinsic_best_kind": ib.get("candidate_kind"),
                "G_structural_at_winner": _safe_float(ib.get("G_structural")),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(PANEL_ROOT / "t45_iterative_v4_results.csv", index=False)

    valid = out_df.dropna(subset=["mIoU_v4_intrinsic", "mIoU_no_reflection"]).copy()
    if not valid.empty:
        d = valid["mIoU_v4_intrinsic"] - valid["mIoU_no_reflection"]
        try:
            t_p = _safe_float(ttest_rel(valid["mIoU_v4_intrinsic"], valid["mIoU_no_reflection"]).pvalue) if len(valid) >= 2 else None
        except Exception:  # noqa: BLE001
            t_p = None
        try:
            w_p = _safe_float(wilcoxon(d.to_numpy(dtype=float)).pvalue) if len(valid) >= 2 else None
        except Exception:  # noqa: BLE001
            w_p = None
    else:
        t_p = None
        w_p = None

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
        "structural_floor": STRUCT_FLOOR,
        "mean_mIoU_A0": _safe_float(valid["mIoU_no_reflection"].mean()) if not valid.empty else None,
        "mean_mIoU_A1": _safe_float(valid["mIoU_A1_single_pass"].mean()) if not valid.empty else None,
        "mean_mIoU_v4_intrinsic": _safe_float(valid["mIoU_v4_intrinsic"].mean()) if not valid.empty else None,
        "mean_mIoU_v4_oracle_best": _safe_float(out_df["mIoU_oracle_best"].dropna().mean()) if not out_df.empty else None,
        "median_mIoU_v4_intrinsic": _safe_float(valid["mIoU_v4_intrinsic"].median()) if not valid.empty else None,
        "share_v4_ge_04": _safe_float((valid["mIoU_v4_intrinsic"] >= 0.4).mean()) if not valid.empty else None,
        "share_v4_ge_05": _safe_float((valid["mIoU_v4_intrinsic"] >= 0.5).mean()) if not valid.empty else None,
        "share_oracle_ge_04": _safe_float((out_df["mIoU_oracle_best"] >= 0.4).mean()) if not out_df["mIoU_oracle_best"].dropna().empty else None,
        "mean_delta_mIoU_v4_vs_A0": _safe_float((valid["mIoU_v4_intrinsic"] - valid["mIoU_no_reflection"]).mean()) if not valid.empty else None,
        "paired_ttest_p_mIoU": t_p,
        "wilcoxon_p_mIoU": w_p,
    }
    _write_json(PANEL_ROOT / "t45_iterative_v4_summary.json", summary)

    report = [
        "# Tunnel 4/5 Iterative Reflection Report (v4 with G_structural)",
        "",
        f"- structural floor: `{STRUCT_FLOOR}`",
        f"- per-round candidate budget: `{summary['max_candidates_per_round']}`",
        f"- rotation grid: `{summary['rotation_steps']}`, rounds <= `{summary['max_rounds']}`",
        "",
        "## Aggregate mIoU",
        "",
        f"- mean mIoU A0: `{summary['mean_mIoU_A0']}`",
        f"- mean mIoU A1 (single-pass): `{summary['mean_mIoU_A1']}`",
        f"- mean mIoU v4 (intrinsic-best with G_structural): `{summary['mean_mIoU_v4_intrinsic']}`",
        f"- mean mIoU v4 oracle (mIoU-best in same pool, **diagnostic**): `{summary['mean_mIoU_v4_oracle_best']}`",
        f"- median v4: `{summary['median_mIoU_v4_intrinsic']}`",
        f"- share v4 >= 0.4: `{summary['share_v4_ge_04']}`",
        f"- share v4 >= 0.5: `{summary['share_v4_ge_05']}`",
        f"- share oracle >= 0.4: `{summary['share_oracle_ge_04']}`",
        f"- mean delta v4 vs A0: `{summary['mean_delta_mIoU_v4_vs_A0']}`",
        f"- paired t-test p (v4 vs A0): `{summary['paired_ttest_p_mIoU']}`",
        f"- Wilcoxon p (v4 vs A0): `{summary['wilcoxon_p_mIoU']}`",
        "",
        "## Per-ring",
        "",
        "| ring_key | weakest@A0 | A0 | A1 | v4 intrinsic | oracle | G_struct@winner | v4 kind |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in out_df.iterrows():
        report.append(
            "| {rk} | {axis} | {a0:s} | {a1:s} | {v4:s} | {orc:s} | {gst:s} | {kind} |".format(
                rk=r["ring_key"],
                axis=r.get("weakest_axis_at_baseline") or "?",
                a0=("{:.4f}".format(r["mIoU_no_reflection"]) if pd.notna(r.get("mIoU_no_reflection")) else "nan"),
                a1=("{:.4f}".format(r["mIoU_A1_single_pass"]) if pd.notna(r.get("mIoU_A1_single_pass")) else "nan"),
                v4=("{:.4f}".format(r["mIoU_v4_intrinsic"]) if pd.notna(r.get("mIoU_v4_intrinsic")) else "nan"),
                orc=("{:.4f}".format(r["mIoU_oracle_best"]) if pd.notna(r.get("mIoU_oracle_best")) else "nan"),
                gst=("{:.4f}".format(r["G_structural_at_winner"]) if pd.notna(r.get("G_structural_at_winner")) else "nan"),
                kind=r.get("intrinsic_best_kind") or "-",
            )
        )
    (PANEL_ROOT / "t45_iterative_v4_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-rounds", type=int, default=2)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--min-delta-proxy", type=float, default=1e-9)
    p.add_argument("--max-candidates", type=int, default=18)
    p.add_argument("--rotation-steps", type=int, default=12)
    p.add_argument("--max-rings", type=int, default=None)
    p.add_argument("--only-rings", type=str, default=None, help="csv of ring_keys, e.g. '4-3/r170,5-2/r144'")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(_main(parse_args()))
