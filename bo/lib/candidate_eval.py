"""Evaluate one Stage A candidate through det+seg and extract proxy features."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.ceiling_gate import REPO_ROOT, setup_sandbox
from lib.guardrail_utils import apply_guardrail_fields
from lib.feature_catalog import PRE7_FEATURES, SEG_REPLAY_FEATURES
from lib.layout_bo import (
    EXCLUDED_TRIAL_METRICS,
    RingContext,
    build_ring_context,
    decode_x,
    evaluate_trial,
)
from lib.direction_select import DIRECTION_META, SELECTION_FILENAME, direction_hypotheses_available

# Logged on every held-out / Stage-A candidate eval (intrinsic branch scorer).
DIRECTION_PROXY_FIELDS = (
    "order_branch",
    "branch_is_minus",
    "direction_select_enabled",
    "direction_score_plus",
    "direction_score_minus",
    "direction_margin",
    "template_margin_minus_plus",
    "template_match_score_plus",
    "template_match_score_minus",
)
from lib.pre_depth_qa import load_depth_3a
from lib.v5_proxy_features import compute_v5_proxy_features, prefix_features


def _import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_pre_extract():
    path = REPO_ROOT / "agents" / "1_preprocessing" / "scripts" / "extract_intrinsics.py"
    return _import_module(path, "extract_pre").extract_preprocessing_metrics


def _import_seg_extract():
    path = REPO_ROOT / "agents" / "3_segmentation" / "scripts" / "extract_intrinsics.py"
    return _import_module(path, "extract_seg").extract_segmentation_metrics


def ring_pre7_features(src_ring: Path) -> dict[str, Any]:
    d3a = load_depth_3a(src_ring)
    extract_pre = _import_pre_extract()
    pre = extract_pre(str(src_ring))
    out = dict(d3a)
    for k in PRE7_FEATURES:
        if k in pre:
            out[k] = pre[k]
        elif k in d3a:
            out[k] = d3a[k]
    return out


def extract_seg_metrics(sandbox_ring: Path) -> dict[str, Any]:
    extract_seg = _import_seg_extract()
    raw = extract_seg(str(sandbox_ring))
    out: dict[str, Any] = {}
    for k in SEG_REPLAY_FEATURES:
        if k not in raw:
            continue
        v = raw[k]
        if k == "seg_ready_for_evaluation":
            out[k] = int(bool(v))
        elif isinstance(v, bool):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def metrics_to_proxy_row(metrics: dict[str, Any], pre7: dict[str, Any]) -> dict[str, Any]:
    """Map trial metrics + ring PRE7 + seg/v5 into proxy feature columns."""
    row: dict[str, Any] = {}
    for k, v in pre7.items():
        row[f"feat_pre_{k}"] = v
    row["feat_intrinsic_n_reclassified_by_r_filter"] = metrics.get("n_reclassified_by_r_filter", 0)
    row["feat_intrinsic_arc_width_entropy"] = metrics.get("arc_width_entropy", 0.0)
    row["param_k_y_frac"] = metrics.get("k_y_frac", 0.0)
    oblique = metrics.get("hough_oblique_threshold", metrics.get("hough_threshold", 37.0))
    row["param_hough_oblique_threshold"] = oblique
    for k, v in metrics.items():
        if k.startswith("seg_"):
            row[k] = v
    for k, v in metrics.items():
        if k.startswith("v5_"):
            row[k] = v
    row["gt_miou"] = metrics.get("gt_miou", 0.0)
    row["agent_error"] = metrics.get("agent_error", True)
    for key in DIRECTION_PROXY_FIELDS:
        if key in metrics:
            row[key] = metrics[key]
    if "order_branch" not in row:
        row["order_branch"] = metrics.get("order_branch", "plus")
    row["det_guardrail_passed"] = apply_guardrail_fields(metrics)["det_guardrail_passed"]
    row["det_ready_for_segmentation"] = metrics.get("det_ready_for_segmentation")
    row["det_y_coverage_pct"] = metrics.get("det_y_coverage_pct")
    return row


def evaluate_candidate(
    ctx: RingContext,
    search_x: list[float] | np.ndarray,
    *,
    candidate_id: int,
    pre7: dict[str, Any] | None = None,
    prefer_branch: str = "plus",
) -> dict[str, Any]:
    """Run det → intrinsic direction_select (plus/minus) → seg; return proxy features.

    Held-out and Stage-A scoring must use this path — not raw seg on a single branch.
    """
    x = np.asarray(search_x, dtype=float)
    k_y, offsets, layout, r_surface_min = decode_x(ctx, x)
    tag = f"cand{candidate_id:02d}"
    metrics = evaluate_trial(
        ctx,
        k_y,
        offsets,
        layout,
        r_surface_min,
        tag=tag,
        order_branch=prefer_branch,
    )
    for k in list(metrics.keys()):
        if k in EXCLUDED_TRIAL_METRICS:
            metrics.pop(k)
    metrics = apply_guardrail_fields(metrics)

    if not metrics.get("agent_error"):
        try:
            seg = extract_seg_metrics(ctx.sandbox_ring)
            metrics.update(seg)
        except Exception as exc:
            metrics["seg_extract_error"] = str(exc)
        try:
            metrics.update(prefix_features(compute_v5_proxy_features(ctx.sandbox_ring)))
        except Exception as exc:
            metrics["v5_extract_error"] = str(exc)

    if pre7 is None:
        pre7 = ring_pre7_features(ctx.src_ring)
    proxy_row = metrics_to_proxy_row(metrics, pre7)
    proxy_row["candidate_id"] = candidate_id
    proxy_row["k_center_norm"] = float(k_y / max(ctx.H, 1))
    proxy_row["layout_k_center_norm"] = proxy_row["k_center_norm"]
    proxy_row["direction_hypotheses_present"] = direction_hypotheses_available(ctx.sandbox_ring)
    sel_path = ctx.out_dir / SELECTION_FILENAME
    if sel_path.is_file():
        proxy_row["direction_selection_path"] = str(sel_path)
    return proxy_row


def check_direction_select_eval(df: pd.DataFrame, ring_dir: Path) -> dict[str, Any]:
    """Gate: held-out candidate eval used intrinsic dual-branch direction_select."""
    criteria: dict[str, bool] = {}
    if df.empty:
        return {"passed": False, "criteria": {"non_empty": False}, "details": {}}
    criteria["non_empty"] = True
    if "direction_select_enabled" in df.columns:
        ok = df["direction_select_enabled"].fillna(False).astype(bool)
        criteria["direction_select_enabled_all"] = bool(ok.all())
    else:
        criteria["direction_select_enabled_all"] = False
    if "direction_score_plus" in df.columns and "direction_score_minus" in df.columns:
        scored = df[~df["agent_error"].astype(bool)] if "agent_error" in df.columns else df
        criteria["branch_scores_logged"] = bool(
            len(scored) > 0
            and scored["direction_score_plus"].notna().any()
            and scored["direction_score_minus"].notna().any()
        )
    else:
        criteria["branch_scores_logged"] = False
    meta = ring_dir / DIRECTION_META
    criteria["direction_hypotheses_meta"] = meta.is_file()
    criteria["order_branch_logged"] = "order_branch" in df.columns
    passed = all(criteria.values())
    return {
        "passed": passed,
        "criteria": criteria,
        "n_candidates": int(len(df)),
        "direction_meta_path": str(meta) if meta.is_file() else None,
    }


def load_ring_context(
    ring_key: str,
    *,
    held_out_root: Path,
    score_root: Path,
) -> tuple[RingContext, dict[str, Any]]:
    tunnel_id, rpart = ring_key.split("/")
    ring_id = int(rpart.lstrip("r"))
    ctx = build_ring_context(
        tunnel_id,
        ring_id,
        source_root=held_out_root,
        run_root=score_root,
    )
    setup_sandbox(ctx.src_ring, ctx.sandbox_ring)
    pre7 = ring_pre7_features(ctx.src_ring)
    return ctx, pre7
