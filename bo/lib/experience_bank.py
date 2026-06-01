"""Merge v3/v4/v5 BO trials into a unified experience bank with normalised features."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.ceiling_gate import REPO_ROOT, blocks_for_segment_count
from lib.layout_bo import offsets_to_arc_widths

LINE_LOG_RE = re.compile(r"Lines:\s+\+(\d+)\s+-(\d+)\s+H(\d+)\s+V(\d+)")

POOLS: dict[str, dict[str, str]] = {
    "v3": {
        "run_root": "logs/bo_experience_v3",
        "source_type": "random",
        "label": "failure_memory",
    },
    "v4": {
        "run_root": "logs/bo_experience_v4_sam4tun_prior",
        "source_type": "SAM4Tun",
        "label": "sam4tun_prior",
    },
    "v5": {
        "run_root": "logs/bo_experience_v5_gt_derived",
        "source_type": "GT-derived",
        "label": "gt_derived",
    },
}

SPARSE_SLOTS = frozenset({"sparse_6", "sparse_7"})


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {r["ring_key"]: r for r in manifest.get("rings", [])}


def _tunnel_diameter(ring_dir: Path) -> float | None:
    prep = ring_dir / "parameters_preprocessing.json"
    if not prep.is_file():
        return None
    return float(json.loads(prep.read_text(encoding="utf-8")).get("tunnel_diameter", float("nan")))


def _depth_static_features(ring_dir: Path) -> dict[str, float]:
    dm_path = ring_dir / "depth_map.npy"
    if not dm_path.is_file():
        return {}
    dm = np.load(dm_path)
    finite = np.isfinite(dm)
    h, w = dm.shape
    row_has = finite.any(axis=1)
    col_has = finite.any(axis=0)
    # largest empty vertical band (angular gaps)
    gap_frac = 0.0
    if h > 0:
        best = cur = 0
        for ok in row_has:
            if not ok:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        gap_frac = best / h
    return {
        "image_height": float(h),
        "image_width": float(w),
        "depth_finite_ratio": float(finite.mean()),
        "depth_row_nonempty_ratio": float(row_has.mean()),
        "depth_col_nonempty_ratio": float(col_has.mean()),
        "depth_blank_band_ratio": float(gap_frac),
        "depth_coverage_ratio": float(finite.mean()),
    }


def _parse_offsets(raw: Any) -> dict[str, float]:
    if isinstance(raw, str):
        data = json.loads(raw)
    else:
        data = raw
    ring0 = data.get("0", data)
    return {str(k): float(v) for k, v in ring0.items()}


def _parse_line_log(log_path: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "oblique_line_count": None,
        "oblique_line_strength_pos": None,
        "oblique_line_strength_neg": None,
        "horizontal_line_count": None,
        "vertical_line_count": None,
    }
    if not log_path.is_file():
        return out
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LINE_LOG_RE.search(line)
        if m:
            pos, neg, horiz, vert = (int(x) for x in m.groups())
            out["oblique_line_count"] = float(pos + neg)
            out["oblique_line_strength_pos"] = float(pos)
            out["oblique_line_strength_neg"] = float(neg)
            out["horizontal_line_count"] = float(horiz)
            out["vertical_line_count"] = float(vert)
            break
    return out


def _guardrail_count(raw: Any) -> int:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return 0
    if isinstance(raw, str):
        try:
            items = json.loads(raw)
            return len(items) if isinstance(items, list) else 0
        except json.JSONDecodeError:
            return 0
    if isinstance(raw, list):
        return len(raw)
    return 0


def _layout_features(
    k_y: float,
    offsets: dict[str, float],
    blocks: list[str],
    image_height: float,
    image_width: float,
) -> dict[str, Any]:
    h = max(image_height, 1.0)
    w = max(image_width, 1.0)
    k_y_f = float(k_y) % h
    off_norm = {b: float(offsets.get(b, 0.0)) % h / h for b in blocks}
    arc_w = offsets_to_arc_widths(blocks, offsets, int(h))
    arc_norm = {blocks[i]: float(arc_w[i]) / h for i in range(len(blocks))}
    k_idx = blocks.index("K") if "K" in blocks else 0
    k_width = float(arc_w[k_idx]) if len(arc_w) else 0.0
    boundaries = []
    ys = [float(offsets[b]) % h for b in blocks]
    for i in range(len(blocks)):
        boundaries.append(((ys[i] + ys[(i + 1) % len(blocks)]) / 2.0) % h / h)
    return {
        "k_center": k_y_f,
        "k_center_norm_h": k_y_f / h,
        "k_center_norm_w": k_y_f / w,
        "k_width": k_width,
        "k_width_norm_h": k_width / h,
        "k_width_norm_w": k_width / w,
        "ab_offset_norm_json": json.dumps(off_norm),
        "arc_width_norm_json": json.dumps(arc_norm),
        "boundary_positions_norm_json": json.dumps(boundaries),
        "segment_order_json": json.dumps(blocks),
    }


def _form_features(row: pd.Series, blocks: list[str], image_height: float) -> dict[str, Any]:
    h = max(image_height, 1.0)
    violations = row.get("det_guardrail_violations")
    n_viol = _guardrail_count(violations)
    missing_hint = 0
    if isinstance(violations, str) and "missing" in violations.lower():
        missing_hint = 1
    coverage = row.get("det_y_coverage_pct")
    coverage_f = float(coverage) if pd.notna(coverage) else np.nan
    min_gap = row.get("det_min_y_gap_px")
    min_gap_f = float(min_gap) if pd.notna(min_gap) else np.nan
    return {
        "form_missing_block_hint": missing_hint,
        "form_guardrail_violation_count": n_viol,
        "form_boundary_gap_norm": (min_gap_f / h) if pd.notna(min_gap) else np.nan,
        "form_segment_coverage_pct": coverage_f,
        "form_y_order_consistency": row.get("det_y_order_consistency"),
        "form_arc_width_entropy": row.get("arc_width_entropy"),
        "form_mask_reclass_count": row.get("n_reclassified_by_r_filter"),
        "form_ready_for_segmentation": row.get("det_ready_for_segmentation"),
    }


def _ring_static_features(case_id: str, manifest_entry: dict, corpus_dir: Path) -> dict[str, Any]:
    tid, rid = case_id.split("/")
    ring_dir = corpus_dir / tid / rid
    iq = manifest_entry.get("intrinsic_quality", {})
    depth = _depth_static_features(ring_dir)
    slot = manifest_entry.get("diversity_slot", "")
    return {
        "ring_id": case_id,
        "segment_count": int(manifest_entry.get("segment_count", 0)),
        "diversity_slot": slot,
        "density_sparsity_slot": "sparse" if slot in SPARSE_SLOTS else "representative",
        "tunnel_id": tid,
        "tunnel_diameter_m": _tunnel_diameter(ring_dir),
        "ceiling_miou_ref": manifest_entry.get("ceiling_miou"),
        "depth_row_nonempty_ratio": depth.get("depth_row_nonempty_ratio", iq.get("row_nonempty")),
        "depth_col_nonempty_ratio": depth.get("depth_col_nonempty_ratio", iq.get("col_nonempty")),
        "depth_finite_ratio": depth.get("depth_finite_ratio", iq.get("finite_ratio")),
        "depth_coverage_ratio": depth.get("depth_coverage_ratio"),
        "depth_blank_band_ratio": depth.get("depth_blank_band_ratio"),
        "image_height": depth.get("image_height", iq.get("shape_h")),
        "image_width": depth.get("image_width", iq.get("shape_w")),
    }


def build_row(
    row: pd.Series,
    *,
    pool_key: str,
    run_root: Path,
    ring_static: dict[str, Any],
    blocks: list[str],
) -> dict[str, Any]:
    pool = POOLS[pool_key]
    case_id = str(row["case_id"])
    trial_id = int(row["trial_id"])
    image_h = float(ring_static.get("image_height") or 1)
    image_w = float(ring_static.get("image_width") or 1)
    offsets = _parse_offsets(row.get("per_ring_offsets"))
    layout = _layout_features(float(row.get("k_y", 0)), offsets, blocks, image_h, image_w)
    form = _form_features(row, blocks, image_h)

    tid, rid = case_id.split("/")
    log_path = run_root / "sandbox" / tid / rid / "logs" / f"trial{trial_id:03d}_2_detection.log"
    lines = _parse_line_log(log_path)

    oblique_total = lines["oblique_line_count"]
    horiz = lines["horizontal_line_count"]
    line_conf_k = row.get("det_k_confidence_avg")
    line_conf_ab = row.get("det_y_order_consistency")

    record: dict[str, Any] = {
        "experience_id": f"{pool_key}:{case_id}:t{trial_id:03d}",
        "experience_pool": pool_key,
        "source_type": pool["source_type"],
        "trial_id": trial_id,
        "trial_kind": row.get("kind"),
        "ring_id": case_id,
        **{f"ring_{k}": v for k, v in ring_static.items() if k != "ring_id"},
        "line_oblique_line_count": oblique_total,
        "line_oblique_line_strength_pos": lines["oblique_line_strength_pos"],
        "line_oblique_line_strength_neg": lines["oblique_line_strength_neg"],
        "line_oblique_angle_consistency": (
            float(line_conf_ab) if pd.notna(line_conf_ab) else np.nan
        ),
        "line_horizontal_line_count": horiz,
        "line_horizontal_spacing_consistency": (
            (horiz / max(image_w, 1.0)) if horiz is not None else np.nan
        ),
        "line_detection_confidence_K": line_conf_k,
        "line_detection_confidence_AB": line_conf_ab,
        "layout_k_center_norm": layout["k_center_norm_h"],
        "layout_k_center_norm_w": layout["k_center_norm_w"],
        "layout_k_width_norm": layout["k_width_norm_h"],
        "layout_k_width_norm_w": layout["k_width_norm_w"],
        "layout_ab_offset_norm_json": layout["ab_offset_norm_json"],
        "layout_arc_width_norm_json": layout["arc_width_norm_json"],
        "layout_boundary_positions_norm_json": layout["boundary_positions_norm_json"],
        "layout_segment_order_json": layout["segment_order_json"],
        "layout_hough_oblique_threshold": row.get("hough_oblique_threshold"),
        "layout_hough_horizontal_threshold": row.get("hough_horizontal_threshold"),
        "layout_line_merge_distance": row.get("line_merge_distance"),
        "layout_line_snap_tolerance_px": row.get("line_snap_tolerance_px"),
        "layout_segmentation_slot_inset_y": row.get("segmentation_slot_inset_y"),
        "layout_r_surface_min": row.get("r_surface_min"),
        "layout_r_surface_min_frac": row.get("r_surface_min_frac"),
        "layout_order_branch": row.get("order_branch"),
        **form,
        "label_gt_miou": row.get("gt_miou"),
        "label_regret_vs_ceiling": row.get("regret_vs_ceiling"),
        "label_agent_error": row.get("agent_error"),
    }
    return record


def load_pool_trials(pool_key: str, repo_root: Path | None = None) -> tuple[pd.DataFrame, Path]:
    root = repo_root or REPO_ROOT
    run_root = (root / POOLS[pool_key]["run_root"]).resolve()
    csv_path = run_root / "bo_trials.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    df["experience_pool"] = pool_key
    df["source_type"] = POOLS[pool_key]["source_type"]
    return df, run_root


def build_experience_bank(
    *,
    manifest_path: Path | None = None,
    corpus_dir: Path | None = None,
    repo_root: Path | None = None,
    pools: list[str] | None = None,
) -> pd.DataFrame:
    root = repo_root or REPO_ROOT
    manifest_path = manifest_path or (root / "data" / "bo_calibration" / "MANIFEST.json")
    corpus_dir = corpus_dir or (root / "data" / "bo_calibration")
    pools = pools or list(POOLS.keys())

    manifest = _load_manifest(manifest_path)
    ring_cache: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for pool_key in pools:
        trials, run_root = load_pool_trials(pool_key, root)
        for _, row in trials.iterrows():
            case_id = str(row["case_id"])
            if case_id not in ring_cache:
                entry = manifest.get(case_id, {})
                ring_cache[case_id] = _ring_static_features(case_id, entry, corpus_dir)
            seg_n = int(ring_cache[case_id].get("segment_count") or row.get("segment_count") or 7)
            blocks = blocks_for_segment_count(seg_n)
            records.append(
                build_row(
                    row,
                    pool_key=pool_key,
                    run_root=run_root,
                    ring_static=ring_cache[case_id],
                    blocks=blocks,
                )
            )

    bank = pd.DataFrame(records)

    # Ranks and success flags within ring (all pools) and within ring+pool
    bank["label_rank_within_ring"] = (
        bank.groupby("ring_id")["label_gt_miou"].rank(method="min", ascending=False)
    )
    bank["label_rank_within_ring_pool"] = (
        bank.groupby(["ring_id", "experience_pool"])["label_gt_miou"].rank(method="min", ascending=False)
    )
    ring_median = bank.groupby("ring_id")["label_gt_miou"].transform("median")
    bank["label_success_flag"] = (bank["label_gt_miou"] >= ring_median).astype(int)
    bank["label_failure_flag"] = (bank["label_gt_miou"] < 0.30).astype(int)

    return bank
