"""Build reusable proposal templates (correction deltas) from experience bank pools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.ceiling_gate import REPO_ROOT

FORM_PARAM_COLS = [
    "layout_hough_oblique_threshold",
    "layout_hough_horizontal_threshold",
    "layout_line_merge_distance",
    "layout_line_snap_tolerance_px",
    "layout_segmentation_slot_inset_y",
    "layout_r_surface_min",
    "layout_r_surface_min_frac",
]

LAYOUT_NORM_COLS = [
    "layout_k_center_norm",
    "layout_k_width_norm",
]


def _parse_json_dict(raw: Any) -> dict[str, float]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    if isinstance(raw, str):
        data = json.loads(raw)
    else:
        data = raw
    return {str(k): float(v) for k, v in data.items()}


def _delta_json(anchor: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(anchor) | set(candidate))
    return {k: candidate.get(k, 0.0) - anchor.get(k, 0.0) for k in keys}


def _anchor_row(v4_ring: pd.DataFrame) -> pd.Series:
    anchor = v4_ring[v4_ring["trial_kind"] == "sam4tun_static"]
    if anchor.empty:
        raise ValueError(f"No sam4tun_static anchor for ring {v4_ring['ring_id'].iloc[0]}")
    return anchor.iloc[0]


def _extract_form_params(row: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in FORM_PARAM_COLS:
        val = row.get(col)
        if pd.notna(val):
            out[col.replace("layout_", "")] = float(val)
    return out


def build_sam4tun_proposal_templates(
    bank: pd.DataFrame,
    *,
    top_frac: float = 0.20,
    min_keep: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (anchors_df, templates_df) for SAM4Tun v4 pool."""
    v4 = bank[bank["experience_pool"] == "v4"].copy()
    if v4.empty:
        raise ValueError("No v4 (SAM4Tun) rows in experience bank")

    anchor_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []

    for ring_id, ring_df in v4.groupby("ring_id"):
        ring_df = ring_df.sort_values("label_gt_miou", ascending=False)
        anchor = _anchor_row(ring_df)
        anchor_id = str(anchor["experience_id"])
        anchor_ab = _parse_json_dict(anchor["layout_ab_offset_norm_json"])
        anchor_arc = _parse_json_dict(anchor["layout_arc_width_norm_json"])
        anchor_form = _extract_form_params(anchor)

        anchor_rows.append({
            "ring_id": ring_id,
            "source_type": "SAM4Tun",
            "anchor_experience_id": anchor_id,
            "anchor_trial_kind": anchor["trial_kind"],
            "anchor_k_center_norm": anchor["layout_k_center_norm"],
            "anchor_k_width_norm": anchor["layout_k_width_norm"],
            "anchor_ab_offset_norm_json": json.dumps(anchor_ab),
            "anchor_arc_width_norm_json": json.dumps(anchor_arc),
            "anchor_form_params_json": json.dumps(anchor_form),
            "anchor_label_gt_miou": anchor["label_gt_miou"],
            "ring_segment_count": anchor["ring_segment_count"],
            "ring_image_height": anchor["ring_image_height"],
            "ring_image_width": anchor["ring_image_width"],
            "deployment_base": "SAM4Tun_prior",
        })

        candidates = ring_df.sort_values("label_gt_miou", ascending=False)
        n_keep = max(min_keep, int(np.ceil(top_frac * len(ring_df))))
        top = candidates.head(n_keep)

        for _, cand in top.iterrows():
            if str(cand["experience_id"]) == anchor_id:
                continue
            cand_ab = _parse_json_dict(cand["layout_ab_offset_norm_json"])
            cand_arc = _parse_json_dict(cand["layout_arc_width_norm_json"])
            cand_form = _extract_form_params(cand)
            delta_ab = _delta_json(anchor_ab, cand_ab)
            delta_arc = _delta_json(anchor_arc, cand_arc)
            delta_form = _delta_json(anchor_form, cand_form)

            template_rows.append({
                "proposal_id": f"sam4tun:{ring_id}:from_{cand['trial_id']:03d}",
                "ring_id": ring_id,
                "source_type": "SAM4Tun",
                "anchor_experience_id": anchor_id,
                "candidate_experience_id": cand["experience_id"],
                "candidate_trial_kind": cand["trial_kind"],
                "delta_k_center_norm": float(cand["layout_k_center_norm"] - anchor["layout_k_center_norm"]),
                "delta_k_width_norm": float(cand["layout_k_width_norm"] - anchor["layout_k_width_norm"]),
                "delta_ab_offset_norm_json": json.dumps(delta_ab),
                "delta_arc_width_norm_json": json.dumps(delta_arc),
                "delta_form_params_json": json.dumps(delta_form),
                "label_gt_miou": cand["label_gt_miou"],
                "label_regret_vs_ceiling": cand["label_regret_vs_ceiling"],
                "label_rank_within_ring_pool": cand["label_rank_within_ring_pool"],
                "label_success_flag": cand["label_success_flag"],
                "top_frac_bucket": top_frac,
                "deployment_recipe": "candidate = SAM4Tun_prior + retrieved_successful_delta",
            })

    return pd.DataFrame(anchor_rows), pd.DataFrame(template_rows)


def load_experience_bank(path: Path | None = None) -> pd.DataFrame:
    path = path or (REPO_ROOT / "methods" / "paper" / "experience" / "experience_bank.csv")
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _quantile_range(series: pd.Series, lo: float = 0.10, hi: float = 0.90) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"lo": float("nan"), "med": float("nan"), "hi": float("nan")}
    return {
        "lo": round(float(s.quantile(lo)), 6),
        "med": round(float(s.quantile(0.5)), 6),
        "hi": round(float(s.quantile(hi)), 6),
    }


def _boundary_spacings(row: pd.Series) -> list[float]:
    raw = row.get("layout_boundary_positions_norm_json")
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, str):
        data = json.loads(raw)
    else:
        data = raw
    if isinstance(data, list):
        bs = sorted(float(v) for v in data)
    elif isinstance(data, dict):
        bs = sorted(float(v) for v in data.values())
    else:
        return []
    if len(bs) < 2:
        return []
    gaps = []
    for i in range(len(bs)):
        gap = (bs[(i + 1) % len(bs)] - bs[i]) % 1.0
        gaps.append(float(gap))
    return gaps


def _arc_widths(row: pd.Series) -> dict[str, float]:
    return _parse_json_dict(row.get("layout_arc_width_norm_json"))


def _ab_separation_stats(rows: pd.DataFrame) -> dict[str, Any]:
    """Vertical separation proxy: non-K arc widths (normalised)."""
    per_block: dict[str, list[float]] = {}
    for _, r in rows.iterrows():
        for block, w in _arc_widths(r).items():
            if block == "K":
                continue
            per_block.setdefault(block, []).append(w)
    return {b: _quantile_range(pd.Series(vals)) for b, vals in sorted(per_block.items())}


def build_gt_good_form_templates(
    bank: pd.DataFrame,
    *,
    top_frac: float = 0.20,
    min_keep: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Learn local good-form ranges from v5 (no GT positions for deployment)."""
    v5 = bank[bank["experience_pool"] == "v5"].copy()
    if v5.empty:
        raise ValueError("No v5 (GT-derived) rows in experience bank")

    template_rows: list[dict[str, Any]] = []
    exemplar_rows: list[dict[str, Any]] = []

    for ring_id, ring_df in v5.groupby("ring_id"):
        ring_df = ring_df.sort_values("label_gt_miou", ascending=False)
        n_keep = max(min_keep, int(np.ceil(top_frac * len(ring_df))))
        good = ring_df.head(n_keep)
        # Form stats from high-mIoU trials; exclude oracle layout kinds from exemplar export only
        non_oracle = good[~good["trial_kind"].astype(str).str.startswith("gt_layout")]

        spacing_vals: list[float] = []
        for _, r in good.iterrows():
            spacing_vals.extend(_boundary_spacings(r))

        form_ranges = {
            "k_width_norm": _quantile_range(good["layout_k_width_norm"]),
            "boundary_spacing_norm": _quantile_range(pd.Series(spacing_vals)),
            "hough_oblique_threshold": _quantile_range(good["layout_hough_oblique_threshold"]),
            "hough_horizontal_threshold": _quantile_range(good["layout_hough_horizontal_threshold"]),
            "line_merge_distance": _quantile_range(good["layout_line_merge_distance"]),
            "line_snap_tolerance_px": _quantile_range(good["layout_line_snap_tolerance_px"]),
            "segmentation_slot_inset_y": _quantile_range(good["layout_segmentation_slot_inset_y"]),
            "r_surface_min_frac": _quantile_range(good["layout_r_surface_min_frac"]),
            "segment_coverage_pct": _quantile_range(good["form_segment_coverage_pct"]),
            "arc_width_entropy": _quantile_range(good["form_arc_width_entropy"]),
            "ab_arc_width_norm": _ab_separation_stats(good),
        }

        template_rows.append({
            "ring_id": ring_id,
            "source_type": "GT-derived",
            "source_pool": "v5",
            "n_good_trials": int(len(good)),
            "n_non_oracle_good_trials": int(len(non_oracle)),
            "good_miou_lo": round(float(good["label_gt_miou"].min()), 4),
            "good_miou_hi": round(float(good["label_gt_miou"].max()), 4),
            "good_form_ranges_json": json.dumps(form_ranges),
            "deployment_recipe": (
                "Apply form ranges locally around SAM4Tun | line-derived | hybrid anchor "
                "(never inject GT k_y / offsets at runtime)"
            ),
            "allowed_anchors_json": json.dumps(["SAM4Tun", "line_derived", "hybrid_sam_line"]),
        })

        for _, r in non_oracle.iterrows():
            exemplar_rows.append({
                "ring_id": ring_id,
                "experience_id": r["experience_id"],
                "trial_kind": r["trial_kind"],
                "label_gt_miou": r["label_gt_miou"],
                "layout_k_width_norm": r["layout_k_width_norm"],
                "form_segment_coverage_pct": r["form_segment_coverage_pct"],
                "form_arc_width_entropy": r["form_arc_width_entropy"],
                "layout_hough_oblique_threshold": r["layout_hough_oblique_threshold"],
                "layout_line_merge_distance": r["layout_line_merge_distance"],
            })

    return pd.DataFrame(template_rows), pd.DataFrame(exemplar_rows)


FAILURE_KINDS = frozenset({
    "perturb_wrong_k",
    "perturb_guardrail_smoke",
    "perturb_misaligned",
    "perturb_offset_shift",
})


def _failure_tags(row: pd.Series, ring_median_cov: float, ring_q25_miou: float) -> list[str]:
    tags: list[str] = []
    kind = str(row.get("trial_kind", ""))
    miou = float(row.get("label_gt_miou", 0))
    cov = row.get("form_segment_coverage_pct")
    k_conf = row.get("line_detection_confidence_K")
    if kind in FAILURE_KINDS or kind.startswith("perturb_wrong"):
        tags.append("bad_layout_perturbation")
    if kind == "perturb_wrong_k":
        tags.append("bad_k_shift")
    if miou <= ring_q25_miou:
        tags.append("low_miou")
    if pd.notna(cov) and float(cov) >= ring_median_cov and miou <= ring_q25_miou:
        tags.append("good_form_wrong_anchor")
    if pd.notna(cov) and pd.notna(k_conf) and float(cov) >= 90 and float(k_conf) < 1.0 and miou < 0.25:
        tags.append("misleading_line_proxy")
    if int(row.get("label_failure_flag", 0)) == 1:
        tags.append("hard_failure")
    return tags or ["low_miou"]


def build_random_failure_memory(
    bank: pd.DataFrame,
    *,
    bottom_frac: float = 0.20,
    min_keep: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Failure memory from v3 random/geometric pool for deploy-time filtering."""
    v3 = bank[bank["experience_pool"] == "v3"].copy()
    if v3.empty:
        raise ValueError("No v3 (random) rows in experience bank")

    memory_rows: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []

    for ring_id, ring_df in v3.groupby("ring_id"):
        ring_df = ring_df.copy()
        ring_median_cov = float(ring_df["form_segment_coverage_pct"].median())
        ring_q25_miou = float(ring_df["label_gt_miou"].quantile(0.25))
        ring_median_k = float(ring_df["layout_k_center_norm"].median())

        n_fail = max(min_keep, int(np.ceil(bottom_frac * len(ring_df))))
        failures = ring_df.sort_values("label_gt_miou").head(n_fail)

        for _, r in failures.iterrows():
            tags = _failure_tags(r, ring_median_cov, ring_q25_miou)
            memory_rows.append({
                "failure_id": f"random:{ring_id}:t{int(r['trial_id']):03d}",
                "ring_id": ring_id,
                "source_type": "random",
                "source_pool": "v3",
                "experience_id": r["experience_id"],
                "trial_kind": r["trial_kind"],
                "failure_tags_json": json.dumps(tags),
                "layout_k_center_norm": r["layout_k_center_norm"],
                "layout_k_width_norm": r["layout_k_width_norm"],
                "layout_ab_offset_norm_json": r["layout_ab_offset_norm_json"],
                "line_oblique_line_count": r.get("line_oblique_line_count"),
                "line_detection_confidence_K": r.get("line_detection_confidence_K"),
                "form_segment_coverage_pct": r.get("form_segment_coverage_pct"),
                "form_arc_width_entropy": r.get("form_arc_width_entropy"),
                "label_gt_miou": r["label_gt_miou"],
                "k_shift_from_ring_median": round(float(r["layout_k_center_norm"] - ring_median_k), 6),
            })

        bad_k_shifts = failures["layout_k_center_norm"] - ring_median_k
        k_fail_lo = float(failures["layout_k_center_norm"].quantile(0.10))
        k_fail_hi = float(failures["layout_k_center_norm"].quantile(0.90))
        rule_rows.append({
            "ring_id": ring_id,
            "source_type": "random",
            "source_pool": "v3",
            "reject_k_center_norm_lo": round(k_fail_lo, 6),
            "reject_k_center_norm_hi": round(k_fail_hi, 6),
            "reject_k_shift_norm_abs_p90": round(float(bad_k_shifts.abs().quantile(0.90)), 6),
            "reject_miou_upper_bound": round(float(failures["label_gt_miou"].quantile(0.90)), 4),
            "penalise_coverage_min": round(ring_median_cov, 2),
            "penalise_miou_max": round(ring_q25_miou, 4),
            "penalise_k_confidence_max": 1.0,
            "n_failure_exemplars": int(len(failures)),
            "filter_recipe": (
                "reject if layout_k_center_norm in [reject_k_center_norm_lo, reject_k_center_norm_hi] "
                "or failure_tags match bad_k_shift; "
                "penalise if form_segment_coverage_pct >= penalise_coverage_min "
                "AND line_detection_confidence_K <= penalise_k_confidence_max "
                "(good form does not guarantee good mIoU when K is wrong)"
            ),
        })

    return pd.DataFrame(memory_rows), pd.DataFrame(rule_rows)
