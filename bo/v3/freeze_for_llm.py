"""Step 5 — Freeze v3 calibration artefacts for the deployment-time LLM loop.

This script reads the stable raw evidence from
``data/v3/calibration/aggregate_summary.json`` (the broad pooled
correlation tables and per-ring breakdown produced upstream by
``bo/v3/finalize_calibration.py``) plus
``data/v3/calibration/intrinsic_units.json`` (Step 1 audit) and
``data/v3/calibration/baseline_vs_bo.csv`` (per-ring scoreboard) and
the per-ring ``intrinsics.csv`` files under ``logs/v3/bo_calibration/``.

It emits a single canonical JSON the LLM loop will read, plus
refreshed companions (sensitive_parameters / diagnostic_intrinsics /
guardrails) that contain only entries that pass the v3 selection
rules. The script is idempotent — re-running it always recomputes the
filtered companions from the raw evidence.

Selection rules (locked in the plan):
  * Tunable parameters: pooled |Spearman| >= 0.2 in the preprocessing-stage
    corpus AND parameter has param_ranges evidence in >= 3 of 6 rings.
  * Diagnostic intrinsics: pooled Spearman ρ >= 0.5 in the preprocessing
    -stage corpus AND >= 90% non-null across successful trials.
  * Guardrail thresholds: p25 of the upper-quartile-mIoU regime
    (miou_fixed_class >= corpus 75th percentile), in the audited units;
    G_stability uses the median (strict) and p75 (operational) of the
    pooled permutation-vs-fixed gap.

Outputs (overwriting prior placeholders):
  * ``data/v3/calibration/llm_loop_frozen.json``
  * ``data/v3/calibration/sensitive_parameters.json``
  * ``data/v3/calibration/diagnostic_intrinsics.json``
  * ``data/v3/calibration/guardrails.json``
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CAL_DIR = REPO_ROOT / "data" / "v3" / "calibration"
LOGS_ROOT = REPO_ROOT / "logs" / "v3" / "bo_calibration"

CALIBRATION_RINGS = [
    "4-9/r365",
    "5-5/r253",
    "4-7/r309",
    "4-8/r336",
    "4-1/r116",
    "5-7/r323",
]

TUNABLE_RHO_ABS_MIN = 0.2
TUNABLE_MIN_RINGS = 3
DIAGNOSTIC_RHO_MIN = 0.5
DIAGNOSTIC_NONNULL_FRACTION_MIN = 0.90

# Hard search-space bounds (mirrored from bo/v3/spaces.py — kept here so the
# frozen artefact carries its own evidence and does not depend on the BO
# package being importable at deployment time).
PRE_HARD_BOUNDS: dict[str, list[float]] = {
    "radius_min": [1.8, 3.8],
    "radius_max": [2.0, 4.4],
    "gradient_threshold": [0.03, 0.40],
    "smoothing_offset": [-0.02, 0.02],
    "curvature_neighbors": [8, 40],
    "interpolation_window": [1, 15],
    "target_distance_1": [0.03, 0.12],
    "target_distance_2": [0.015, 0.06],
    "target_distance_3": [0.008, 0.04],
    "outlier_interpolation_radius": [0.01, 0.08],
    "outlier_num_interpolations": [1, 5],
    "outlier_depth_map_window": [1, 9],
    "outlier_neighbors": [8, 40],
}


def _load_pooled_intrinsics() -> pd.DataFrame:
    frames = []
    for csvp in LOGS_ROOT.glob("*/r*/*/bo30/intrinsics.csv"):
        frames.append(pd.read_csv(csvp))
    if not frames:
        raise SystemExit("no intrinsics.csv files found under logs/v3/bo_calibration/")
    return pd.concat(frames, ignore_index=True)


def _seed_default(param_name: str) -> Any:
    """Return the R4Tun-seed default value for a given preprocessing param.

    Reads the baseline parameter sandbox written by the v3 baseline run
    (any ring's baseline file works because seeds are shared after schema
    mapping at the same diameter). The target_distance_* keys are the
    sorted-descending elements of the seed's ``target_distances`` list.
    """
    seed_path = LOGS_ROOT / "4-9" / "r365" / "baseline" / "sandbox" / "4-9" / "r365" / "parameters_preprocessing.json"
    seed = json.loads(seed_path.read_text())
    if param_name.startswith("target_distance_"):
        td = sorted([float(t) for t in seed.get("target_distances", [0.08, 0.04, 0.02])], reverse=True)
        idx = int(param_name.rsplit("_", 1)[-1]) - 1
        if 0 <= idx < len(td):
            return float(td[idx])
        return None
    val = seed.get(param_name)
    if isinstance(val, (int, float)):
        return float(val) if isinstance(val, float) else int(val)
    return val


def _build_tunable_parameters(agg: dict[str, Any]) -> list[dict[str, Any]]:
    pre = agg["per_stage"]["preprocessing"]
    pooled_rho = pre.get("pooled_param_spearman_vs_miou", {})
    pooled_envelope = pre.get("pooled_param_envelope", {})
    per_ring = pre.get("per_ring", {})
    out: list[dict[str, Any]] = []
    for name, rho in pooled_rho.items():
        if rho is None or abs(float(rho)) < TUNABLE_RHO_ABS_MIN:
            continue
        rings_with_evidence = [
            ring for ring, payload in per_ring.items()
            if name in (payload.get("param_ranges") or {})
        ]
        if len(rings_with_evidence) < TUNABLE_MIN_RINGS:
            continue
        env = pooled_envelope.get(name) or {}
        entry = {
            "range_low_p25": env.get("p25"),
            "range_high_p75": env.get("p75"),
        }
        rings = rings_with_evidence
        hard = PRE_HARD_BOUNDS.get(name)
        seed_default = _seed_default(name)
        clipped = seed_default
        clipped_note = None
        if seed_default is not None and hard is not None:
            sd = float(seed_default)
            if sd < hard[0]:
                clipped = hard[0]
                clipped_note = f"R4Tun-seed default {sd} below BO hard_min {hard[0]}; deploy-time default clipped to hard_min."
            elif sd > hard[1]:
                clipped = hard[1]
                clipped_note = f"R4Tun-seed default {sd} above BO hard_max {hard[1]}; deploy-time default clipped to hard_max."
            # Preserve the int dtype where appropriate.
            if isinstance(seed_default, int) and not isinstance(seed_default, bool):
                clipped = int(round(float(clipped)))
        record: dict[str, Any] = {
            "name": name,
            "stage": "preprocessing",
            "default_r4tun_seed": seed_default,
            "default_deployable": clipped,
            "soft_bounds_p25_p75": [
                float(entry.get("range_low_p25")) if entry.get("range_low_p25") is not None else None,
                float(entry.get("range_high_p75")) if entry.get("range_high_p75") is not None else None,
            ],
            "hard_bounds_min_max": [float(hard[0]), float(hard[1])] if hard else None,
            "pooled_spearman_vs_miou": float(rho),
            "evidence_rings": rings,
        }
        if clipped_note is not None:
            record["clip_note"] = clipped_note
        out.append(record)
    out.sort(key=lambda r: abs(r["pooled_spearman_vs_miou"]), reverse=True)
    return out


def _build_diagnostic_intrinsics(agg: dict[str, Any],
                                 units: dict[str, Any],
                                 ok: pd.DataFrame) -> list[dict[str, Any]]:
    n_total = len(ok)
    pre = agg["per_stage"]["preprocessing"]
    pooled_rho = pre.get("pooled_intrinsic_spearman_vs_miou", {})
    per_ring = pre.get("per_ring", {})
    out: list[dict[str, Any]] = []
    for name, rho in pooled_rho.items():
        if rho is None or float(rho) < DIAGNOSTIC_RHO_MIN:
            continue
        if name not in ok.columns:
            continue
        col = ok[name]
        nn = (col.notna() if col.dtype == bool else pd.to_numeric(col, errors="coerce").notna()).sum()
        if n_total == 0 or (nn / n_total) < DIAGNOSTIC_NONNULL_FRACTION_MIN:
            continue
        u = units.get("intrinsics", {}).get(name, {})
        top_q = u.get("top_quartile") or {}
        rings_with_evidence = [
            ring for ring, payload in per_ring.items()
            if name in (payload.get("intrinsic_spearman_vs_miou") or {})
        ]
        out.append({
            "name": name,
            "stage_source": u.get("source_stage"),
            "units": u.get("units"),
            "dtype": u.get("dtype"),
            "pooled_spearman_vs_miou": float(rho),
            "min_good_threshold_p25_top_quartile": top_q.get("p25"),
            "permissive_threshold_top_quartile_min": top_q.get("min"),
            "evidence_rings": rings_with_evidence,
        })
    out.sort(key=lambda r: r["pooled_spearman_vs_miou"], reverse=True)
    return out


def _build_guardrails(intrinsics: list[dict[str, Any]], units: dict[str, Any], ok: pd.DataFrame) -> dict[str, Any]:
    by_name = {i["name"]: i for i in intrinsics}

    def thr_strict(name: str) -> Any:
        return by_name.get(name, {}).get("min_good_threshold_p25_top_quartile")

    def thr_permissive(name: str) -> Any:
        return by_name.get(name, {}).get("permissive_threshold_top_quartile_min")

    g_pre = {
        "intrinsics": ["pre_valid_ratio", "pre_depth_shape_w"],
        "rule": "all_of_min",
        "thresholds_strict": {
            "pre_valid_ratio": thr_strict("pre_valid_ratio"),
            "pre_depth_shape_w": thr_strict("pre_depth_shape_w"),
        },
        "thresholds_permissive": {
            "pre_valid_ratio": thr_permissive("pre_valid_ratio"),
            "pre_depth_shape_w": thr_permissive("pre_depth_shape_w"),
        },
        "rationale": "Preprocessing produced a usable depth map; both intrinsics must clear the upper-quartile-mIoU floor.",
    }
    g_layout = {
        "intrinsics": [
            "seg_segment_type_completeness",
            "seg_ring_completeness_avg",
            "seg_mask_coverage_pct",
        ],
        "rule": "all_of_min",
        "thresholds_strict": {
            "seg_segment_type_completeness": thr_strict("seg_segment_type_completeness"),
            "seg_ring_completeness_avg": thr_strict("seg_ring_completeness_avg"),
            "seg_mask_coverage_pct": thr_strict("seg_mask_coverage_pct"),
        },
        "thresholds_permissive": {
            "seg_segment_type_completeness": thr_permissive("seg_segment_type_completeness"),
            "seg_ring_completeness_avg": thr_permissive("seg_ring_completeness_avg"),
            "seg_mask_coverage_pct": thr_permissive("seg_mask_coverage_pct"),
        },
        "rationale": "Segmentation produced a structurally complete ring (all 7 block types, ring-level completeness, sufficient mask coverage).",
    }
    # G_stability: pooled permutation-vs-fixed gap distribution.
    gap = (
        pd.to_numeric(ok["miou_permutation"], errors="coerce")
        - pd.to_numeric(ok["miou_fixed_class"], errors="coerce")
    ).dropna().to_numpy()
    g_stability = {
        "intrinsics": ["miou_perm_minus_fixed_gap"],
        "rule": "max_below",
        "thresholds_strict": {"miou_perm_minus_fixed_gap": float(np.median(gap))} if gap.size else None,
        "thresholds_permissive": {"miou_perm_minus_fixed_gap": float(np.quantile(gap, 0.75))} if gap.size else None,
        "warning_threshold_p90": {"miou_perm_minus_fixed_gap": float(np.quantile(gap, 0.90))} if gap.size else None,
        "evidence_n_pooled_trials": int(len(gap)),
        "evidence_min_max": [float(gap.min()), float(gap.max())] if gap.size else None,
        "rationale": "Canonical-anchoring residual: a large permutation-vs-fixed mIoU gap means the segments are correct but mislabeled. Threshold uses pooled corpus median (strict) and p75 (operational).",
    }
    return {
        "G_pre": g_pre,
        "G_layout": g_layout,
        "G_stability": g_stability,
    }


def _failure_modes(pooled: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stage in ("preprocessing", "detection"):
        sdf = pooled[pooled.get("stage") == stage] if "stage" in pooled.columns else pd.DataFrame()
        if sdf.empty:
            continue
        n_total = int(len(sdf))
        n_failed = int((sdf["status"] != "ok").sum())
        modes = Counter(sdf[sdf["status"] != "ok"]["failure_mode"].dropna().astype(str).tolist())
        out[stage] = {
            "n_attempted": n_total,
            "n_failed": n_failed,
            "failure_rate_pct": round(100.0 * n_failed / n_total, 3) if n_total else None,
            "modes": dict(modes),
        }
    return out


def _baseline_vs_bo() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    csvp = CAL_DIR / "baseline_vs_bo.csv"
    if not csvp.exists():
        return rows
    with open(csvp, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main() -> int:
    agg = json.loads((CAL_DIR / "aggregate_summary.json").read_text())
    units = json.loads((CAL_DIR / "intrinsic_units.json").read_text())
    pooled = _load_pooled_intrinsics()
    if "stage" not in pooled.columns:
        pooled["stage"] = "unknown"
    ok = pooled[pooled["status"] == "ok"].copy()

    tunable = _build_tunable_parameters(agg)
    intrinsics = _build_diagnostic_intrinsics(agg, units, ok)
    guardrails = _build_guardrails(intrinsics, units, ok)
    failure_modes = _failure_modes(pooled)
    bvb = _baseline_vs_bo()

    det = agg.get("per_stage", {}).get("detection", {})
    detection_top_rho_abs = max(
        (abs(float(v or 0.0)) for v in (det.get("pooled_param_spearman_vs_miou") or {}).values()),
        default=0.0,
    )
    n_det_beats = sum(
        1 for r in bvb
        if float(r.get("detection_bo_fixed", 0.0)) > float(r.get("baseline_fixed", 0.0))
    )
    n_total_rings = len(bvb)

    frozen = {
        "version": "v3",
        "evidence": {
            "calibration_panel_rings": CALIBRATION_RINGS,
            "successful_trials_preprocessing": int((ok["__stage" if "__stage" in ok.columns else "stage"] == "preprocessing").sum()) if not ok.empty else 0,
            "successful_trials_detection": int((ok["__stage" if "__stage" in ok.columns else "stage"] == "detection").sum()) if not ok.empty else 0,
            "baseline_source": "r4tun/sample (regular reference)",
            "baseline_anchored_default": True,
            "selection_rules": {
                "tunable_parameters": {
                    "min_pooled_abs_spearman": TUNABLE_RHO_ABS_MIN,
                    "min_evidence_rings": TUNABLE_MIN_RINGS,
                    "stage_filter": "preprocessing",
                },
                "diagnostic_intrinsics": {
                    "min_pooled_spearman": DIAGNOSTIC_RHO_MIN,
                    "min_nonnull_fraction": DIAGNOSTIC_NONNULL_FRACTION_MIN,
                    "stage_filter": "preprocessing",
                },
                "guardrail_thresholds": "p25 of upper-quartile-mIoU regime (strict); top-quartile min (permissive)",
            },
            "failure_modes": failure_modes,
            "detection_top_pooled_abs_spearman": detection_top_rho_abs,
        },
        "tunable_parameters": tunable,
        "frozen_parameters": {
            "stage_detection": {
                "policy": "Use r4tun/sample defaults at deployment (no LLM tuning).",
                "rationale": (
                    f"All 21 detection knobs have pooled |Spearman| <= {detection_top_rho_abs:.3f}; "
                    "calibration evidence does not support LLM tuning of detection. "
                    f"Per-ring detection-BO winners on the calibration panel beat baseline on "
                    f"{n_det_beats} of {n_total_rings} rings; "
                    "the calibration record is preserved in baseline_vs_bo.csv but is not promoted to the deployment policy."
                ),
                "source": "r4tun/sample/parameters_detection.json",
            },
            "stage_preprocessing_other": {
                "policy": "Preprocessing knobs not in tunable_parameters use r4tun/sample defaults at deployment.",
                "source": "r4tun/sample/parameters_preprocessing.json",
            },
        },
        "diagnostic_intrinsics": intrinsics,
        "guardrails": guardrails,
        "calibration_scoreboard": bvb,
    }

    # Recompute trial counts using the per-CSV stage tag; the path-derived
    # stage column is more reliable than the per-row stage column.
    pre_csvs = list(LOGS_ROOT.glob("*/r*/preprocessing/bo30/intrinsics.csv"))
    det_csvs = list(LOGS_ROOT.glob("*/r*/detection/bo30/intrinsics.csv"))
    pre_ok = sum(int((pd.read_csv(p)["status"] == "ok").sum()) for p in pre_csvs)
    det_ok = sum(int((pd.read_csv(p)["status"] == "ok").sum()) for p in det_csvs)
    frozen["evidence"]["successful_trials_preprocessing"] = pre_ok
    frozen["evidence"]["successful_trials_detection"] = det_ok

    # Emit canonical artefact + refreshed companions.
    (CAL_DIR / "llm_loop_frozen.json").write_text(json.dumps(frozen, indent=2, default=str) + "\n")
    (CAL_DIR / "sensitive_parameters.json").write_text(
        json.dumps(
            {
                "selection_rule": frozen["evidence"]["selection_rules"]["tunable_parameters"],
                "preprocessing": tunable,
                "detection_frozen_at_deployment": {
                    "reason": frozen["frozen_parameters"]["stage_detection"]["rationale"],
                    "top_pooled_abs_spearman": detection_top_rho_abs,
                },
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    (CAL_DIR / "diagnostic_intrinsics.json").write_text(
        json.dumps(
            {
                "selection_rule": frozen["evidence"]["selection_rules"]["diagnostic_intrinsics"],
                "intrinsics": intrinsics,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    (CAL_DIR / "guardrails.json").write_text(
        json.dumps(
            {
                "selection_rule": frozen["evidence"]["selection_rules"]["guardrail_thresholds"],
                "bundles": guardrails,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"wrote {CAL_DIR}/llm_loop_frozen.json (+ refreshed companions)")
    print(f"  tunable_parameters: {len(tunable)}")
    print(f"  diagnostic_intrinsics: {len(intrinsics)}")
    print(f"  guardrail bundles: {len(guardrails)}")
    print(f"  preprocessing successful trials: {pre_ok}")
    print(f"  detection successful trials: {det_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
