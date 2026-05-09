from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bo.v3._paths import assert_writable
from bo.v3.intrinsics import collect_trial_intrinsics
from bo.v3.objectives import (
    DETECTION_CLI,
    PREPROCESSING_CLI,
    REQUIRED_DET_ARTEFACTS,
    REQUIRED_PRE_ARTEFACTS,
    REQUIRED_SEG_ARTEFACTS,
    SEGMENTATION_CLI,
    VENV_PYTHON,
    _run_subprocess,
)
from bo.v3.ontology import compute_j_reflect_v3, evaluate_ontology
from bo.v3.r4tun_seed import load_r4tun_detection
from bo.v3.run_binary_order_model_search import ORDER, _best_template_score, build_feature_table
from bo.v3.run_binary_order_proxy_search import BRANCH_FEATURES, RING_FEATURES, build_branch_table

RUN_ROOT = REPO / "logs" / "v3_arm_c_reflection_pilot_v1"
ARM_B_SCOREBOARD = REPO / "logs" / "v3_arm_b_proxy_stabilisation_v1" / "arm_b_final_scoreboard.csv"
DIR_STAB_ROOT = REPO / "logs" / "v3_direction_stabilisation_v1"
TEMPLATES_PATH = REPO / "data" / "v3" / "calibration" / "k_rotation_templates.json"

DEFAULT_PILOT_RINGS = [
    "4-7/r308",
    "4-4/r212",
    "4-3/r177",
    "5-4/r227",
    "5-5/r251",
    "4-2/r142",
    "4-4/r215",
]

PARAM_SPECS: dict[str, dict[str, float]] = {
    "radius_max": {"hard_min": 2.00, "hard_max": 4.40, "soft_min": 3.11, "soft_max": 4.14},
    "target_distance_2": {"hard_min": 0.015, "hard_max": 0.060, "soft_min": 0.031, "soft_max": 0.054},
    "outlier_neighbors": {"hard_min": 8.0, "hard_max": 40.0, "soft_min": 14.75, "soft_max": 34.00},
    "target_distance_1": {"hard_min": 0.030, "hard_max": 0.120, "soft_min": 0.051, "soft_max": 0.094},
    "interpolation_window": {"hard_min": 1.0, "hard_max": 15.0, "soft_min": 5.0, "soft_max": 12.0},
}
HIGH_SENSITIVITY = ("radius_max", "target_distance_2")

THRESHOLDS = {
    "g_pre_valid_ratio": 0.0005,
    "g_pre_depth_shape_w": 366.0,
    "g_layout_ring_completeness": 0.8571,
    "g_layout_mask_coverage_pct": 2.2383,
    "k_confidence_min": 0.3333,
    "flip_margin_min": 0.03,
    "min_proxy_gain": 0.005,
}


@dataclass
class BranchEval:
    branch: str
    status: str
    proxy_miou: float | None
    miou_fixed: float | None
    miou_perm: float | None
    ontology: dict[str, Any]
    intrinsics: dict[str, Any]
    errors: list[str]


def _ring_parts(ring_key: str) -> tuple[str, int]:
    tid, rid = ring_key.split("/r", 1)
    return tid, int(rid)


def _ring_seed_dir(ring_key: str) -> Path:
    tid, rid = _ring_parts(ring_key)
    return DIR_STAB_ROOT / tid / f"r{rid}"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _check_required(ring_dir: Path, names: tuple[str, ...]) -> list[str]:
    return [n for n in names if not (ring_dir / n).exists()]


def _stage_input_ring(ring_dir: Path, tunnel_id: str, ring_id: int) -> None:
    candidates = (
        REPO / "data" / "v3" / "panels" / "heldout" / "rings" / f"{tunnel_id.replace('-', '_')}_ring{ring_id}.txt",
        REPO / "data" / "v3" / "panels" / "bo" / "rings" / f"{tunnel_id.replace('-', '_')}_ring{ring_id}.txt",
        REPO / "data" / "rings" / f"{tunnel_id.replace('-', '_')}_ring{ring_id}.txt",
    )
    for cand in candidates:
        if cand.exists():
            dst = ring_dir / f"{tunnel_id}_r{ring_id}.txt"
            if not dst.exists():
                shutil.copy2(cand, dst)
            return
    raise FileNotFoundError(f"input point cloud not found for {tunnel_id}/r{ring_id}")


def _render_pre_params(seed: dict[str, Any], updates: dict[str, float]) -> dict[str, Any]:
    out = dict(seed)
    for k, v in updates.items():
        if k in {"radius_max", "outlier_neighbors", "target_distance_1", "target_distance_2", "interpolation_window"}:
            out[k] = float(v) if k != "interpolation_window" else int(round(v))
    # Keep legacy fields consistent.
    if "interpolation_window" in updates:
        out["interpolation_window"] = int(round(float(updates["interpolation_window"])))
    if "outlier_neighbors" in updates:
        out["outlier_neighbors"] = int(round(float(updates["outlier_neighbors"])))
    td = list(out.get("target_distances", [0.08, 0.04, 0.02]))
    while len(td) < 3:
        td.append(0.02)
    td[0] = float(out.get("target_distance_1", td[0]))
    td[1] = float(out.get("target_distance_2", td[1]))
    out["target_distances"] = sorted([float(td[0]), float(td[1]), float(td[2])], reverse=True)
    return out


def _bound_value(name: str, value: float) -> float:
    spec = PARAM_SPECS[name]
    v = float(max(spec["hard_min"], min(spec["hard_max"], value)))
    if name in {"outlier_neighbors", "interpolation_window"}:
        return float(int(round(v)))
    return v


def _normalized_distance(a: dict[str, float], b: dict[str, float]) -> float:
    s = 0.0
    for k, spec in PARAM_SPECS.items():
        span = max(1e-9, spec["hard_max"] - spec["hard_min"])
        s += abs(float(a[k]) - float(b[k])) / span
    return float(s / len(PARAM_SPECS))


def _mode_for_round(round_index_1based: int) -> str:
    if round_index_1based <= 3:
        return "explore"
    if round_index_1based <= 5:
        return "exploit"
    return "consolidate"


def _propose_parameters(
    *,
    rng: np.random.Generator,
    mode: str,
    current: dict[str, float],
    best: dict[str, float],
    best_guardrails_weak: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    updates = dict(current)
    changed: list[str] = []

    if mode == "explore":
        n_changes = int(rng.integers(2, 4))
        primary = HIGH_SENSITIVITY[int(rng.integers(0, len(HIGH_SENSITIVITY)))]
        change_pool = [primary] + [k for k in PARAM_SPECS if k != primary]
        chosen = []
        for k in change_pool:
            if k not in chosen:
                chosen.append(k)
            if len(chosen) >= n_changes:
                break
        for name in chosen:
            spec = PARAM_SPECS[name]
            span = spec["hard_max"] - spec["hard_min"]
            frac_lo, frac_hi = (0.25, 0.50) if name in HIGH_SENSITIVITY else (0.15, 0.35)
            mag = float(rng.uniform(frac_lo, frac_hi) * span)
            sign = -1.0 if bool(rng.integers(0, 2)) else 1.0
            candidate = _bound_value(name, float(current[name]) + sign * mag)
            if abs(candidate - float(current[name])) > 1e-9:
                updates[name] = candidate
                changed.append(name)
        why = "Wide move to escape local basin and test a distinct failure mode."
    elif mode == "exploit":
        anchor = dict(best)
        for name, spec in PARAM_SPECS.items():
            span = spec["hard_max"] - spec["hard_min"]
            jitter = float(rng.uniform(-0.12, 0.12) * span)
            target = float(anchor[name]) + jitter
            if best_guardrails_weak:
                soft_mid = 0.5 * (spec["soft_min"] + spec["soft_max"])
                target = 0.7 * target + 0.3 * soft_mid
            candidate = _bound_value(name, target)
            if abs(candidate - float(current[name])) > 1e-9:
                updates[name] = candidate
                changed.append(name)
        why = "Narrow around best trajectory while restoring guardrail strength."
    else:
        # Consolidate near best and soft interior.
        for name, spec in PARAM_SPECS.items():
            soft_mid = 0.5 * (spec["soft_min"] + spec["soft_max"])
            target = 0.75 * float(best[name]) + 0.25 * soft_mid
            candidate = _bound_value(name, target)
            if abs(candidate - float(current[name])) > 1e-9:
                updates[name] = candidate
                changed.append(name)
        why = "Conservative variant of best setting to avoid one-iteration spikes."

    if not changed:
        # Force one movement if the random draw produced no net change.
        name = HIGH_SENSITIVITY[int(rng.integers(0, len(HIGH_SENSITIVITY)))]
        spec = PARAM_SPECS[name]
        span = spec["hard_max"] - spec["hard_min"]
        bump = 0.25 * span if mode == "explore" else 0.08 * span
        updates[name] = _bound_value(name, float(current[name]) + bump)
        changed = [name]

    return updates, {
        "mode": mode,
        "changed_parameters": changed,
        "expected_effect": "increase robust intrinsic proxy while preserving structure",
        "risk": "may reduce completeness on sparse regions",
        "why_this_is_not_a_small_local_move": why,
        "stop_or_continue": "continue",
    }


def _run_stage_command(
    *,
    cmd: list[str],
    timeout_sec: float,
    mem_cap_bytes: int | None,
    log_path: Path,
) -> tuple[bool, str]:
    info = _run_subprocess(
        cmd,
        timeout_sec=float(timeout_sec),
        mem_cap_bytes=mem_cap_bytes,
        log_path=log_path,
    )
    if info["timed_out"] or info["oom"] or info["returncode"] != 0:
        return False, str(info.get("failure_detail") or "stage failed")
    return True, ""


def _extract_z_features_for_branch(ring_dir: Path, branch: str, templates: dict[str, dict[str, float]]) -> dict[str, float]:
    final_path = ring_dir / f"final_direction_{branch}.csv"
    if not final_path.exists():
        return {
            "branch_z_score": 0.0,
            "k_z_rel": 0.0,
            "k_intensity_rel": 0.0,
            "k_r_rel": 0.0,
        }
    df = pd.read_csv(final_path, usecols=["pred", "z", "intensity", "r"])
    df = df[df["pred"].notna()].copy()
    if df.empty:
        return {
            "branch_z_score": 0.0,
            "k_z_rel": 0.0,
            "k_intensity_rel": 0.0,
            "k_r_rel": 0.0,
        }
    df["pred"] = df["pred"].astype(int)
    class_to_block = {i + 1: b for i, b in enumerate(ORDER)}
    z_by_block: dict[str, float] = {}
    i_by_block: dict[str, float] = {}
    r_by_block: dict[str, float] = {}
    for cls, blk in class_to_block.items():
        sub = df[df["pred"] == cls]
        z_by_block[blk] = float(sub["z"].mean()) if not sub.empty else 0.0
        i_by_block[blk] = float(sub["intensity"].mean()) if not sub.empty else 0.0
        r_by_block[blk] = float(sub["r"].mean()) if not sub.empty else 0.0
    zvals = list(z_by_block.values())
    ivals = list(i_by_block.values())
    rvals = list(r_by_block.values())
    z_span = max(1e-9, max(zvals) - min(zvals))
    return {
        "branch_z_score": _best_template_score(z_by_block, templates),
        "k_z_rel": float((z_by_block["K"] - min(zvals)) / z_span),
        "k_intensity_rel": float((i_by_block["K"] - float(np.mean(ivals))) / max(1e-9, float(np.std(ivals)))),
        "k_r_rel": float((r_by_block["K"] - float(np.mean(rvals))) / max(1e-9, float(np.std(rvals)))),
    }


def _health_from_intr_onto(intr: dict[str, Any], ontology: dict[str, Any]) -> float:
    bd = ontology.get("breakdown", {}) if isinstance(ontology, dict) else {}
    hard_pass = all(bool((bd.get(k) or {}).get("passed")) for k in ("O_block_set", "O_block_count", "O_no_duplicates"))
    one_k_pass = bool((bd.get("O_one_K_unique") or {}).get("passed"))
    gate = 1.0 if (hard_pass and one_k_pass) else 0.0
    structural = float(ontology.get("structural_score") or 0.0)
    seg_type = 1.0 if bool(intr.get("seg_segment_type_completeness")) else 0.0
    ring_comp = float(intr.get("seg_ring_completeness_avg") or 0.0)
    mask_cov = float(intr.get("seg_mask_coverage_pct") or 0.0) / 100.0
    var_ratio = intr.get("seg_block_size_variance_ratio")
    if var_ratio is None:
        balance = 0.0
    else:
        vr = float(var_ratio)
        balance = 1.0 if 3.0 <= vr <= 20.0 else max(0.0, 1.0 - min(abs(vr - 10.0) / 20.0, 1.0))
    return float(gate * (0.4 * structural + 0.2 * seg_type + 0.15 * ring_comp + 0.1 * mask_cov + 0.15 * balance))


def _width_features_from_ring_dir(ring_dir: Path) -> dict[str, float]:
    bnd = _load_json(ring_dir / "boundaries_per_ring_direction_plus.json", {}) or {}
    if not bnd:
        return {"k_width_rank_norm": 1.0, "k_width_ratio": 0.0, "width_cv": 0.0}
    entries = next(iter(bnd.values()), [])
    if not entries:
        return {"k_width_rank_norm": 1.0, "k_width_ratio": 0.0, "width_cv": 0.0}
    ordered = sorted(entries, key=lambda e: float(e["y"]))
    det = _load_json(ring_dir / "single_ring_detection_meta.json", {}) or {}
    h = float(det.get("image_height") or 0.0)
    if h <= 0:
        h = max(float(e["y"]) for e in ordered) + 1.0
    vals = []
    k_width = None
    n = len(ordered)
    for i, e in enumerate(ordered):
        w = (float(ordered[(i + 1) % n]["y"]) - float(e["y"])) % h
        vals.append(float(w))
        if str(e["block"]) == "K":
            k_width = float(w)
    if k_width is None:
        k_width = vals[0]
    rank_norm = sorted(vals).index(k_width) / max(1, len(vals) - 1)
    nonk = [v for v in vals if v != k_width]
    return {
        "k_width_rank_norm": float(rank_norm),
        "k_width_ratio": float(k_width / max(1e-9, float(np.mean(nonk)) if nonk else 1.0)),
        "width_cv": float(np.std(vals) / max(1e-9, float(np.mean(vals)))),
    }


def _ring_features_for_proxy(ring_dir: Path, plus_intr: dict[str, Any], z_plus: dict[str, float]) -> dict[str, float]:
    det = _load_json(ring_dir / "single_ring_detection_meta.json", {}) or {}
    h = float(det.get("image_height") or 1.0)
    pos = float(det.get("positive_line_count") or 0.0)
    neg = float(det.get("negative_line_count") or 0.0)
    width = _width_features_from_ring_dir(ring_dir)
    return {
        "det_k_confidence": float(det.get("k_confidence") or 0.0),
        "det_pos_count": pos,
        "det_neg_count": neg,
        "det_line_diff_pos_minus_neg": pos - neg,
        "det_abs_line_diff": abs(pos - neg),
        "det_horizontal_count": float(det.get("horizontal_line_count") or 0.0),
        "det_selected_pos_count": float(det.get("selected_positive_count") or 0.0),
        "det_selected_neg_count": float(det.get("selected_negative_count") or 0.0),
        "det_k_y_rel": float(det.get("k_y") or 0.0) / max(1.0, h),
        "pre_valid_ratio": float(plus_intr.get("pre_valid_ratio") or 0.0),
        "pre_empty_row_band_ratio": float(plus_intr.get("pre_empty_row_band_ratio") or 0.0),
        "det_y_coverage_pct": float(plus_intr.get("det_y_coverage_pct") or 0.0),
        "det_min_y_gap_px": float(plus_intr.get("det_min_y_gap_px") or 0.0),
        "det_k_x_spacing_cv": float(plus_intr.get("det_k_x_spacing_cv") or 0.0),
        "seg_mask_coverage_pct": float(plus_intr.get("seg_mask_coverage_pct") or 0.0),
        "seg_ring_completeness_avg": float(plus_intr.get("seg_ring_completeness_avg") or 0.0),
        "seg_k_size_ratio": float(plus_intr.get("seg_k_size_ratio") or 0.0),
        "seg_block_size_variance_ratio": float(plus_intr.get("seg_block_size_variance_ratio") or 0.0),
        "k_width_rank_norm": float(width["k_width_rank_norm"]),
        "k_width_ratio": float(width["k_width_ratio"]),
        "width_cv": float(width["width_cv"]),
        "k_z_rel": float(z_plus["k_z_rel"]),
        "k_intensity_rel": float(z_plus["k_intensity_rel"]),
        "k_r_rel": float(z_plus["k_r_rel"]),
    }


def _branch_feature_vector(
    *,
    branch: str,
    ring_features: dict[str, float],
    z_plus_score: float,
    z_minus_score: float,
    health_plus: float,
    health_minus: float,
) -> dict[str, float]:
    is_minus = int(branch == "minus")
    branch_z = z_minus_score if is_minus else z_plus_score
    opp_z = z_plus_score if is_minus else z_minus_score
    health_adv = (health_minus - health_plus) if is_minus else (health_plus - health_minus)
    row = {
        "is_minus_branch": float(is_minus),
        "branch_z_score": float(branch_z),
        "opponent_z_score": float(opp_z),
        "z_advantage": float(branch_z - opp_z),
        "health_advantage": float(health_adv),
    }
    row.update(ring_features)
    return row


def _compute_guardrails(
    *,
    selected_intr: dict[str, Any],
    selected_ontology: dict[str, Any],
    selected_branch: str,
    arm_b_branch: str,
    proxy_plus: float,
    proxy_minus: float,
    baseline_proxy: float,
) -> dict[str, Any]:
    g_pre = (
        float(selected_intr.get("pre_valid_ratio") or 0.0) >= THRESHOLDS["g_pre_valid_ratio"]
        and float(selected_intr.get("pre_depth_shape_w") or 0.0) >= THRESHOLDS["g_pre_depth_shape_w"]
    )
    g_layout = (
        float(selected_intr.get("seg_ring_completeness_avg") or 0.0) >= THRESHOLDS["g_layout_ring_completeness"]
        and float(selected_intr.get("seg_mask_coverage_pct") or 0.0) >= THRESHOLDS["g_layout_mask_coverage_pct"]
    )
    hard_pass = len(selected_ontology.get("hard_failures") or []) == 0
    k_conf = float((selected_intr.get("det_k_confidence_avg") or 0.0))
    margin = abs(float(proxy_minus) - float(proxy_plus))
    branch_flip = selected_branch != arm_b_branch
    flip_ok = (not branch_flip) or (margin >= THRESHOLDS["flip_margin_min"])
    g_frame = bool(hard_pass and (k_conf >= THRESHOLDS["k_confidence_min"]) and flip_ok)
    selected_proxy = max(float(proxy_plus), float(proxy_minus))
    g_gain = bool(selected_proxy >= float(baseline_proxy) + THRESHOLDS["min_proxy_gain"])
    return {
        "G_pre": bool(g_pre),
        "G_layout": bool(g_layout),
        "G_frame_robustness": bool(g_frame),
        "G_proxy_gain": bool(g_gain),
        "branch_flip": bool(branch_flip),
        "proxy_margin": float(margin),
        "k_confidence_avg": float(k_conf),
    }


def _compute_j_arm_c_proxy(
    *,
    selected_proxy: float,
    ontology: dict[str, Any],
    guardrails: dict[str, Any],
    param_move_norm: float,
) -> dict[str, float]:
    j_reflect = compute_j_reflect_v3(
        ontology_verdict=ontology,
        g_pre_pass=bool(guardrails["G_pre"]),
        g_layout_pass=bool(guardrails["G_layout"]),
        g_stability_pass=bool(guardrails["G_frame_robustness"]),
    )
    direction_reflect = 0.0
    direction_reflect += 0.45 * float(guardrails["G_frame_robustness"])
    direction_reflect += 0.35 * min(1.0, float(guardrails["proxy_margin"]) / 0.1)
    direction_reflect += 0.20 * float(not bool(guardrails["branch_flip"]))
    penalties = 0.0
    penalties += 0.10 * len(ontology.get("hard_failures") or [])
    penalties += 0.08 * float(param_move_norm)
    if bool(guardrails["branch_flip"]) and float(guardrails["proxy_margin"]) < THRESHOLDS["flip_margin_min"]:
        penalties += 0.10
    j_total = float(selected_proxy + 0.20 * j_reflect + 0.12 * direction_reflect + 0.05 * float(guardrails["G_proxy_gain"]) - penalties)
    return {
        "J_reflect_v3": float(j_reflect),
        "J_direction_reflect": float(direction_reflect),
        "J_arm_c_proxy": float(j_total),
        "penalties": float(penalties),
    }


def _write_reflection_packet(
    *,
    packet_path: Path,
    ring_key: str,
    iteration: int,
    mode: str,
    current_params: dict[str, float],
    proposal: dict[str, Any],
    previous_record: dict[str, Any] | None,
) -> None:
    previous_json = json.dumps(previous_record or {}, indent=2, default=str)
    proposal_json = json.dumps(proposal, indent=2, default=str)
    params_json = json.dumps(current_params, indent=2, default=str)
    text = f"""# Arm C Reflection Packet

- ring: `{ring_key}`
- iteration: `{iteration}`
- mode: `{mode}`

## Current Parameters
```json
{params_json}
```

## Previous Iteration Summary
```json
{previous_json}
```

## Instruction
```text
You are not tuning for cosmetic small changes. The current parameters may be trapped in a local basin.
For this iteration, propose a meaningful parameter move within BO hard bounds.

Use only current proxy score, guardrails, artefact status, parameter trajectory, and previous proxy improvement.
Do not use ground truth or mIoU to choose parameters.

For exploration rounds:
- At least one high-sensitivity parameter must move by 25-50% of its BO range.
- You may change up to three critical preprocessing parameters together.
- Explain which failure mode this move tests.

For exploitation rounds:
- Move around the best previous proxy-scoring setting.
- Prefer smaller adjustments and guardrail preservation.
```

## Deterministic Proposal
```json
{proposal_json}
```
"""
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(text, encoding="utf-8")


def _train_proxy_model(model_dir: Path) -> RandomForestRegressor:
    model_dir.mkdir(parents=True, exist_ok=True)
    ring_df = build_feature_table(model_dir)
    branch_df = build_branch_table(ring_df, model_dir)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=4,
        random_state=11,
    )
    model.fit(branch_df[BRANCH_FEATURES].fillna(0.0), branch_df["branch_miou"])
    return model


def _iter_record_to_row(rec: dict[str, Any]) -> dict[str, Any]:
    out = dict(rec)
    for key in ("proposal", "parameters_used", "errors"):
        if key in out:
            out[key] = json.dumps(out[key], default=str)
    return out


def _run_iteration(
    *,
    ring_key: str,
    iter_index: int,
    iter_dir: Path,
    pre_params: dict[str, Any],
    det_params: dict[str, Any],
    model: RandomForestRegressor,
    templates: dict[str, dict[str, float]],
    timeout_sec: float,
    mem_cap_bytes: int | None,
    arm_b_branch: str,
    baseline_proxy: float,
    param_move_norm: float,
) -> dict[str, Any]:
    tid, rid = _ring_parts(ring_key)
    ring_dir = iter_dir / tid / f"r{rid}"
    ring_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "logs").mkdir(parents=True, exist_ok=True)

    _stage_input_ring(ring_dir, tid, rid)
    (ring_dir / "parameters_preprocessing.json").write_text(json.dumps(pre_params, indent=2) + "\n", encoding="utf-8")
    (ring_dir / "parameters_detection.json").write_text(json.dumps(det_params, indent=2) + "\n", encoding="utf-8")
    (ring_dir / "parameters_segmentation.json").write_text(json.dumps({"k_cap": 130, "ab_cap": 390}, indent=2) + "\n", encoding="utf-8")

    errors: list[str] = []
    ok, detail = _run_stage_command(
        cmd=[str(VENV_PYTHON), str(PREPROCESSING_CLI), tid, str(rid), "--data-dir", str(iter_dir)],
        timeout_sec=timeout_sec,
        mem_cap_bytes=mem_cap_bytes,
        log_path=iter_dir / "logs" / "preprocessing.log",
    )
    if not ok:
        errors.append(f"preprocessing: {detail}")
    if not errors:
        miss = _check_required(ring_dir, REQUIRED_PRE_ARTEFACTS)
        if miss:
            errors.append(f"missing preprocessing artefacts: {miss}")

    if not errors:
        ok, detail = _run_stage_command(
            cmd=[str(VENV_PYTHON), str(DETECTION_CLI), tid, str(rid), "--data-dir", str(iter_dir)],
            timeout_sec=timeout_sec,
            mem_cap_bytes=mem_cap_bytes,
            log_path=iter_dir / "logs" / "detection.log",
        )
        if not ok:
            errors.append(f"detection: {detail}")
        miss = _check_required(ring_dir, REQUIRED_DET_ARTEFACTS)
        if miss:
            errors.append(f"missing detection artefacts: {miss}")

    branch_results: dict[str, BranchEval] = {}
    for branch in ("plus", "minus"):
        if errors:
            branch_results[branch] = BranchEval(
                branch=branch,
                status="failed",
                proxy_miou=None,
                miou_fixed=None,
                miou_perm=None,
                ontology={},
                intrinsics={},
                errors=list(errors),
            )
            continue
        seg_csv_name = f"all_segments_direction_{branch}.csv"
        bnd_name = f"boundaries_per_ring_direction_{branch}.json"
        if not (ring_dir / seg_csv_name).exists() or not (ring_dir / bnd_name).exists():
            branch_results[branch] = BranchEval(
                branch=branch,
                status="failed",
                proxy_miou=None,
                miou_fixed=None,
                miou_perm=None,
                ontology={},
                intrinsics={},
                errors=[f"direction artefacts missing for {branch}"],
            )
            continue
        shutil.copy2(ring_dir / bnd_name, ring_dir / "boundaries_per_ring.json")
        ok, detail = _run_stage_command(
            cmd=[
                str(VENV_PYTHON),
                str(SEGMENTATION_CLI),
                tid,
                str(rid),
                "--data-dir",
                str(iter_dir),
                "--segments-file",
                seg_csv_name,
            ],
            timeout_sec=timeout_sec,
            mem_cap_bytes=mem_cap_bytes,
            log_path=iter_dir / "logs" / f"segmentation_{branch}.log",
        )
        if not ok:
            branch_results[branch] = BranchEval(
                branch=branch,
                status="failed",
                proxy_miou=None,
                miou_fixed=None,
                miou_perm=None,
                ontology={},
                intrinsics={},
                errors=[f"segmentation {branch}: {detail}"],
            )
            continue
        miss = _check_required(ring_dir, REQUIRED_SEG_ARTEFACTS)
        if miss:
            branch_results[branch] = BranchEval(
                branch=branch,
                status="failed",
                proxy_miou=None,
                miou_fixed=None,
                miou_perm=None,
                ontology={},
                intrinsics={},
                errors=[f"missing segmentation artefacts {branch}: {miss}"],
            )
            continue

        intr = collect_trial_intrinsics(ring_dir)
        miou_fixed = intr.pop("miou_fixed_class", None)
        miou_perm = intr.pop("miou_permutation", None)
        onto = evaluate_ontology(ring_dir)
        shutil.copy2(ring_dir / "final.csv", ring_dir / f"final_direction_{branch}.csv")
        (ring_dir / f"intrinsics_direction_{branch}.json").write_text(
            json.dumps(intr | {"miou_fixed_class": miou_fixed, "miou_permutation": miou_perm}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (ring_dir / f"ontology_direction_{branch}.json").write_text(
            json.dumps(onto, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        branch_results[branch] = BranchEval(
            branch=branch,
            status="ok",
            proxy_miou=None,
            miou_fixed=miou_fixed,
            miou_perm=miou_perm,
            ontology=onto,
            intrinsics=intr,
            errors=[],
        )

    if branch_results["plus"].status != "ok" or branch_results["minus"].status != "ok":
        failure_errs = list(errors)
        failure_errs.extend(branch_results["plus"].errors)
        failure_errs.extend(branch_results["minus"].errors)
        return {
            "iteration": int(iter_index),
            "status": "failed",
            "selected_branch": arm_b_branch,
            "proxy_plus_miou": None,
            "proxy_minus_miou": None,
            "selected_proxy_miou": None,
            "selected_gt_miou": None,
            "G_pre": False,
            "G_layout": False,
            "G_frame_robustness": False,
            "G_proxy_gain": False,
            "J_reflect_v3": 0.0,
            "J_direction_reflect": 0.0,
            "J_arm_c_proxy": -1e9,
            "penalties": 1.0,
            "branch_flip": False,
            "proxy_margin": 0.0,
            "k_confidence_avg": 0.0,
            "hard_failures_count": 99,
            "param_move_norm": float(param_move_norm),
            "errors": failure_errs,
        }

    z_plus = _extract_z_features_for_branch(ring_dir, "plus", templates)
    z_minus = _extract_z_features_for_branch(ring_dir, "minus", templates)
    health_plus = _health_from_intr_onto(branch_results["plus"].intrinsics, branch_results["plus"].ontology)
    health_minus = _health_from_intr_onto(branch_results["minus"].intrinsics, branch_results["minus"].ontology)
    ring_features = _ring_features_for_proxy(ring_dir, branch_results["plus"].intrinsics, z_plus)

    row_plus = _branch_feature_vector(
        branch="plus",
        ring_features=ring_features,
        z_plus_score=z_plus["branch_z_score"],
        z_minus_score=z_minus["branch_z_score"],
        health_plus=health_plus,
        health_minus=health_minus,
    )
    row_minus = _branch_feature_vector(
        branch="minus",
        ring_features=ring_features,
        z_plus_score=z_plus["branch_z_score"],
        z_minus_score=z_minus["branch_z_score"],
        health_plus=health_plus,
        health_minus=health_minus,
    )
    proxy_plus = float(np.clip(model.predict(pd.DataFrame([row_plus])[BRANCH_FEATURES].fillna(0.0))[0], 0.0, 1.0))
    proxy_minus = float(np.clip(model.predict(pd.DataFrame([row_minus])[BRANCH_FEATURES].fillna(0.0))[0], 0.0, 1.0))
    selected_branch = "minus" if proxy_minus > proxy_plus else "plus"
    selected_eval = branch_results[selected_branch]
    guardrails = _compute_guardrails(
        selected_intr=selected_eval.intrinsics,
        selected_ontology=selected_eval.ontology,
        selected_branch=selected_branch,
        arm_b_branch=arm_b_branch,
        proxy_plus=proxy_plus,
        proxy_minus=proxy_minus,
        baseline_proxy=baseline_proxy,
    )
    selected_proxy = max(proxy_plus, proxy_minus)
    j_terms = _compute_j_arm_c_proxy(
        selected_proxy=selected_proxy,
        ontology=selected_eval.ontology,
        guardrails=guardrails,
        param_move_norm=param_move_norm,
    )
    return {
        "iteration": int(iter_index),
        "status": "ok",
        "selected_branch": selected_branch,
        "proxy_plus_miou": proxy_plus,
        "proxy_minus_miou": proxy_minus,
        "selected_proxy_miou": selected_proxy,
        "selected_gt_miou": selected_eval.miou_fixed,
        "plus_gt_miou": branch_results["plus"].miou_fixed,
        "minus_gt_miou": branch_results["minus"].miou_fixed,
        "G_pre": bool(guardrails["G_pre"]),
        "G_layout": bool(guardrails["G_layout"]),
        "G_frame_robustness": bool(guardrails["G_frame_robustness"]),
        "G_proxy_gain": bool(guardrails["G_proxy_gain"]),
        "J_reflect_v3": float(j_terms["J_reflect_v3"]),
        "J_direction_reflect": float(j_terms["J_direction_reflect"]),
        "J_arm_c_proxy": float(j_terms["J_arm_c_proxy"]),
        "penalties": float(j_terms["penalties"]),
        "branch_flip": bool(guardrails["branch_flip"]),
        "proxy_margin": float(guardrails["proxy_margin"]),
        "k_confidence_avg": float(guardrails["k_confidence_avg"]),
        "hard_failures_count": int(len(selected_eval.ontology.get("hard_failures") or [])),
        "param_move_norm": float(param_move_norm),
        "errors": [],
    }


def _prepare_seed_params(ring_key: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    seed_dir = _ring_seed_dir(ring_key)
    pre = _load_json(seed_dir / "parameters_preprocessing.json", {})
    det = _load_json(seed_dir / "parameters_detection.json", {})
    if not pre:
        raise RuntimeError(f"missing Arm B preprocessing seed params for {ring_key}")
    if not det:
        tid, _ = _ring_parts(ring_key)
        diameter = 7.5
        pre_candidates = REPO / "agents" / "1_preprocessing" / "parameters" / tid
        if pre_candidates.exists():
            diameter = float(pre.get("tunnel_diameter", 7.5))
        det = load_r4tun_detection(target_tunnel_diameter=diameter)
    critical = {
        "radius_max": float(pre.get("radius_max", 4.0)),
        "target_distance_2": float(pre.get("target_distance_2", 0.04)),
        "outlier_neighbors": float(pre.get("outlier_neighbors", 20.0)),
        "target_distance_1": float(pre.get("target_distance_1", 0.08)),
        "interpolation_window": float(pre.get("interpolation_window", 9.0)),
    }
    for k in critical:
        critical[k] = _bound_value(k, critical[k])
    return pre, det, critical


def _pilot_decision(summary_df: pd.DataFrame) -> dict[str, Any]:
    if summary_df.empty:
        return {"ready_for_rollout": False, "reason": "no rings executed"}
    agg_lift = float(summary_df["lift_gt_vs_baseline"].mean())
    high_risk = summary_df[summary_df["ring_group"] == "high_risk"]
    high_risk_ok = bool((high_risk["lift_gt_vs_baseline"] >= -0.01).all()) if not high_risk.empty else True
    align_rate = float((summary_df["best_iter_proxy_rank_matches_gt_rank"]).mean())
    guardrail_nonworse = bool((summary_df["guardrail_failure_delta"] <= 0).mean() >= 0.7)
    ready = bool(agg_lift > 0.0 and high_risk_ok and align_rate >= 0.5 and guardrail_nonworse)
    return {
        "ready_for_rollout": ready,
        "aggregate_gt_lift_vs_arm_b_baseline": agg_lift,
        "high_risk_non_damage": high_risk_ok,
        "proxy_gt_alignment_rate": align_rate,
        "guardrail_nonworse_ratio": float((summary_df["guardrail_failure_delta"] <= 0).mean()),
        "decision": "proceed pilot->broader rollout" if ready else "refine proxy/guardrails before 40-ring rollout",
    }


def _write_report(
    *,
    out_root: Path,
    summary_df: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    lines = []
    lines.append("# Arm C Reflection Pilot Report")
    lines.append("")
    lines.append("## Aggregate")
    lines.append(f"- rings: {len(summary_df)}")
    lines.append(f"- mean Arm B baseline GT mIoU: {summary_df['baseline_gt_miou'].mean():.4f}")
    lines.append(f"- mean best Arm C GT mIoU: {summary_df['best_gt_miou'].mean():.4f}")
    lines.append(f"- mean GT lift vs Arm B baseline: {summary_df['lift_gt_vs_baseline'].mean():+.4f}")
    lines.append(f"- proxy/GT best-iteration alignment rate: {summary_df['best_iter_proxy_rank_matches_gt_rank'].mean():.3f}")
    lines.append("")
    lines.append("## Ring-Level")
    for _, r in summary_df.sort_values("ring_key").iterrows():
        lines.append(
            f"- {r['ring_key']}: baseline={r['baseline_gt_miou']:.4f}, "
            f"best={r['best_gt_miou']:.4f}, lift={r['lift_gt_vs_baseline']:+.4f}, "
            f"best_iter={int(r['best_iteration'])}, group={r['ring_group']}"
        )
    lines.append("")
    lines.append("## Limitations")
    lines.append("- Proxy score is a learned estimate and can mis-rank iterations on some rings.")
    lines.append("- Pilot evidence is limited to a small subset; generalization is suggestive, not conclusive.")
    lines.append("- Arm C currently tunes preprocessing only; detection/segmentation adaptation is out of scope.")
    lines.append("")
    lines.append("## Confidence")
    lines.append("- Medium confidence in per-ring qualitative behavior (explore/exploit schedule is auditable).")
    lines.append("- Low-to-medium confidence for full 40-ring rollout until proxy/guardrail agreement improves.")
    lines.append("")
    lines.append("## Rollout Decision")
    lines.append(f"- ready_for_rollout: {decision['ready_for_rollout']}")
    lines.append(f"- decision: {decision['decision']}")
    (out_root / "arm_c_pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Arm C reflection pilot with GT-free proxy/guardrail selection")
    p.add_argument("--run-root", default=str(RUN_ROOT))
    p.add_argument("--rings", nargs="*", default=DEFAULT_PILOT_RINGS)
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--stagnation-rounds", type=int, default=3)
    p.add_argument("--min-meaningful-improvement", type=float, default=0.005)
    p.add_argument("--timeout-sec", type=float, default=900.0)
    p.add_argument("--mem-cap-gb", type=float, default=16.0)
    p.add_argument("--fresh", action="store_true")
    ns = p.parse_args(argv)

    out_root = assert_writable(Path(ns.run_root).resolve())
    out_root.mkdir(parents=True, exist_ok=True)
    model = _train_proxy_model(out_root / "proxy_model")
    arm_b_df = pd.read_csv(ARM_B_SCOREBOARD)
    arm_b_map = {
        row["ring_key"]: {
            "selected_order": str(row["selected_order"]),
            "selected_proxy_miou": float(row["selected_proxy_miou"]),
            "selected_gt_miou": float(row["selected_proxy_miou"]),
        }
        for _, row in arm_b_df.iterrows()
    }
    templates = (_load_json(TEMPLATES_PATH, {}) or {}).get("templates", {})
    mem_cap_bytes = int(float(ns.mem_cap_gb) * (1024**3)) if float(ns.mem_cap_gb) > 0 else None

    ring_summary_rows: list[dict[str, Any]] = []
    pilot_cfg = {
        "rings": list(ns.rings),
        "max_rounds": int(ns.max_rounds),
        "stagnation_rounds": int(ns.stagnation_rounds),
        "min_meaningful_improvement": float(ns.min_meaningful_improvement),
        "thresholds": THRESHOLDS,
        "param_specs": PARAM_SPECS,
        "sandbox_path": str(out_root),
    }
    (out_root / "pilot_config.json").write_text(json.dumps(pilot_cfg, indent=2) + "\n", encoding="utf-8")

    for ring_key in ns.rings:
        tid, rid = _ring_parts(ring_key)
        ring_root = out_root / tid / f"r{rid}"
        assert_writable(ring_root)
        if ring_root.exists() and ns.fresh:
            shutil.rmtree(ring_root)
        ring_root.mkdir(parents=True, exist_ok=True)
        (ring_root / "reflection_packets").mkdir(parents=True, exist_ok=True)

        pre_seed, det_seed, current_critical = _prepare_seed_params(ring_key)
        arm_b = arm_b_map.get(ring_key, {"selected_order": "plus", "selected_proxy_miou": 0.0})
        arm_b_branch = str(arm_b["selected_order"])

        # Iteration loop: iter_0 is baseline re-run, iter_1..iter_6 use reflection schedule.
        records: list[dict[str, Any]] = []
        best_j = -1e9
        best_idx = -1
        best_critical = dict(current_critical)
        stagnation = 0

        for iter_idx in range(0, int(ns.max_rounds) + 1):
            mode = "baseline" if iter_idx == 0 else _mode_for_round(iter_idx)
            prev = records[-1] if records else None
            if iter_idx == 0:
                proposal_meta = {
                    "mode": "baseline",
                    "parameter_updates": {},
                    "expected_effect": "establish Arm B baseline in Arm C sandbox",
                    "risk": "none",
                    "why_this_is_not_a_small_local_move": "baseline anchor run",
                    "stop_or_continue": "continue",
                }
                proposal_critical = dict(current_critical)
            else:
                rng = np.random.default_rng(abs(hash((ring_key, iter_idx))) % (2**32))
                best_guardrails_weak = not bool(records[best_idx]["G_pre"] and records[best_idx]["G_layout"] and records[best_idx]["G_frame_robustness"]) if best_idx >= 0 else False
                proposal_critical, proposal_meta = _propose_parameters(
                    rng=rng,
                    mode=mode,
                    current=current_critical,
                    best=best_critical,
                    best_guardrails_weak=best_guardrails_weak,
                )
            packet = {
                "mode": mode if iter_idx > 0 else "baseline",
                "parameter_updates": {
                    k: float(v) for k, v in proposal_critical.items() if abs(float(v) - float(current_critical[k])) > 1e-9
                },
                "expected_effect": proposal_meta["expected_effect"],
                "risk": proposal_meta["risk"],
                "why_this_is_not_a_small_local_move": proposal_meta["why_this_is_not_a_small_local_move"],
                "stop_or_continue": proposal_meta["stop_or_continue"],
            }
            _write_reflection_packet(
                packet_path=ring_root / "reflection_packets" / f"iter_{iter_idx}.md",
                ring_key=ring_key,
                iteration=iter_idx,
                mode=mode,
                current_params=current_critical,
                proposal=packet,
                previous_record=prev,
            )

            iter_dir = ring_root / f"iter_{iter_idx}"
            if iter_dir.exists() and ns.fresh:
                shutil.rmtree(iter_dir)
            iter_dir.mkdir(parents=True, exist_ok=True)
            pre_params = _render_pre_params(pre_seed, proposal_critical)
            move_norm = _normalized_distance(proposal_critical, current_critical if iter_idx > 0 else proposal_critical)
            rec = _run_iteration(
                ring_key=ring_key,
                iter_index=iter_idx,
                iter_dir=iter_dir,
                pre_params=pre_params,
                det_params=det_seed,
                model=model,
                templates=templates,
                timeout_sec=float(ns.timeout_sec),
                mem_cap_bytes=mem_cap_bytes,
                arm_b_branch=arm_b_branch,
                baseline_proxy=float(arm_b["selected_proxy_miou"]),
                param_move_norm=move_norm,
            )
            rec["proposal"] = packet
            rec["parameters_used"] = {k: float(v) for k, v in proposal_critical.items()}
            records.append(rec)
            current_critical = dict(proposal_critical)

            if rec["status"] == "ok" and float(rec["J_arm_c_proxy"]) > best_j:
                if float(rec["J_arm_c_proxy"]) >= best_j + float(ns.min_meaningful_improvement):
                    stagnation = 0
                else:
                    stagnation += 1
                best_j = float(rec["J_arm_c_proxy"])
                best_idx = iter_idx
                best_critical = dict(proposal_critical)
            else:
                stagnation += 1

            if iter_idx >= 1 and stagnation >= int(ns.stagnation_rounds):
                break

        # Persist per-ring scoreboard.
        if records:
            fields = list(_iter_record_to_row(records[0]).keys())
            with (ring_root / "scoreboard.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for rec in records:
                    w.writerow(_iter_record_to_row(rec))

        ok_records = [r for r in records if r["status"] == "ok"]
        baseline = next((r for r in records if r["iteration"] == 0), None)
        best = records[best_idx] if (best_idx >= 0 and best_idx < len(records)) else baseline
        if baseline is None or best is None:
            raise RuntimeError(f"failed to produce baseline/best records for {ring_key}")

        if ok_records:
            proxy_sorted = sorted(ok_records, key=lambda r: float(r["J_arm_c_proxy"]), reverse=True)
            gt_sorted = sorted(ok_records, key=lambda r: float(r.get("selected_gt_miou") or -1e9), reverse=True)
            align = int(proxy_sorted[0]["iteration"] == gt_sorted[0]["iteration"])
            guardrail_fail_baseline = int(not (baseline["G_pre"] and baseline["G_layout"] and baseline["G_frame_robustness"]))
            guardrail_fail_best = int(not (best["G_pre"] and best["G_layout"] and best["G_frame_robustness"]))
        else:
            align = 0
            guardrail_fail_baseline = 1
            guardrail_fail_best = 1

        ring_group = "neutral"
        if ring_key in {"4-7/r308", "4-4/r212", "4-3/r177"}:
            ring_group = "high_risk"
        elif ring_key in {"5-4/r227", "5-5/r251", "4-2/r142"}:
            ring_group = "proxy_helped"

        ring_summary_rows.append(
            {
                "ring_key": ring_key,
                "ring_group": ring_group,
                "baseline_iteration": int(baseline["iteration"]),
                "best_iteration": int(best["iteration"]),
                "baseline_branch": str(baseline["selected_branch"]),
                "best_branch": str(best["selected_branch"]),
                "baseline_proxy_miou": float(baseline["selected_proxy_miou"] or 0.0),
                "best_proxy_miou": float(best["selected_proxy_miou"] or 0.0),
                "baseline_gt_miou": float(baseline["selected_gt_miou"] or 0.0),
                "best_gt_miou": float(best["selected_gt_miou"] or 0.0),
                "lift_gt_vs_baseline": float((best["selected_gt_miou"] or 0.0) - (baseline["selected_gt_miou"] or 0.0)),
                "proxy_lift_vs_baseline": float((best["selected_proxy_miou"] or 0.0) - (baseline["selected_proxy_miou"] or 0.0)),
                "best_iter_proxy_rank_matches_gt_rank": int(align),
                "guardrail_failure_delta": int(guardrail_fail_best - guardrail_fail_baseline),
                "n_iterations_executed": int(len(records)),
            }
        )

    summary_df = pd.DataFrame(ring_summary_rows)
    summary_df.sort_values("ring_key").to_csv(out_root / "arm_c_pilot_summary.csv", index=False)
    decision = _pilot_decision(summary_df)
    summary_json = {
        "n_rings": int(len(summary_df)),
        "mean_baseline_gt_miou": float(summary_df["baseline_gt_miou"].mean()) if not summary_df.empty else None,
        "mean_best_gt_miou": float(summary_df["best_gt_miou"].mean()) if not summary_df.empty else None,
        "mean_gt_lift_vs_baseline": float(summary_df["lift_gt_vs_baseline"].mean()) if not summary_df.empty else None,
        "mean_proxy_lift_vs_baseline": float(summary_df["proxy_lift_vs_baseline"].mean()) if not summary_df.empty else None,
        "decision": decision,
    }
    (out_root / "arm_c_pilot_summary.json").write_text(json.dumps(summary_json, indent=2) + "\n", encoding="utf-8")
    _write_report(out_root=out_root, summary_df=summary_df, decision=decision)
    print(json.dumps(summary_json, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
