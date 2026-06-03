"""Composite Stage A selection: intrinsic proxy + structural guardrails."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.ceiling_gate import REPO_ROOT
from lib.guardrail_utils import guardrail_passed
from lib.held_out_common import RELATIVE_MARGIN
from lib.proxy_a3_v5 import load_a3_slim_model, load_p11_model, predict_proxy
from lib.relative_proxy_train import load_rel_v2_model, predict_delta
from lib.v5_relative_proxy import RELATIVE_FEATURES, FEATURE_DIRECTION

# Interpretable guardrail weights (non-learned).
W_RHO = 0.05
W_GUARDRAIL = 0.25
W_LINE_ANCHOR = 0.10
W_FAILURE = 0.15
FAILURE_K_SCALE = 0.08


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if np.isfinite(f) else default


def load_failure_tables(experience_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fail = pd.read_csv(experience_root / "failure_memory_random.csv")
    rules = pd.read_csv(experience_root / "failure_memory_random_rules.csv")
    return fail, rules


def failure_similarity_penalty(
    k_center_norm: float,
    *,
    nearest_calib_ring: str,
    failures: pd.DataFrame,
    rules: pd.DataFrame,
    coverage_pct: float | None = None,
    k_confidence: float | None = None,
) -> float:
    """Penalty for proximity to v3 random failure exemplars."""
    sub = failures[failures["ring_id"] == nearest_calib_ring]
    if sub.empty:
        sub = failures
    if sub.empty:
        return 0.0

    k_dists = (sub["layout_k_center_norm"].astype(float) - float(k_center_norm)).abs()
    min_dist = float(k_dists.min()) if len(k_dists) else 1.0
    sim = float(np.exp(-min_dist / FAILURE_K_SCALE))
    penalty = W_FAILURE * sim

    rules_row = rules[rules["ring_id"] == nearest_calib_ring]
    if not rules_row.empty:
        rr = rules_row.iloc[0]
        lo = _safe_float(rr.get("reject_k_center_norm_lo"), float("nan"))
        hi = _safe_float(rr.get("reject_k_center_norm_hi"), float("nan"))
        if np.isfinite(lo) and np.isfinite(hi) and lo <= k_center_norm <= hi:
            penalty += W_FAILURE * 0.5
        cov_min = _safe_float(rr.get("penalise_coverage_min"), float("nan"))
        miou_max = _safe_float(rr.get("penalise_miou_max"), float("nan"))
        k_conf_max = _safe_float(rr.get("penalise_k_confidence_max"), float("nan"))
        if (
            coverage_pct is not None
            and k_confidence is not None
            and np.isfinite(cov_min)
            and float(coverage_pct) >= cov_min
            and float(k_confidence) <= k_conf_max
        ):
            penalty += W_FAILURE * 0.25

    return float(penalty)


def anchor_plausibility_bonus(
    *,
    rho_k: float,
    rho_ab: float,
    det_guardrail_passed: bool,
    valid_line_anchor: bool,
) -> float:
    bonus = W_RHO * (float(rho_k) + float(rho_ab)) / 2.0
    if det_guardrail_passed:
        bonus += W_GUARDRAIL
    if valid_line_anchor:
        bonus += W_LINE_ANCHOR
    return float(bonus)


def baseline_regression_risk(proxy_score: float, proxy_c0: float, *, margin: float = RELATIVE_MARGIN) -> float:
    return float(max(0.0, proxy_c0 - proxy_score + margin))


def score_candidate_row(
    row: pd.Series,
    *,
    proxy_c0: float,
    nearest_calib_ring: str,
    failures: pd.DataFrame,
    rules: pd.DataFrame,
    rho_k: float,
    rho_ab: float,
    valid_line_anchor: bool,
    model: dict[str, Any],
) -> dict[str, Any]:
    proxy_score = predict_proxy(model, row)
    bonus = anchor_plausibility_bonus(
        rho_k=rho_k,
        rho_ab=rho_ab,
        det_guardrail_passed=guardrail_passed(row),
        valid_line_anchor=valid_line_anchor,
    )
    penalty = failure_similarity_penalty(
        _safe_float(row.get("layout_k_center_norm")),
        nearest_calib_ring=nearest_calib_ring,
        failures=failures,
        rules=rules,
        coverage_pct=_safe_float(row.get("det_y_coverage_pct"), float("nan")),
        k_confidence=_safe_float(row.get("det_k_confidence_avg"), float("nan")),
    )
    risk = baseline_regression_risk(proxy_score, proxy_c0)
    final_score = proxy_score + bonus - penalty - risk
    return {
        "proxy_score": proxy_score,
        "anchor_plausibility_bonus": bonus,
        "failure_penalty": penalty,
        "baseline_regression_risk": risk,
        "final_score": final_score,
    }


def select_from_pool(
    df: pd.DataFrame,
    *,
    model: dict[str, Any],
    variant: str,
    nearest_calib_ring: str,
    failures: pd.DataFrame,
    rules: pd.DataFrame,
    rho_k: float,
    rho_ab: float,
    valid_line_anchor: bool,
    margin: float = RELATIVE_MARGIN,
) -> dict[str, Any]:
    """Pick best candidate or abstain to C0."""
    if df.empty:
        raise ValueError("empty candidate pool")

    c0_rows = df[df["candidate_id"] == 0]
    if c0_rows.empty:
        c0_row = df.iloc[0]
    else:
        c0_row = c0_rows.iloc[0]

    scored_rows: list[dict[str, Any]] = []
    proxy_c0 = predict_proxy(model, c0_row)
    c0_terms = score_candidate_row(
        c0_row,
        proxy_c0=proxy_c0,
        nearest_calib_ring=nearest_calib_ring,
        failures=failures,
        rules=rules,
        rho_k=rho_k,
        rho_ab=rho_ab,
        valid_line_anchor=valid_line_anchor,
        model=model,
    )
    c0_final = c0_terms["final_score"]

    for _, row in df.iterrows():
        rec = row.to_dict()
        terms = score_candidate_row(
            row,
            proxy_c0=proxy_c0,
            nearest_calib_ring=nearest_calib_ring,
            failures=failures,
            rules=rules,
            rho_k=rho_k,
            rho_ab=rho_ab,
            valid_line_anchor=valid_line_anchor,
            model=model,
        )
        rec.update(terms)
        rec["variant"] = variant
        scored_rows.append(rec)

    scored_df = pd.DataFrame(scored_rows)
    best_idx = scored_df["final_score"].idxmax()
    best = scored_df.loc[best_idx]
    if int(best["candidate_id"]) == 0:
        abstained = False
    else:
        abstained = bool(
            float(best["final_score"]) <= c0_final + margin
            or not guardrail_passed(best)
        )
    if abstained:
        selected = scored_df[scored_df["candidate_id"] == 0].iloc[0]
    else:
        selected = best

    oracle_idx = scored_df["gt_miou"].idxmax()
    oracle = scored_df.loc[oracle_idx]

    return {
        "variant": variant,
        "selected_candidate_id": int(selected["candidate_id"]),
        "abstained_to_c0": abstained,
        "selected_gt_miou": float(selected["gt_miou"]),
        "c0_gt_miou": float(c0_row["gt_miou"]),
        "oracle_gt_miou": float(oracle["gt_miou"]),
        "oracle_candidate_id": int(oracle["candidate_id"]),
        "regret_vs_oracle": float(oracle["gt_miou"]) - float(selected["gt_miou"]),
        "lift_vs_c0": float(selected["gt_miou"]) - float(c0_row["gt_miou"]),
        "selected_final_score": float(selected["final_score"]),
        "c0_final_score": float(c0_final),
        "selected_proxy_score": float(selected["proxy_score"]),
        "scored_df": scored_df,
    }


def _directed_delta(col: str, delta: float) -> float:
    direction = FEATURE_DIRECTION.get(col, 1)
    if direction == 0:
        return -abs(delta)
    return direction * delta


def compute_row_deltas(c0: pd.Series, row: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    rel_sum = 0.0
    n = 0
    for col in RELATIVE_FEATURES:
        try:
            v = float(row.get(col, 0.0) or 0.0)
            v0 = float(c0.get(col, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        d = _directed_delta(col, v - v0)
        out[f"rel_{col}"] = d
        rel_sum += d
        n += 1
    out["rel_feature_mean"] = rel_sum / max(n, 1)
    return out


def select_from_pool_rel_v2(
    df: pd.DataFrame,
    *,
    model: dict[str, Any],
    nearest_calib_ring: str,
    failures: pd.DataFrame,
    rules: pd.DataFrame,
    rho_k: float,
    rho_ab: float,
    valid_line_anchor: bool,
    margin: float | None = None,
) -> dict[str, Any]:
    margin = RELATIVE_MARGIN if margin is None else margin
    c0_row = df[df["candidate_id"] == 0].iloc[0] if (df["candidate_id"] == 0).any() else df.iloc[0]

    scored_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        deltas = compute_row_deltas(c0_row, row)
        rec.update(deltas)
        pred_delta = predict_delta(model, pd.Series(rec))
        bonus = anchor_plausibility_bonus(
            rho_k=rho_k,
            rho_ab=rho_ab,
            det_guardrail_passed=guardrail_passed(row),
            valid_line_anchor=valid_line_anchor,
        )
        penalty = failure_similarity_penalty(
            _safe_float(row.get("layout_k_center_norm")),
            nearest_calib_ring=nearest_calib_ring,
            failures=failures,
            rules=rules,
            coverage_pct=_safe_float(row.get("det_y_coverage_pct"), float("nan")),
            k_confidence=_safe_float(row.get("det_k_confidence_avg"), float("nan")),
        )
        final_score = pred_delta + bonus - penalty
        rec.update({
            "pred_delta_miou": pred_delta,
            "proxy_score": pred_delta,
            "anchor_plausibility_bonus": bonus,
            "failure_penalty": penalty,
            "baseline_regression_risk": 0.0,
            "final_score": final_score,
            "variant": "rel_v2",
        })
        scored_rows.append(rec)

    scored_df = pd.DataFrame(scored_rows)
    best_idx = scored_df["final_score"].idxmax()
    best = scored_df.loc[best_idx]
    c0_delta = float(scored_df[scored_df["candidate_id"] == c0_row["candidate_id"]]["pred_delta_miou"].iloc[0])

    if int(best["candidate_id"]) == 0:
        abstained = False
    else:
        abstained = bool(
            float(best["pred_delta_miou"]) <= margin
            or not guardrail_passed(best)
        )
    selected = scored_df[scored_df["candidate_id"] == 0].iloc[0] if abstained else best
    oracle = scored_df.loc[scored_df["gt_miou"].idxmax()]

    return {
        "variant": "rel_v2",
        "selected_candidate_id": int(selected["candidate_id"]),
        "abstained_to_c0": abstained,
        "selected_gt_miou": float(selected["gt_miou"]),
        "c0_gt_miou": float(c0_row["gt_miou"]),
        "oracle_gt_miou": float(oracle["gt_miou"]),
        "oracle_candidate_id": int(oracle["candidate_id"]),
        "regret_vs_oracle": float(oracle["gt_miou"]) - float(selected["gt_miou"]),
        "lift_vs_c0": float(selected["gt_miou"]) - float(c0_row["gt_miou"]),
        "selected_final_score": float(selected["final_score"]),
        "c0_final_score": float(scored_df[scored_df["candidate_id"] == c0_row["candidate_id"]]["final_score"].iloc[0]),
        "selected_proxy_score": float(selected["pred_delta_miou"]),
        "c0_pred_delta": c0_delta,
        "scored_df": scored_df,
    }


def default_models(include_rel_v2: bool = True) -> dict[str, dict[str, Any]]:
    models = {"p11": load_p11_model(), "a3_slim": load_a3_slim_model()}
    if include_rel_v2:
        manifest = REPO_ROOT / "logs" / "bo_proxy_v2_v1" / "PROXY_REL_V2_MANIFEST.json"
        if manifest.is_file():
            models["rel_v2"] = load_rel_v2_model(manifest)
    return models
