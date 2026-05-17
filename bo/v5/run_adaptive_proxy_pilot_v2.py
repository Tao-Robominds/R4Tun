from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "logs" / "v5_adaptive_proxy_pilot_v1"
RUN_ROOT = REPO_ROOT / "logs" / "v5_proxy_improvement_v1"
FORMULA_JSON = RUN_ROOT / "proxy_formula.json"

DEPTH_RISK_CONTROLS = {"4-6/r276", "5-6/r285"}
COARSE_ANCHORS = {0.10, 0.26, 0.42, 0.58, 0.74, 0.90}
EXPAND_ANCHORS = {round(x, 3) for x in np.linspace(0.04, 0.96, 12).tolist()}


def _dist_circ(a: float, b: float) -> float:
    d = abs(float(a) - float(b))
    return float(min(d, 1.0 - d))


def _ensure_feature_columns(cands: pd.DataFrame) -> pd.DataFrame:
    out = cands.copy()
    out["branch_is_minus"] = out["branch"].astype(str).str.lower().eq("minus").astype(float)
    out["rotation_shift_num"] = pd.to_numeric(out["rotation_shift"], errors="coerce")
    return out


def _score_proxy(cands: pd.DataFrame, formula: dict[str, Any]) -> pd.Series:
    vals = np.full(len(cands), float(formula["intercept"]), dtype=float)
    for feat, coef in zip(formula["features"], formula["coefficients"]):
        if feat in cands.columns:
            s = pd.to_numeric(cands[feat], errors="coerce")
        else:
            s = pd.Series([np.nan] * len(cands), index=cands.index, dtype=float)
        med = float(s.median()) if s.notna().any() else 0.0
        vals += float(coef) * s.fillna(med).to_numpy(dtype=float)
    return pd.Series(vals, index=cands.index)


def _choose_action(cands: pd.DataFrame, depth_gate_pass: bool) -> str:
    ranked = cands.sort_values(["proxy_new", "det_tag", "branch", "rotation_shift"], ascending=[False, True, True, True]).reset_index(drop=True)
    top = float(ranked.iloc[0]["proxy_new"])
    second = float(ranked.iloc[1]["proxy_new"]) if len(ranked) > 1 else top
    margin = top - second
    if not bool(depth_gate_pass):
        return "flag_unstable"
    if margin >= 0.03 and top >= 0.18:
        return "refine_top1"
    if margin < 0.01 or top < 0.12:
        return "expand_global"
    return "refine_top3"


def _subset_by_action(ring_cands: pd.DataFrame, action: str) -> pd.DataFrame:
    if action == "expand_global":
        sub = ring_cands[ring_cands["anchor_frac"].round(3).isin(EXPAND_ANCHORS)].copy()
        return sub if not sub.empty else ring_cands.copy()
    ranked = ring_cands.sort_values(["proxy_new", "det_tag", "branch", "rotation_shift"], ascending=[False, True, True, True]).reset_index(drop=True)
    if action == "refine_top1":
        a0 = float(ranked.iloc[0]["anchor_frac"])
        sub = ring_cands[ring_cands["anchor_frac"].map(lambda x: _dist_circ(float(x), a0) <= 0.021)].copy()
        return sub if not sub.empty else ring_cands.copy()
    # refine_top3 and flag_unstable: neighborhood around top-3 unique anchors.
    anchors = ranked["anchor_frac"].drop_duplicates().head(3).astype(float).tolist()
    sub = ring_cands[ring_cands["anchor_frac"].map(lambda x: min(_dist_circ(float(x), a) for a in anchors) <= 0.016)].copy()
    return sub if not sub.empty else ring_cands.copy()


def _failure_mode(ring_row: pd.Series, ring_cands: pd.DataFrame, selected: pd.Series, action: str) -> tuple[str, bool]:
    ranked = ring_cands.sort_values(["proxy_new", "det_tag", "branch", "rotation_shift"], ascending=[False, True, True, True]).reset_index(drop=True)
    top = float(ranked.iloc[0]["proxy_new"])
    second = float(ranked.iloc[1]["proxy_new"]) if len(ranked) > 1 else top
    margin = top - second
    cats: list[str] = []
    if not bool(ring_row["depth_pass"]):
        cats.append("depth_quality_failure")
    if margin < 0.01:
        cats.append("weak_proxy_margin")
    if float(selected.get("struct_missing_ids_before_n", 0.0)) > 0:
        cats.append("structural_incompleteness")
    if float(selected.get("geom_boundary_gap_cv", 0.0)) > 0.35:
        cats.append("boundary_spacing_ambiguity")
    top5 = ranked.head(min(5, len(ranked)))
    if top5["branch"].nunique() > 1 or top5["rotation_shift"].nunique() > 1:
        if margin < 0.02:
            cats.append("branch_or_rotation_ambiguity")
    stab = float(ring_row["stabilised_miou"])
    fin = float(selected["miou"])
    proxy_failed = bool(np.isfinite(stab) and np.isfinite(fin) and fin < stab)
    if proxy_failed:
        cats.append("proxy_regression_below_stabilised")
    if not cats:
        cats.append("pass_or_no_clear_failure")
    return ";".join(cats), proxy_failed


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if not FORMULA_JSON.exists():
        raise RuntimeError(f"Missing trained proxy formula: {FORMULA_JSON}")
    formula = json.loads(FORMULA_JSON.read_text(encoding="utf-8"))

    pilot = pd.read_csv(SRC_ROOT / "pilot_ring_list.csv")
    cands = pd.read_csv(SRC_ROOT / "pilot_candidates.csv")
    cands = _ensure_feature_columns(cands)
    cands["proxy_new"] = _score_proxy(cands, formula)
    cands["proxy_old"] = (
        0.015213
        + (-0.031164) * pd.to_numeric(cands.get("struct_missing_ids_before_n"), errors="coerce").fillna(0.0)
        + 0.196603 * pd.to_numeric(cands.get("depth_row_nonempty_ratio_audit"), errors="coerce").fillna(0.0)
        + 0.081910 * pd.to_numeric(cands.get("geom_boundary_gap_cv"), errors="coerce").fillna(0.0)
    )

    rows: list[dict[str, Any]] = []
    explanations: list[dict[str, Any]] = []
    for r in pilot.itertuples(index=False):
        ring_key = str(r.ring_key)
        ring_cands = cands[cands["ring_key"].eq(ring_key)].copy()
        if ring_cands.empty:
            continue
        coarse = ring_cands[ring_cands["anchor_frac"].round(2).isin({round(x, 2) for x in COARSE_ANCHORS})].copy()
        if coarse.empty:
            coarse = ring_cands.copy()
        action = _choose_action(coarse, bool(r.depth_pass))
        search_set = _subset_by_action(ring_cands, action)
        selected = search_set.sort_values(["proxy_new", "det_tag", "branch", "rotation_shift"], ascending=[False, True, True, True]).iloc[0]
        fm, proxy_failed = _failure_mode(
            ring_row=pd.Series(
                {
                    "depth_pass": bool(r.depth_pass),
                    "stabilised_miou": float(r.stabilised_miou),
                }
            ),
            ring_cands=ring_cands,
            selected=selected,
            action=action,
        )
        rows.append(
            {
                "ring_key": ring_key,
                "tunnel_id": r.tunnel_id,
                "family": int(r.family),
                "is_depth_risk_control": bool(r.is_depth_risk_control),
                "depth_gate_pass": bool(r.depth_pass),
                "stabilised_miou": float(r.stabilised_miou),
                "final_intrinsic_miou": float(selected["miou"]),
                "proxy_score": float(selected["proxy_new"]),
                "selected_det_tag": str(selected["det_tag"]),
                "selected_branch": str(selected["branch"]),
                "selected_rotation_shift": int(selected["rotation_shift"]),
                "adaptive_action": action,
                "proxy_failed": bool(proxy_failed),
                "failure_mode": fm,
            }
        )
        explanations.append(
            {
                "ring_key": ring_key,
                "tunnel_id": r.tunnel_id,
                "adaptive_action": action,
                "proxy_margin_top2": float(
                    ring_cands.sort_values(["proxy_new"], ascending=[False]).head(2)["proxy_new"].pipe(
                        lambda s: float(s.iloc[0] - s.iloc[1]) if len(s) > 1 else 0.0
                    )
                ),
                "categories": fm.split(";"),
                "proxy_failed": bool(proxy_failed),
            }
        )

    out = pd.DataFrame(rows).sort_values(["family", "tunnel_id"]).reset_index(drop=True)
    out.to_csv(RUN_ROOT / "pilot_scoreboard.csv", index=False)
    with (RUN_ROOT / "failure_mode_explanations.jsonl").open("w", encoding="utf-8") as f:
        for rec in explanations:
            f.write(json.dumps(rec) + "\n")

    non_risk = out[~out["is_depth_risk_control"].astype(bool)].copy()
    pass_mask = (
        (non_risk["final_intrinsic_miou"] >= 0.5)
        | ((non_risk["stabilised_miou"] < 0.5) & (non_risk["final_intrinsic_miou"] >= non_risk["stabilised_miou"]))
    )
    summary = {
        "n_pilot_rings": int(len(out)),
        "n_non_depth_risk": int(len(non_risk)),
        "n_depth_risk_controls": int(out["is_depth_risk_control"].astype(bool).sum()),
        "proxy_failed_count": int(out["proxy_failed"].astype(bool).sum()),
        "mean_stabilised_miou": float(out["stabilised_miou"].mean()),
        "mean_final_intrinsic_miou": float(out["final_intrinsic_miou"].mean()),
        "pilot_gate_pass_non_depth_risk": bool(pass_mask.all()) if len(non_risk) > 0 else False,
        "pilot_gate_failed_rings": non_risk.loc[~pass_mask, "ring_key"].astype(str).tolist(),
        "depth_risk_control_rows": out[out["is_depth_risk_control"].astype(bool)][
            ["ring_key", "stabilised_miou", "final_intrinsic_miou", "failure_mode"]
        ].to_dict(orient="records"),
    }
    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
