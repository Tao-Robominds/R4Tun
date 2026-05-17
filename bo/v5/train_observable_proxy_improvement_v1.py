from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bo.v5 import train_ridge_geometric_proxy_v2 as geo
from bo.v5 import train_ridge_intrinsic_selector_v1 as base

RUN_ROOT = REPO_ROOT / "logs" / "v5_proxy_improvement_v1"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
PILOT_CANDIDATES = REPO_ROOT / "logs" / "v5_adaptive_proxy_pilot_v1" / "pilot_candidates.csv"


def _pred_distribution_features(final_csv: Path) -> dict[str, float]:
    if not final_csv.exists():
        return {}
    try:
        pred = pd.read_csv(final_csv, usecols=["pred"])["pred"].dropna().astype(int)
    except Exception:
        return {}
    pred = pred[(pred >= 1) & (pred <= 7)]
    if pred.empty:
        return {}
    counts = pred.value_counts().reindex(range(1, 8), fill_value=0).astype(float).to_numpy()
    p = counts / counts.sum()
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(7.0))
    present_ratio = float((counts > 0).mean())
    cv = float(np.std(counts) / (np.mean(counts) + 1e-9))
    max_share = float(counts.max() / (counts.sum() + 1e-9))
    return {
        "feat_present_ratio": present_ratio,
        "feat_entropy": entropy,
        "feat_cv": cv,
        "feat_max_share": max_share,
        "feat_nonzero_classes": float((counts > 0).sum()),
    }


def _extract_candidate_paths(row: pd.Series) -> tuple[Path | None, Path | None, Path | None]:
    if "final_csv" in row and isinstance(row["final_csv"], str):
        final_csv = (REPO_ROOT / row["final_csv"]).resolve()
    else:
        ring_dir = (REPO_ROOT / "logs" / "v5_adaptive_proxy_pilot_v1" / str(row["tunnel_id"]) / f"r{int(str(row['ring_key']).split('/r')[-1])}").resolve()
        det_tag = str(row.get("det_tag", ""))
        branch = str(row.get("branch", "plus"))
        final_csv = (ring_dir / f"final_{det_tag}_{branch}.csv").resolve()
    if final_csv is None or not final_csv.exists():
        return None, None, None
    ring_dir = final_csv.parent
    det_tag = str(row.get("det_tag", ""))
    branch = str(row.get("branch", "plus"))
    bnd = ring_dir / f"boundaries_{det_tag}_{branch}.json"
    seg = ring_dir / f"all_segments_{det_tag}_{branch}.csv"
    if not bnd.exists():
        bnd = ring_dir / "boundaries_per_ring.json"
    if not seg.exists():
        seg = ring_dir / "all_segments.csv"
    return final_csv, bnd, seg


def _k_anchor_dist_frac(anchor_frac: float, bnd_json: Path, ring_height: int) -> float:
    if not bnd_json.exists():
        return float("nan")
    data = json.loads(bnd_json.read_text(encoding="utf-8"))
    entries = data.get("0") or data.get(0) or []
    k_entries = [e for e in entries if str(e.get("block", "")).upper() == "K"]
    if not k_entries:
        return float("nan")
    ky = float(k_entries[0].get("y", 0.0)) % float(ring_height)
    kf = ky / float(max(1, ring_height))
    d = abs(float(anchor_frac) - float(kf))
    return float(min(d, 1.0 - d))


def _boundary_features(bnd_json: Path, ring_height: int) -> dict[str, float]:
    if not bnd_json.exists():
        return {}
    data = json.loads(bnd_json.read_text(encoding="utf-8"))
    entries = data.get("0") or data.get(0) or []
    ys = sorted(float(e.get("y", 0.0)) for e in entries)
    if len(ys) < 2:
        return {}
    h = float(max(1, ring_height))
    diffs = np.array([(ys[(i + 1) % len(ys)] - ys[i]) % h for i in range(len(ys))], dtype=float)
    exp = h / len(ys)
    return {
        "geom_boundary_gap_cv": float(np.std(diffs) / (np.mean(diffs) + 1e-9)),
        "geom_boundary_min_gap_frac": float(np.min(diffs) / h),
        "geom_boundary_max_gap_frac": float(np.max(diffs) / h),
        "geom_boundary_expected_resid_frac": float(np.mean(np.abs(diffs - exp)) / h),
    }


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rows: list[dict[str, float]] = []
    for r in out.itertuples(index=False):
        s = pd.Series(r._asdict())
        final_csv, bnd_json, seg_csv = _extract_candidate_paths(s)
        feats: dict[str, float] = {}
        if final_csv is None:
            rows.append(feats)
            continue
        ring_dir = final_csv.parent
        depth_path = ring_dir / "depth_map.npy"
        ring_height = int(np.load(depth_path).shape[0]) if depth_path.exists() else 1

        feats.update(_pred_distribution_features(final_csv))
        feats.update(_boundary_features(bnd_json, ring_height=ring_height))
        feats["k_anchor_dist_frac"] = _k_anchor_dist_frac(float(s.get("anchor_frac", np.nan)), bnd_json, ring_height)
        feats["branch_is_minus"] = float(str(s.get("branch", "")).lower() == "minus")
        feats["rotation_shift_num"] = float(s.get("rotation_shift", np.nan))
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    out = pd.concat([out.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    # Keep first occurrence if duplicate column names appear.
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def _load_prior_pool() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in base.TRAIN_FILES:
        if p.exists():
            x = base._normalize_schema(base._load_one(p))
            x["pool"] = "prior"
            frames.append(x)
    if not frames:
        raise RuntimeError("No prior candidate files found.")
    out = pd.concat(frames, ignore_index=True)
    out = out[out["miou"].notna()].copy()
    out = geo._attach_geometric_features(out)
    return out


def _load_pilot_pool() -> pd.DataFrame:
    if not PILOT_CANDIDATES.exists():
        raise RuntimeError(f"Missing pilot candidates: {PILOT_CANDIDATES}")
    x = pd.read_csv(PILOT_CANDIDATES)
    x["pool"] = "pilot"
    x["ring_key"] = x["ring_key"].astype(str)
    x["tunnel_id"] = x["ring_key"].map(lambda s: s.split("/")[0])
    x["is_t3"] = x["tunnel_id"].str.startswith("3-").astype(int)
    return x


def _feature_variation_audit(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        nunq = (
            df.assign(_v=s)
            .groupby("ring_key")["_v"]
            .nunique(dropna=True)
        )
        vary_rings = int((nunq > 1).sum())
        corr = float(s.corr(pd.to_numeric(df["miou"], errors="coerce"))) if s.notna().sum() > 5 else float("nan")
        rows.append(
            {
                "feature": c,
                "non_null_rows": int(s.notna().sum()),
                "non_null_ratio": float(s.notna().mean()),
                "rings_with_candidate_variation": vary_rings,
                "abs_corr_with_miou": float(abs(corr)) if np.isfinite(corr) else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["rings_with_candidate_variation", "abs_corr_with_miou"], ascending=[False, False]
    )
    out.to_csv(RUN_ROOT / "feature_variation_audit.csv", index=False)
    return out


def _evaluate_selection(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ring_key, g in df.groupby("ring_key"):
        g2 = g.sort_values([pred_col, "det_tag", "branch", "rotation_shift"], ascending=[False, True, True, True]).reset_index(drop=True)
        sel = g2.iloc[0]
        oracle = g.loc[g["miou"].idxmax()]
        rows.append(
            {
                "ring_key": ring_key,
                "selected_miou": float(sel["miou"]),
                "oracle_miou": float(oracle["miou"]),
                "oracle_gap": float(oracle["miou"] - sel["miou"]),
            }
        )
    out = pd.DataFrame(rows)
    out["good_oracle"] = out["oracle_miou"] >= 0.75
    out["missed_good"] = out["good_oracle"] & (out["selected_miou"] < 0.75)
    return out


def main() -> int:
    prior = _load_prior_pool()
    pilot = _load_pilot_pool()
    # Attach shared geometric features to pilot if missing.
    for col in ["depth_row_nonempty_ratio_audit", "struct_missing_ids_before_n", "geom_boundary_gap_cv"]:
        if col not in pilot.columns:
            pilot[col] = np.nan

    prior = _engineer_features(prior)
    pilot = _engineer_features(pilot)
    pooled = pd.concat([prior, pilot], ignore_index=True)
    pooled.to_csv(RUN_ROOT / "proxy_training_pool.csv", index=False)

    feature_candidates = [
        "depth_row_nonempty_ratio_audit",
        "struct_missing_ids_before_n",
        "geom_boundary_gap_cv",
        "geom_boundary_min_gap_frac",
        "geom_boundary_max_gap_frac",
        "geom_boundary_expected_resid_frac",
        "k_anchor_dist_frac",
        "feat_present_ratio",
        "feat_entropy",
        "feat_cv",
        "feat_max_share",
        "feat_nonzero_classes",
        "branch_is_minus",
        "rotation_shift_num",
    ]
    audit = _feature_variation_audit(pooled, feature_candidates)
    selected = audit[
        (audit["rings_with_candidate_variation"] >= 10) & (audit["non_null_ratio"] >= 0.2)
    ]["feature"].astype(str).tolist()
    if len(selected) < 3:
        selected = ["geom_boundary_gap_cv", "k_anchor_dist_frac", "feat_entropy", "feat_cv", "feat_max_share"]

    train_df = pooled[pooled["pool"].eq("prior")].copy()
    eval_df = pooled[pooled["pool"].eq("pilot")].copy()
    selected = [f for f in selected if f in train_df.columns and pd.to_numeric(train_df[f], errors="coerce").notna().any()]
    x_train = train_df[selected].apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(train_df["miou"], errors="coerce").to_numpy()
    x_eval = eval_df[selected].apply(pd.to_numeric, errors="coerce")

    best: dict[str, Any] | None = None
    best_model: Pipeline | None = None
    for alpha in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        model = Pipeline([("impute", SimpleImputer(strategy="median")), ("ridge", Ridge(alpha=alpha, random_state=0))])
        model.fit(x_train, y_train)
        scored = eval_df.copy()
        scored["proxy_new"] = model.predict(x_eval)
        ev = _evaluate_selection(scored, "proxy_new")
        metric = {
            "alpha": float(alpha),
            "mean_selected_miou": float(ev["selected_miou"].mean()),
            "mean_oracle_gap": float(ev["oracle_gap"].mean()),
            "missed_good": int(ev["missed_good"].sum()),
            "rings_ge_0_5": int((ev["selected_miou"] >= 0.5).sum()),
        }
        key = (-metric["missed_good"], metric["mean_selected_miou"], -metric["mean_oracle_gap"], metric["rings_ge_0_5"])
        if best is None or key > (
            -best["missed_good"],
            best["mean_selected_miou"],
            -best["mean_oracle_gap"],
            best["rings_ge_0_5"],
        ):
            best = metric
            best_model = model

    if best is None or best_model is None:
        raise RuntimeError("Training failed.")

    ridge = best_model.named_steps["ridge"]
    formula_terms = [f"{ridge.coef_[i]:+.6f}*{selected[i]}" for i in range(len(selected))]
    formula = f"proxy_miou = {ridge.intercept_:.6f} " + " ".join(formula_terms)

    (RUN_ROOT / "proxy_formula.txt").write_text(
        formula + "\n\n# Runtime: median-impute missing features, then apply linear formula.\n",
        encoding="utf-8",
    )
    (RUN_ROOT / "proxy_formula.json").write_text(
        json.dumps(
            {
                "intercept": float(ridge.intercept_),
                "features": selected,
                "coefficients": [float(x) for x in ridge.coef_.tolist()],
                "alpha": float(best["alpha"]),
                "training_pool": "prior_candidates_only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Offline comparison on pilot candidates: old proxy vs new proxy vs stabilised baseline.
    eval_scored = eval_df.copy()
    eval_scored["proxy_old"] = (
        0.015213
        + (-0.031164) * pd.to_numeric(eval_scored.get("struct_missing_ids_before_n"), errors="coerce").fillna(0.0)
        + 0.196603 * pd.to_numeric(eval_scored.get("depth_row_nonempty_ratio_audit"), errors="coerce").fillna(0.0)
        + 0.081910 * pd.to_numeric(eval_scored.get("geom_boundary_gap_cv"), errors="coerce").fillna(0.0)
    )
    eval_scored["proxy_new"] = best_model.predict(x_eval)
    ev_old = _evaluate_selection(eval_scored, "proxy_old").rename(columns={"selected_miou": "selected_miou_old", "oracle_gap": "oracle_gap_old"})
    ev_new = _evaluate_selection(eval_scored, "proxy_new").rename(columns={"selected_miou": "selected_miou_new", "oracle_gap": "oracle_gap_new"})
    ring_stab = (
        pd.read_csv(REPO_ROOT / "logs" / "v5_adaptive_proxy_pilot_v1" / "pilot_ring_list.csv")[
            ["ring_key", "stabilised_miou", "is_depth_risk_control"]
        ]
        .copy()
    )
    comp = ev_old[["ring_key", "selected_miou_old", "oracle_miou", "oracle_gap_old"]].merge(
        ev_new[["ring_key", "selected_miou_new", "oracle_gap_new"]],
        on="ring_key",
        how="left",
    ).merge(ring_stab, on="ring_key", how="left")
    comp["lift_new_vs_old"] = comp["selected_miou_new"] - comp["selected_miou_old"]
    comp["lift_new_vs_stabilised"] = comp["selected_miou_new"] - comp["stabilised_miou"]
    comp.to_csv(RUN_ROOT / "offline_proxy_comparison.csv", index=False)

    summary = {
        "selected_features": selected,
        "best_alpha": float(best["alpha"]),
        "new_proxy_formula": formula,
        "offline_mean_selected_old": float(comp["selected_miou_old"].mean()),
        "offline_mean_selected_new": float(comp["selected_miou_new"].mean()),
        "offline_mean_lift_new_vs_old": float(comp["lift_new_vs_old"].mean()),
        "offline_proxy_failed_old": int((comp["selected_miou_old"] < comp["stabilised_miou"]).sum()),
        "offline_proxy_failed_new": int((comp["selected_miou_new"] < comp["stabilised_miou"]).sum()),
    }
    (RUN_ROOT / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
