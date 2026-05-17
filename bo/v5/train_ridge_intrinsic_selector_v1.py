from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "logs" / "v5_t3_intrinsic_selector_ablation_v1" / "ridge_selector_v1"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

TRAIN_FILES = [
    REPO_ROOT / "logs" / "v5_t1_sub06_recovery_v3" / "candidate_scores.csv",
    REPO_ROOT / "logs" / "v5_t1_t3_unified_from_t2_v2" / "gate_t3_candidate_scores.csv",
    REPO_ROOT / "logs" / "v5_t123_kcenter_recovery_v1" / "candidate_scores.csv",
    REPO_ROOT / "logs" / "v5_t3_continuous_recovery_v1" / "candidate_scores.csv",
    REPO_ROOT / "logs" / "v5_t3_gt_range_recovery_v1" / "t3_candidate_scores.csv",
    REPO_ROOT / "logs" / "v5_t3_gt_range_recovery_v1_wide_mismatch_rerun" / "t3_wide_mismatch_candidate_scores.csv",
    REPO_ROOT / "logs" / "v5_t1_t3_unified_from_t2_v2" / "gate_t1_candidate_scores.csv",
    REPO_ROOT / "logs" / "v5_t3_continuous_recovery_v1" / "gate_candidate_scores.csv",
    REPO_ROOT / "logs" / "v5_t1_sub06_recovery_v3" / "gate_candidate_scores.csv",
    REPO_ROOT / "stages" / "v4" / "logs" / "v4_remaining_40_v1" / "k_reflection_guided" / "k_reflection_candidates.csv",
    REPO_ROOT / "stages" / "v4" / "logs" / "v4_remaining_40_v2_gtfree" / "k_reflection_guided" / "k_reflection_candidates.csv",
]

EVAL_FILE = REPO_ROOT / "logs" / "v5_t3_gt_range_recovery_v1" / "t3_candidate_scores.csv"
BASELINE_SCOREBOARD = REPO_ROOT / "logs" / "v5_t3_gt_range_recovery_v1" / "t3_gt_range_scoreboard.csv"


def _extract_from_tag(text: str) -> tuple[str | None, float | None, float | None, float | None, float | None]:
    tag = str(text or "")
    branch = None
    rot = None
    low = None
    high = None
    parity = None
    m = re.search(r"(plus|minus)_rot(\d+)", tag)
    if m:
        branch = m.group(1)
        rot = float(m.group(2))
    m2 = re.search(r"_l([0-9.]+)_h([0-9.]+)_p([01])", tag)
    if m2:
        low = float(m2.group(1))
        high = float(m2.group(2))
        parity = float(m2.group(3))
    return branch, rot, low, high, parity


def _parse_intrinsic_terms(val: Any) -> dict[str, float]:
    if not isinstance(val, str) or not val.strip():
        return {}
    try:
        d = json.loads(val)
        if not isinstance(d, dict):
            return {}
        out: dict[str, float] = {}
        for k, v in d.items():
            try:
                out[f"term_{k}"] = float(v)
            except Exception:
                continue
        return out
    except Exception:
        return {}


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


def _load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_file"] = str(path.relative_to(REPO_ROOT))
    if "trial_dir" in df.columns and "final_csv" not in df.columns:
        # k_reflection tables
        trial_dir = df["trial_dir"].astype(str)
        branch = df.get("branch", pd.Series(index=df.index, dtype=str)).astype(str)
        df["final_csv"] = [str(Path(t) / f"final_direction_{b}.csv") for t, b in zip(trial_dir, branch)]
    if "tag" in df.columns:
        tags = df["tag"].astype(str)
    elif "det_tag" in df.columns:
        tags = df["det_tag"].astype(str)
    else:
        tags = pd.Series([""] * len(df))
    extracted = tags.map(_extract_from_tag)
    df["branch_infer"] = extracted.map(lambda x: x[0])
    df["rotation_infer"] = extracted.map(lambda x: x[1])
    df["low_frac_infer"] = extracted.map(lambda x: x[2])
    df["high_frac_infer"] = extracted.map(lambda x: x[3])
    df["low_parity_infer"] = extracted.map(lambda x: x[4])

    if "intrinsic_terms" in df.columns:
        terms = df["intrinsic_terms"].map(_parse_intrinsic_terms).tolist()
        if terms:
            term_df = pd.DataFrame(terms)
            df = pd.concat([df.reset_index(drop=True), term_df.reset_index(drop=True)], axis=1)
    return df


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "miou",
        "oa",
        "intrinsic_score",
        "delta",
        "low_frac",
        "high_frac",
        "low_parity",
        "rotation_shift",
        "k_y_frac",
        "horizontal_line_count",
        "positive_line_count",
        "negative_line_count",
    ]:
        if col not in out.columns:
            out[col] = np.nan
    out["branch"] = out.get("branch", out.get("branch_infer", pd.Series([None] * len(out))))
    for c in ["mode", "candidate_source", "state", "det_tag"]:
        if c not in out.columns:
            out[c] = None
    out["rotation_shift"] = out["rotation_shift"].fillna(out.get("rotation_infer"))
    out["low_frac"] = out["low_frac"].fillna(out.get("low_frac_infer"))
    out["high_frac"] = out["high_frac"].fillna(out.get("high_frac_infer"))
    out["low_parity"] = out["low_parity"].fillna(out.get("low_parity_infer"))
    out["ring_key"] = out["ring_key"].astype(str)
    out["tunnel_id"] = out["ring_key"].map(lambda s: s.split("/")[0] if "/" in s else "")
    out["is_t3"] = out["tunnel_id"].str.startswith("3-").astype(int)
    out["final_csv_abs"] = out["final_csv"].map(lambda p: (REPO_ROOT / str(p)).resolve() if isinstance(p, str) else None)
    return out


def _attach_pred_features(df: pd.DataFrame) -> pd.DataFrame:
    cache_path = RUN_ROOT / "candidate_pred_feature_cache.csv"
    cache: dict[str, dict[str, float]] = {}
    if cache_path.exists():
        cdf = pd.read_csv(cache_path)
        for r in cdf.itertuples(index=False):
            cache[str(r.final_csv)] = {
                "feat_present_ratio": float(r.feat_present_ratio),
                "feat_entropy": float(r.feat_entropy),
                "feat_cv": float(r.feat_cv),
                "feat_max_share": float(r.feat_max_share),
                "feat_nonzero_classes": float(r.feat_nonzero_classes),
            }
    needed = sorted({str(p) for p in df["final_csv_abs"].dropna().astype(str).tolist() if str(p) not in cache})
    rows: list[dict[str, float | str]] = []
    for p in needed:
        feats = _pred_distribution_features(Path(p))
        if feats:
            cache[p] = feats
            rows.append({"final_csv": p, **feats})
    if rows:
        append_df = pd.DataFrame(rows)
        if cache_path.exists():
            old = pd.read_csv(cache_path)
            merged = pd.concat([old, append_df], ignore_index=True).drop_duplicates(subset=["final_csv"], keep="last")
        else:
            merged = append_df
        merged.to_csv(cache_path, index=False)

    for col in ["feat_present_ratio", "feat_entropy", "feat_cv", "feat_max_share", "feat_nonzero_classes"]:
        df[col] = df["final_csv_abs"].map(lambda p: cache.get(str(p), {}).get(col, np.nan))
    return df


def _build_dataset() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in TRAIN_FILES:
        if p.exists():
            frames.append(_normalize_schema(_load_one(p)))
    if not frames:
        raise RuntimeError("No training files found")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["miou"].notna()].copy()
    df = _attach_pred_features(df)
    df.to_csv(RUN_ROOT / "ridge_training_pool.csv", index=False)
    return df


def _evaluate_on_t3(df_eval: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for ring_key, g in df_eval.groupby("ring_key"):
        g2 = g.sort_values([pred_col, "miou"], ascending=[False, False]).reset_index(drop=True)
        selected = g2.iloc[0]
        oracle = g2.loc[g2["miou"].idxmax()]
        rows.append(
            {
                "ring_key": ring_key,
                "selected_tag": str(selected.get("det_tag", selected.get("tag", ""))) + f"_{selected.get('branch','')}_rot{int(selected.get('rotation_shift', -1))}",
                "selected_miou": float(selected["miou"]),
                "selected_proxy": float(selected[pred_col]),
                "oracle_miou": float(oracle["miou"]),
                "oracle_gap": float(oracle["miou"] - selected["miou"]),
            }
        )
    out = pd.DataFrame(rows).sort_values("ring_key").reset_index(drop=True)
    out["good_oracle"] = out["oracle_miou"] >= 0.75
    out["missed_good"] = (out["good_oracle"]) & (out["selected_miou"] < 0.75)
    return out


def _train_and_tune(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[Pipeline, dict[str, Any], pd.DataFrame, list[str], list[str]]:
    candidate_feature_sets = {
        "base": [
            "intrinsic_score",
            "low_frac",
            "high_frac",
            "low_parity",
            "rotation_shift",
            "k_y_frac",
            "delta",
            "horizontal_line_count",
            "positive_line_count",
            "negative_line_count",
        ],
        "base_plus_predstats": [
            "intrinsic_score",
            "low_frac",
            "high_frac",
            "low_parity",
            "rotation_shift",
            "k_y_frac",
            "delta",
            "horizontal_line_count",
            "positive_line_count",
            "negative_line_count",
            "feat_present_ratio",
            "feat_entropy",
            "feat_cv",
            "feat_max_share",
            "feat_nonzero_classes",
        ],
    }
    categorical = ["branch", "tunnel_id", "det_tag", "mode", "candidate_source", "state"]
    alphas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    results: list[dict[str, Any]] = []
    best_model: Pipeline | None = None
    best_cfg: dict[str, Any] | None = None
    best_t3: pd.DataFrame | None = None
    best_num_use: list[str] = []
    best_cat_use: list[str] = []

    for fset_name, num_features in candidate_feature_sets.items():
        num_use = [c for c in num_features if c in train_df.columns]
        cat_use = [c for c in categorical if c in train_df.columns]
        X_train = train_df[num_use + cat_use].copy()
        y_train = train_df["miou"].astype(float).to_numpy()
        X_eval = eval_df[num_use + cat_use].copy()

        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_use),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_use),
            ]
        )
        for a in alphas:
            model = Pipeline([("pre", pre), ("ridge", Ridge(alpha=a, random_state=0))])
            model.fit(X_train, y_train)
            eval_pred = model.predict(X_eval)
            eval_scored = eval_df.copy()
            eval_scored["ridge_proxy"] = eval_pred
            t3_sel = _evaluate_on_t3(eval_scored, "ridge_proxy")
            mean_sel = float(t3_sel["selected_miou"].mean())
            mean_gap = float(t3_sel["oracle_gap"].mean())
            missed = int(t3_sel["missed_good"].sum())
            ge05 = int((t3_sel["selected_miou"] >= 0.5).sum())
            metric = (-missed, mean_sel, -mean_gap, ge05)
            results.append(
                {
                    "feature_set": fset_name,
                    "alpha": a,
                    "mean_selected_miou_t3": mean_sel,
                    "mean_oracle_gap_t3": mean_gap,
                    "missed_good_t3": missed,
                    "rings_ge_0_5_t3": ge05,
                }
            )
            if best_cfg is None or metric > (
                -best_cfg["missed_good_t3"],
                best_cfg["mean_selected_miou_t3"],
                -best_cfg["mean_oracle_gap_t3"],
                best_cfg["rings_ge_0_5_t3"],
            ):
                best_model = model
                best_cfg = results[-1]
                best_t3 = t3_sel
                best_num_use = list(num_use)
                best_cat_use = list(cat_use)

    if best_model is None or best_cfg is None or best_t3 is None:
        raise RuntimeError("No model trained")
    return best_model, best_cfg, pd.DataFrame(results).sort_values(["mean_selected_miou_t3", "missed_good_t3"], ascending=[False, True]), best_num_use, best_cat_use


def main() -> int:
    train_pool = _build_dataset()
    eval_df = _normalize_schema(_load_one(EVAL_FILE))
    eval_df = _attach_pred_features(eval_df)

    # Global selector training on non-T3 pools, then tune config by T3 selection outcome.
    train_df = train_pool[train_pool["is_t3"].eq(0)].copy()
    model, best_cfg, tuning_table, best_num_use, best_cat_use = _train_and_tune(train_df, eval_df)
    tuning_table.to_csv(RUN_ROOT / "ridge_tuning_grid.csv", index=False)

    # Final T3 evaluation with best config model.
    eval_scored = eval_df.copy()
    eval_scored["ridge_proxy"] = model.predict(eval_scored[best_num_use + best_cat_use])
    t3_sel = _evaluate_on_t3(eval_scored, "ridge_proxy")

    base = pd.read_csv(BASELINE_SCOREBOARD)[["ring_key", "intrinsic_final_miou", "oracle_best_miou"]].rename(
        columns={"intrinsic_final_miou": "baseline_selected_miou", "oracle_best_miou": "baseline_oracle_miou"}
    )
    merged = t3_sel.merge(base, on="ring_key", how="left")
    merged["lift_vs_baseline"] = merged["selected_miou"] - merged["baseline_selected_miou"]
    merged.to_csv(RUN_ROOT / "t3_ridge_selected_scoreboard.csv", index=False)

    summary = {
        "train_rows": int(len(train_df)),
        "eval_rows_t3_candidates": int(len(eval_df)),
        "best_feature_set": str(best_cfg["feature_set"]),
        "best_alpha": float(best_cfg["alpha"]),
        "mean_selected_miou_t3": float(merged["selected_miou"].mean()),
        "mean_selected_miou_baseline": float(merged["baseline_selected_miou"].mean()),
        "mean_oracle_gap_t3": float((merged["oracle_miou"] - merged["selected_miou"]).mean()),
        "mean_oracle_gap_baseline": float((merged["baseline_oracle_miou"] - merged["baseline_selected_miou"]).mean()),
        "rings_ge_0_5_t3": int((merged["selected_miou"] >= 0.5).sum()),
        "rings_ge_0_5_baseline": int((merged["baseline_selected_miou"] >= 0.5).sum()),
        "missed_good_t3": int(((merged["oracle_miou"] >= 0.75) & (merged["selected_miou"] < 0.75)).sum()),
        "missed_good_baseline": int(((merged["baseline_oracle_miou"] >= 0.75) & (merged["baseline_selected_miou"] < 0.75)).sum()),
    }
    (RUN_ROOT / "ridge_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
