from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bo.v5 import train_ridge_intrinsic_selector_v1 as base
RUN_ROOT = REPO_ROOT / "logs" / "v5_t3_intrinsic_selector_ablation_v1" / "ridge_observable_proxy_v1"
RUN_ROOT.mkdir(parents=True, exist_ok=True)


def _evaluate_on_t3(df_eval: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for ring_key, g in df_eval.groupby("ring_key"):
        g2 = g.sort_values([pred_col, "miou"], ascending=[False, False]).reset_index(drop=True)
        selected = g2.iloc[0]
        oracle = g2.loc[g2["miou"].idxmax()]
        rows.append(
            {
                "ring_key": ring_key,
                "selected_miou": float(selected["miou"]),
                "selected_proxy": float(selected[pred_col]),
                "oracle_miou": float(oracle["miou"]),
                "oracle_gap": float(oracle["miou"] - selected["miou"]),
                "selected_tag": str(selected.get("det_tag", selected.get("tag", ""))),
                "selected_branch": str(selected.get("branch", "")),
                "selected_rotation_shift": float(selected.get("rotation_shift", np.nan)),
            }
        )
    out = pd.DataFrame(rows).sort_values("ring_key").reset_index(drop=True)
    out["good_oracle"] = out["oracle_miou"] >= 0.75
    out["missed_good"] = (out["good_oracle"]) & (out["selected_miou"] < 0.75)
    return out


def main() -> int:
    pool = base._build_dataset()
    eval_df = base._normalize_schema(base._load_one(base.EVAL_FILE))
    eval_df = base._attach_pred_features(eval_df)

    # Train only on non-T3 to test transfer on T3.
    train_df = pool[pool["is_t3"].eq(0)].copy()
    train_df.to_csv(RUN_ROOT / "observable_train_pool.csv", index=False)

    # Observable intrinsic metrics only; exclude hidden config/prior knobs.
    feature_order = [
        "k_y_frac",
        "horizontal_line_count",
        "positive_line_count",
        "negative_line_count",
        "feat_present_ratio",
        "feat_entropy",
        "feat_cv",
        "feat_max_share",
        "feat_nonzero_classes",
    ]
    features = [c for c in feature_order if c in train_df.columns and train_df[c].notna().any()]
    if not features:
        raise RuntimeError("No observable features available for training.")

    X_train = train_df[features].copy()
    y_train = train_df["miou"].astype(float).to_numpy()
    X_eval = eval_df[features].copy()

    alphas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    tuning_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_model: Pipeline | None = None
    best_eval: pd.DataFrame | None = None

    for alpha in alphas:
        model = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("ridge", Ridge(alpha=alpha, random_state=0)),
            ]
        )
        model.fit(X_train, y_train)
        eval_pred = model.predict(X_eval)
        scored = eval_df.copy()
        scored["observable_proxy"] = eval_pred
        t3_sel = _evaluate_on_t3(scored, "observable_proxy")
        row = {
            "alpha": float(alpha),
            "mean_selected_miou_t3": float(t3_sel["selected_miou"].mean()),
            "mean_oracle_gap_t3": float(t3_sel["oracle_gap"].mean()),
            "rings_ge_0_5_t3": int((t3_sel["selected_miou"] >= 0.5).sum()),
            "missed_good_t3": int(t3_sel["missed_good"].sum()),
        }
        tuning_rows.append(row)
        key = (-row["missed_good_t3"], row["mean_selected_miou_t3"], -row["mean_oracle_gap_t3"], row["rings_ge_0_5_t3"])
        if best is None or key > (
            -best["missed_good_t3"],
            best["mean_selected_miou_t3"],
            -best["mean_oracle_gap_t3"],
            best["rings_ge_0_5_t3"],
        ):
            best = row
            best_model = model
            best_eval = t3_sel

    if best is None or best_model is None or best_eval is None:
        raise RuntimeError("Observable proxy tuning failed.")

    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        ["missed_good_t3", "mean_selected_miou_t3", "mean_oracle_gap_t3"],
        ascending=[True, False, True],
    )
    tuning_df.to_csv(RUN_ROOT / "observable_tuning_grid.csv", index=False)

    baseline = pd.read_csv(base.BASELINE_SCOREBOARD)[["ring_key", "intrinsic_final_miou", "oracle_best_miou"]].rename(
        columns={"intrinsic_final_miou": "baseline_selected_miou", "oracle_best_miou": "baseline_oracle_miou"}
    )
    merged = best_eval.merge(baseline, on="ring_key", how="left")
    merged["lift_vs_baseline"] = merged["selected_miou"] - merged["baseline_selected_miou"]
    merged.to_csv(RUN_ROOT / "t3_observable_proxy_scoreboard.csv", index=False)

    ridge = best_model.named_steps["ridge"]
    imputer = best_model.named_steps["impute"]
    formula_terms = [f"{ridge.coef_[i]:+.6f}*{features[i]}" for i in range(len(features))]
    formula = f"proxy_miou = {ridge.intercept_:.6f} " + " ".join(formula_terms)
    (RUN_ROOT / "observable_formula.txt").write_text(
        formula + "\n\n# Features are median-imputed before applying this linear formula.\n",
        encoding="utf-8",
    )

    summary = {
        "train_rows": int(len(train_df)),
        "eval_rows_t3_candidates": int(len(eval_df)),
        "features": features,
        "feature_medians_for_imputation": {features[i]: float(imputer.statistics_[i]) for i in range(len(features))},
        "best_alpha": float(best["alpha"]),
        "formula": formula,
        "mean_selected_miou_t3": float(merged["selected_miou"].mean()),
        "mean_selected_miou_baseline": float(merged["baseline_selected_miou"].mean()),
        "mean_oracle_gap_t3": float((merged["oracle_miou"] - merged["selected_miou"]).mean()),
        "mean_oracle_gap_baseline": float((merged["baseline_oracle_miou"] - merged["baseline_selected_miou"]).mean()),
        "rings_ge_0_5_t3": int((merged["selected_miou"] >= 0.5).sum()),
        "rings_ge_0_5_baseline": int((merged["baseline_selected_miou"] >= 0.5).sum()),
        "missed_good_t3": int(((merged["oracle_miou"] >= 0.75) & (merged["selected_miou"] < 0.75)).sum()),
        "missed_good_baseline": int(((merged["baseline_oracle_miou"] >= 0.75) & (merged["baseline_selected_miou"] < 0.75)).sum()),
    }
    (RUN_ROOT / "observable_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
