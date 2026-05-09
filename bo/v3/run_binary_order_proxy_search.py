from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bo.v3._paths import assert_writable
from bo.v3.run_binary_order_model_search import build_feature_table

RUN_ROOT = REPO / "logs" / "v3_binary_order_bo_r4tun_v1" / "proxy"

RING_FEATURES = [
    "det_k_confidence",
    "det_pos_count",
    "det_neg_count",
    "det_line_diff_pos_minus_neg",
    "det_abs_line_diff",
    "det_horizontal_count",
    "det_selected_pos_count",
    "det_selected_neg_count",
    "det_k_y_rel",
    "pre_valid_ratio",
    "pre_empty_row_band_ratio",
    "det_y_coverage_pct",
    "det_min_y_gap_px",
    "det_k_x_spacing_cv",
    "seg_mask_coverage_pct",
    "seg_ring_completeness_avg",
    "seg_k_size_ratio",
    "seg_block_size_variance_ratio",
    "k_width_rank_norm",
    "k_width_ratio",
    "width_cv",
    "k_z_rel",
    "k_intensity_rel",
    "k_r_rel",
]

BRANCH_FEATURES = [
    "is_minus_branch",
    "branch_z_score",
    "opponent_z_score",
    "z_advantage",
    "health_advantage",
    *RING_FEATURES,
]

BO_PROXY_NAMES = [
    "bias",
    "minus_bias",
    "w_branch_z",
    "w_z_adv",
    "w_health_adv",
    "w_conf",
    "w_ring_complete",
    "w_mask",
    "w_width_cv",
    "w_k_width_ratio",
    "w_k_y",
]

BO_BOUNDS = np.asarray(
    [
        [0.0, 0.7],
        [-0.5, 0.5],
        [-0.5, 0.8],
        [-0.8, 0.8],
        [-0.8, 0.8],
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, 0.5],
    ],
    dtype=float,
)


@dataclass
class ProxyEval:
    strategy: str
    predictions: pd.DataFrame
    summary: dict[str, Any]


def _summary(ring_df: pd.DataFrame, pred_minus: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    plus = ring_df["plus_miou"].to_numpy(dtype=float)
    minus = ring_df["minus_miou"].to_numpy(dtype=float)
    oracle = ring_df["oracle_miou"].to_numpy(dtype=float)
    minus_better = ring_df["minus_better"].to_numpy(dtype=int)
    lifts = selected - plus
    return {
        "n": int(len(ring_df)),
        "mean_plus_s0": float(np.mean(plus)),
        "mean_selected": float(np.mean(selected)),
        "mean_oracle": float(np.mean(oracle)),
        "lift_vs_s0": float(np.mean(selected) - np.mean(plus)),
        "oracle_recovered_fraction": float((np.mean(selected) - np.mean(plus)) / max(1e-9, np.mean(oracle) - np.mean(plus))),
        "mean_oracle_regret": float(np.mean(oracle - selected)),
        "n_degrade_lt_minus_0p01": int(np.sum(lifts < -0.01)),
        "n_minus_selected": int(np.sum(pred_minus)),
        "direction_accuracy": float(np.mean(pred_minus.astype(int) == minus_better)),
    }


def build_branch_table(ring_df: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in ring_df.iterrows():
        for branch in ("plus", "minus"):
            is_minus = branch == "minus"
            z_branch = float(row["z_score_minus"] if is_minus else row["z_score_plus"])
            z_opp = float(row["z_score_plus"] if is_minus else row["z_score_minus"])
            health_adv = -float(row["health_score_diff_plus_minus"]) if is_minus else float(row["health_score_diff_plus_minus"])
            out = {
                "ring_key": row["ring_key"],
                "section": row["section"],
                "branch": branch,
                "is_minus_branch": int(is_minus),
                "branch_miou": float(row["minus_miou"] if is_minus else row["plus_miou"]),
                "opponent_miou": float(row["plus_miou"] if is_minus else row["minus_miou"]),
                "branch_z_score": z_branch,
                "opponent_z_score": z_opp,
                "z_advantage": z_branch - z_opp,
                "health_advantage": health_adv,
            }
            for f in RING_FEATURES:
                out[f] = float(row[f])
            rows.append(out)
    branch_df = pd.DataFrame(rows)
    branch_df.to_csv(out_root / "branch_proxy_features.csv", index=False)
    return branch_df


def _ring_predictions_from_branch(branch_pred: pd.DataFrame, ring_df: pd.DataFrame, strategy: str) -> ProxyEval:
    rows = []
    for _, ring in ring_df.iterrows():
        sub = branch_pred[branch_pred["ring_key"] == ring["ring_key"]]
        plus_score = float(sub[sub["branch"] == "plus"]["proxy_miou"].iloc[0])
        minus_score = float(sub[sub["branch"] == "minus"]["proxy_miou"].iloc[0])
        pred_minus = minus_score > plus_score
        selected = float(ring["minus_miou"] if pred_minus else ring["plus_miou"])
        rows.append(
            {
                "strategy": strategy,
                "section": ring["section"],
                "ring_key": ring["ring_key"],
                "proxy_plus_miou": plus_score,
                "proxy_minus_miou": minus_score,
                "proxy_margin_minus_plus": minus_score - plus_score,
                "pred_minus": pred_minus,
                "minus_better": bool(ring["minus_better"]),
                "plus_miou": float(ring["plus_miou"]),
                "minus_miou": float(ring["minus_miou"]),
                "selected_miou": selected,
                "oracle_miou": float(ring["oracle_miou"]),
                "lift_vs_s0": selected - float(ring["plus_miou"]),
            }
        )
    pred = pd.DataFrame(rows)
    summary = _summary(ring_df, pred["pred_minus"].to_numpy(dtype=bool), pred["selected_miou"].to_numpy(dtype=float))
    return ProxyEval(strategy=strategy, predictions=pred, summary=summary)


def evaluate_regression_proxies(ring_df: pd.DataFrame, branch_df: pd.DataFrame) -> dict[str, ProxyEval]:
    models: dict[str, Any] = {
        "proxy_ridge": make_pipeline(StandardScaler(), Ridge(alpha=2.0)),
        "proxy_rf": RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=4, random_state=11),
        "proxy_gbdt": GradientBoostingRegressor(max_depth=2, min_samples_leaf=4, learning_rate=0.04, n_estimators=220, random_state=11),
    }
    out: dict[str, ProxyEval] = {}
    for name, model in models.items():
        pred_rows = []
        for section in sorted(ring_df["section"].unique()):
            train = branch_df[branch_df["section"] != section].copy()
            test = branch_df[branch_df["section"] == section].copy()
            model.fit(train[BRANCH_FEATURES].fillna(0.0), train["branch_miou"])
            test = test.copy()
            test["proxy_miou"] = np.clip(model.predict(test[BRANCH_FEATURES].fillna(0.0)), 0.0, 1.0)
            pred_rows.append(test[["ring_key", "section", "branch", "branch_miou", "proxy_miou"]])
        branch_pred = pd.concat(pred_rows, ignore_index=True)
        eval_result = _ring_predictions_from_branch(branch_pred, ring_df, name)
        eval_result.summary["branch_proxy_mae"] = float(mean_absolute_error(branch_pred["branch_miou"], branch_pred["proxy_miou"]))
        eval_result.summary["branch_proxy_rmse"] = float(mean_squared_error(branch_pred["branch_miou"], branch_pred["proxy_miou"]) ** 0.5)
        eval_result.summary["branch_proxy_corr"] = float(pd.Series(branch_pred["branch_miou"]).corr(pd.Series(branch_pred["proxy_miou"])))
        out[name] = eval_result
    return out


def _bo_proxy_scores(branch_df: pd.DataFrame, params: np.ndarray, train_stats: tuple[pd.Series, pd.Series]) -> np.ndarray:
    mu, sd = train_stats
    x = (branch_df[BRANCH_FEATURES].fillna(0.0) - mu) / sd.replace(0.0, 1.0)
    p = dict(zip(BO_PROXY_NAMES, params.tolist()))
    score = (
        p["bias"]
        + p["minus_bias"] * branch_df["is_minus_branch"].to_numpy(dtype=float)
        + p["w_branch_z"] * x["branch_z_score"].to_numpy(dtype=float)
        + p["w_z_adv"] * x["z_advantage"].to_numpy(dtype=float)
        + p["w_health_adv"] * x["health_advantage"].to_numpy(dtype=float)
        + p["w_conf"] * x["det_k_confidence"].to_numpy(dtype=float)
        + p["w_ring_complete"] * x["seg_ring_completeness_avg"].to_numpy(dtype=float)
        + p["w_mask"] * x["seg_mask_coverage_pct"].to_numpy(dtype=float)
        + p["w_width_cv"] * x["width_cv"].to_numpy(dtype=float)
        + p["w_k_width_ratio"] * x["k_width_ratio"].to_numpy(dtype=float)
        + p["w_k_y"] * x["det_k_y_rel"].to_numpy(dtype=float)
    )
    return np.clip(score, 0.0, 1.0)


def evaluate_bo_proxy_params(ring_df: pd.DataFrame, branch_df: pd.DataFrame, params: np.ndarray, strategy: str = "proxy_bo") -> ProxyEval:
    pred_rows = []
    for section in sorted(ring_df["section"].unique()):
        train = branch_df[branch_df["section"] != section].copy()
        test = branch_df[branch_df["section"] == section].copy()
        mu = train[BRANCH_FEATURES].fillna(0.0).mean()
        sd = train[BRANCH_FEATURES].fillna(0.0).std().replace(0.0, 1.0)
        test = test.copy()
        test["proxy_miou"] = _bo_proxy_scores(test, params, (mu, sd))
        pred_rows.append(test[["ring_key", "section", "branch", "branch_miou", "proxy_miou"]])
    branch_pred = pd.concat(pred_rows, ignore_index=True)
    eval_result = _ring_predictions_from_branch(branch_pred, ring_df, strategy)
    mae = float(mean_absolute_error(branch_pred["branch_miou"], branch_pred["proxy_miou"]))
    rmse = float(mean_squared_error(branch_pred["branch_miou"], branch_pred["proxy_miou"]) ** 0.5)
    corr = float(pd.Series(branch_pred["branch_miou"]).corr(pd.Series(branch_pred["proxy_miou"])))
    eval_result.summary.update({"branch_proxy_mae": mae, "branch_proxy_rmse": rmse, "branch_proxy_corr": corr})
    return eval_result


def _objective_for_params(ring_df: pd.DataFrame, branch_df: pd.DataFrame, params: np.ndarray) -> float:
    ev = evaluate_bo_proxy_params(ring_df, branch_df, params)
    s = ev.summary
    return float(
        s["lift_vs_s0"]
        - 0.08 * s["n_degrade_lt_minus_0p01"]
        - 0.25 * s["branch_proxy_rmse"]
        + 0.05 * max(0.0, s["branch_proxy_corr"])
    )


def run_lightweight_bo_proxy(ring_df: pd.DataFrame, branch_df: pd.DataFrame, out_root: Path, *, seed: int, init_trials: int, bo_trials: int) -> ProxyEval:
    rng = np.random.default_rng(seed)
    lows = BO_BOUNDS[:, 0]
    highs = BO_BOUNDS[:, 1]

    x_rows = rng.uniform(lows, highs, size=(init_trials, len(BO_PROXY_NAMES)))
    y_rows = np.asarray([_objective_for_params(ring_df, branch_df, x) for x in x_rows], dtype=float)

    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-4)
    for _ in range(bo_trials):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed, alpha=1e-6)
        gp.fit(x_rows, y_rows)
        candidates = rng.uniform(lows, highs, size=(1200, len(BO_PROXY_NAMES)))
        mu, std = gp.predict(candidates, return_std=True)
        next_x = candidates[int(np.argmax(mu + 1.5 * std))]
        next_y = _objective_for_params(ring_df, branch_df, next_x)
        x_rows = np.vstack([x_rows, next_x])
        y_rows = np.append(y_rows, next_y)

    order = np.argsort(-y_rows)
    trial_rows = []
    for rank, idx in enumerate(order):
        params = {name: float(v) for name, v in zip(BO_PROXY_NAMES, x_rows[idx])}
        trial_rows.append({"rank": int(rank), "trial_index": int(idx), "objective": float(y_rows[idx]), "params": params})
    (out_root / "proxy_bo_trials.json").write_text(json.dumps(trial_rows, indent=2) + "\n", encoding="utf-8")

    best_params = x_rows[order[0]]
    ev = evaluate_bo_proxy_params(ring_df, branch_df, best_params, strategy="proxy_bo")
    ev.summary["best_objective"] = float(y_rows[order[0]])
    ev.summary["best_params"] = {name: float(v) for name, v in zip(BO_PROXY_NAMES, best_params)}
    return ev


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Learn branch mIoU proxies and choose the higher predicted order")
    p.add_argument("--run-root", default=str(RUN_ROOT))
    p.add_argument("--seed", type=int, default=20260509)
    p.add_argument("--init-trials", type=int, default=60)
    p.add_argument("--bo-trials", type=int, default=50)
    ns = p.parse_args(argv)

    out_root = assert_writable(Path(ns.run_root).resolve())
    out_root.mkdir(parents=True, exist_ok=True)
    ring_df = build_feature_table(out_root)
    branch_df = build_branch_table(ring_df, out_root)

    evaluations = evaluate_regression_proxies(ring_df, branch_df)
    evaluations["proxy_bo"] = run_lightweight_bo_proxy(
        ring_df,
        branch_df,
        out_root,
        seed=int(ns.seed),
        init_trials=int(ns.init_trials),
        bo_trials=int(ns.bo_trials),
    )

    prediction_rows = []
    summary = {"strategies": {}}
    for name, ev in evaluations.items():
        summary["strategies"][name] = ev.summary
        prediction_rows.extend(ev.predictions.to_dict(orient="records"))

    with (out_root / "proxy_loso_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "strategy",
            "section",
            "ring_key",
            "proxy_plus_miou",
            "proxy_minus_miou",
            "proxy_margin_minus_plus",
            "pred_minus",
            "minus_better",
            "plus_miou",
            "minus_miou",
            "selected_miou",
            "oracle_miou",
            "lift_vs_s0",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in prediction_rows:
            w.writerow({k: row.get(k) for k in fields})

    (out_root / "proxy_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
