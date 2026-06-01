"""Post-process layout BO trials: sensitivity + intrinsic guardrails."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = REPO_ROOT / "logs" / "v7_detection_bo_v1"

INTRINSIC_CANDIDATES = [
    "det_y_coverage_pct",
    "det_block_count_per_ring",
    "det_ready_for_segmentation",
    "det_k_count_match",
    "det_min_y_gap_px",
    "det_y_order_consistency",
    "k_y_frac",
    "arc_width_entropy",
    "r_surface_min_frac",
    "n_reclassified_by_r_filter",
]


def _gp_ard_importance(X: np.ndarray, y: np.ndarray, seed: int = 7) -> np.ndarray:
    n_dims = X.shape[1]
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(n_dims), nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=seed)
    gp.fit(X, y)
    ls = gp.kernel_.k2.length_scale
    if np.isscalar(ls):
        return np.array([float(ls)])
    return np.asarray(ls, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=str(DEFAULT_RUN))
    ap.add_argument("--rho-threshold", type=float, default=0.3)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    trials_path = run_root / "bo_trials.csv"
    if not trials_path.exists():
        raise FileNotFoundError(f"Missing {trials_path}; run BO first.")

    df = pd.read_csv(trials_path)
    df["gt_miou"] = pd.to_numeric(df["gt_miou"], errors="coerce")

    search_cols = [c for c in ["k_y_frac", "r_surface_min_frac", "arc_width_entropy"] if c in df.columns]
    sens_lines = ["# Parameter sensitivity (detection layout BO)\n"]
    if search_cols and df["gt_miou"].notna().sum() >= 10:
        X = df[search_cols].apply(pd.to_numeric, errors="coerce").fillna(0.5).to_numpy()
        y = df["gt_miou"].fillna(0).to_numpy()
        ard = _gp_ard_importance(X, y)
        sens_lines.append("## GP ARD lengthscales (smaller = more sensitive)\n")
        for name, ls in zip(search_cols, ard):
            sens_lines.append(f"- `{name}`: lengthscale={ls:.4f}")
        sens_lines.append("\n## Expected critical parameters\n")
        sens_lines.append("| Parameter | Stage | Evidence |")
        sens_lines.append("|---|---|---|")
        sens_lines.append("| `k_y_positions[0]` | Detection | `k_y_frac` ARD + trial spread |")
        sens_lines.append("| `per_ring_offsets` | Detection | `arc_width_entropy` + offsets in trials |")
        sens_lines.append("| `r_surface_min` | Segmentation | `r_surface_min_frac` ARD + reclass count |")

        per_ring = []
        for case_id, g in df.groupby("case_id"):
            sub = g[search_cols + ["gt_miou"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 8:
                continue
            ard_r = _gp_ard_importance(sub[search_cols].to_numpy(), sub["gt_miou"].to_numpy())
            per_ring.append({"case_id": case_id, **{f"ard_{c}": v for c, v in zip(search_cols, ard_r)}})
        if per_ring:
            sens_lines.append("\n## Per-ring ARD lengthscales\n")
            pr_df = pd.DataFrame(per_ring)
            sens_lines.append(pr_df.to_string(index=False))
    else:
        sens_lines.append("Insufficient data for GP ARD analysis.\n")

    (run_root / "parameter_sensitivity.md").write_text("\n".join(sens_lines) + "\n", encoding="utf-8")

    corr_rows = []
    for col in INTRINSIC_CANDIDATES:
        if col not in df.columns:
            continue
        series = df[col]
        if series.dtype == bool or series.dropna().isin([True, False, 0, 1]).all():
            numeric = series.map(lambda v: float(v) if v is True or v == 1 else (0.0 if pd.isna(v) or v is False else float(v)))
        else:
            numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.notna() & df["gt_miou"].notna()
        if valid.sum() < 12:
            continue
        rho, pval = spearmanr(numeric[valid], df.loc[valid, "gt_miou"])
        per_ring_sign = []
        for case_id, g in df.groupby("case_id"):
            m = numeric.loc[g.index].notna() & g["gt_miou"].notna()
            if m.sum() < 8:
                continue
            r, _ = spearmanr(numeric.loc[g.index][m], g.loc[m, "gt_miou"])
            if np.isfinite(r):
                per_ring_sign.append(np.sign(r))
        stable = len(per_ring_sign) >= 4 and len(set(per_ring_sign)) == 1
        corr_rows.append({
            "metric": col,
            "spearman_rho": round(float(rho), 4) if np.isfinite(rho) else None,
            "p_value": round(float(pval), 6) if np.isfinite(pval) else None,
            "n": int(valid.sum()),
            "stable_sign_4plus_rings": stable,
            "keep": bool(np.isfinite(rho) and abs(rho) >= args.rho_threshold and stable),
        })

    corr_df = pd.DataFrame(corr_rows).sort_values("spearman_rho", key=abs, ascending=False)
    corr_df.to_csv(run_root / "intrinsic_correlation.csv", index=False)

    guardrails: dict[str, Any] = {
        "metrics": {},
        "source": str(trials_path.resolve().relative_to(REPO_ROOT.resolve())),
    }
    kept = corr_df[corr_df["keep"]]["metric"].tolist()
    if not kept:
        kept = corr_df.head(5)["metric"].tolist()

    for metric in kept:
        if metric not in df.columns:
            continue
        numeric = pd.to_numeric(df[metric], errors="coerce") if df[metric].dtype != bool else df[metric].astype(float)
        thresholds = []
        for case_id, g in df.groupby("case_id"):
            sub = g.assign(_m=numeric.loc[g.index], _miou=g["gt_miou"])
            sub = sub.dropna(subset=["_m", "_miou"])
            if sub.empty:
                continue
            q75 = sub["_miou"].quantile(0.75)
            top = sub[sub["_miou"] >= q75]
            if top.empty:
                top = sub.nlargest(max(4, len(sub) // 4), "_miou")
            thresholds.append({"case_id": case_id, "p10": float(top["_m"].quantile(0.10)), "p90": float(top["_m"].quantile(0.90))})
        if thresholds:
            p10 = float(np.median([t["p10"] for t in thresholds]))
            p90 = float(np.median([t["p90"] for t in thresholds]))
            guardrails["metrics"][metric] = {
                "min": round(p10, 4),
                "max": round(p90, 4),
                "spearman_rho": float(corr_df.loc[corr_df["metric"] == metric, "spearman_rho"].iloc[0]),
                "per_ring_thresholds": thresholds,
            }

    (run_root / "detection_guardrails.json").write_text(json.dumps(guardrails, indent=2) + "\n", encoding="utf-8")

    summary_lines = [
        "# BO experience summary (detection layout)\n",
        f"- Total trials: {len(df)}",
        f"- Rings: {df['case_id'].nunique() if 'case_id' in df.columns else 'n/a'}",
        f"- mIoU range: {df['gt_miou'].min():.4f} – {df['gt_miou'].max():.4f}",
        f"- Metrics with |ρ|≥{args.rho_threshold}: {len(corr_df[corr_df['keep']])}",
        f"- Guardrails promoted: {list(guardrails['metrics'].keys())}",
        "\n## Handoff (Step 03+)\n",
        f"- `{run_root / 'bo_trials.csv'}` → proxy training",
        f"- `{run_root / 'detection_guardrails.json'}` → runtime QC",
        f"- `{run_root / 'parameter_sensitivity.md'}` → candidate bounds",
    ]
    (run_root / "bo_experience_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "n_trials": len(df),
        "metrics_kept": kept,
        "guardrail_count": len(guardrails["metrics"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
