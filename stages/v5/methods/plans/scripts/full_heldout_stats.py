#!/usr/bin/env python3
"""Compute full-heldout statistics and significance tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "logs" / "gravity_v1" / "full_heldout"
COMPARE_CSV = OUT_ROOT / "full_heldout_variant_compare.csv"
REGISTRY_CSV = OUT_ROOT / "full_heldout_registry.csv"


def _bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    boots = []
    n = len(arr)
    for _ in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        boots.append(float(np.mean(sample)))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return lo, hi


def _paired_stats(a: pd.Series, b: pd.Series, name: str) -> dict[str, float | int | str]:
    va = a.astype(float).to_numpy()
    vb = b.astype(float).to_numpy()
    m = np.isfinite(va) & np.isfinite(vb)
    va = va[m]
    vb = vb[m]
    if len(va) == 0:
        return {"comparison": name, "n": 0}
    d = vb - va
    mean_d = float(np.mean(d))
    med_d = float(np.median(d))
    ci_lo, ci_hi = _bootstrap_ci(d)
    try:
        t_p = float(ttest_rel(vb, va).pvalue) if len(d) >= 2 else float("nan")
    except Exception:  # noqa: BLE001
        t_p = float("nan")
    try:
        w_p = float(wilcoxon(d).pvalue) if len(d) >= 2 else float("nan")
    except Exception:  # noqa: BLE001
        w_p = float("nan")
    return {
        "comparison": name,
        "n": int(len(d)),
        "mean_delta": mean_d,
        "median_delta": med_d,
        "bootstrap95_lo": ci_lo,
        "bootstrap95_hi": ci_hi,
        "t_test_p": t_p,
        "wilcoxon_p": w_p,
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(COMPARE_CSV)
    reg = pd.read_csv(REGISTRY_CSV)
    reg_small = reg[["ring_key", "segment_count", "difficulty", "density_group"]].rename(columns={"ring_key": "ring"})
    df = comp.merge(reg_small, on="ring", how="left")

    # Variant-level summary
    valid_bg = df.dropna(subset=["A0_baseline_canonical_mIoU", "A0_gravity_canonical_mIoU"]).copy()
    valid_gi = df.dropna(subset=["A0_gravity_canonical_mIoU", "A2_iter_canonical_mIoU"]).copy()
    valid_bi = df.dropna(subset=["A0_baseline_canonical_mIoU", "A2_iter_canonical_mIoU"]).copy()

    variant_rows = [
        {
            "variant": "A0_baseline",
            "n": int(df["A0_baseline_canonical_mIoU"].notna().sum()),
            "mean_canonical_mIoU": float(df["A0_baseline_canonical_mIoU"].mean()),
            "median_canonical_mIoU": float(df["A0_baseline_canonical_mIoU"].median()),
            "share_ge_040": float((df["A0_baseline_canonical_mIoU"] >= 0.40).mean()),
            "share_ge_050": float((df["A0_baseline_canonical_mIoU"] >= 0.50).mean()),
        },
        {
            "variant": "A0_gravity",
            "n": int(df["A0_gravity_canonical_mIoU"].notna().sum()),
            "mean_canonical_mIoU": float(df["A0_gravity_canonical_mIoU"].mean()),
            "median_canonical_mIoU": float(df["A0_gravity_canonical_mIoU"].median()),
            "share_ge_040": float((df["A0_gravity_canonical_mIoU"] >= 0.40).mean()),
            "share_ge_050": float((df["A0_gravity_canonical_mIoU"] >= 0.50).mean()),
        },
        {
            "variant": "A2_iter",
            "n": int(df["A2_iter_canonical_mIoU"].notna().sum()),
            "mean_canonical_mIoU": float(df["A2_iter_canonical_mIoU"].mean()),
            "median_canonical_mIoU": float(df["A2_iter_canonical_mIoU"].median()),
            "share_ge_040": float((df["A2_iter_canonical_mIoU"] >= 0.40).mean()),
            "share_ge_050": float((df["A2_iter_canonical_mIoU"] >= 0.50).mean()),
        },
    ]
    variant_df = pd.DataFrame(variant_rows)
    variant_df.to_csv(OUT_ROOT / "stats_variant_summary.csv", index=False)

    pair_df = pd.DataFrame(
        [
            _paired_stats(valid_bg["A0_baseline_canonical_mIoU"], valid_bg["A0_gravity_canonical_mIoU"], "A0->gravity"),
            _paired_stats(valid_gi["A0_gravity_canonical_mIoU"], valid_gi["A2_iter_canonical_mIoU"], "gravity->iter"),
            _paired_stats(valid_bi["A0_baseline_canonical_mIoU"], valid_bi["A2_iter_canonical_mIoU"], "A0->iter"),
        ]
    )
    pair_df.to_csv(OUT_ROOT / "stats_paired_tests.csv", index=False)

    # Stratified by segment_count / difficulty / density
    strat_cols = ["segment_count", "difficulty", "density_group"]
    strat_rows: list[dict[str, object]] = []
    for col in strat_cols:
        for key, sub in df.groupby(col):
            strat_rows.append(
                {
                    "stratum": col,
                    "value": key,
                    "n": int(len(sub)),
                    "A0_mean": float(sub["A0_baseline_canonical_mIoU"].mean()),
                    "gravity_mean": float(sub["A0_gravity_canonical_mIoU"].mean()),
                    "iter_mean": float(sub["A2_iter_canonical_mIoU"].mean()),
                    "delta_gravity_vs_A0": float((sub["A0_gravity_canonical_mIoU"] - sub["A0_baseline_canonical_mIoU"]).mean()),
                    "delta_iter_vs_gravity": float((sub["A2_iter_canonical_mIoU"] - sub["A0_gravity_canonical_mIoU"]).mean()),
                }
            )
    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(OUT_ROOT / "stats_stratified.csv", index=False)

    # Per-tunnel summary
    tunnel_df = (
        df.groupby("tunnel", as_index=False)
        .agg(
            n=("ring", "count"),
            A0_mean=("A0_baseline_canonical_mIoU", "mean"),
            gravity_mean=("A0_gravity_canonical_mIoU", "mean"),
            iter_mean=("A2_iter_canonical_mIoU", "mean"),
        )
    )
    tunnel_df["delta_gravity_vs_A0"] = tunnel_df["gravity_mean"] - tunnel_df["A0_mean"]
    tunnel_df["delta_iter_vs_gravity"] = tunnel_df["iter_mean"] - tunnel_df["gravity_mean"]
    tunnel_df.to_csv(OUT_ROOT / "stats_by_tunnel.csv", index=False)

    # json summary for manuscript injection
    summary = {
        "n_total_eligible": int(len(df)),
        "n_baseline": int(df["A0_baseline_canonical_mIoU"].notna().sum()),
        "n_gravity": int(df["A0_gravity_canonical_mIoU"].notna().sum()),
        "n_iter": int(df["A2_iter_canonical_mIoU"].notna().sum()),
        "variant_summary": variant_rows,
        "paired_tests": pair_df.to_dict(orient="records"),
    }
    (OUT_ROOT / "stats_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # human-readable
    md = []
    md.append("# Full-heldout statistical summary\n")
    md.append(f"- Eligible rings: **{len(df)}**")
    md.append(f"- Baseline available: **{summary['n_baseline']}**")
    md.append(f"- Gravity available: **{summary['n_gravity']}**")
    md.append(f"- Iterative available: **{summary['n_iter']}**")
    def _md_table(df_in: pd.DataFrame) -> str:
        cols = list(df_in.columns)
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, row in df_in.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    md.append("\n## Variant means\n")
    md.append(_md_table(variant_df))
    md.append("\n## Paired tests\n")
    md.append(_md_table(pair_df))
    (OUT_ROOT / "stats_summary.md").write_text("\n".join(md) + "\n")

    print("saved:")
    for p in [
        "stats_variant_summary.csv",
        "stats_paired_tests.csv",
        "stats_stratified.csv",
        "stats_by_tunnel.csv",
        "stats_summary.json",
        "stats_summary.md",
    ]:
        print("-", OUT_ROOT / p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
