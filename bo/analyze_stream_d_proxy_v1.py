#!/usr/bin/env python3
"""Stream D proxy calibration: Spearman vs gt_miou, direction_tier strata, panel table."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_BO = Path(__file__).resolve().parent
REPO_ROOT = _BO.parent
if str(_BO) not in sys.path:
    sys.path.insert(0, str(_BO))

D_PROXY_CANDIDATES = [
    "direction_margin",
    "template_margin_minus_plus",
    "template_match_score_plus",
    "template_match_score_minus",
    "direction_score_plus",
    "direction_score_minus",
    "gt_miou_plus",
    "gt_miou_minus",
    "oracle_branch_hit",
]


def _spearman_col(df: pd.DataFrame, col: str, target: str = "gt_miou") -> float | None:
    if col not in df.columns:
        return None
    sub = df[[col, target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 8 or sub[col].std() < 1e-12:
        return None
    r, _ = spearmanr(sub[col], sub[target])
    return float(r) if np.isfinite(r) else None


def _panel_comparison_table(run_root: Path) -> pd.DataFrame:
    stream_k = REPO_ROOT / "logs" / "proxy4tun" / "stream_k"
    rows = []
    for ring_dir in sorted((REPO_ROOT / "data" / "bo_calibration").glob("*/*")):
        if not ring_dir.name.startswith("r"):
            continue
        tunnel = ring_dir.parent.name
        case_id = f"{tunnel}/{ring_dir.name}"
        sk_best = d_best = None
        sk_path = stream_k / tunnel / ring_dir.name / "k_best_for_stream_d.json"
        if sk_path.is_file():
            sk_best = float(json.loads(sk_path.read_text())["best_bo_miou"])
        dtrials = run_root / tunnel / ring_dir.name / "bo_trials.csv"
        twin_spread = oracle_hit_rate = None
        if dtrials.is_file():
            ddf = pd.read_csv(dtrials)
            i = ddf["gt_miou"].idxmax()
            d_best = float(ddf.loc[i, "gt_miou"])
            base = ddf[ddf["kind"] == "twin_baseline"]
            if not base.empty:
                row = base.iloc[0]
                mp, mm = row.get("gt_miou_plus"), row.get("gt_miou_minus")
                if pd.notna(mp) and pd.notna(mm):
                    twin_spread = abs(float(mp) - float(mm))
            if "oracle_branch_hit" in ddf.columns:
                hits = ddf["oracle_branch_hit"].dropna()
                if len(hits):
                    oracle_hit_rate = float(hits.astype(bool).mean())
        rows.append({
            "case_id": case_id,
            "stream_k_best_miou": round(sk_best, 4) if sk_best is not None else None,
            "stream_d_best_miou": round(d_best, 4) if d_best is not None else None,
            "lift_vs_stream_k": round(d_best - sk_best, 4) if d_best is not None and sk_best else None,
            "twin_miou_spread": round(twin_spread, 4) if twin_spread is not None else None,
            "oracle_hit_rate": round(oracle_hit_rate, 4) if oracle_hit_rate is not None else None,
            "direction_tier_gt": (
                str(ddf["direction_tier_gt"].iloc[0])
                if dtrials.is_file() and "direction_tier_gt" in ddf.columns
                else None
            ),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default=str(REPO_ROOT / "logs" / "proxy4tun" / "stream_d"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "logs" / "proxy4tun" / "analysis"))
    ap.add_argument("--rho-threshold", type=float, default=0.15)
    ap.add_argument("--guardrail-threshold", type=float, default=0.3)
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(run_root.glob("*/*/bo_trials.csv"))
    if not paths:
        print(f"No trials under {run_root}", file=sys.stderr)
        return 1
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    panel = _panel_comparison_table(run_root)
    panel.to_csv(out_dir / "stream_d_panel_comparison.csv", index=False)

    ring_rows = []
    for case_id, g in df.groupby("case_id"):
        rec = {"case_id": case_id, "n_trials": len(g)}
        for feat in D_PROXY_CANDIDATES:
            r = _spearman_col(g, feat)
            rec[feat] = r
            rec[f"abs_{feat}"] = abs(r) if r is not None else None
        ring_rows.append(rec)
    ring_df = pd.DataFrame(ring_rows)
    ring_df.to_csv(out_dir / "stream_d_within_ring_spearman.csv", index=False)

    pooled = []
    for feat in D_PROXY_CANDIDATES:
        r = _spearman_col(df, feat)
        pooled.append({
            "feature": feat,
            "pooled_spearman": r,
            "abs": abs(r) if r is not None else None,
        })
    pooled_df = pd.DataFrame(pooled).sort_values("abs", ascending=False, na_position="last")
    pooled_df.to_csv(out_dir / "stream_d_pooled_spearman.csv", index=False)

    strata: dict[str, list] = {}
    if "direction_tier_gt" in df.columns:
        for tier, sg in df.groupby("direction_tier_gt"):
            strata[str(tier)] = [
                {"feature": f, "pooled_spearman": round(_spearman_col(sg, f) or 0.0, 4)}
                for f in D_PROXY_CANDIDATES[:6]
            ]

    rings_pass = 0
    for _, row in ring_df.iterrows():
        cols = [c for c in ring_df.columns if c.startswith("abs_")]
        if any((row.get(c) or 0) >= args.rho_threshold for c in cols):
            rings_pass += 1

    guardrails = pooled_df[pooled_df["abs"] >= args.guardrail_threshold]["feature"].tolist()
    gate = {
        "run_root": str(run_root),
        "n_trials": int(len(df)),
        "n_rings": int(df["case_id"].nunique()),
        "rings_with_rho_ge_threshold": rings_pass,
        "rho_threshold": args.rho_threshold,
        "top_pooled_features": pooled_df.head(8).to_dict(orient="records"),
        "stratified_pooled": strata,
        "guardrails_promoted": guardrails,
        "panel_comparison_csv": str(out_dir / "stream_d_panel_comparison.csv"),
    }
    (out_dir / "stream_d_proxy_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    (out_dir / "stream_d_guardrails.json").write_text(
        json.dumps({"promoted": guardrails}, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stream D proxy report",
        "",
        f"Trials: {len(df)} across {df['case_id'].nunique()} rings",
        "",
        "## Panel comparison",
        "",
        panel.to_csv(index=False),
        "",
        "## Top pooled Spearman",
        "",
        pooled_df.head(10).to_csv(index=False),
    ]
    (out_dir / "stream_d_proxy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
