#!/usr/bin/env python3
"""Stream L proxy calibration: Spearman vs gt_miou on layout/form features."""
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

from lib.proxy4tun_train import L_PROXY_ALLOWLIST  # noqa: E402

L_PROXY_CANDIDATES = L_PROXY_ALLOWLIST


def _spearman_col(df: pd.DataFrame, col: str, target: str = "gt_miou") -> float | None:
    if col not in df.columns:
        return None
    sub = df[[col, target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 8 or sub[col].std() < 1e-12:
        return None
    r, _ = spearmanr(sub[col], sub[target])
    return float(r) if np.isfinite(r) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default=str(REPO_ROOT / "logs/proxy4tun/stream_l"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "logs/proxy4tun/analysis"))
    ap.add_argument("--rho-threshold", type=float, default=0.15)
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trials_path = run_root / "bo_trials.csv"
    if not trials_path.is_file():
        raise FileNotFoundError(f"Missing {trials_path}")

    df = pd.read_csv(trials_path, low_memory=False)
    df["gt_miou"] = pd.to_numeric(df["gt_miou"], errors="coerce")

    ring_rows = []
    for case_id, g in df.groupby("case_id"):
        row = {"case_id": case_id, "n_trials": len(g)}
        for col in L_PROXY_CANDIDATES:
            r = _spearman_col(g, col)
            if r is not None:
                row[f"rho_{col}"] = round(r, 4)
        ring_rows.append(row)
    pd.DataFrame(ring_rows).to_csv(out_dir / "stream_l_within_ring_spearman.csv", index=False)

    pooled = []
    for col in L_PROXY_CANDIDATES:
        r = _spearman_col(df, col)
        if r is not None:
            pooled.append({"feature": col, "pooled_spearman": round(r, 4), "abs": round(abs(r), 4)})
    pooled_df = pd.DataFrame(pooled).sort_values("abs", ascending=False)
    pooled_df.to_csv(out_dir / "stream_l_pooled_spearman.csv", index=False)

    n_rings_pass = 0
    for case_id, g in df.groupby("case_id"):
        ok = any(
            abs(_spearman_col(g, c) or 0) >= args.rho_threshold
            for c in ("arc_width_entropy", "r_surface_min_frac", "hough_oblique_threshold")
        )
        if ok:
            n_rings_pass += 1

    gate = {
        "run_root": str(run_root),
        "n_trials": int(len(df)),
        "n_rings": int(df["case_id"].nunique()),
        "rings_with_rho_ge_threshold": n_rings_pass,
        "rho_threshold": args.rho_threshold,
        "top_pooled_features": pooled_df.head(8).to_dict(orient="records"),
    }
    (out_dir / "stream_l_proxy_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")

    lines = ["# Stream L proxy analysis\n", f"Trials: {len(df)} | Rings: {df['case_id'].nunique()}\n"]
    lines.append("## Pooled Spearman (top)\n")
    lines.append(pooled_df.head(10).to_csv(index=False))
    (out_dir / "stream_l_proxy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"rings_pass": n_rings_pass, "top": gate["top_pooled_features"][:4]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
