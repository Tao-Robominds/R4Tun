#!/usr/bin/env python3
"""Stream K proxy calibration: Spearman vs gt_miou, regular/irregular strata, panel table."""
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

K_PROXY_CANDIDATES = [
    "k_y_frac",
    "layout_k_center_norm",
    "k_anchor_dist_sam_frac",
    "k_anchor_dist_line_frac",
    "line_detection_confidence_K",
    "rho_K",
    "det_k_confidence_avg",
    "det_k_count_match",
    "det_min_y_gap_px",
    "det_y_coverage_pct",
    "finite_ratio",
    "row_nonempty_ratio",
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
    rows = []
    v4_root = REPO_ROOT / "logs" / "bo_experience_v4_sam4tun_prior"
    stream_l = REPO_ROOT / "logs" / "proxy4tun" / "stream_l"
    for ring_dir in sorted((REPO_ROOT / "data" / "bo_calibration").glob("*/*")):
        if not ring_dir.name.startswith("r"):
            continue
        tunnel = ring_dir.parent.name
        case_id = f"{tunnel}/{ring_dir.name}"
        H = int(np.load(ring_dir / "depth_map.npy").shape[0])
        gt_k = float(json.loads((ring_dir / "gt_layout.json").read_text())["k_y"])
        gt_f = gt_k / H
        sam_path = REPO_ROOT / "logs" / "proxy4tun" / "sam4tun_prior" / case_id.replace("/", "_") / "sam4tun_prior.json"
        sam_f = None
        if sam_path.is_file():
            sam_f = float(json.loads(sam_path.read_text())["k_y"]) / H
        k_best_f = v4_best = stream_l_best = None
        k_trials = run_root / tunnel / ring_dir.name / "bo_trials.csv"
        if k_trials.is_file():
            kdf = pd.read_csv(k_trials)
            i = kdf["gt_miou"].idxmax()
            k_best_f = float(kdf.loc[i, "k_y_frac"])
            k_best_m = float(kdf.loc[i, "gt_miou"])
        else:
            k_best_m = None
        v4t = v4_root / tunnel / ring_dir.name / "bo_trials.csv"
        if v4t.is_file():
            vdf = pd.read_csv(v4t)
            v4_best = float(vdf.loc[vdf["gt_miou"].idxmax(), "gt_miou"])
        sl = stream_l / tunnel / ring_dir.name / "best_bo_trial.json"
        if sl.is_file():
            stream_l_best = float(json.loads(sl.read_text())["best_bo_miou"])
        rows.append({
            "case_id": case_id,
            "gt_k_frac": round(gt_f, 4),
            "sam_k_frac": round(sam_f, 4) if sam_f is not None else None,
            "stream_k_best_k_frac": round(k_best_f, 4) if k_best_f is not None else None,
            "abs_best_minus_gt": round(abs(k_best_f - gt_f), 4) if k_best_f is not None else None,
            "stream_k_best_miou": round(k_best_m, 4) if k_best_m is not None else None,
            "stream_l_best_miou": round(stream_l_best, 4) if stream_l_best is not None else None,
            "v4_best_miou": round(v4_best, 4) if v4_best is not None else None,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default=str(REPO_ROOT / "logs" / "proxy4tun" / "stream_k"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "logs" / "proxy4tun" / "analysis"))
    ap.add_argument("--rho-threshold", type=float, default=0.15)
    ap.add_argument("--guardrail-threshold", type=float, default=0.3)
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trials_path = run_root / "bo_trials.csv"
    if not trials_path.is_file():
        raise FileNotFoundError(f"Missing {trials_path}")

    df = pd.read_csv(trials_path, low_memory=False)
    df["gt_miou"] = pd.to_numeric(df["gt_miou"], errors="coerce")
    if "ring_is_regular" in df.columns:
        df["ring_is_regular"] = df["ring_is_regular"].astype(str).str.lower().isin(
            ("true", "1", "yes")
        )

    panel = _panel_comparison_table(run_root)
    panel.to_csv(out_dir / "stream_k_panel_comparison.csv", index=False)

    ring_rows = []
    for case_id, g in df.groupby("case_id"):
        row = {"case_id": case_id, "n_trials": len(g)}
        for col in K_PROXY_CANDIDATES:
            r = _spearman_col(g, col)
            if r is not None:
                row[f"rho_{col}"] = round(r, 4)
        ring_rows.append(row)
    ring_df = pd.DataFrame(ring_rows)
    ring_df.to_csv(out_dir / "stream_k_within_ring_spearman.csv", index=False)

    pooled = []
    for col in K_PROXY_CANDIDATES:
        r = _spearman_col(df, col)
        if r is not None:
            pooled.append({"feature": col, "pooled_spearman": round(r, 4), "abs": round(abs(r), 4)})
    pooled_df = pd.DataFrame(pooled).sort_values("abs", ascending=False)
    pooled_df.to_csv(out_dir / "stream_k_pooled_spearman.csv", index=False)

    strata = {}
    if "ring_is_regular" in df.columns:
        for label, sub in [("regular", df[df["ring_is_regular"]]), ("irregular", df[~df["ring_is_regular"]])]:
            srows = []
            for col in K_PROXY_CANDIDATES:
                r = _spearman_col(sub, col)
                if r is not None:
                    srows.append({"feature": col, "pooled_spearman": round(r, 4)})
            strata[label] = srows

    n_rings_pass = 0
    for case_id, g in df.groupby("case_id"):
        ok = any(
            abs(_spearman_col(g, c) or 0) >= args.rho_threshold
            for c in ("k_y_frac", "layout_k_center_norm", "rho_K")
        )
        if ok:
            n_rings_pass += 1

    guardrails = pooled_df[pooled_df["abs"] >= args.guardrail_threshold]["feature"].tolist()

    gate = {
        "run_root": str(run_root),
        "n_trials": int(len(df)),
        "n_rings": int(df["case_id"].nunique()),
        "rings_with_rho_ge_threshold": n_rings_pass,
        "rho_threshold": args.rho_threshold,
        "top_pooled_features": pooled_df.head(8).to_dict(orient="records"),
        "stratified_pooled": strata,
        "guardrails_promoted": guardrails,
        "panel_comparison_csv": str(out_dir / "stream_k_panel_comparison.csv"),
    }
    (out_dir / "stream_k_proxy_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    (out_dir / "stream_k_guardrails.json").write_text(
        json.dumps({"features": guardrails, "threshold": args.guardrail_threshold}, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = ["# Stream K proxy analysis\n", f"Trials: {len(df)} | Rings: {df['case_id'].nunique()}\n"]
    lines.append("## Pooled Spearman (top)\n")
    lines.append(pooled_df.head(10).to_csv(index=False))
    lines.append("\n## Stratified\n")
    for k, v in strata.items():
        lines.append(f"### {k}\n")
        if v:
            lines.append(pd.DataFrame(v).to_csv(index=False))
        lines.append("")
    (out_dir / "stream_k_proxy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"gate": gate["top_pooled_features"][:4], "rings_pass": n_rings_pass}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
