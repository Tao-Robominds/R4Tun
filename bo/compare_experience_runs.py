#!/usr/bin/env python3
"""Compare honest BO experience runs (e.g. v3 vs v4 SAM4Tun prior)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BO_DIR = Path(__file__).resolve().parent
REPO_ROOT = _BO_DIR.parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))


def _load_trials(run_root: Path) -> pd.DataFrame:
    panel = run_root / "bo_trials.csv"
    if panel.is_file():
        return pd.read_csv(panel)
    parts = sorted(run_root.glob("*/*/bo_trials.csv"))
    if not parts:
        raise FileNotFoundError(f"No trials under {run_root}")
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def _ring_stratum(case_id: str, manifest: dict) -> str:
    for entry in manifest.get("rings", []):
        if entry.get("ring_key") == case_id:
            sc = entry.get("segment_count")
            return f"{sc}-seg" if sc else "unknown"
    return "unknown"


def _k_error(case_id: str, k_y: float) -> float | None:
    parts = case_id.split("/")
    if len(parts) != 2:
        return None
    gt_path = REPO_ROOT / "data" / "bo_calibration" / parts[0] / parts[1] / "gt_layout.json"
    if not gt_path.is_file():
        return None
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    H_path = REPO_ROOT / "data" / "bo_calibration" / parts[0] / parts[1] / "depth_map.npy"
    import numpy as np
    H = int(np.load(H_path).shape[0]) if H_path.is_file() else int(gt.get("H") or 0)
    if H <= 0:
        return None
    return float((k_y - float(gt["k_y"]) + H / 2) % H - H / 2)


def _ceiling_miou(case_id: str, manifest: dict) -> float | None:
    for entry in manifest.get("rings", []):
        if entry.get("ring_key") == case_id:
            return float(entry["ceiling_miou"]) if entry.get("ceiling_miou") is not None else None
    return None


def summarize_run(df: pd.DataFrame, manifest: dict, label: str) -> pd.DataFrame:
    rows = []
    for case_id, g in df.groupby("case_id"):
        bo = g[g["kind"].astype(str) == "bo"]
        warm_geo = g[g["kind"] == "geometric_0"]
        warm_sam = g[g["kind"] == "sam4tun_static"]
        warm_gt = g[g["kind"] == "gt_layout_ceiling_r"]
        if not warm_gt.empty:
            warm = warm_gt
        elif not warm_sam.empty:
            warm = warm_sam
        else:
            warm = warm_geo
        ceiling = _ceiling_miou(case_id, manifest)
        best_bo = float(g["gt_miou"].max())
        best_bo_pure = float(bo["gt_miou"].max()) if not bo.empty else best_bo
        warm_miou = float(warm["gt_miou"].iloc[0]) if not warm.empty else None
        best_k = g.loc[g["gt_miou"].idxmax(), "k_y"] if "k_y" in g.columns else None
        k_err = _k_error(case_id, float(best_k)) if best_k is not None and pd.notna(best_k) else None
        regret = float(ceiling - best_bo) if ceiling is not None else None
        rows.append({
            "run": label,
            "case_id": case_id,
            "stratum": _ring_stratum(case_id, manifest),
            "warm_kind": warm["kind"].iloc[0] if not warm.empty else None,
            "warm_miou": round(warm_miou, 4) if warm_miou is not None else None,
            "best_bo_miou": round(best_bo, 4),
            "best_bo_pure_gp_miou": round(best_bo_pure, 4),
            "regret_vs_ceiling": round(regret, 4) if regret is not None else None,
            "n_oracle": int(g["kind"].astype(str).str.startswith("gt_layout").sum()),
            "k_error_best_px": round(k_err, 1) if k_err is not None else None,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(REPO_ROOT / "logs" / "bo_experience_v3"))
    ap.add_argument("--candidate", default=str(REPO_ROOT / "logs" / "bo_experience_v4_sam4tun_prior"))
    ap.add_argument("--gt-derived", default=None, help="Optional v5 GT-anchor run root for 3-way compare")
    ap.add_argument("--manifest", default=str(REPO_ROOT / "data" / "bo_calibration" / "MANIFEST.json"))
    ap.add_argument("--out", default=None, help="Write vs_v3_summary.md under candidate run root")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    baseline_root = Path(args.baseline).resolve()
    candidate_root = Path(args.candidate).resolve()
    gt_root = Path(args.gt_derived).resolve() if args.gt_derived else None
    out_root = gt_root or candidate_root

    df_v3 = _load_trials(baseline_root)
    df_v4 = _load_trials(candidate_root)
    s3 = summarize_run(df_v3, manifest, "v3")
    s4 = summarize_run(df_v4, manifest, "v4")
    merged = s3.merge(
        s4,
        on="case_id",
        suffixes=("_v3", "_v4"),
        how="outer",
    )
    merged["stratum"] = merged["stratum_v4"].fillna(merged["stratum_v3"])
    merged["warm_miou_lift"] = merged["warm_miou_v4"] - merged["warm_miou_v3"]
    merged["best_bo_lift"] = merged["best_bo_miou_v4"] - merged["best_bo_miou_v3"]

    if gt_root is not None:
        df_v5 = _load_trials(gt_root)
        s5 = summarize_run(df_v5, manifest, "v5_gt")
        s5r = s5.rename(columns={c: f"{c}_v5" for c in s5.columns if c != "case_id"})
        merged = merged.merge(s5r, on="case_id", how="outer")
        merged["warm_miou_lift_v5_vs_v3"] = merged["warm_miou_v5"] - merged["warm_miou_v3"]
        merged["best_bo_lift_v5_vs_v3"] = merged["best_bo_miou_v5"] - merged["best_bo_miou_v3"]

    csv_path = out_root / ("vs_v3_v4_v5_comparison.csv" if gt_root else "vs_v3_comparison.csv")
    merged.to_csv(csv_path, index=False)

    def _mean(col: str, stratum: str | None = None) -> float | None:
        sub = merged
        if stratum:
            sub = merged[merged["stratum"] == stratum]
        if col not in sub.columns or sub[col].isna().all():
            return None
        return round(float(sub[col].mean()), 4)

    if gt_root is not None:
        lines = [
            "# v3 / v4 / v5 GT-anchor experience comparison",
            "",
            f"- v3 (failure memory): `{baseline_root}`",
            f"- v4 (SAM4Tun prior): `{candidate_root}`",
            f"- v5 (GT-derived anchor): `{gt_root}`",
            "",
            "## Panel summary",
            "",
            "| Metric | v3 | v4 | v5 |",
            "|--------|---:|---:|---:|",
            f"| Mean best BO mIoU | {_mean('best_bo_miou_v3')} | {_mean('best_bo_miou_v4')} | {_mean('best_bo_miou_v5')} |",
            f"| Mean warm-start mIoU | {_mean('warm_miou_v3')} | {_mean('warm_miou_v4')} | {_mean('warm_miou_v5')} |",
            f"| Mean regret vs ceiling | {_mean('regret_vs_ceiling_v3')} | {_mean('regret_vs_ceiling_v4')} | {_mean('regret_vs_ceiling_v5')} |",
            f"| Oracle / gt_layout trials | {int(s3['n_oracle'].sum())} | {int(s4['n_oracle'].sum())} | {int(s5['n_oracle'].sum())} |",
            "",
            "## Per ring",
            "",
            "| case_id | warm_v3 | warm_v4 | warm_v5 | best_v3 | best_v4 | best_v5 |",
            "|---------|--------:|--------:|--------:|--------:|--------:|--------:|",
        ]
        for _, r in merged.iterrows():
            lines.append(
                f"| {r['case_id']} | {r.get('warm_miou_v3')} | {r.get('warm_miou_v4')} | {r.get('warm_miou_v5')} | "
                f"{r.get('best_bo_miou_v3')} | {r.get('best_bo_miou_v4')} | {r.get('best_bo_miou_v5')} |"
            )
        md_path = Path(args.out) if args.out else out_root / "vs_v3_v4_v5_summary.md"
    else:
        lines = [
            "# v4 SAM4Tun prior vs v3 geometric warm-start",
            "",
            f"- Baseline: `{baseline_root}`",
            f"- Candidate: `{candidate_root}`",
            "",
            "## Panel summary",
            "",
            "| Metric | v3 | v4 | lift |",
            "|--------|-----|-----|------|",
            f"| Mean best BO mIoU (all) | {_mean('best_bo_miou_v3')} | {_mean('best_bo_miou_v4')} | {_mean('best_bo_lift')} |",
            f"| Mean warm-start mIoU | {_mean('warm_miou_v3')} | {_mean('warm_miou_v4')} | {_mean('warm_miou_lift')} |",
            f"| Mean regret vs ceiling | {_mean('regret_vs_ceiling_v3')} | {_mean('regret_vs_ceiling_v4')} | — |",
            f"| Oracle trials (sum) | {int(s3['n_oracle'].sum())} | {int(s4['n_oracle'].sum())} | — |",
            "",
            "## 6-seg stratum",
            "",
            f"| Mean best BO mIoU | {_mean('best_bo_miou_v3', '6-seg')} | {_mean('best_bo_miou_v4', '6-seg')} | {_mean('best_bo_lift', '6-seg')} |",
            f"| Mean warm-start mIoU | {_mean('warm_miou_v3', '6-seg')} | {_mean('warm_miou_v4', '6-seg')} | {_mean('warm_miou_lift', '6-seg')} |",
            "",
            "## 7-seg stratum",
            "",
            f"| Mean best BO mIoU | {_mean('best_bo_miou_v3', '7-seg')} | {_mean('best_bo_miou_v4', '7-seg')} | {_mean('best_bo_lift', '7-seg')} |",
            f"| Mean warm-start mIoU | {_mean('warm_miou_v3', '7-seg')} | {_mean('warm_miou_v4', '7-seg')} | {_mean('warm_miou_lift', '7-seg')} |",
            "",
            "## Per ring",
            "",
            "| case_id | stratum | warm_v3 | warm_v4 | best_bo_v3 | best_bo_v4 | warm_lift | bo_lift |",
            "|---------|---------|---------|---------|------------|------------|-----------|---------|",
        ]
        for _, r in merged.iterrows():
            lines.append(
                f"| {r['case_id']} | {r.get('stratum', '')} | {r.get('warm_miou_v3')} | {r.get('warm_miou_v4')} | "
                f"{r.get('best_bo_miou_v3')} | {r.get('best_bo_miou_v4')} | {r.get('warm_miou_lift')} | {r.get('best_bo_lift')} |"
            )
        md_path = Path(args.out) if args.out else out_root / "vs_v3_summary.md"
    lines.extend(["",])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
