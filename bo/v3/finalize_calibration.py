"""Step 4 — finalize calibration artefacts under data/v3/calibration/.

This script:
  1. Reads per-ring intrinsics CSVs collected by ``run_v3_calibration``.
  2. Re-runs the lightweight aggregation in :mod:`bo.v3.aggregate_calibration`.
  3. Derives empirical guardrail thresholds from the calibration corpus
     (overriding the manuscript-anchored placeholders): for each top-
     diagnostic intrinsic we report (a) the value at the 25th-percentile
     of mIoU(fixed) — i.e. the "min good" cut, and (b) Spearman vs mIoU.
  4. Adds a per-ring baseline-vs-BO summary to the report.

Outputs (overwrites prior aggregation):
  data/v3/calibration/
    sensitive_parameters.json
    diagnostic_intrinsics.json
    guardrails.json                 (now empirical)
    aggregate_summary.json
    calibration_report.md           (extended with per-ring scoreboard)
    baseline_vs_bo.csv              (NEW)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
logger = logging.getLogger("bo.v3.finalize")

from bo.v3 import aggregate_calibration as agg_mod


def _load_intrinsics(ring_key: str, stage: str, label: str) -> pd.DataFrame | None:
    tid, ring = ring_key.split("/", 1)
    path = REPO_ROOT / "logs" / "v3" / "bo_calibration" / tid / ring / stage / label / "intrinsics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_baseline(rings: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    bsum = REPO_ROOT / "logs" / "v3" / "bo_calibration" / "baseline_summary.csv"
    if not bsum.exists():
        return out
    with open(bsum, newline="") as f:
        for row in csv.DictReader(f):
            rk = row["ring_key"]
            if rk in rings:
                out[rk] = {
                    "fixed": float(row["miou_fixed_class"]),
                    "perm": float(row["miou_permutation"]),
                }
    return out


def _empirical_guardrails(
    rings: list[str], stage: str, label: str, top_intrinsics: list[str]
) -> dict[str, dict[str, Any]]:
    """For each top-diagnostic intrinsic, derive an empirical lower bound
    that picks out the upper-quartile-mIoU trials across the calibration
    corpus.
    """
    rows = []
    for rk in rings:
        df = _load_intrinsics(rk, stage, label)
        if df is None or df.empty:
            continue
        ok = df[df["status"] == "ok"]
        if ok.empty:
            continue
        rows.append(ok)
    if not rows:
        return {}
    pooled = pd.concat(rows, ignore_index=True)
    miou = pd.to_numeric(pooled["miou_fixed_class"], errors="coerce")
    if miou.dropna().empty:
        return {}
    top_mask = miou >= miou.quantile(0.75)
    bot_mask = miou <= miou.quantile(0.25)

    out: dict[str, dict[str, Any]] = {}
    for intr in top_intrinsics:
        if intr not in pooled.columns:
            continue
        col = pooled[intr]
        if col.dtype == bool:
            col = col.astype(float)
        v = pd.to_numeric(col, errors="coerce").astype(float)
        if v.dropna().empty:
            continue
        # Best separating cut: the top-quartile's 25th-percentile.
        cut_top = float(v[top_mask].dropna().quantile(0.25)) if v[top_mask].dropna().size else float("nan")
        cut_pooled = float(v.dropna().quantile(0.50))
        spearman_y = pd.Series(miou).rank().to_numpy()
        spearman_x = pd.Series(v).rank().to_numpy()
        mask = np.isfinite(v) & np.isfinite(miou)
        rho = (
            float(np.corrcoef(spearman_x[mask], spearman_y[mask])[0, 1])
            if mask.sum() > 4 and pd.Series(spearman_x[mask]).std() > 0
            and pd.Series(spearman_y[mask]).std() > 0
            else None
        )
        out[intr] = {
            "stage": stage,
            "min_good_p25_of_top_quartile": cut_top,
            "median_pooled": cut_pooled,
            "pooled_spearman_vs_miou": rho,
            "n_pooled_trials": int(mask.sum()),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize v3 BO calibration artefacts")
    parser.add_argument("--label", default="bo30")
    parser.add_argument("--stages", nargs="+", default=["preprocessing", "detection"])
    parser.add_argument("--out-dir", default="data/v3/calibration")
    args = parser.parse_args(argv)

    panel = agg_mod._load_panel()
    rings = list(panel.keys())
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = agg_mod.aggregate(rings, stages=args.stages, out_dir=out_dir, label=args.label)

    diag = json.loads((out_dir / "diagnostic_intrinsics.json").read_text())

    # Empirical guardrails per stage from top-5 diagnostic intrinsics.
    guard: dict[str, Any] = {}
    for stage in args.stages:
        top_names = [d["name"] for d in diag.get(stage, [])[:5]]
        guard[stage] = _empirical_guardrails(rings, stage, args.label, top_names)
    # Group preprocessing intrinsics into G_pre; segmentation into G_layout;
    # cross-stage permutation gap into G_stability.
    g_pre_candidates = {k: v for k, v in guard.get("preprocessing", {}).items() if k.startswith("pre_")}
    g_layout_candidates = {k: v for k, v in guard.get("preprocessing", {}).items() if k.startswith("seg_")}
    # Permutation gap (perm - fixed) computed over pooled corpus.
    pooled = []
    for rk in rings:
        for stage in args.stages:
            df = _load_intrinsics(rk, stage, args.label)
            if df is None:
                continue
            ok = df[df["status"] == "ok"]
            pooled.append(ok)
    pdf = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()
    if not pdf.empty:
        gap = pd.to_numeric(pdf["miou_permutation"], errors="coerce") - pd.to_numeric(pdf["miou_fixed_class"], errors="coerce")
        median_gap = float(gap.dropna().median()) if not gap.dropna().empty else None
    else:
        median_gap = None
    guard_out = {
        "G_pre": {
            "intrinsic_candidates": g_pre_candidates,
            "rationale": "G_pre groups preprocessing-side intrinsics (pre_valid_ratio, "
            "pre_depth_shape_w) calibrated to the upper-quartile mIoU regime.",
        },
        "G_layout": {
            "intrinsic_candidates": g_layout_candidates,
            "rationale": "G_layout groups deterministic-segmentation completeness/coverage "
            "intrinsics (seg_segment_type_completeness, seg_ring_completeness_avg, "
            "seg_mask_coverage_pct) calibrated to the upper-quartile mIoU regime.",
        },
        "G_stability": {
            "median_perm_minus_fixed_gap": median_gap,
            "rationale": "G_stability tracks the permutation-vs-fixed mIoU gap, a proxy "
            "for canonical-anchoring residual under gravity unwrap.",
        },
        "raw_per_stage_intrinsic_thresholds": guard,
    }
    (out_dir / "guardrails.json").write_text(json.dumps(guard_out, indent=2, default=str) + "\n")

    baseline = _load_baseline(rings)
    bo_pre = {
        rk: json.loads((REPO_ROOT / "logs" / "v3" / "bo_calibration" / rk.split("/")[0] / rk.split("/")[1] / "preprocessing" / args.label / "summary.json").read_text())["best"]
        for rk in rings
    }
    bo_det = {
        rk: json.loads((REPO_ROOT / "logs" / "v3" / "bo_calibration" / rk.split("/")[0] / rk.split("/")[1] / "detection" / args.label / "summary.json").read_text())["best"]
        for rk in rings
    }
    bvb_path = out_dir / "baseline_vs_bo.csv"
    with open(bvb_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ring_key", "regime_label",
            "baseline_fixed", "baseline_perm",
            "preproc_bo_fixed", "preproc_bo_perm", "preproc_bo_trial",
            "detection_bo_fixed", "detection_bo_perm", "detection_bo_trial",
        ])
        for rk in rings:
            b = baseline.get(rk, {"fixed": None, "perm": None})
            p = bo_pre[rk]
            d = bo_det[rk]
            w.writerow([
                rk, panel[rk].get("regime_label"),
                b["fixed"], b["perm"],
                p["miou_fixed_class"], p["miou_permutation"], p["trial_index"],
                d["miou_fixed_class"], d["miou_permutation"], d["trial_index"],
            ])

    # Extend calibration_report.md with the per-ring scoreboard.
    report_path = out_dir / "calibration_report.md"
    body = report_path.read_text() if report_path.exists() else ""
    extra = ["", "## Per-ring baseline → BO scoreboard", ""]
    extra.append("| ring | regime | baseline fixed | preproc BO fixed | detection BO fixed | preproc Δ | detection Δ |")
    extra.append("|---|---|---:|---:|---:|---:|---:|")
    for rk in rings:
        b = baseline.get(rk, {"fixed": None})
        p = bo_pre[rk]["miou_fixed_class"]
        d = bo_det[rk]["miou_fixed_class"]
        bf = b["fixed"]
        dpre = (p - bf) if bf is not None else None
        ddet = (d - bf) if bf is not None else None
        extra.append(
            f"| `{rk}` | {panel[rk].get('regime_label','—')} | "
            f"{bf:.3f} | {p:.3f} | {d:.3f} | "
            f"{dpre:+.3f} | {ddet:+.3f} |"
        )
    extra += ["", "## Empirical guardrail thresholds", ""]
    extra.append("Thresholds are derived from the upper-quartile mIoU regime over the 6-ring × 30-trial × 2-stage corpus.")
    for grp_name, grp in (("G_pre", g_pre_candidates), ("G_layout", g_layout_candidates)):
        if not grp:
            continue
        extra.append("")
        extra.append(f"### {grp_name}")
        extra.append("| intrinsic | min-good cut (p25 of top quartile) | pooled median | ρ vs mIoU |")
        extra.append("|---|---:|---:|---:|")
        for k, v in grp.items():
            extra.append(
                f"| `{k}` | {v['min_good_p25_of_top_quartile']:.4g} | "
                f"{v['median_pooled']:.4g} | "
                f"{v['pooled_spearman_vs_miou'] if v['pooled_spearman_vs_miou'] is not None else 'n/a'} |"
            )
    extra += [
        "",
        "### G_stability",
        f"- Median permutation-vs-fixed mIoU gap: {median_gap:.3f}" if median_gap is not None else "- (no data)",
    ]
    report_path.write_text(body + "\n" + "\n".join(extra) + "\n")

    logger.info("finalize complete; outputs at %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
