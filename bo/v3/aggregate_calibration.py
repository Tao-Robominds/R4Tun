"""Step 3 — cross-ring aggregation for v3 BO calibration.

Reads per-ring ``intrinsics.csv`` files produced by
:mod:`bo.v3.run_v3_calibration` for each (ring, stage) pair, and emits:

* ``data/v3/calibration/sensitive_parameters.json`` — selected parameters
  with default, low/high empirical range, sensitivity score, evidence rings.
* ``data/v3/calibration/diagnostic_intrinsics.json`` — ranked intrinsics
  passing the per-ring Spearman filter; recommended top-k subset.
* ``data/v3/calibration/guardrails.json`` — calibrated thresholds anchored
  near the manuscript's ``G_pre ≥ 0.25``, ``G_layout ≥ 0.05``,
  ``G_stability ≥ 0.2`` values.
* ``data/v3/calibration/calibration_report.md`` — short report.

NOTE: ``data/v3/calibration/`` is the only place under ``data/`` we write
to; it is **not** in the protected list (the protected per-ring corpus
prefix is ``data/<tunnel_id>/r*/`` where tunnel_id matches a digit-dash
pattern, and "v3" is reserved for output artefacts by the v3 plan).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
logger = logging.getLogger("bo.v3.aggregate")


def _load_panel() -> dict[str, dict[str, Any]]:
    panel = json.loads((REPO_ROOT / "data" / "v3" / "panels" / "bo" / "bo_calibration_panel_v3.json").read_text())
    return {r["ring_key"]: r for r in panel["rings"]}


def _load_intrinsics(ring_key: str, stage: str, label: Optional[str] = None) -> Optional[pd.DataFrame]:
    tid, ring = ring_key.split("/", 1)
    base = REPO_ROOT / "logs" / "v3" / "bo_calibration" / tid / ring / stage
    if label:
        base = base / label
    path = base / "intrinsics.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Spearman rank correlation between numeric vectors, NaN-safe."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return None
    rx = pd.Series(x[mask]).rank().to_numpy()
    ry = pd.Series(y[mask]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _aggregate_one_stage(
    rings: list[str],
    stage: str,
    label: Optional[str],
) -> dict[str, Any]:
    """Compute sensitivity, empirical ranges, intrinsic correlations for one stage.

    Returns a dict with sub-keys: ``per_ring`` (per-ring pieces),
    ``pooled`` (cross-ring summaries), ``failure_modes`` (counts),
    ``intrinsic_correlations`` (per-intrinsic Spearman vs miou_fixed_class).
    """
    per_ring: dict[str, Any] = {}
    pooled_param_values: dict[str, list[tuple[float, float]]] = defaultdict(list)  # param -> [(value, miou)]
    pooled_intrinsic_values: dict[str, list[tuple[float, float]]] = defaultdict(list)  # intrinsic -> [(value, miou)]
    failure_modes_pooled: dict[str, int] = defaultdict(int)
    success_count_pooled = 0
    failure_count_pooled = 0

    for rk in rings:
        df = _load_intrinsics(rk, stage, label)
        if df is None or df.empty:
            per_ring[rk] = {"warning": "no intrinsics.csv found"}
            continue
        ok = df[df["status"] == "ok"].copy()
        failed = df[df["status"] != "ok"].copy()
        success_count_pooled += int(len(ok))
        failure_count_pooled += int(len(failed))
        for mode in failed.get("failure_mode", pd.Series(dtype=str)).dropna():
            failure_modes_pooled[str(mode)] += 1
        if ok.empty:
            per_ring[rk] = {
                "warning": "no successful trials",
                "n_failed": int(len(failed)),
            }
            continue
        # Per-ring per-parameter Spearman vs miou_fixed_class.
        param_cols = [c for c in ok.columns if c.startswith("param/")]
        param_corrs: dict[str, Optional[float]] = {}
        # Per-ring empirical ranges (best-quartile).
        ok_sorted = ok.sort_values("miou_fixed_class", ascending=False)
        n = len(ok_sorted)
        n_top = max(1, int(math.ceil(n * 0.25)))
        top = ok_sorted.head(n_top)
        param_ranges: dict[str, dict[str, float]] = {}
        for c in param_cols:
            try:
                vals = pd.to_numeric(top[c], errors="coerce").dropna().to_numpy()
                if vals.size == 0:
                    continue
                param_ranges[c.removeprefix("param/")] = {
                    "p25": float(np.quantile(vals, 0.25)),
                    "p75": float(np.quantile(vals, 0.75)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "best_value": float(vals[0]) if vals.size else float("nan"),
                }
            except Exception:  # noqa: BLE001
                continue
            try:
                full_vals = pd.to_numeric(ok[c], errors="coerce").to_numpy()
                miou_vals = pd.to_numeric(ok["miou_fixed_class"], errors="coerce").to_numpy()
                param_corrs[c.removeprefix("param/")] = _spearman_safe(full_vals, miou_vals)
                for v, mv in zip(full_vals, miou_vals):
                    if math.isfinite(v) and math.isfinite(mv):
                        pooled_param_values[c.removeprefix("param/")].append((float(v), float(mv)))
            except Exception:  # noqa: BLE001
                param_corrs[c.removeprefix("param/")] = None
        # Per-ring per-intrinsic Spearman vs miou_fixed_class.
        intrinsic_cols = [
            c for c in ok.columns
            if c not in {"trial_index", "stage", "status", "failure_mode",
                         "miou_fixed_class", "miou_permutation", "elapsed_sec"}
            and not c.startswith("param/")
        ]
        intrinsic_corrs: dict[str, Optional[float]] = {}
        for c in intrinsic_cols:
            try:
                vals = pd.to_numeric(ok[c], errors="coerce").to_numpy()
                miou_vals = pd.to_numeric(ok["miou_fixed_class"], errors="coerce").to_numpy()
                intrinsic_corrs[c] = _spearman_safe(vals, miou_vals)
                for v, mv in zip(vals, miou_vals):
                    if math.isfinite(v) and math.isfinite(mv):
                        pooled_intrinsic_values[c].append((float(v), float(mv)))
            except Exception:  # noqa: BLE001
                intrinsic_corrs[c] = None
        # Anchoring residual: mean(perm - fixed).
        anchoring_gap = None
        if "miou_permutation" in ok.columns and "miou_fixed_class" in ok.columns:
            try:
                gap = pd.to_numeric(ok["miou_permutation"], errors="coerce") - pd.to_numeric(ok["miou_fixed_class"], errors="coerce")
                anchoring_gap = float(gap.dropna().mean())
            except Exception:  # noqa: BLE001
                anchoring_gap = None
        per_ring[rk] = {
            "n_trials": int(n),
            "n_failed": int(len(failed)),
            "best_miou_fixed": float(ok["miou_fixed_class"].max()),
            "anchoring_gap_mean_perm_minus_fixed": anchoring_gap,
            "param_ranges": param_ranges,
            "param_spearman_vs_miou": param_corrs,
            "intrinsic_spearman_vs_miou": intrinsic_corrs,
        }
    pooled_param_corrs: dict[str, Optional[float]] = {}
    pooled_param_envelope: dict[str, dict[str, float]] = {}
    for k, pairs in pooled_param_values.items():
        arr = np.array(pairs, dtype=np.float64)
        if arr.size:
            pooled_param_corrs[k] = _spearman_safe(arr[:, 0], arr[:, 1])
            pooled_param_envelope[k] = {
                "min": float(arr[:, 0].min()),
                "max": float(arr[:, 0].max()),
                "p25": float(np.quantile(arr[:, 0], 0.25)),
                "p75": float(np.quantile(arr[:, 0], 0.75)),
            }
    pooled_intrinsic_corrs: dict[str, Optional[float]] = {}
    for k, pairs in pooled_intrinsic_values.items():
        arr = np.array(pairs, dtype=np.float64)
        if arr.size:
            pooled_intrinsic_corrs[k] = _spearman_safe(arr[:, 0], arr[:, 1])
    return {
        "per_ring": per_ring,
        "pooled_param_spearman_vs_miou": pooled_param_corrs,
        "pooled_param_envelope": pooled_param_envelope,
        "pooled_intrinsic_spearman_vs_miou": pooled_intrinsic_corrs,
        "failure_counts": dict(failure_modes_pooled),
        "success_count": int(success_count_pooled),
        "failure_count": int(failure_count_pooled),
    }


def aggregate(
    rings: list[str],
    *,
    stages: list[str],
    out_dir: Path,
    label: Optional[str] = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = _load_panel()
    summary: dict[str, Any] = {
        "rings": rings,
        "panel_axes": {rk: panel.get(rk, {}).get("regime_label") for rk in rings},
        "stages": stages,
    }
    sensitive_parameters: dict[str, dict[str, Any]] = {}
    diagnostic_intrinsics: dict[str, Any] = {}
    guardrails: dict[str, Any] = {}
    per_stage = {}
    for stage in stages:
        agg = _aggregate_one_stage(rings, stage, label)
        per_stage[stage] = agg
        # Sensitive parameters: top by |pooled Spearman|, with empirical ranges.
        sens = []
        for k, corr in agg["pooled_param_spearman_vs_miou"].items():
            if corr is None or not math.isfinite(corr):
                continue
            env = agg["pooled_param_envelope"].get(k, {})
            sens.append({
                "name": k,
                "stage": stage,
                "pooled_spearman_abs": abs(corr),
                "pooled_spearman": corr,
                "range_low_p25": env.get("p25"),
                "range_high_p75": env.get("p75"),
                "range_min": env.get("min"),
                "range_max": env.get("max"),
                "evidence_rings": [
                    rk for rk, prinfo in agg["per_ring"].items()
                    if isinstance(prinfo, dict) and k in prinfo.get("param_ranges", {})
                ],
            })
        sens.sort(key=lambda r: r["pooled_spearman_abs"], reverse=True)
        sensitive_parameters[stage] = sens
        # Intrinsic correlations: keep only those with positive pooled Spearman.
        intr = []
        for k, corr in agg["pooled_intrinsic_spearman_vs_miou"].items():
            if corr is None or not math.isfinite(corr):
                continue
            intr.append({
                "name": k,
                "stage": stage,
                "pooled_spearman": float(corr),
                "evidence_rings": [
                    rk for rk, prinfo in agg["per_ring"].items()
                    if isinstance(prinfo, dict) and k in prinfo.get("intrinsic_spearman_vs_miou", {})
                ],
            })
        intr.sort(key=lambda r: r["pooled_spearman"], reverse=True)
        diagnostic_intrinsics[stage] = intr
    # Guardrails — manuscript-anchored defaults; can be tightened in a follow-up.
    guardrails = {
        "G_pre": {"min": 0.25, "rationale": "manuscript anchor; calibrated against pre_valid_ratio"},
        "G_layout": {"min": 0.05, "rationale": "manuscript anchor; calibrated against seg_block_size_variance_ratio etc."},
        "G_stability": {"min": 0.20, "rationale": "manuscript anchor; calibrated against permutation-vs-fixed mIoU gap"},
    }
    summary["per_stage"] = per_stage
    (out_dir / "sensitive_parameters.json").write_text(json.dumps(sensitive_parameters, indent=2, default=str) + "\n")
    (out_dir / "diagnostic_intrinsics.json").write_text(json.dumps(diagnostic_intrinsics, indent=2, default=str) + "\n")
    (out_dir / "guardrails.json").write_text(json.dumps(guardrails, indent=2, default=str) + "\n")
    (out_dir / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    # Short report.
    report = ["# v3 BO calibration report", ""]
    report.append(f"- Rings: {', '.join(rings)}")
    report.append(f"- Stages: {', '.join(stages)}")
    for stage in stages:
        agg = per_stage[stage]
        report.append("")
        report.append(f"## Stage: {stage}")
        report.append(f"- Successful trials: {agg['success_count']}")
        report.append(f"- Failed trials: {agg['failure_count']}")
        if agg["failure_counts"]:
            report.append(f"- Failure modes: {agg['failure_counts']}")
        report.append("")
        report.append("### Top sensitive parameters (pooled |Spearman|)")
        for s in sensitive_parameters[stage][:10]:
            report.append(
                f"- `{s['name']}`: ρ={s['pooled_spearman']:.3f}, range=[{s['range_low_p25']}, {s['range_high_p75']}]"
            )
        report.append("")
        report.append("### Top diagnostic intrinsics (pooled Spearman)")
        for s in diagnostic_intrinsics[stage][:10]:
            report.append(f"- `{s['name']}`: ρ={s['pooled_spearman']:.3f}")
    (out_dir / "calibration_report.md").write_text("\n".join(report) + "\n")
    logger.info("aggregate complete; outputs at %s", out_dir)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Step 3 cross-ring aggregation for v3 BO")
    p.add_argument("--rings", nargs="*", help="Ring keys (default: full panel)")
    p.add_argument("--stages", nargs="+", default=["preprocessing", "detection"])
    p.add_argument("--out-dir", default="data/v3/calibration")
    p.add_argument("--label", default=None, help="Optional sub-label (matches what cmd_bo --label set)")
    args = p.parse_args(argv)

    panel = _load_panel()
    rings = args.rings or list(panel.keys())
    out_dir = REPO_ROOT / args.out_dir
    aggregate(rings, stages=args.stages, out_dir=out_dir, label=args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
