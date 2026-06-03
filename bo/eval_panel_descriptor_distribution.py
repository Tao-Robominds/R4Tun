#!/usr/bin/env python3
"""Corpus-backed descriptor marginals for BO calibration (6) and held-out (50)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BO = Path(__file__).resolve().parent
REPO = _BO.parent
sys.path.insert(0, str(_BO))

from lib.ceiling_gate import REPO_ROOT, derive_gt_layout
from lib.held_out_descriptors import (
    _direction_from_spatial_order,
    _load_spatial_order,
    _load_depth_audit,
)
from lib.ring_site_params import resolve_ring_site_params

SUMMARY_PATH = REPO_ROOT / "data" / "rings" / "summary.json"

# Descriptor thresholds (documented in paper Table~\ref{tab:dataset_panels})
COVERAGE_GAP_FULL_MAX = 0.02
COVERAGE_FINITE_FALLBACK_MIN = 0.85
K_SPAN_NARROW_MAX_DEG = 22.0
K_SPAN_WIDE_MIN_DEG = 39.0


def _summary_index() -> dict[tuple[str, int], dict]:
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    out: dict[tuple[str, int], dict] = {}
    for s in data["samples"]:
        tid = str(s["file"]).replace("_", "-")
        out[(tid, int(s["ring_id"]))] = s
    return out


def _density_tier(reason: str | None) -> str:
    if not reason:
        return "unknown"
    if reason == "density_dense":
        return "dense"
    if reason == "density_medium":
        return "medium"
    if reason == "density_low":
        return "low"
    if reason == "density_sparse":
        return "low"
    return "unknown"


def _coverage_tier(angular_gap_frac: float | None, ring_dir: Path) -> str:
    if angular_gap_frac is not None:
        return "full" if float(angular_gap_frac) <= COVERAGE_GAP_FULL_MAX else "partial"
    audit = _load_depth_audit(ring_dir)
    finite = float(audit.get("finite_ratio", 0.0))
    return "full" if finite >= COVERAGE_FINITE_FALLBACK_MIN else "partial"


def _theta_span_deg(ring_dir: Path) -> float | None:
    enh = ring_dir / "enhanced.csv"
    if not enh.is_file():
        return None
    df = pd.read_csv(enh, usecols=lambda c: c in {"theta", "h"})
    if "theta" not in df.columns:
        return None
    t = pd.to_numeric(df["theta"], errors="coerce").dropna()
    if len(t) < 10:
        return None
    return float(t.max() - t.min())


def _k_span_from_gt(ring_dir: Path, segment_count: int) -> float | None:
    gt_path = ring_dir / "gt_layout.json"
    if gt_path.is_file():
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
    else:
        try:
            gt = derive_gt_layout(ring_dir, ring_dir, segment_count)
        except Exception:
            return None
    H = int(gt.get("H") or np.load(ring_dir / "depth_map.npy").shape[0])
    k_y = float(gt["k_y"])
    offsets = gt["offsets"]
    blocks = sorted(offsets.keys(), key=lambda b: float(offsets[b]))
    if "K" not in offsets or len(blocks) < 2:
        return None
    k_idx = blocks.index("K")
    next_b = blocks[(k_idx + 1) % len(blocks)]
    span_px = float(offsets[next_b]) - float(offsets["K"])
    if span_px < 0:
        span_px += H
    return float(span_px / max(H, 1) * 360.0)


def _k_span_tier(deg: float | None) -> str:
    if deg is None:
        return "unknown"
    if deg <= K_SPAN_NARROW_MAX_DEG:
        return "narrow"
    if deg >= K_SPAN_WIDE_MIN_DEG:
        return "wide"
    return "normal"


def _resolve_k_span_deg(ring_dir: Path, segment_count: int, meta: dict) -> tuple[float | None, str]:
    if meta.get("k_span_deg") is not None:
        deg = float(meta["k_span_deg"])
        return deg, "summary"
    gt_deg = _k_span_from_gt(ring_dir, segment_count)
    if gt_deg is not None:
        return gt_deg, "gt_layout"
    theta_deg = _theta_span_deg(ring_dir)
    if theta_deg is not None:
        return theta_deg, "theta_enhanced"
    return None, "missing"


def _ring_keys_bo() -> list[str]:
    manifest = json.loads((REPO_ROOT / "data" / "bo_calibration" / "MANIFEST.json").read_text())
    return [str(r["ring_key"]) for r in manifest["rings"]]


def _ring_keys_held() -> list[str]:
    panel = REPO_ROOT / "data" / "held-out" / "_manifests" / "data_v6_50ring_calibration_panel.csv"
    return pd.read_csv(panel)["ring_key"].astype(str).tolist()


def _describe_ring(ring_key: str, data_root: Path, panel: str, summary: dict) -> dict:
    tunnel_id, ring_part = ring_key.split("/")
    ring_id = int(ring_part.lstrip("r"))
    ring_dir = data_root / tunnel_id / f"r{ring_id}"
    site = resolve_ring_site_params(ring_key, ring_dir)
    seg = int(site["segment_count"])
    meta = summary.get((tunnel_id, ring_id), {})
    k_deg, k_src = _resolve_k_span_deg(ring_dir, seg, meta)
    spatial = _load_spatial_order(ring_dir, seg)
    direction_tier, _ = _direction_from_spatial_order(spatial, seg)
    return {
        "ring_key": ring_key,
        "panel": panel,
        "density_tier": _density_tier(meta.get("reason")),
        "coverage_tier": _coverage_tier(meta.get("angular_gap_frac"), ring_dir),
        "k_span_deg": k_deg,
        "k_span_source": k_src,
        "k_span_tier": _k_span_tier(k_deg),
        "direction_tier": direction_tier,
    }


def _marginals(df: pd.DataFrame, col: str, order: list[str]) -> dict[str, int]:
    vc = df[col].value_counts()
    return {k: int(vc.get(k, 0)) for k in order}


def _format_slash(counts: dict[str, int], keys: list[str]) -> str:
    return " / ".join(str(counts.get(k, 0)) for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "logs" / "panel_descriptor_distribution_v1")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = _summary_index()
    rows = []
    for rk in _ring_keys_bo():
        rows.append(_describe_ring(rk, REPO_ROOT / "data" / "bo_calibration", "bo_calibration", summary))
    for rk in _ring_keys_held():
        rows.append(_describe_ring(rk, REPO_ROOT / "data" / "held-out", "held_out", summary))
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "ring_descriptors.csv", index=False)

    density_keys = ["dense", "medium", "low"]
    coverage_keys = ["full", "partial"]
    kspan_keys = ["narrow", "normal", "wide"]
    direction_keys = ["plus", "minus"]

    panels = {}
    for panel in ("bo_calibration", "held_out"):
        sub = df[df["panel"] == panel]
        panels[panel] = {
            "n_rings": len(sub),
            "density": _marginals(sub, "density_tier", density_keys),
            "coverage": _marginals(sub, "coverage_tier", coverage_keys),
            "k_span": _marginals(sub, "k_span_tier", kspan_keys),
            "direction": _marginals(sub, "direction_tier", direction_keys),
        }

    latex = {
        "bo_calibration": {
            "density": _format_slash(panels["bo_calibration"]["density"], density_keys),
            "coverage": _format_slash(panels["bo_calibration"]["coverage"], coverage_keys),
            "k_span": _format_slash(panels["bo_calibration"]["k_span"], kspan_keys),
            "direction": _format_slash(panels["bo_calibration"]["direction"], direction_keys),
        },
        "held_out": {
            "density": _format_slash(panels["held_out"]["density"], density_keys),
            "coverage": _format_slash(panels["held_out"]["coverage"], coverage_keys),
            "k_span": _format_slash(panels["held_out"]["k_span"], kspan_keys),
            "direction": _format_slash(panels["held_out"]["direction"], direction_keys),
        },
    }

    thresholds = {
        "density": {
            "source": "data/rings/summary.json field reason",
            "dense": "density_dense",
            "medium": "density_medium",
            "low": ["density_low", "density_sparse"],
        },
        "coverage": {
            "full_if_angular_gap_frac_lte": COVERAGE_GAP_FULL_MAX,
            "fallback_full_if_finite_ratio_gte": COVERAGE_FINITE_FALLBACK_MIN,
            "partial": "otherwise",
        },
        "k_span_deg": {
            "priority": ["summary.k_span_deg", "GT keystone arc span", "enhanced theta span"],
            "narrow_deg_lte": K_SPAN_NARROW_MAX_DEG,
            "normal_deg_range": f"({K_SPAN_NARROW_MAX_DEG}, {K_SPAN_WIDE_MIN_DEG})",
            "wide_deg_gte": K_SPAN_WIDE_MIN_DEG,
        },
        "direction": {
            "source": "gt_layout.json spatial_order_by_label",
            "plus": "cyclic rotation of (1,2,...,n)",
            "minus": "cyclic rotation of (n,n-1,...,1)",
        },
    }
    out = {
        "panels": panels,
        "latex_table_rows": latex,
        "thresholds": thresholds,
    }
    (args.out_dir / "panel_descriptor_counts.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "latex_table_snippet.tex").write_text(
        _latex_table_panels(latex, panels, thresholds),
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2))


def _latex_table_panels(latex: dict, panels: dict, thresholds: dict) -> str:
    bo = latex["bo_calibration"]
    ev = latex["held_out"]
    cov = thresholds["coverage"]
    ksp = thresholds["k_span_deg"]
    cap = (
        "Composition of the BO calibration and evaluation panels (counts from frozen corpora). "
        "\\textbf{Density:} pre-registered point-count regime in \\texttt{data/rings/summary.json} "
        "(\\texttt{density\\_dense}$\\rightarrow$dense, \\texttt{density\\_medium}$\\rightarrow$medium, "
        "\\texttt{density\\_low}/\\texttt{density\\_sparse}$\\rightarrow$low). "
        f"\\textbf{{Coverage:}} full if angular gap fraction $\\leq {cov['full_if_angular_gap_frac_lte']:.2f}$, "
        f"else partial; if gap unlisted, full if depth-map finite ratio "
        f"$\\geq {cov['fallback_full_if_finite_ratio_gte']:.2f}$. "
        f"\\textbf{{K-span:}} circumferential keystone span in degrees "
        f"(\\texttt{{k\\_span\\_deg}} when listed, else GT keystone arc, else enhanced $\\theta$ span); "
        f"narrow $\\leq {ksp['narrow_deg_lte']:.0f}^{{\\circ}}$, "
        f"normal ${ksp['narrow_deg_lte']:.0f}^{{\\circ}}<s<{ksp['wide_deg_gte']:.0f}^{{\\circ}}$, "
        f"wide $\\geq {ksp['wide_deg_gte']:.0f}^{{\\circ}}$. "
        "\\textbf{Direction:} GT circumferential block order; "
        "\\emph{plus} $=$ rotation of $1,\\ldots,n$; \\emph{minus} $=$ rotation of $n,\\ldots,1$."
    )
    return f"""\\begin{{table*}}[t]
\\caption{{{cap}}}
\\label{{tab:dataset_panels}}
\\centering
\\footnotesize
\\begin{{tabular}}{{lcc}}
\\toprule
Descriptor & BO calibration & Evaluation \\\\
\\midrule
Rings & {panels['bo_calibration']['n_rings']} & {panels['held_out']['n_rings']} \\\\
Density (dense / medium / low) & {bo['density']} & {ev['density']} \\\\
Coverage (full / partial) & {bo['coverage']} & {ev['coverage']} \\\\
K-span (narrow / normal / wide) & {bo['k_span']} & {ev['k_span']} \\\\
Direction (plus / minus) & {bo['direction']} & {ev['direction']} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


if __name__ == "__main__":
    main()
