#!/usr/bin/env python3
"""Intrinsic QA over all ring-native depth maps (no GT).

Scans canonical rings from ``data/rings/*_ring*.txt``, loads each
``data/{tunnel}/r{ring}/depth_map.npy`` when present, computes coverage and
empty-structure metrics, assigns PASS / WARN / FAIL, and writes JSON + Markdown.

Run::

    ./venv/bin/python agents/1_preprocessing/scripts/qa_depth_maps_corpus.py
    ./venv/bin/python agents/1_preprocessing/scripts/qa_depth_maps_corpus.py --json-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RINGS_DIR = PROJECT_ROOT / "data" / "rings"
DATA_ROOT = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "data" / "preprocessing_qa"
RING_TXT_RE = re.compile(r"^(\d+(?:_\d+)+)_ring(\d+)\.txt$")


def stem_to_tid(stem: str) -> str:
    return stem.replace("_", "-")


def discover_pairs() -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    for p in sorted(RINGS_DIR.glob("*_ring*.txt")):
        m = RING_TXT_RE.match(p.name)
        if m:
            pairs.append((stem_to_tid(m.group(1)), int(m.group(2))))
    return pairs


def compute_intrinsic_metrics(depth_map: np.ndarray) -> Dict[str, Any]:
    """Same spirit as methods/ablation/scripts/_preprocessing_metrics but GT-free."""
    finite = np.isfinite(depth_map)
    valid = finite & (depth_map > 0.0)
    total = int(depth_map.size)
    valid_count = int(np.count_nonzero(valid))
    valid_ratio = float(valid_count / total) if total else 0.0

    empty = (~valid).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(empty, connectivity=8)
    largest_empty = 0
    for lab in range(1, n_labels):
        largest_empty = max(largest_empty, int(stats[lab, cv2.CC_STAT_AREA]))
    largest_empty_ratio = float(largest_empty / total) if total else 0.0

    col_occ = np.mean(valid, axis=0) if valid.ndim == 2 else np.array([], dtype=float)
    row_occ = np.mean(valid, axis=1) if valid.ndim == 2 else np.array([], dtype=float)
    empty_col_bands = int(np.count_nonzero(col_occ < 0.01)) if col_occ.size else 0
    empty_row_bands = int(np.count_nonzero(row_occ < 0.01)) if row_occ.size else 0

    h, w = int(depth_map.shape[0]), int(depth_map.shape[1]) if depth_map.ndim == 2 else (0, 0)
    col_p50 = float(np.percentile(col_occ, 50.0)) if col_occ.size else 0.0
    row_p50 = float(np.percentile(row_occ, 50.0)) if row_occ.size else 0.0
    col_std = float(np.std(col_occ)) if col_occ.size else 0.0

    metrics: Dict[str, Any] = {
        "shape_h": h,
        "shape_w": w,
        "total_pixels": total,
        "valid_pixels": valid_count,
        "valid_ratio": valid_ratio,
        "largest_empty_ratio": largest_empty_ratio,
        "empty_col_bands": empty_col_bands,
        "empty_row_bands": empty_row_bands,
        "col_occupancy_p50": col_p50,
        "row_occupancy_p50": row_p50,
        "col_occupancy_std": col_std,
    }
    return metrics


def classify(metrics: Dict[str, Any], missing_file: bool) -> Tuple[str, str]:
    """Return (status, reason_snippet)."""
    if missing_file:
        return "FAIL", "missing_depth_map"

    h, w = metrics["shape_h"], metrics["shape_w"]
    if h < 8 or w < 8:
        return "FAIL", "collapsed_dimensions"

    vr = metrics["valid_ratio"]
    ler = metrics["largest_empty_ratio"]
    erb = metrics["empty_row_bands"]
    ecb = metrics["empty_col_bands"]

    # Severe: almost no signal or one giant hole
    if vr < 0.03:
        return "FAIL", "near_empty_valid_ratio"
    if ler > 0.92:
        return "FAIL", "dominant_empty_component"
    # Full-width banding: many consecutive empty-ish rows (horizontal stripes)
    if h > 0 and erb > 0.45 * h:
        return "FAIL", "many_empty_row_bands"

    if vr < 0.12 or ler > 0.75:
        return "WARN", "low_coverage_or_large_hole"
    if ecb > max(8, int(0.25 * w)):
        return "WARN", "many_empty_col_bands"

    return "PASS", ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-only", action="store_true", help="Write corpus_metrics.json only.")
    args = ap.parse_args()

    pairs = discover_pairs()
    rows: List[Dict[str, Any]] = []
    for tid, rid in pairs:
        ring_dir = DATA_ROOT / tid / f"r{rid}"
        npy_path = ring_dir / "depth_map.npy"
        rel_key = f"{tid}/r{rid}"
        if not npy_path.is_file():
            row = {
                "tunnel_id": tid,
                "ring_id": rid,
                "relative_key": rel_key,
                "status": "FAIL",
                "reason": "missing_depth_map",
                "depth_map_path": str(npy_path.relative_to(PROJECT_ROOT)),
            }
            rows.append(row)
            continue
        dm = np.load(npy_path)
        m = compute_intrinsic_metrics(dm)
        status, reason = classify(m, False)
        row = {
            "tunnel_id": tid,
            "ring_id": rid,
            "relative_key": rel_key,
            "status": status,
            "reason": reason or None,
            "metrics": m,
            "depth_map_path": str(npy_path.relative_to(PROJECT_ROOT)),
        }
        rows.append(row)

    n_total = len(rows)
    by_status = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_total": n_total,
        "n_pass": by_status.get("PASS", 0),
        "n_warn": by_status.get("WARN", 0),
        "n_fail": by_status.get("FAIL", 0),
        "rings": rows,
    }
    json_path = OUT_DIR / "corpus_metrics.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(PROJECT_ROOT)}")

    if args.json_only:
        return 0

    md_lines = [
        "# Ring-native preprocessing — intrinsic depth-map QA (299 rings)",
        "",
        f"- Total rings (canonical TXT list): **{n_total}**",
        f"- PASS: **{summary['n_pass']}**",
        f"- WARN: **{summary['n_warn']}**",
        f"- FAIL: **{summary['n_fail']}**",
        "",
        "Metrics: `valid_ratio`, `largest_empty_ratio`, empty row/column bands (occupancy < 1%).",
        "Classification is intrinsic-only (no GT).",
        "",
        "## FAIL rings",
        "",
        "| tunnel | ring | reason | depth_map.png |",
        "|---|---:|---|---|",
    ]
    fails = [r for r in rows if r["status"] == "FAIL"]
    warns = [r for r in rows if r["status"] == "WARN"]
    for r in sorted(fails, key=lambda x: (x["tunnel_id"], x["ring_id"])):
        png = f"`data/{r['tunnel_id']}/r{r['ring_id']}/depth_map.png`"
        reason = r.get("reason") or r.get("metrics", {})
        if isinstance(reason, dict):
            reason = "metrics"
        md_lines.append(
            f"| {r['tunnel_id']} | {r['ring_id']} | {r.get('reason', '')} | {png} |"
        )
    md_lines.extend(["", "## WARN rings (sample)", "", "| tunnel | ring | reason | valid_ratio | largest_empty_ratio |", "|---|---:|---|---:|---:|"])
    for r in sorted(warns, key=lambda x: (x["tunnel_id"], x["ring_id"]))[:40]:
        m = r.get("metrics") or {}
        md_lines.append(
            f"| {r['tunnel_id']} | {r['ring_id']} | {r.get('reason', '')} | "
            f"{m.get('valid_ratio', 0):.4f} | {m.get('largest_empty_ratio', 0):.4f} |"
        )
    if len(warns) > 40:
        md_lines.append(f"| … | … | *({len(warns) - 40} more WARN rows in JSON)* | | |")

    md_path = OUT_DIR / "report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
