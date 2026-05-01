"""Scan ring depth maps for empty space / distortion and emit a per-ring metrics report.

For every ring with ``data/<tid>/r<rid>/depth_map.npy``, compute:

  * ``nan_ratio``           = NaN pixels / total pixels in the depth map
  * ``row_empty_ratio``     = fraction of rows that are entirely NaN (top/bottom holes)
  * ``col_empty_ratio``     = fraction of columns that are entirely NaN (axial gaps)
  * ``height_px``, ``width_px`` shape of the depth map
  * ``r_median``            = median radial coord from ``unwrapped.csv``
  * ``frac_within_05m``     = fraction of unwrapped points with |r - D/2| < 0.5
  * ``denoised_keep_pct``   = pred != 0 fraction in ``denoised.csv``

Flag thresholds (severe failures only — partial-arc rings naturally have ~0.30
``nan_ratio``, so we only flag genuine outliers):

  * ``nan_ratio > 0.55``           -> ``empty_space``
  * ``row_empty_ratio > 0.50``     -> ``row_distortion``
  * ``col_empty_ratio > 0.20``     -> ``axial_gap``
  * ``frac_within_05m < 0.50``     -> ``bad_unwrap``
  * ``denoised_keep_pct < 0.20``   -> ``over_denoise``

Plus per-tunnel outlier flags (relative deviation from the tunnel's median):

  * ``nan_outlier``           = ``nan_ratio > tunnel_median + 0.20``
  * ``row_empty_outlier``     = ``row_empty_ratio > tunnel_median + 0.20``

Outputs:

  * ``data/rings/preprocessing_metrics.csv`` (one row per ring)
  * ``data/rings/preprocessing_report.md``   (per-tunnel summary + flagged-ring list)

Run::

    ./venv/bin/python agents/1_preprocessing/scripts/scan_depth_maps.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RINGS_DIR = PROJECT_ROOT / "data" / "rings"
DATA_ROOT = PROJECT_ROOT / "data"
PARAMS_ROOT = PROJECT_ROOT / "agents" / "1_preprocessing" / "parameters"
METRICS_CSV = RINGS_DIR / "preprocessing_metrics.csv"
REPORT_MD = RINGS_DIR / "preprocessing_report.md"

THRESH_NAN = 0.55
THRESH_ROW_EMPTY = 0.50
THRESH_COL_EMPTY = 0.20
THRESH_FRAC_WITHIN = 0.50
THRESH_DENOISE_KEEP = 0.20
THRESH_NAN_OUTLIER_DELTA = 0.20
THRESH_ROW_OUTLIER_DELTA = 0.20

FLAG_LABELS = {
    "empty_space": f"Empty space (depth NaN > {THRESH_NAN:.2f})",
    "row_distortion": f"Row distortion (row_empty > {THRESH_ROW_EMPTY:.2f})",
    "axial_gap": f"Axial gap (col_empty > {THRESH_COL_EMPTY:.2f})",
    "bad_unwrap": f"Bad unwrap (frac_within_05m < {THRESH_FRAC_WITHIN:.2f})",
    "over_denoise": f"Over-aggressive denoise (keep_pct < {THRESH_DENOISE_KEEP:.2f})",
    "nan_outlier": f"Per-tunnel NaN outlier (>{THRESH_NAN_OUTLIER_DELTA:.2f} above tunnel median)",
    "row_empty_outlier": f"Per-tunnel row-empty outlier (>{THRESH_ROW_OUTLIER_DELTA:.2f} above tunnel median)",
}


def load_diameter(tunnel_id: str, ring_id: int) -> Optional[float]:
    candidates = [
        PARAMS_ROOT / tunnel_id / f"r{ring_id}" / "parameters_preprocessing.json",
        PARAMS_ROOT / tunnel_id / "parameters_preprocessing.json",
    ]
    for p in candidates:
        if p.is_file():
            try:
                with p.open() as f:
                    data = json.load(f)
                if "tunnel_diameter" in data:
                    return float(data["tunnel_diameter"])
            except Exception:
                continue
    return None


def scan_ring(tunnel_id: str, ring_id: int) -> Optional[Dict[str, object]]:
    out_dir = DATA_ROOT / tunnel_id / f"r{ring_id}"
    npy = out_dir / "depth_map.npy"
    if not npy.is_file():
        return None
    dm = np.load(npy)
    height_px, width_px = (int(dm.shape[0]), int(dm.shape[1])) if dm.ndim == 2 else (0, 0)
    if dm.size == 0:
        nan_ratio = float("nan")
        row_empty = float("nan")
        col_empty = float("nan")
    else:
        nan_mask = np.isnan(dm)
        nan_ratio = float(nan_mask.sum()) / float(dm.size)
        row_empty = float(nan_mask.all(axis=1).mean())
        col_empty = float(nan_mask.all(axis=0).mean())

    diameter = load_diameter(tunnel_id, ring_id)

    r_median = float("nan")
    frac_within_05m = float("nan")
    unwrapped = out_dir / "unwrapped.csv"
    if unwrapped.is_file():
        try:
            ud = pd.read_csv(unwrapped, usecols=["r"])
            if not ud.empty:
                r_median = float(ud["r"].median())
                if diameter is not None:
                    frac_within_05m = float((ud["r"] - diameter / 2.0).abs().lt(0.5).mean())
        except Exception:
            pass

    denoised_keep_pct = float("nan")
    denoised = out_dir / "denoised.csv"
    if denoised.is_file():
        try:
            dd = pd.read_csv(denoised, usecols=["pred"])
            if len(dd):
                denoised_keep_pct = float((dd["pred"] != 0).mean())
        except Exception:
            pass

    flags: List[str] = []
    if not np.isnan(nan_ratio) and nan_ratio > THRESH_NAN:
        flags.append("empty_space")
    if not np.isnan(row_empty) and row_empty > THRESH_ROW_EMPTY:
        flags.append("row_distortion")
    if not np.isnan(col_empty) and col_empty > THRESH_COL_EMPTY:
        flags.append("axial_gap")
    if not np.isnan(frac_within_05m) and frac_within_05m < THRESH_FRAC_WITHIN:
        flags.append("bad_unwrap")
    if not np.isnan(denoised_keep_pct) and denoised_keep_pct < THRESH_DENOISE_KEEP:
        flags.append("over_denoise")

    return {
        "tunnel_id": tunnel_id,
        "ring_id": ring_id,
        "tunnel_diameter": diameter if diameter is not None else "",
        "height_px": height_px,
        "width_px": width_px,
        "nan_ratio": nan_ratio,
        "row_empty_ratio": row_empty,
        "col_empty_ratio": col_empty,
        "r_median": r_median,
        "frac_within_05m": frac_within_05m,
        "denoised_keep_pct": denoised_keep_pct,
        "n_flags": len(flags),
        "flags": ",".join(flags),
    }


def discover_pairs() -> List[Tuple[str, int]]:
    """All canonical (tid, rid) pairs from data/rings/summary.json."""
    summary = RINGS_DIR / "summary.json"
    if not summary.is_file():
        return []
    with summary.open() as f:
        data = json.load(f)
    out = [(s["file"], int(s["ring_id"])) for s in data.get("samples", [])]
    out.sort()
    return out


def fmt(v: object, kind: str = "f") -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        if kind == "p":
            return f"{v * 100:.2f}%"
        return f"{v:.4f}"
    return "" if v is None else str(v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tunnels",
        nargs="*",
        default=None,
        help="Optional tunnel allow-list.",
    )
    args = parser.parse_args()

    pairs = discover_pairs()
    if args.tunnels:
        wanted = set(args.tunnels)
        pairs = [(t, r) for (t, r) in pairs if t in wanted]
    if not pairs:
        print("[scan] no canonical pairs found")
        return 1

    rows: List[Dict[str, object]] = []
    missing: List[Tuple[str, int]] = []
    for tid, rid in pairs:
        m = scan_ring(tid, rid)
        if m is None:
            missing.append((tid, rid))
            continue
        rows.append(m)

    by_tid_for_median: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        by_tid_for_median.setdefault(str(r["tunnel_id"]), []).append(r)
    for tid, tr in by_tid_for_median.items():
        nans = [float(r["nan_ratio"]) for r in tr
                if isinstance(r["nan_ratio"], float) and not np.isnan(float(r["nan_ratio"]))]
        rows_e = [float(r["row_empty_ratio"]) for r in tr
                  if isinstance(r["row_empty_ratio"], float) and not np.isnan(float(r["row_empty_ratio"]))]
        med_nan = float(np.median(nans)) if nans else float("nan")
        med_row = float(np.median(rows_e)) if rows_e else float("nan")
        for r in tr:
            extra: List[str] = []
            v = r["nan_ratio"]
            if isinstance(v, float) and not np.isnan(v) and not np.isnan(med_nan):
                if v > med_nan + THRESH_NAN_OUTLIER_DELTA:
                    extra.append("nan_outlier")
            v = r["row_empty_ratio"]
            if isinstance(v, float) and not np.isnan(v) and not np.isnan(med_row):
                if v > med_row + THRESH_ROW_OUTLIER_DELTA:
                    extra.append("row_empty_outlier")
            if extra:
                cur = str(r["flags"]).split(",") if r["flags"] else []
                cur = [c for c in cur if c]
                cur.extend(extra)
                r["flags"] = ",".join(cur)
                r["n_flags"] = int(r["n_flags"]) + len(extra)

    fields = [
        "tunnel_id", "ring_id", "tunnel_diameter",
        "height_px", "width_px",
        "nan_ratio", "row_empty_ratio", "col_empty_ratio",
        "r_median", "frac_within_05m", "denoised_keep_pct",
        "n_flags", "flags",
    ]
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row_out = {k: r.get(k, "") for k in fields}
            for k in ("nan_ratio", "row_empty_ratio", "col_empty_ratio",
                      "r_median", "frac_within_05m", "denoised_keep_pct"):
                v = row_out[k]
                if isinstance(v, float) and np.isnan(v):
                    row_out[k] = ""
            w.writerow(row_out)

    by_tunnel: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        by_tunnel.setdefault(str(r["tunnel_id"]), []).append(r)

    flagged = [r for r in rows if int(r["n_flags"]) > 0]
    by_flag: Dict[str, List[Dict[str, object]]] = {}
    for r in flagged:
        for flag in str(r["flags"]).split(","):
            if flag:
                by_flag.setdefault(flag, []).append(r)

    lines: List[str] = []
    lines.append("# All-rings preprocessing report")
    lines.append("")
    lines.append(f"- Canonical rings: {len(pairs)}")
    lines.append(f"- Scanned rings (with depth_map.npy): {len(rows)}")
    lines.append(f"- Missing rings (no depth_map.npy): {len(missing)}")
    lines.append(f"- Flagged rings (>=1 issue): {len(flagged)}")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    for k, v in FLAG_LABELS.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Per-tunnel summary")
    lines.append("")
    lines.append("| tunnel | n_rings | mean NaN | max NaN | mean row_empty | flagged |")
    lines.append("|--------|--------:|---------:|--------:|---------------:|--------:|")
    for tid in sorted(by_tunnel.keys()):
        tr = by_tunnel[tid]
        nans = [float(r["nan_ratio"]) for r in tr if isinstance(r["nan_ratio"], float) and not np.isnan(float(r["nan_ratio"]))]
        rows_ = [float(r["row_empty_ratio"]) for r in tr if isinstance(r["row_empty_ratio"], float) and not np.isnan(float(r["row_empty_ratio"]))]
        n_flagged = sum(1 for r in tr if int(r["n_flags"]) > 0)
        mean_nan = (sum(nans) / len(nans)) if nans else float("nan")
        max_nan = max(nans) if nans else float("nan")
        mean_row = (sum(rows_) / len(rows_)) if rows_ else float("nan")
        lines.append(
            f"| {tid} | {len(tr)} | {fmt(mean_nan)} | {fmt(max_nan)} | {fmt(mean_row)} | {n_flagged} |"
        )
    lines.append("")

    if missing:
        lines.append("## Missing depth maps")
        lines.append("")
        for tid, rid in missing:
            lines.append(f"- {tid} r{rid}")
        lines.append("")

    lines.append("## Flagged rings by failure mode")
    lines.append("")
    if not flagged:
        lines.append("None.")
    else:
        for flag in ["empty_space", "row_distortion", "axial_gap", "bad_unwrap",
                     "over_denoise", "nan_outlier", "row_empty_outlier"]:
            entries = by_flag.get(flag, [])
            lines.append(f"### {flag}  ({len(entries)})")
            lines.append("")
            if not entries:
                lines.append("(none)")
                lines.append("")
                continue
            lines.append("| tunnel | ring | nan_ratio | row_empty | col_empty | frac_within_05m | denoised_keep_pct | flags |")
            lines.append("|--------|-----:|----------:|----------:|----------:|----------------:|------------------:|-------|")
            for r in sorted(entries, key=lambda x: (str(x["tunnel_id"]), int(x["ring_id"]))):
                lines.append(
                    "| {tid} | {rid} | {nan} | {row} | {col} | {fwm} | {dkp} | {flags} |".format(
                        tid=r["tunnel_id"],
                        rid=r["ring_id"],
                        nan=fmt(r["nan_ratio"]),
                        row=fmt(r["row_empty_ratio"]),
                        col=fmt(r["col_empty_ratio"]),
                        fwm=fmt(r["frac_within_05m"]),
                        dkp=fmt(r["denoised_keep_pct"]),
                        flags=str(r["flags"]),
                    )
                )
            lines.append("")

    REPORT_MD.write_text("\n".join(lines))
    print(f"[scan] wrote {METRICS_CSV.relative_to(PROJECT_ROOT)} ({len(rows)} rows)")
    print(f"[scan] wrote {REPORT_MD.relative_to(PROJECT_ROOT)}")
    print(f"[scan] flagged {len(flagged)}/{len(rows)} rings; missing {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
