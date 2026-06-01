#!/usr/bin/env python3
"""Compare BO experience v1 (fixed r) vs v2 (searchable r_surface_min)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_BO_DIR = Path(__file__).resolve().parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))

from lib.ceiling_gate import REPO_ROOT


def _ring_stats(trials_path: Path, r_col: str) -> dict:
    if not trials_path.exists():
        return {"missing": True}
    df = pd.read_csv(trials_path)
    best_idx = df["gt_miou"].idxmax()
    best = df.loc[best_idx]
    r_series = df[r_col] if r_col in df.columns else pd.Series(dtype=float)
    return {
        "n_trials": int(len(df)),
        "best_bo_miou": float(best["gt_miou"]),
        "regret_vs_ceiling": float(best.get("regret_vs_ceiling", float("nan"))),
        "best_r_surface_min": float(best[r_col]) if r_col in best else None,
        "r_surface_min_std": float(r_series.std()) if len(r_series) > 1 else 0.0,
        "r_surface_min_unique": int(r_series.nunique()) if len(r_series) else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-root", type=Path, default=REPO_ROOT / "logs" / "bo_experience_v1")
    ap.add_argument("--v2-root", type=Path, default=REPO_ROOT / "logs" / "bo_experience_v2")
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "bo_calibration" / "MANIFEST.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "logs" / "bo_experience_v2" / "v1_v2_comparison.md")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lines = ["# BO experience v1 vs v2 comparison\n\n", "| ring | v1 best | v2 best | v1 regret | v2 regret | v1 r std | v2 r std | v2 best r |\n", "|---|---:|---:|---:|---:|---:|---:|---:|\n"]

    rows = []
    for entry in manifest.get("rings", []):
        rk = entry["ring_key"]
        tid, rid = rk.split("/")
        v1 = _ring_stats(args.v1_root / tid / rid / "bo_trials.csv", "r_surface_min_fixed")
        v2 = _ring_stats(args.v2_root / tid / rid / "bo_trials.csv", "r_surface_min")
        rows.append({"ring_key": rk, "v1": v1, "v2": v2})
        if v1.get("missing") or v2.get("missing"):
            lines.append(f"| {rk} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |\n")
            continue
        lines.append(
            f"| {rk} | {v1['best_bo_miou']:.3f} | {v2['best_bo_miou']:.3f} | "
            f"{v1['regret_vs_ceiling']:.3f} | {v2['regret_vs_ceiling']:.3f} | "
            f"{v1['r_surface_min_std']:.3f} | {v2['r_surface_min_std']:.3f} | "
            f"{v2['best_r_surface_min']:.4f} |\n"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(lines), encoding="utf-8")
    summary = {"rings": rows, "report": str(args.out)}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
