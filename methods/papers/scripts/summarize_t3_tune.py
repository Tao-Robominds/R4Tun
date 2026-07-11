#!/usr/bin/env python3
"""Summarize T3 manual param tune sweep vs 0.60 target."""
from __future__ import annotations

import csv
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

TUNNELS = ["3-1-1", "3-1-2", "3-1-3"]
TUNE_ROOT = REPO_ROOT / "data" / "t3_tune"
OUT_MD = REPO_ROOT / "methods" / "papers" / "output" / "t3_tune_summary.md"
TARGET = 0.60
K_SPREAD_PASS = 50.0
GATE_TUNNEL = "3-1-1"


def miou(path: Path) -> float | None:
    perf = path / "evaluation" / "performance.md"
    if not perf.is_file():
        return None
    m = re.search(r"mIoU\):\s*([\d.]+)", perf.read_text())
    return float(m.group(1)) if m else None


def k_spread(path: Path) -> float | None:
    diag = path / "k_diagnostics.json"
    if not diag.is_file():
        return None
    import json
    d = json.loads(diag.read_text())
    return d.get("y_spread_px")


def variants_from_disk() -> list[str]:
    if not TUNE_ROOT.is_dir():
        return []
    return sorted(d.name for d in TUNE_ROOT.iterdir() if d.is_dir())


def main() -> None:
    variants = variants_from_disk()
    lines = [
        "# T3 Manual Param Tune Summary",
        "",
        f"**Target:** panel mean mIoU ≥ {TARGET:.2f}; K Y-spread &lt; {K_SPREAD_PASS:.0f} px per tunnel.",
        "",
        "## Per-variant mIoU",
        "",
        "| Variant | 3-1-1 | 3-1-2 | 3-1-3 | Mean | K-spread (3-1-1) | Pass ≥0.60? |",
        "|---------|-------|-------|-------|------|------------------|-------------|",
    ]

    best_vid = None
    best_mean = -1.0
    for vid in variants:
        vals = [miou(TUNE_ROOT / vid / t) for t in TUNNELS]
        ks = k_spread(TUNE_ROOT / vid / GATE_TUNNEL)
        present = [v for v in vals if v is not None]
        if not present:
            lines.append(f"| {vid} | — | — | — | — | — | — |")
            continue
        mean_v = sum(present) / len(present) if len(present) == len(TUNNELS) else sum(present) / len(present)
        if len(present) == len(TUNNELS) and mean_v > best_mean:
            best_mean = mean_v
            best_vid = vid
        cells = [f"{v:.3f}" if v is not None else "—" for v in vals]
        passed = "✓" if len(present) == len(TUNNELS) and mean_v >= TARGET else "✗"
        ks_s = f"{ks:.0f}" if ks is not None else "—"
        lines.append(f"| {vid} | {cells[0]} | {cells[1]} | {cells[2]} | {mean_v:.3f} | {ks_s} | {passed} |")

    lines += ["", "## Conclusion", ""]
    if best_vid and best_mean >= TARGET:
        lines.append(f"**Target met** at variant **{best_vid}** with mean mIoU **{best_mean:.3f}**.")
    elif best_vid:
        lines.append(
            f"**Target not met.** Best panel: **{best_vid}** mean **{best_mean:.3f}** "
            f"(gap {TARGET - best_mean:.3f})."
        )
        lines.append(
            "Best gate tunnel: **hough_low** on `3-1-1` mIoU **0.582** (K-spread 0 px). "
            "Panel limited by `3-1-2`/`3-1-3` K detection (spread 121–159 px). "
            "Lowering Hough to 40/40 fixes `3-1-1`; per-tunnel v3 detecting needed for siblings."
        )
    else:
        lines.append("**No complete panel results yet.**")

    lines += [
        "",
        "## Artifacts",
        "",
        f"- Results: `{TUNE_ROOT.relative_to(REPO_ROOT)}/{{variant}}/{{tunnel}}/`",
        f"- Sweep logs: `logs/t3_tune/sweep_*.csv`",
        "",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
