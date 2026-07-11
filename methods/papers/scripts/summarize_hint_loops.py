#!/usr/bin/env python3
"""Aggregate detection + SAM hint-loop results into a single markdown table."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TUNNELS = [
    "1-1", "1-2", "1-3", "1-4", "1-5",
    "2-1", "2-2", "2-3", "2-4", "2-5",
]
DET_LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
SAM_LEVELS = ["S0", "S1", "S2", "S3", "S4", "S5a", "S5b", "S5"]
OUT = REPO / "methods" / "papers" / "output" / "regular_hint_combined_matrix.md"


def miou(root: Path, level: str, tunnel: str) -> float | None:
    p = root / level / tunnel / "evaluation" / "performance.md"
    if not p.is_file():
        return None
    m = re.search(r"mIoU\):\s*([\d.]+)", p.read_text())
    return float(m.group(1)) if m else None


def level_stats(root: Path, levels: list[str]) -> list[str]:
    lines = []
    for lvl in levels:
        vals = [miou(root, lvl, t) for t in TUNNELS]
        present = [v for v in vals if v is not None]
        if not present:
            continue
        n_pass = sum(1 for v in present if v >= 0.8)
        mean = sum(present) / len(present)
        lines.append(
            f"| {lvl} | {len(present)}/10 | {n_pass}/10 | {mean:.3f} |"
        )
    return lines


def tunnel_table(root: Path, levels: list[str], title: str) -> list[str]:
    header = "| Tunnel | " + " | ".join(levels) + " |"
    sep = "|--------|" + "|".join(["------:" for _ in levels]) + "|"
    rows = [f"## {title}", "", header, sep]
    for t in TUNNELS:
        cells = []
        for lvl in levels:
            v = miou(root, lvl, t)
            cells.append(f"{v:.3f}" if v is not None else "—")
        rows.append(f"| {t} | " + " | ".join(cells) + " |")
    return rows


def main() -> None:
    det_root = REPO / "data" / "regular_hint_loop"
    sam_root = REPO / "data" / "regular_sam_hint_loop"

    lines = [
        "# Regular Hint Loops — Combined Matrix (1-*, 2-*)",
        "",
        "Gate pair: **2-2** + **1-3** | Model: opus4.6 | Upstream: `data/ablation_anthropic`",
        "",
        "## Detection hints (`regular_hint_loop`)",
        "",
        "| Level | Coverage | Pass ≥0.8 | Mean mIoU |",
        "|-------|----------|-----------|-----------|",
        *level_stats(det_root, DET_LEVELS),
        "",
        *tunnel_table(det_root, DET_LEVELS, "Per-tunnel detection mIoU"),
        "",
        "## SAM hints (`regular_sam_hint_loop`)",
        "",
        "| Level | Coverage | Pass ≥0.8 | Mean mIoU |",
        "|-------|----------|-----------|-----------|",
        *level_stats(sam_root, SAM_LEVELS),
        "",
        *tunnel_table(sam_root, SAM_LEVELS, "Per-tunnel SAM mIoU"),
        "",
        "## Minimum-hint summary",
        "",
        "- **Detection alone:** L0 optimal; no level reaches 8/10 at 0.8.",
        "- **SAM partial (S5a):** GT for K+A2+A3 — 1/10 pass (2-2 only at 0.801).",
        "- **SAM minimum for 8/10:** S5b `oracle_swap` — 10/10 pass, mean 0.863.",
        "",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
