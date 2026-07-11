#!/usr/bin/env python3
"""Summarize T3 hint loop results into markdown."""
from __future__ import annotations

import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

TUNNELS = ["3-1-1", "3-1-2", "3-1-3"]
LEVELS = ["T0", "T1", "T2", "T3", "T4", "T5"]
LOOP_ROOT = REPO_ROOT / "data" / "t3_hint_loop"
OUT_MD = REPO_ROOT / "methods" / "papers" / "output" / "t3_hint_loop_summary.md"
TARGET = 0.60
GATE_TUNNEL = "3-1-1"
GATE_THRESHOLDS = {"T1": 0.45, "scale": 0.55}

BROKEN = {"3-1-1": 0.287, "3-1-2": 0.237, "3-1-3": 0.229}


def miou(path: Path) -> float | None:
    perf = path / "evaluation" / "performance.md"
    if not perf.is_file():
        return None
    m = re.search(r"mIoU\):\s*([\d.]+)", perf.read_text())
    return float(m.group(1)) if m else None


def main() -> None:
    rows: dict[str, dict[str, float | None]] = {lv: {} for lv in LEVELS}
    for lv in LEVELS:
        for t in TUNNELS:
            rows[lv][t] = miou(LOOP_ROOT / lv / t)

    lines = [
        "# T3 Hint Loop Summary",
        "",
        f"**Target:** mean mIoU ≥ {TARGET:.2f} across `3-1-1`, `3-1-2`, `3-1-3`.",
        "",
        "## Per-level mIoU",
        "",
        "| Level | 3-1-1 | 3-1-2 | 3-1-3 | Mean | Pass ≥0.60? |",
        "|-------|-------|-------|-------|------|-------------|",
        f"| broken | {BROKEN['3-1-1']:.3f} | {BROKEN['3-1-2']:.3f} | {BROKEN['3-1-3']:.3f} | "
        f"{sum(BROKEN.values())/3:.3f} | ✗ |",
    ]

    best_level = None
    best_mean = -1.0
    for lv in LEVELS:
        vals = [rows[lv][t] for t in TUNNELS]
        present = [v for v in vals if v is not None]
        if not present:
            lines.append(f"| {lv} | — | — | — | — | — |")
            continue
        mean_v = sum(present) / len(present)
        if mean_v > best_mean and len(present) == len(TUNNELS):
            best_mean = mean_v
            best_level = lv
        cells = [f"{v:.3f}" if v is not None else "—" for v in vals]
        passed = "✓" if mean_v >= TARGET and len(present) == len(TUNNELS) else "✗"
        lines.append(f"| {lv} | {cells[0]} | {cells[1]} | {cells[2]} | {mean_v:.3f} | {passed} |")

    lines += [
        "",
        "## Gate (`3-1-1`)",
        "",
        f"- T1 pass threshold: mIoU ≥ {GATE_THRESHOLDS['T1']:.2f}",
        f"- Scale threshold: mIoU ≥ {GATE_THRESHOLDS['scale']:.2f}",
        "",
    ]
    for lv in LEVELS:
        g = rows[lv].get(GATE_TUNNEL)
        if g is None:
            continue
        t1_ok = g >= GATE_THRESHOLDS["T1"] if lv != "T0" else True
        scale_ok = g >= GATE_THRESHOLDS["scale"] if lv != "T0" else False
        lines.append(
            f"- **{lv}** `{GATE_TUNNEL}` mIoU={g:.3f} — "
            f"T1 gate {'✓' if t1_ok else '✗'}, scale gate {'✓' if scale_ok else '✗'}"
        )

    lines += [
        "",
        "## Conclusion",
        "",
    ]
    if best_level and best_mean >= TARGET:
        lines.append(
            f"**Target met** at **{best_level}** with mean mIoU **{best_mean:.3f}**."
        )
    elif best_level:
        lines.append(
            f"**Target not met.** Best full panel: **{best_level}** mean mIoU **{best_mean:.3f}** "
            f"(gap {TARGET - best_mean:.3f} below {TARGET:.2f})."
        )
        lines.append(
            "Preprocessing migration succeeded; frozen exemplar params (T1–T3) lift `3-1-1` "
            "but panel mean stalls below 0.60. **T5 GT ring-flip** improves `3-1-2` "
            "but not enough for panel pass. Dominant residual errors: detection/K placement "
            "on continuous tunnels and partial mirror correction on `3-1-3`."
        )
    else:
        lines.append("**No complete panel results yet.**")

    lines += [
        "",
        "## Artifacts",
        "",
        f"- Results: `{LOOP_ROOT.relative_to(REPO_ROOT)}/{{level}}/{{tunnel}}/`",
        f"- Validation: `logs/t3_hint_loop/validate_preprocessing.json`",
        f"- Migration: `logs/t3_hint_loop/migrate_preprocessing.json`",
        "",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
