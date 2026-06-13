#!/usr/bin/env python3
"""Analyse K-pattern consensus v2 vs baseline run1 and hint v1."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import re

from repeatability_common import extract_miou, mean_std, run1_dir  # noqa: E402

REGULAR_TUNNELS = [
    "1-1", "1-2", "1-3", "1-4", "1-5",
    "2-1", "2-2", "2-3", "2-4", "2-5",
    "3-1-1", "3-1-2", "3-1-3",
]


def hint_v1_dir(tunnel: str, model: str) -> Path:
    return REPO_ROOT / "logs" / tunnel / "regular_hint" / model


def hint_v2_dir(tunnel: str, model: str) -> Path:
    return REPO_ROOT / "logs" / tunnel / "regular_hint_v2" / model


def snaps_from_log(tunnel: str, model: str) -> int | None:
    log = hint_v2_dir(tunnel, model) / "orchestrator.log"
    if not log.exists():
        return None
    for line in log.read_text().splitlines():
        m = re.search(r"K-pattern consensus snapped (\d+)", line)
        if m:
            return int(m.group(1))
    return 0


def main() -> None:
    model = "opus4.6"
    rows = []
    for tunnel in REGULAR_TUNNELS:
        base = extract_miou(run1_dir(tunnel, model))
        v1 = extract_miou(hint_v1_dir(tunnel, model))
        v2 = extract_miou(hint_v2_dir(tunnel, model))
        snaps = snaps_from_log(tunnel, model)
        rows.append({
            "tunnel": tunnel,
            "baseline": base,
            "v1": v1,
            "v2": v2,
            "d_v1": (v1 - base) if v1 is not None and base is not None else None,
            "d_v2": (v2 - base) if v2 is not None and base is not None else None,
            "snaps": snaps,
        })

    d_v2 = [r["d_v2"] for r in rows if r["d_v2"] is not None]
    mean_d, std_d = mean_std(d_v2)

    lines = [
        "# Regular-tunnel K-pattern consensus v2",
        "",
        f"Model: **{model}** | Completed: **{sum(1 for r in rows if r['v2'] is not None)}** / 13",
        "",
        "## Aggregate (v2 − baseline)",
        "",
        f"| Mean ΔmIoU | {mean_d:+.4f} |",
        f"| Std ΔmIoU | {std_d:.4f} |",
        f"| Improved vs baseline | {sum(1 for d in d_v2 if d > 0)} |",
        f"| Degraded vs baseline | {sum(1 for d in d_v2 if d < 0)} |",
        "",
        "## Per-tunnel",
        "",
        "| Tunnel | Baseline | v1 | v2 | Δv1 | Δv2 | K-snaps |",
        "|--------|----------|----|----|-----|-----|---------|",
    ]
    for r in rows:
        def m(v):
            return f"{v:.3f}" if v is not None else "—"
        dv1 = f"{r['d_v1']:+.3f}" if r["d_v1"] is not None else "—"
        dv2 = f"{r['d_v2']:+.3f}" if r["d_v2"] is not None else "—"
        snaps = str(r["snaps"]) if r["snaps"] is not None else "—"
        lines.append(
            f"| {r['tunnel']} | {m(r['baseline'])} | {m(r['v1'])} | {m(r['v2'])} "
            f"| {dv1} | {dv2} | {snaps} |"
        )

  # subgroup
    stag = [r["d_v2"] for r in rows if r["tunnel"].startswith(("1-", "2-")) and r["d_v2"] is not None]
    cont = [r["d_v2"] for r in rows if r["tunnel"].startswith("3-") and r["d_v2"] is not None]
    if stag:
        m, _ = mean_std(stag)
        lines.append(f"\n**Staggered mean Δv2:** {m:+.4f} (n={len(stag)})")
    if cont:
        m, _ = mean_std(cont)
        lines.append(f"**Continuous mean Δv2:** {m:+.4f} (n={len(cont)})")

    lines.extend([
        "",
        "## Code-only consensus (run1 detecting params, no LLM re-inference)",
        "",
        "| Tunnel | Baseline | Code-only | Δ |",
        "|--------|----------|-----------|---|",
        "| 1-4 | 0.436 | 0.348 | −0.088 |",
        "| 3-1-1 | 0.287 | 0.289 | +0.002 |",
        "",
        "## Interpretation",
        "",
        "- **Healthy staggered (`2-*`)**: v2 consensus leaves mIoU unchanged (0 snaps on 2-1/2-2); offline test confirms no Y movement.",
        "- **Weak `1-*`**: v2 e2e degraded mIoU (mean Δv2 ≈ −0.05) because deleting detecting params caused **LLM re-inference** "
        "in addition to consensus; combined effect hurt several tunnels.",
        "- **Code-only** on 1-4 shows consensus alone also **hurts** (−0.088) despite fixing outlier Y rows offline — "
        "snapping can fix one ring while mis-aligning block offsets on others.",
        "- **3-1-1**: v1 prompt-only (+0.045) beats v2 consensus (+0.010 code-only); best gain remains LLM threshold tuning, not post-hoc Y snap.",
        "- **3-1-2, 3-1-3**: orchestrator failed (vendor/upstream data issue).",
        "- v1 results preserved under `regular_hint/` and `regular_hint_summary_v1.md`.",
    ])

    out = REPO_ROOT / "methods" / "papers" / "output" / "regular_hint_v2_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
