#!/usr/bin/env python3
"""Analyse continuous-tunnel v3 (prompt-only) vs run1 and v1."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from repeatability_common import REPO_ROOT, extract_miou, load_flat_params, mean_std, run1_dir  # noqa: E402

CONTINUOUS = ["3-1-1", "3-1-2", "3-1-3"]
MODEL = "opus4.6"
OUT = REPO_ROOT / "methods" / "papers" / "output" / "regular_hint_v3_summary.md"
FEAS = REPO_ROOT / "logs" / "walk_direction_feasibility.txt"


def v3_dir(tunnel: str) -> Path:
    return REPO_ROOT / "logs" / tunnel / "regular_hint_v3" / MODEL


def v1_dir(tunnel: str) -> Path:
    return REPO_ROOT / "logs" / tunnel / "regular_hint" / MODEL


def _fmt(x: float | None, prec: int = 3) -> str:
    return f"{x:.{prec}f}" if x is not None else "—"


def load_type_dist(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    swa = data.get("sam_workflow_analysis", {})
    raw = (
        swa.get("prompt_distribution", {}).get("type_distribution")
        or swa.get("prompt_effectiveness", {})
        .get("segmentation_effectiveness", {})
        .get("prompt_type_distribution", {})
        or {}
    )
    return {str(k): int(v) for k, v in raw.items()}


def char_path(tunnel: str) -> Path:
    d = v3_dir(tunnel)
    p = d / "detected_characteristics.json"
    if p.exists():
        return p
    return d / "data" / "characteristics" / "detected_characteristics.json"


def y_spread(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        yr = (
            data.get("sam_workflow_analysis", {})
            .get("prompt_distribution", {})
            .get("sam_template_distribution", {})
            .get("spatial_bounds", {})
            .get("y_range")
        )
        if yr and len(yr) == 2:
            return float(yr[1]) - float(yr[0])
    except (json.JSONDecodeError, OSError, TypeError):
        pass
    return None


def detecting_diff(base: dict, new: dict) -> list[str]:
    out = []
    for k in sorted(set(base) | set(new)):
        if not k.startswith("detecting."):
            continue
        if base.get(k) != new.get(k):
            out.append(f"`{k}`: {base.get(k)} → {new.get(k)}")
    return out


def per_ring_mirror(final_csv: Path) -> list[tuple[int, int, float, bool]]:
    per = defaultdict(lambda: ([], []))

    def trans(p, s, k):
        return ((p - 1) * s + k) % 6 + 1

    with open(final_csv) as f:
        for row in csv.DictReader(f):
            try:
                seg = int(float(row["segment"]))
                pred = int(float(row["pred"]))
                pr = int(float(row["pred_ring"]))
            except ValueError:
                continue
            if seg > 0 and pred > 0:
                per[pr][0].append(seg)
                per[pr][1].append(pred)

    rows = []
    for pr in sorted(per):
        segs, preds = per[pr]
        best = (1, 0, 0)
        for s in (1, -1):
            for k in range(6):
                ok = sum(1 for a, b in zip(segs, preds) if trans(b, s, k) == a)
                if ok > best[2]:
                    best = (s, k, ok)
        s, k, n = best
        rows.append((pr, k, n / len(segs), s == -1 and k == 0))
    return rows


def main() -> None:
    lines = [
        "# Continuous-tunnel v3 summary (prompt-only detecting docs)",
        "",
        f"Model: **{MODEL}**. Tunnels: `3-1-1`, `3-1-2`, `3-1-3`.",
        "Walk-direction code **not** deployed (GT-free cue accuracy <80% gate).",
        "",
        "## mIoU",
        "",
        "| tunnel | run1 | v1 hint | v3 hint | Δv3 vs run1 |",
        "|--------|------|---------|---------|-------------|",
    ]

    deltas: list[float] = []
    for t in CONTINUOUS:
        r1 = extract_miou(run1_dir(t, MODEL))
        v1 = extract_miou(v1_dir(t))
        v3 = extract_miou(v3_dir(t))
        d = (v3 - r1) if v3 is not None and r1 is not None else None
        if d is not None:
            deltas.append(d)
        lines.append(
            f"| {t} | {_fmt(r1)} | {_fmt(v1)} | {_fmt(v3)} | "
            f"{f'{d:+.3f}' if d is not None else '—'} |"
        )

    if deltas:
        m, s = mean_std(deltas)
        lines += ["", f"**Mean Δv3 vs run1:** {m:+.4f} (std {s:.4f}, n={len(deltas)})"]

    lines += [
        "",
        "## Detecting state (GT-free signals, v3 run)",
        "",
        "| tunnel | fallback rate | Y spread (px) | midpoint rate |",
        "|--------|---------------|---------------|---------------|",
    ]
    for t in CONTINUOUS:
        dist = load_type_dist(char_path(t))
        total = sum(dist.values()) or 1
        fb = (dist.get("default", 0) + dist.get("assume", 0)) / total
        mp = dist.get("midpoint", 0) / total
        ys = y_spread(char_path(t))
        lines.append(f"| {t} | {fb:.0%} | {_fmt(ys, 0)} | {mp:.0%} |")

    lines += ["", "## Detecting parameter changes (v3 vs run1)", ""]
    for t in CONTINUOUS:
        base_p = load_flat_params(run1_dir(t, MODEL) / "parameters", MODEL)
        v3_p = load_flat_params(v3_dir(t) / "parameters", MODEL)
        diff = detecting_diff(base_p, v3_p)
        lines.append(f"### {t}")
        if diff:
            lines.extend(f"- {d}" for d in diff)
        else:
            lines.append("- (no detecting param changes or run missing)")
        lines.append("")

    lines += [
        "## Per-ring handedness (evaluation only — uses GT)",
        "",
        "Mirrored = best transform has s=-1, k=0.",
        "",
    ]
    for t in CONTINUOUS:
        final = v3_dir(t) / "data" / "final.csv"
        if not final.exists():
            lines.append(f"### {t}\n\n(no final.csv)\n")
            continue
        rings = per_ring_mirror(final)
        n_mir = sum(1 for _, k, _, m in rings if k == 0 and m)
        n_clean = sum(1 for _, k, _, _ in rings if k == 0)
        lines.append(f"### {t}")
        lines.append(f"- k=0 rings: {n_clean}; mirrored among those: {n_mir}")
        lines.append("")
        lines.append("| pred_ring | k | acc | mirrored |")
        lines.append("|-----------|---|-----|----------|")
        for pr, k, acc, mir in rings:
            lines.append(f"| {pr} | {k} | {acc:.2f} | {mir} |")
        lines.append("")

    if FEAS.exists():
        lines += [
            "## Walk-direction feasibility (design-time)",
            "",
            "```",
            FEAS.read_text().strip(),
            "```",
            "",
            "**Verdict:** slope-sign rule 50%; majority vote 50–60% on k=0 rings.",
            "Below 80% gate → no `walk_direction` code change.",
        ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
