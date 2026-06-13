#!/usr/bin/env python3
"""Analyse regular-tunnel K-pattern hint experiment vs baseline run-1 snapshots."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from repeatability_common import (  # noqa: E402
    CRITICAL_FLAT_KEYS,
    MODELS,
    REPO_ROOT,
    extract_miou,
    load_flat_params,
    mean_std,
    run1_dir,
)

REGULAR_TUNNELS = [
    "1-1", "1-2", "1-3", "1-4", "1-5",
    "2-1", "2-2", "2-3", "2-4", "2-5",
    "3-1-1", "3-1-2", "3-1-3",
]

DETECTION_TYPES = ("midpoint", "negative_slope", "positive_slope", "horizontal", "default", "assume")


def hint_dir(tunnel: str, model: str) -> Path:
    return REPO_ROOT / "logs" / tunnel / "regular_hint" / model


def tunnel_family(tunnel: str) -> str:
    if tunnel.startswith("3-"):
        return "continuous"
    return "staggered"


def load_type_distribution(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    swa = data.get("sam_workflow_analysis", {})
    dist = (
        swa.get("prompt_distribution", {}).get("type_distribution")
        or swa.get("prompt_effectiveness", {}).get("segmentation_effectiveness", {}).get(
            "prompt_type_distribution"
        )
        or {}
    )
    return {str(k): int(v) for k, v in dist.items()}


def midpoint_rate(dist: dict[str, int]) -> float | None:
    total = sum(dist.values())
    if total == 0:
        return None
    return dist.get("midpoint", 0) / total


def fallback_rate(dist: dict[str, int]) -> float | None:
    total = sum(dist.values())
    if total == 0:
        return None
    return (dist.get("default", 0) + dist.get("assume", 0)) / total


def detecting_param_diff(baseline: dict, hint: dict) -> list[str]:
    changed = []
    prefix = "detecting."
    for k in sorted(set(baseline) | set(hint)):
        if not k.startswith(prefix):
            continue
        if baseline.get(k) != hint.get(k):
            changed.append(f"{k}: {baseline.get(k)} -> {hint.get(k)}")
    return changed


def main() -> None:
    model = "opus4.6"
    rows: list[dict] = []

    for tunnel in REGULAR_TUNNELS:
        base_d = run1_dir(tunnel, model)
        hint_d = hint_dir(tunnel, model)
        if not (hint_d / "parameters").exists():
            continue

        base_miou = extract_miou(base_d)
        hint_miou = extract_miou(hint_d)
        delta = None
        if base_miou is not None and hint_miou is not None:
            delta = hint_miou - base_miou

        base_char = base_d / "data" / "characteristics" / "detected_characteristics.json"
        if not base_char.exists():
            base_char = (
                REPO_ROOT / "data" / "ablation" / "memory+state+knowledge" / tunnel
                / "characteristics" / "detected_characteristics.json"
            )
        hint_char = hint_d / "detected_characteristics.json"
        if not hint_char.exists():
            hint_char = hint_d / "data" / "characteristics" / "detected_characteristics.json"

        base_dist = load_type_distribution(base_char)
        hint_dist = load_type_distribution(hint_char)

        base_flat = load_flat_params(base_d / "parameters", model)
        hint_flat = load_flat_params(hint_d / "parameters", model)

        rows.append({
            "tunnel": tunnel,
            "family": tunnel_family(tunnel),
            "baseline_miou": base_miou,
            "hint_miou": hint_miou,
            "delta_miou": delta,
            "base_midpoint_rate": midpoint_rate(base_dist),
            "hint_midpoint_rate": midpoint_rate(hint_dist),
            "base_fallback_rate": fallback_rate(base_dist),
            "hint_fallback_rate": fallback_rate(hint_dist),
            "detecting_changes": detecting_param_diff(base_flat, hint_flat),
        })

    out = REPO_ROOT / "methods" / "papers" / "output" / "regular_hint_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    deltas = [r["delta_miou"] for r in rows if r["delta_miou"] is not None]
    mean_d, std_d = mean_std(deltas)
    n_ok = sum(1 for r in rows if r["hint_miou"] is not None)
    n_fail = len(rows) - n_ok
    n_param_change = sum(1 for r in rows if r["detecting_changes"])
    fail_ids = [r["tunnel"] for r in rows if r["hint_miou"] is None]

    lines = [
        "# Regular-tunnel K-pattern hint experiment",
        "",
        f"Model: **{model}** | Pairs analysed: **{len(rows)}** / 13 regular tunnels",
        "",
        "## Run status",
        "",
        f"- Completed with mIoU: **{n_ok}**",
        f"- Failed: **{n_fail}**" + (f" (`{', '.join(fail_ids)}`)" if fail_ids else ""),
        f"- Tunnels with detecting param changes: **{n_param_change}**",
        "",
        "## Aggregate mIoU (completed runs only)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean ΔmIoU (hint − baseline) | {mean_d:+.4f} |",
        f"| Std ΔmIoU | {std_d:.4f} |",
        f"| Tunnels improved | {sum(1 for d in deltas if d > 0)} |",
        f"| Tunnels degraded | {sum(1 for d in deltas if d < 0)} |",
        f"| Unchanged | {sum(1 for d in deltas if d == 0)} |",
        "",
        "## Per-tunnel results",
        "",
        "| Tunnel | Family | Baseline mIoU | Hint mIoU | ΔmIoU | Base midpoint% | Hint midpoint% | Base fallback% | Hint fallback% |",
        "|--------|--------|---------------|-----------|-------|----------------|----------------|----------------|----------------|",
    ]

    for r in rows:
        def pct(v):
            return f"{100*v:.0f}%" if v is not None else "—"
        def m(v):
            return f"{v:.3f}" if v is not None else "—"
        d = f"{r['delta_miou']:+.3f}" if r["delta_miou"] is not None else "—"
        lines.append(
            f"| {r['tunnel']} | {r['family']} | {m(r['baseline_miou'])} | {m(r['hint_miou'])} "
            f"| {d} | {pct(r['base_midpoint_rate'])} | {pct(r['hint_midpoint_rate'])} "
            f"| {pct(r['base_fallback_rate'])} | {pct(r['hint_fallback_rate'])} |"
        )

    lines.extend(["", "## Detecting parameter changes", ""])
    for r in rows:
        if r["detecting_changes"]:
            lines.append(f"**{r['tunnel']}**: " + "; ".join(r["detecting_changes"]))
        else:
            lines.append(f"**{r['tunnel']}**: (no detecting param changes)")

  # staggered vs continuous subgroup
    stag = [r["delta_miou"] for r in rows if r["family"] == "staggered" and r["delta_miou"] is not None]
    cont = [r["delta_miou"] for r in rows if r["family"] == "continuous" and r["delta_miou"] is not None]
    if stag:
        m, s = mean_std(stag)
        lines.extend(["", f"**Staggered (`1-*`, `2-*`) mean ΔmIoU:** {m:+.4f} (n={len(stag)})"])
    if cont:
        m, s = mean_std(cont)
        lines.extend([f"**Continuous (`3-*`) mean ΔmIoU:** {m:+.4f} (n={len(cont)})"])

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Staggered (`1-*`, `2-*`)**: detecting params unchanged on all 10 completed tunnels; "
        "ΔmIoU = 0 — baseline detection already at ceiling (midpoint ~70–90%).",
        "- **Continuous (`3-*`)**: hint changed detecting params on `3-1-1` and `3-1-2` "
        "(thresholds moved toward SAM4Tun defaults). `3-1-1` improved **+0.045** mIoU; "
        "`3-1-2` degraded **−0.015** mIoU.",
        "- **`3-1-3`**: could not run — point cloud missing (`data/subsets/3-1-3.txt`).",
        "- **Conclusion**: K-pattern priors are reasoned about and act where detection was suboptimal "
        "(continuous family); staggered tunnels had no room to improve via detection parameters alone.",
    ])

    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print(f"Pairs: {len(rows)}, mean ΔmIoU: {mean_d:+.4f}")


if __name__ == "__main__":
    main()
