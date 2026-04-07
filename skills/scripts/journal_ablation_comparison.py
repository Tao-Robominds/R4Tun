#!/usr/bin/env python3
"""
Emit methods/journals-style ablation comparison markdown (paired t-test vs sam4tun).

Reads data/ablation/{sam4tun,memory,memory+state,memory+state+knowledge}/<tunnel>/evaluation/performance.md

Usage:
  ./venv/bin/python skills/scripts/journal_ablation_comparison.py \\
    --out methods/journals/comparison_openai.md \\
    --summary "OpenAI GPT-5.4 …"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "configurable"))

from pipeline_data import ABLATION_CONDITIONS  # noqa: E402

MIOU_PATTERN = re.compile(r"Mean IoU \(mIoU\):\s*([\d.]+)")

COND_ORDER = ["sam4tun", "m", "m_s", "m_s_k"]
COND_LABEL = {
    "sam4tun": "sam4tun (baseline)",
    "m": "memory",
    "m_s": "memory+state",
    "m_s_k": "memory+state+knowledge",
}
FOLDER = {c: REPO_ROOT / ABLATION_CONDITIONS[c]["out_prefix"] for c in COND_ORDER}


def tunnel_sort_key(tid: str) -> tuple[int, ...]:
    return tuple(int(p) for p in tid.split("-"))


def discover_tunnels() -> list[str]:
    base = FOLDER["m"]
    if not base.is_dir():
        return []
    return sorted(
        (d.name for d in base.iterdir() if d.is_dir() and re.match(r"^\d+-", d.name)),
        key=tunnel_sort_key,
    )


def parse_miou(path: Path) -> float | None:
    if not path.is_file():
        return None
    m = MIOU_PATTERN.search(path.read_text())
    return float(m.group(1)) if m else None


def miou(cond: str, tunnel_id: str) -> float | None:
    return parse_miou(FOLDER[cond] / tunnel_id / "evaluation" / "performance.md")


def tunnel_type(tid: str) -> str:
    p = tid.split("-")[0]
    if p in ("1", "2"):
        return "reg"
    if p == "3":
        return "con"
    return "com"


def subset_mask(tunnels: list[str], name: str) -> np.ndarray:
    masks = {
        "all": np.ones(len(tunnels), dtype=bool),
        "regular_additional": np.array(
            [tid.split("-")[0] in ("1", "2", "3") for tid in tunnels]
        ),
        "alternated": np.array([tid.split("-")[0] in ("1", "2") for tid in tunnels]),
        "continuous": np.array([tid.split("-")[0] == "3" for tid in tunnels]),
        "complex": np.array([tid.split("-")[0] in ("4", "5") for tid in tunnels]),
    }
    return masks[name]


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return "p<0.0001"
    if p < 0.001:
        return "p=0.000"
    return f"p={p:.3f}"


def fmt_delta(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.3f}"


def paired_stats(
    baseline: np.ndarray,
    cond: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, float, float] | None:
    m = mask & np.isfinite(baseline) & np.isfinite(cond)
    if m.sum() < 2:
        return None
    b, c = baseline[m], cond[m]
    deltas = c - b
    mean_miou = float(np.mean(c))
    mean_d = float(np.mean(deltas))
    std_d = float(np.std(deltas, ddof=1))
    from scipy.stats import ttest_rel

    p = float(ttest_rel(c, b).pvalue)
    return mean_miou, mean_d, std_d, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "methods" / "journals" / "comparison_openai.md",
    )
    ap.add_argument(
        "--summary",
        required=True,
        help="One paragraph for the ## Summary section (model / run description).",
    )
    args = ap.parse_args()

    tunnels = discover_tunnels()
    if len(tunnels) != 30:
        print(f"warning: expected 30 tunnels, found {len(tunnels)}", file=sys.stderr)

    arrays = {c: np.array([miou(c, t) for t in tunnels], dtype=float) for c in COND_ORDER}
    for c in COND_ORDER:
        arrays[c][np.isnan(arrays[c])] = np.nan

    base = arrays["sam4tun"]

    def summary_row(name: str, label: str) -> str:
        """label is full first-column markdown, e.g. '**Overall**' or '- **Alternated (n=10)**'."""
        mask = subset_mask(tunnels, name)
        parts = []
        for code in ("m", "m_s", "m_s_k"):
            st = paired_stats(base, arrays[code], mask)
            if st is None:
                parts.append("—")
                continue
            _, mean_d, _, p = st
            parts.append(f"{fmt_delta(mean_d)} ({fmt_p(p)})")
        return f"| {label:<25} | {parts[0]:<18} | {parts[1]:<24} | {parts[2]:<34} |"

    def family_block(title: str, subset: str) -> list[str]:
        mask = subset_mask(tunnels, subset)
        lines = [
            "",
            f"### {title}",
            "",
            "",
            "| condition              | mean_mIoU | delta vs baseline | std (delta) | p-value  |",
            "| ---------------------- | --------- | ----------------- | ----------- | -------- |",
        ]
        bmask = mask & np.isfinite(base)
        if bmask.sum() < 2:
            lines.append("| (insufficient paired data) | — | — | — | — |")
            return lines

        b_only = base[bmask]
        base_mean = float(np.mean(b_only))
        lines.append(
            f"| {COND_LABEL['sam4tun']:<22} | {base_mean:.3f}     | —                 | —           | —        |"
        )

        for code in ("m", "m_s", "m_s_k"):
            st = paired_stats(base, arrays[code], mask)
            if st is None:
                lines.append(f"| {COND_LABEL[code]:<22} | —         | —                 | —           | —        |")
                continue
            mean_miou, mean_d, std_d, p = st
            sign = "+" if mean_d >= 0 else ""
            lines.append(
                f"| {COND_LABEL[code]:<22} | {mean_miou:.3f}     | {sign}{mean_d:.3f}            | {std_d:.3f}       | {fmt_p(p):<8} |"
            )
        return lines

    # Overall summary sentence: three deltas + p for all 30
    st_m = paired_stats(base, arrays["m"], subset_mask(tunnels, "all"))
    st_ms = paired_stats(base, arrays["m_s"], subset_mask(tunnels, "all"))
    st_msk = paired_stats(base, arrays["m_s_k"], subset_mask(tunnels, "all"))
    if st_m and st_ms and st_msk:
        summ_extra = (
            f"Overall vs baseline: memory {fmt_delta(st_m[1])} ({fmt_p(st_m[3])}), "
            f"memory+state {fmt_delta(st_ms[1])} ({fmt_p(st_ms[3])}), "
            f"memory+state+knowledge {fmt_delta(st_msk[1])} ({fmt_p(st_msk[3])})."
        )
    else:
        summ_extra = "(Insufficient paired evaluation data for full summary.)"

    lines_out: list[str] = [
        "# Ablation Comparison Report",
        "",
        "Baseline: sam4tun | Conditions: memory (m), memory+state (m_s), memory+state+knowledge (m_s_k)",
        "Tunnels: 30 — **Regular additional** = regular ∪ continuous (n=13); breakdown still 10 regular, 3 continuous, 17 complex",
        "Test: paired t-test (two-sided) vs baseline per tunnel subset",
        "",
        "## Summary",
        "",
        args.summary.strip(),
        "",
        summ_extra,
        "",
        "",
        "|                         | memory vs baseline | memory+state vs baseline | memory+state+knowledge vs baseline |",
        "| ----------------------- | ------------------ | ------------------------ | ---------------------------------- |",
        summary_row("all", "**Overall**"),
        summary_row("regular_additional", "**Regular (n=13)**"),
        summary_row("alternated", "- **Alternated (n=10)**"),
        summary_row("continuous", "- **Continuous (n=3)**"),
        summary_row("complex", "**Complex (n=17)**"),
        "",
        "## Family-level Statistics",
    ]
    lines_out.extend(family_block("Overall (n=30)", "all"))
    lines_out.extend(
        family_block("Regular additional — regular ∪ continuous (n=13)", "regular_additional")
    )
    lines_out.extend(family_block("regular (n=10)", "alternated"))
    lines_out.extend(family_block("continuous (n=3)", "continuous"))
    lines_out.extend(family_block("complex (n=17)", "complex"))

    lines_out.extend(
        [
            "",
            "## Per-tunnel mIoU",
            "",
            "",
            "| tunnel_id | type | sam4tun | memory | delta_m | memory+state | delta_ms | m_s_k | delta_msk |",
            "| --------- | ---- | ------- | ------ | ------- | ------------ | -------- | ----- | --------- |",
        ]
    )

    for tid in tunnels:
        ty = tunnel_type(tid)
        vals = [miou(c, tid) for c in COND_ORDER]
        if any(v is None for v in vals):
            row = f"| {tid:<9} | {ty:<3} |" + " — |" * 8
            lines_out.append(row)
            continue
        s, m_, ms, msk = vals
        row = (
            f"| {tid:<9} | {ty:<3} | {s:.3f}   | {m_:.3f}  | {fmt_delta(m_ - s):>7}   | {ms:.3f}        | {fmt_delta(ms - s):>8}   | {msk:.3f} | {fmt_delta(msk - s):>9}    |"
        )
        lines_out.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines_out) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
