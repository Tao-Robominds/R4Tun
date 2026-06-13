#!/usr/bin/env python3
"""
Compute Wilcoxon signed-rank tests on paired mIoU differences from comparison journals.

Reads per-tunnel tables in:
  methods/journals/comparison_openai.md
  methods/journals/comparison_anthropic.md
  methods/journals/comparison_gemini.md

Writes:
  methods/papers/output/wilcoxon_vs_ttest.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]

JOURNAL_FILES = [
    ("GPT-5.4", REPO_ROOT / "methods" / "journals" / "comparison_openai.md"),
    ("Claude Opus 4.6", REPO_ROOT / "methods" / "journals" / "comparison_anthropic.md"),
    ("Gemini 3 Flash", REPO_ROOT / "methods" / "journals" / "comparison_gemini.md"),
]

def parse_tunnel_table(text: str) -> list[dict]:
    """Parse markdown table: tunnel_id | type | sam4tun | memory | ... | m_s_k |."""
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or "tunnel_id" in line or line.startswith("|---"):
            continue
        parts = [p.strip() for p in line.split("|")]
        # leading/trailing empty from split
        parts = [p for p in parts if p != ""]
        if len(parts) < 8:
            continue
        tid, typ = parts[0], parts[1]
        if typ not in ("reg", "con", "com"):
            continue
        try:
            b0 = float(parts[2])
            m0 = float(parts[3])
            ms = float(parts[5])
            msk = float(parts[7])
        except (ValueError, IndexError):
            continue
        rows.append(
            {
                "tunnel_id": tid,
                "type": typ,
                "sam4tun": b0,
                "memory": m0,
                "memory+state": ms,
                "memory+state+knowledge": msk,
            }
        )
    return rows


def family_filter(rows: list[dict], family: str) -> list[dict]:
    if family == "overall":
        return rows
    if family == "regular_all":
        return [r for r in rows if r["type"] in ("reg", "con")]
    if family == "alternated":
        return [r for r in rows if r["type"] == "reg"]
    if family == "continuous":
        return [r for r in rows if r["type"] == "con"]
    if family == "complex":
        return [r for r in rows if r["type"] == "com"]
    raise ValueError(family)


def paired_tests(
    baseline: np.ndarray, treatment: np.ndarray
) -> tuple[float, float, float]:
    """Return (mean_delta, t_pvalue, wilcoxon_pvalue)."""
    d = treatment - baseline
    mean_d = float(np.mean(d))
    if len(d) < 2:
        return mean_d, float("nan"), float("nan")
    t_res = stats.ttest_rel(treatment, baseline, alternative="two-sided")
    t_p = float(t_res.pvalue)
    # Wilcoxon: require at least one non-zero difference for exact test
    if np.allclose(d, 0):
        return mean_d, t_p, 1.0
    try:
        w_res = stats.wilcoxon(
            treatment,
            baseline,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        w_p = float(w_res.pvalue)
    except ValueError:
        w_p = float("nan")
    return mean_d, t_p, w_p


def format_p(p: float) -> str:
    if p != p:  # nan
        return "—"
    if p < 1e-4:
        return "p<0.0001"
    return f"p={p:.4g}"


def run_all(rows: list[dict]) -> list[str]:
    families = [
        ("overall", "Overall (n=30)"),
        ("regular_all", "Regular ∪ continuous (n=13)"),
        ("alternated", "Alternated (n=10)"),
        ("continuous", "Continuous (n=3)"),
        ("complex", "Complex (n=17)"),
    ]
    conditions = [
        ("memory", "memory vs baseline"),
        ("memory+state", "memory+state vs baseline"),
        ("memory+state+knowledge", "m_s_k vs baseline"),
    ]
    lines: list[str] = []
    for fam_key, fam_label in families:
        sub = family_filter(rows, fam_key)
        if len(sub) < 2 and fam_key != "continuous":
            continue
        b = np.array([r["sam4tun"] for r in sub])
        lines.append(f"### {fam_label}")
        lines.append("")
        lines.append(
            "| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |"
        )
        lines.append("|-----------|---|------------|---------------|----------|")
        for cond_key, cond_label in conditions:
            y = np.array([r[cond_key] for r in sub])
            mean_d, t_p, w_p = paired_tests(b, y)
            lines.append(
                f"| {cond_label} | {len(sub)} | {mean_d:+.3f} | {format_p(t_p)} | {format_p(w_p)} |"
            )
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "methods" / "papers" / "output" / "wilcoxon_vs_ttest.md",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    out_parts: list[str] = [
        "# Paired t-test vs Wilcoxon signed-rank (mIoU)",
        "",
        "Per-tunnel paired comparisons vs **sam4tun** baseline. "
        "t-test and Wilcoxon are both two-sided. "
        "Source tables: `methods/journals/comparison_*.md`.",
        "",
    ]

    for llm_name, path in JOURNAL_FILES:
        if not path.is_file():
            out_parts.append(f"## {llm_name}\n\n_Missing file: {path}_\n\n")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rows = parse_tunnel_table(text)
        out_parts.append(f"## {llm_name}\n\n")
        if len(rows) != 30:
            out_parts.append(f"_Warning: parsed {len(rows)} tunnels (expected 30)._\n\n")
        out_parts.extend(run_all(rows))

    args.out.write_text("\n".join(out_parts), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
