#!/usr/bin/env python3
"""
Bootstrap CIs, paired Cohen's d, binomial sign tests, and within-family spread
from per-tunnel mIoU tables in comparison journals.

Writes:
  methods/papers/output/confidence_analysis.md
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


def bootstrap_mean_ci(
    deltas: np.ndarray, n_boot: int = 10_000, seed: int = 42
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) for mean(deltas) via paired bootstrap on indices."""
    n = len(deltas)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = float(np.mean(deltas[idx]))
    return float(np.mean(deltas)), float(np.percentile(means, 2.5)), float(
        np.percentile(means, 97.5)
    )


def paired_cohens_d(deltas: np.ndarray) -> float:
    """Paired Cohen's d = mean(d) / std(d)."""
    if len(deltas) < 2:
        return float("nan")
    sd = float(np.std(deltas, ddof=1))
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(deltas) / sd)


def format_ci(lo: float, hi: float) -> str:
    if lo != lo or hi != hi:
        return "—"
    return f"[{lo:.3f}, {hi:.3f}]"


def binomial_sign_test_positives(deltas: np.ndarray) -> tuple[int, int, float]:
    """One-sided binomial test H0: p=0.5 vs H1: p>0.5 for positive deltas."""
    n = len(deltas)
    pos = int(np.sum(deltas > 0))
    # scipy binomtest: successes = positive, n_trials = n (ties count as non-success for one-sided >0.5 on strict positive)
    if n == 0:
        return 0, 0, float("nan")
    res = stats.binomtest(pos, n, p=0.5, alternative="greater")
    return pos, n, float(res.pvalue)


def format_p_binom(p: float) -> str:
    if p != p:
        return "p=—"
    if p < 1e-4:
        return "p<0.0001"
    return f"p={p:.4g}"


def analyze_llm(
    rows: list[dict], llm_name: str, n_boot: int = 10_000
) -> tuple[list[str], dict]:
    """Return markdown lines and a dict of summary stats for aggregation."""
    lines: list[str] = []
    b = np.array([r["sam4tun"] for r in rows])
    mem = np.array([r["memory"] for r in rows])
    ms = np.array([r["memory+state"] for r in rows])
    msk = np.array([r["memory+state+knowledge"] for r in rows])

    d_mem = mem - b
    d_ms = ms - b
    d_msk = msk - b
    d_know = msk - ms  # knowledge increment on top of memory+state

    lines.append(f"## {llm_name}")
    lines.append("")
    lines.append("### Bootstrap 95% CI on mean paired ΔmIoU (n=30 tunnels, 10 000 resamples)")
    lines.append("")
    lines.append("| Contrast | mean Δ | 95% CI | paired Cohen's d |")
    lines.append("|----------|--------|--------|------------------|")

    for label, d in [
        ("memory − baseline", d_mem),
        ("memory+state − baseline", d_ms),
        ("m+s+k − baseline", d_msk),
        ("m+s+k − memory+state (knowledge increment)", d_know),
    ]:
        mean_d, lo, hi = bootstrap_mean_ci(d, n_boot=n_boot)
        cd = paired_cohens_d(d)
        lines.append(
            f"| {label} | {mean_d:+.3f} | {format_ci(lo, hi)} | {cd:.3f} |"
        )
    lines.append("")

    pos, n_tot, p_bin = binomial_sign_test_positives(d_know)
    lines.append("### Knowledge increment: per-tunnel sign (m+s+k − m+s)")
    lines.append("")
    lines.append(
        f"- Tunnels with strictly positive increment: **{pos}/{n_tot}** "
        f"(one-sided binomial vs p=0.5, {format_p_binom(p_bin)})"
    )
    com_rows = [r for r in rows if r["type"] == "com"]
    if com_rows:
        ms_c = np.array([r["memory+state"] for r in com_rows])
        msk_c = np.array([r["memory+state+knowledge"] for r in com_rows])
        dk_c = msk_c - ms_c
        pos_c, n_c, p_c = binomial_sign_test_positives(dk_c)
        lines.append(
            f"- Complex only (n={n_c}): **{pos_c}/{n_c}** positive "
            f"({format_p_binom(p_c)})"
        )
    lines.append("")

    summary = {
        "d_mem": d_mem,
        "d_ms": d_ms,
        "d_msk": d_msk,
        "d_know": d_know,
        "mem_ci": bootstrap_mean_ci(d_mem, n_boot=n_boot),
        "ms_ci": bootstrap_mean_ci(d_ms, n_boot=n_boot),
        "msk_ci": bootstrap_mean_ci(d_msk, n_boot=n_boot),
        "know_ci": bootstrap_mean_ci(d_know, n_boot=n_boot),
        "know_pos": pos,
        "know_n": n_tot,
        "know_p": p_bin,
        "b": b,
        "mem": mem,
        "ms": ms,
        "msk": msk,
        "rows": rows,
    }
    return lines, summary


def std_per_condition(vals: np.ndarray) -> float:
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def table_4a_block(summaries: list[dict]) -> list[str]:
    """Mean across 3 LLMs of per-condition tunnel-level statistics."""
    cond_keys = [
        ("sam4tun", lambda s: s["b"]),
        ("memory", lambda s: s["mem"]),
        ("memory+state", lambda s: s["ms"]),
        ("m+s+k", lambda s: s["msk"]),
    ]
    lines: list[str] = []
    lines.append("## Aggregated across three LLMs (mean of per-LLM statistics)")
    lines.append("")
    lines.append("### Table 4a inputs: overall (n=30 per LLM)")
    lines.append("")
    lines.append(
        "| Metric | sam4tun | memory | memory+state | m+s+k |"
    )
    lines.append("|--------|---------|--------|--------------|-------|")

    # Mean mIoU
    row_mean = []
    for _k, getv in cond_keys:
        row_mean.append(float(np.mean([float(np.mean(getv(s))) for s in summaries])))
    lines.append(
        "| Mean mIoU | "
        + " | ".join(f"{x:.3f}" for x in row_mean)
        + " |"
    )

    # Std of per-tunnel mIoU (mean across LLMs)
    row_std = []
    for _k, getv in cond_keys:
        stds = [std_per_condition(getv(s)) for s in summaries]
        row_std.append(float(np.mean(stds)))
    lines.append(
        "| Std of per-tunnel mIoU | "
        + " | ".join(f"{x:.3f}" for x in row_std)
        + " |"
    )

    row_min = []
    for _k, getv in cond_keys:
        mins = [float(np.min(getv(s))) for s in summaries]
        row_min.append(float(np.mean(mins)))
    lines.append(
        "| Min tunnel mIoU | "
        + " | ".join(f"{x:.3f}" for x in row_min)
        + " |"
    )

    row_max = []
    for _k, getv in cond_keys:
        maxs = [float(np.max(getv(s))) for s in summaries]
        row_max.append(float(np.mean(maxs)))
    lines.append(
        "| Max tunnel mIoU | "
        + " | ".join(f"{x:.3f}" for x in row_max)
        + " |"
    )

    # Rise in the performance floor: mean across LLMs of (min mIoU_cond − min mIoU_baseline)
    row_floor = ["—"]
    for _k, getv in cond_keys[1:]:
        gains = []
        for s in summaries:
            b = s["b"]
            v = getv(s)
            gains.append(float(np.min(v) - np.min(b)))
        row_floor.append(f"{float(np.mean(gains)):+.3f}")
    lines.append(
        "| Δ min mIoU vs baseline (floor lift, mean across LLMs) | "
        + " | ".join(row_floor)
        + " |"
    )
    lines.append("")
    return lines


def within_family_table(summaries: list[dict]) -> list[str]:
    lines: list[str] = []
    lines.append("### Within-family std of per-tunnel mIoU (mean across 3 LLMs)")
    lines.append("")
    lines.append(
        "| Family | n | sam4tun | memory | memory+state | m+s+k |"
    )
    lines.append("|--------|---|---------|--------|--------------|-------|")

    def std_for_family(s: dict, getv, pred) -> float:
        idx = [i for i, r in enumerate(s["rows"]) if pred(r)]
        if len(idx) < 2:
            return float("nan")
        v = getv(s)[idx]
        return std_per_condition(v)

    reg_pred = lambda r: r["type"] in ("reg", "con")
    com_pred = lambda r: r["type"] == "com"

    for fam_name, pred, n_exp in [
        ("Regular ∪ continuous", reg_pred, 13),
        ("Complex", com_pred, 17),
    ]:
        row = [fam_name, str(n_exp)]
        for _k, getv in [
            ("sam4tun", lambda s: s["b"]),
            ("memory", lambda s: s["mem"]),
            ("memory+state", lambda s: s["ms"]),
            ("m+s+k", lambda s: s["msk"]),
        ]:
            stds = [std_for_family(s, getv, pred) for s in summaries]
            stds = [x for x in stds if x == x]
            row.append(f"{float(np.mean(stds)):.3f}" if stds else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "methods" / "papers" / "output" / "confidence_analysis.md",
    )
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    out: list[str] = [
        "# Confidence analysis (bootstrap CI, Cohen's d, sign tests, spread)",
        "",
        "Source: per-tunnel mIoU in `methods/journals/comparison_*.md`. "
        "Bootstrap: resample tunnels with replacement (10 000 iterations), "
        "mean of paired ΔmIoU each draw; report 2.5th and 97.5th percentiles.",
        "",
    ]

    summaries: list[dict] = []
    for llm_name, path in JOURNAL_FILES:
        if not path.is_file():
            out.append(f"## {llm_name}\n\n_Missing: {path}_\n\n")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rows = parse_tunnel_table(text)
        if len(rows) != 30:
            out.append(f"_Warning {llm_name}: parsed {len(rows)} tunnels (expected 30)._\n\n")
        block, summ = analyze_llm(rows, llm_name, n_boot=args.n_boot)
        out.extend(block)
        summaries.append(summ)

    if summaries:
        out.extend(table_4a_block(summaries))
        out.extend(within_family_table(summaries))

        out.append("## Cross-LLM summary: knowledge increment bootstrap CI")
        out.append("")
        for i, (name, _) in enumerate(JOURNAL_FILES[: len(summaries)]):
            m, lo, hi = summaries[i]["know_ci"]
            out.append(f"- **{name}:** mean Δ = {m:+.3f}, 95% CI {format_ci(lo, hi)}")
        out.append("")

    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
