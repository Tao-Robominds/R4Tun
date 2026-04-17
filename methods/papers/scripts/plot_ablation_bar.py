#!/usr/bin/env python3
"""
Generate step-wise ablation bar chart (Fig. 4) split by tunnel family and LLM.
Reads per-tunnel mIoU from comparison journals and computes per-family means
with bootstrap 95% CIs for error bars.

Output: methods/papers/figs/fig4_ablation_bar.pdf
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]

JOURNAL_FILES = [
    ("Opus-4.6", REPO_ROOT / "methods" / "journals" / "comparison_anthropic.md"),
    ("GPT-5.4", REPO_ROOT / "methods" / "journals" / "comparison_openai.md"),
    ("Gemini-3-Flash", REPO_ROOT / "methods" / "journals" / "comparison_gemini.md"),
]

CONDITIONS = ["sam4tun", "memory", "memory+state", "memory+state+knowledge"]
COND_LABELS = ["sam4tun", "m", "m+s", "m+s+k"]
FAMILIES = {
    "Regular": lambda r: r["type"] in ("reg", "con"),
    "Complex": lambda r: r["type"] == "com",
}
BASELINES = {"Regular": 0.291, "Complex": 0.042}

LLM_COLORS = ["#2C73D2", "#FF6B6B", "#44BBA4"]
N_BOOT = 10_000
SEED = 42


def parse_tunnel_table(text: str) -> list[dict]:
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or "tunnel_id" in line or line.startswith("|---"):
            continue
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        if len(parts) < 8:
            continue
        tid, typ = parts[0], parts[1]
        if typ not in ("reg", "con", "com"):
            continue
        try:
            row = {
                "tunnel_id": tid,
                "type": typ,
                "sam4tun": float(parts[2]),
                "memory": float(parts[3]),
                "memory+state": float(parts[5]),
                "memory+state+knowledge": float(parts[7]),
            }
        except (ValueError, IndexError):
            continue
        rows.append(row)
    return rows


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(values)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = np.mean(values[idx])
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    all_data: dict[str, list[dict]] = {}
    for llm_name, path in JOURNAL_FILES:
        text = path.read_text(encoding="utf-8", errors="replace")
        all_data[llm_name] = parse_tunnel_table(text)
        assert len(all_data[llm_name]) == 30, f"{llm_name}: got {len(all_data[llm_name])} tunnels"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.0), sharey=False)
    llm_handles, llm_labels = [], []

    for ax_idx, (fam_name, fam_pred) in enumerate(FAMILIES.items()):
        ax = axes[ax_idx]
        n_cond = len(CONDITIONS)
        n_llm = len(JOURNAL_FILES)
        bar_width = 0.22
        x = np.arange(n_cond)

        for llm_idx, (llm_name, _) in enumerate(JOURNAL_FILES):
            rows = all_data[llm_name]
            fam_rows = [r for r in rows if fam_pred(r)]

            means = []
            ci_lo = []
            ci_hi = []
            for cond in CONDITIONS:
                vals = np.array([r[cond] for r in fam_rows])
                m = np.mean(vals)
                lo, hi = bootstrap_ci(vals)
                means.append(m)
                ci_lo.append(m - lo)
                ci_hi.append(hi - m)

            offset = (llm_idx - 1) * bar_width
            bars = ax.bar(
                x + offset, means, bar_width,
                yerr=[ci_lo, ci_hi],
                capsize=3,
                color=LLM_COLORS[llm_idx], alpha=0.85,
                edgecolor="white", linewidth=0.5,
                error_kw={"linewidth": 0.8, "capthick": 0.8},
            )
            if ax_idx == 0:
                llm_handles.append(bars)
                llm_labels.append(llm_name)

            for bar, m_val, hi_err in zip(bars, means, ci_hi):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    m_val + hi_err + 0.006,
                    f".{int(round(m_val * 1000)):03d}" if m_val < 1 else f"{m_val:.3f}",
                    ha="center", va="bottom", fontsize=8,
                    color="#333333", rotation=90,
                )

        baseline_val = BASELINES[fam_name]
        msk_vals = []
        for llm_name_inner, _ in JOURNAL_FILES:
            fam_rows_inner = [r for r in all_data[llm_name_inner] if fam_pred(r)]
            msk_vals.append(np.mean([r["memory+state+knowledge"] for r in fam_rows_inner]))
        msk_mean = np.mean(msk_vals)

        ax.axhline(y=baseline_val, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7,
                    label=f"Baseline = {baseline_val:.3f}")
        ax.axhline(y=msk_mean, color="#E8871E", linestyle="-.", linewidth=1.0, alpha=0.7,
                    label=f"m+s+k mean = {msk_mean:.3f}")
        ax.legend(fontsize=10, loc="upper left", framealpha=0.85, edgecolor="none")

        n_fam = sum(1 for r in all_data[JOURNAL_FILES[0][0]] if fam_pred(r))
        ax.set_title(f"{fam_name} ($n={n_fam}$)", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(COND_LABELS, fontsize=11)
        ax.set_ylabel("Mean mIoU" if ax_idx == 0 else "", fontsize=12)
        ax.set_xlabel("Ablation condition", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylim(0, 0.82)
    axes[1].set_ylim(0, 0.34)

    fig.legend(
        llm_handles, llm_labels,
        loc="upper center", ncol=3, fontsize=11,
        frameon=False, bbox_to_anchor=(0.5, 1.03),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = REPO_ROOT / "methods" / "papers" / "figs" / "ablation_bar.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
