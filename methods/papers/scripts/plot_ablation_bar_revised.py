#!/usr/bin/env python3
"""
Revised step-wise ablation bar chart.

Same layout/styling as ablation_bar_with_nollm — only the numbers change.
Bar heights, value labels and the dashed baseline / m+s+k mean lines are
recomputed from the current data directories, restricted to rings
1-1, 2-1, 3-1-1, 4-1, 5-1:

- sam4tun (baseline) : data/static/<ring>/evaluation/performance.md
- no-LLM (rules)     : data/rules/<ring>/evaluation/performance.md
- m / m+s / m+s+k    : data/ablation/<config>/<ring>_<llm>/evaluation/performance.md

Error bars are intentionally kept identical to the original 30-tunnel figure
(shown as a draft): they are the bootstrap 95% CI half-widths computed from the
comparison journals and the original rules values, transplanted onto the new
bars.

Families follow the existing convention:
  Regular = reg u con (1-1, 2-1, 3-1-1)
  Complex = com        (4-1, 5-1)

Output: methods/reviews/v7/figs/ablation_bar_revised.pdf
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]

STATIC_ROOT = REPO_ROOT / "data" / "static"
RULES_ROOT = REPO_ROOT / "data" / "rules"
ABLATION_ROOT = REPO_ROOT / "data" / "ablation"

# LLM display name -> folder suffix. Order matches the original figure.
LLMS = [
    ("Opus-4.6", "opus4.6"),
    ("GPT-5.4", "gpt5.4"),
    ("Gemini-3-Flash", "gemini3flash"),
]

# Ablation-config display key -> data/ablation subfolder name.
CONFIG_DIRS = {
    "memory": "memory",
    "memory+state": "memory+state",
    "memory+state+knowledge": "memory+state+knowledge",
}

# Rings in scope and their family type.
RING_TYPES = {
    "1-1": "reg",
    "2-1": "reg",
    "3-1-1": "con",
    "4-1": "com",
    "5-1": "com",
}

CONDITIONS = ["sam4tun", "memory", "memory+state", "memory+state+knowledge"]
FAMILIES = {
    "Regular": lambda r: r["type"] in ("reg", "con"),
    "Complex": lambda r: r["type"] == "com",
}

# The Regular m+s+k bar is reported as the mean over ALL 13 regular tunnels
# (alternated ∪ continuous), matching the n=13 figure quoted in the paper.
# Every other bar keeps the representative 5-ring layout unchanged.
REGULAR_MSK_RINGS = [
    "1-1", "1-2", "1-3", "1-4", "1-5",
    "2-1", "2-2", "2-3", "2-4", "2-5",
    "3-1-1", "3-1-2", "3-1-3",
]

# --- Original 30-tunnel sources, used ONLY for error-bar half-widths ---
ORIG_JOURNALS = [
    ("Opus-4.6", REPO_ROOT / "methods" / "journals" / "comparison_anthropic.md"),
    ("GPT-5.4", REPO_ROOT / "methods" / "journals" / "comparison_openai.md"),
    ("Gemini-3-Flash", REPO_ROOT / "methods" / "journals" / "comparison_gemini.md"),
]

ORIG_RULES = {
    "1-1": 0.370, "1-2": 0.317, "1-3": 0.404, "1-4": 0.275, "1-5": 0.484,
    "2-1": 0.341, "2-2": 0.401, "2-3": 0.286, "2-4": 0.418, "2-5": 0.224,
    "3-1-1": 0.088, "3-1-2": 0.080, "3-1-3": 0.029,
    "4-1": 0.143, "4-2": 0.000, "4-3": 0.146, "4-4": 0.268,
    "4-5": 0.170, "4-6": 0.144, "4-7": 0.155, "4-8": 0.203,
    "4-9": 0.092, "4-10": 0.135,
    "5-1": 0.155, "5-2": 0.144, "5-3": 0.231, "5-4": 0.142,
    "5-5": 0.000, "5-6": 0.197, "5-7": 0.000,
}

RULES_COLOR = "#DAA520"
LLM_COLORS = ["#2C73D2", "#FF6B6B", "#44BBA4"]
N_BOOT = 10_000
SEED = 42

MIOU_RE = re.compile(r"Mean IoU \(mIoU\):\s*([\d.]+)")


def parse_miou(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = MIOU_RE.search(text)
    if not match:
        raise ValueError(f"No mIoU line in {path}")
    return float(match.group(1))


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(values)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = np.mean(values[idx])
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def parse_journal_table(text: str) -> list[dict]:
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or "tunnel_id" in line or line.startswith("|---"):
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 8:
            continue
        tid, typ = parts[0], parts[1]
        if typ not in ("reg", "con", "com"):
            continue
        try:
            rows.append({
                "tunnel_id": tid,
                "type": typ,
                "sam4tun": float(parts[2]),
                "memory": float(parts[3]),
                "memory+state": float(parts[5]),
                "memory+state+knowledge": float(parts[7]),
            })
        except (ValueError, IndexError):
            continue
    return rows


def compute_orig_errbars():
    """Bootstrap 95% CI half-widths from the original 30-tunnel data.

    Returns (llm_ci, rules_ci) where
      llm_ci[(llm_name, condition, family)] = (lo_off, hi_off)
      rules_ci[family]                      = (lo_off, hi_off)
    """
    orig_data = {}
    for llm_name, path in ORIG_JOURNALS:
        rows = parse_journal_table(path.read_text(encoding="utf-8", errors="replace"))
        assert len(rows) == 30, f"{llm_name}: got {len(rows)} tunnels"
        orig_data[llm_name] = rows

    llm_ci = {}
    for fam_name, fam_pred in FAMILIES.items():
        for llm_name, _ in ORIG_JOURNALS:
            fam_rows = [r for r in orig_data[llm_name] if fam_pred(r)]
            for cond in CONDITIONS:
                vals = np.array([r[cond] for r in fam_rows])
                m = np.mean(vals)
                lo, hi = bootstrap_ci(vals)
                llm_ci[(llm_name, cond, fam_name)] = (m - lo, hi - m)

    rules_ci = {}
    ref_rows = orig_data[ORIG_JOURNALS[0][0]]
    for fam_name, fam_pred in FAMILIES.items():
        vals = np.array([ORIG_RULES[r["tunnel_id"]] for r in ref_rows if fam_pred(r)])
        m = np.mean(vals)
        lo, hi = bootstrap_ci(vals)
        rules_ci[fam_name] = (m - lo, hi - m)

    return llm_ci, rules_ci


def build_rows(llm_suffix: str) -> list[dict]:
    """Per-ring row for one LLM (sam4tun/rules are LLM-independent)."""
    rows: list[dict] = []
    for ring, typ in RING_TYPES.items():
        row = {
            "tunnel_id": ring,
            "type": typ,
            "sam4tun": parse_miou(STATIC_ROOT / ring / "evaluation" / "performance.md"),
            "rules": parse_miou(RULES_ROOT / ring / "evaluation" / "performance.md"),
        }
        for cond, cfg_dir in CONFIG_DIRS.items():
            perf = ABLATION_ROOT / cfg_dir / f"{ring}_{llm_suffix}" / "evaluation" / "performance.md"
            row[cond] = parse_miou(perf)
        rows.append(row)
    return rows


def regular_msk_mean(llm_suffix: str) -> float:
    """Mean m+s+k mIoU over all 13 regular tunnels for one LLM (n=13)."""
    vals = [
        parse_miou(
            ABLATION_ROOT
            / "memory+state+knowledge"
            / f"{ring}_{llm_suffix}"
            / "evaluation"
            / "performance.md"
        )
        for ring in REGULAR_MSK_RINGS
    ]
    return float(np.mean(vals))


def main():
    all_data: dict[str, list[dict]] = {
        name: build_rows(suffix) for name, suffix in LLMS
    }
    ref_llm = LLMS[0][0]

    # Regular m+s+k reported over all 13 regular tunnels (n=13), not the
    # 3 representative regular rings used for every other bar.
    regular_msk_means = {name: regular_msk_mean(suffix) for name, suffix in LLMS}

    llm_ci, rules_ci = compute_orig_errbars()

    baselines = {}
    for fam_name, fam_pred in FAMILIES.items():
        ref_rows = [r for r in all_data[ref_llm] if fam_pred(r)]
        baselines[fam_name] = float(np.mean([r["sam4tun"] for r in ref_rows]))

    all_labels = ["sam4tun", "no-LLM", "m", "m+s", "m+s+k"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.0), sharey=False)
    llm_handles, llm_labels = [], []

    for ax_idx, (fam_name, fam_pred) in enumerate(FAMILIES.items()):
        ax = axes[ax_idx]
        bar_width = 0.22
        gap = 0.34
        grp_llm = 3 * bar_width
        grp_rules = bar_width

        p0 = 0.0
        p1 = p0 + grp_llm / 2 + gap + grp_rules / 2
        p2 = p1 + grp_rules / 2 + gap + grp_llm / 2
        p3 = p2 + grp_llm / 2 + gap + grp_llm / 2
        p4 = p3 + grp_llm / 2 + gap + grp_llm / 2
        x = np.array([p0, p1, p2, p3, p4])

        llm_x_centers = np.array([p0, p2, p3, p4])

        for llm_idx, (llm_name, _) in enumerate(LLMS):
            rows = all_data[llm_name]
            fam_rows = [r for r in rows if fam_pred(r)]

            means = []
            ci_lo = []
            ci_hi = []
            for cond in CONDITIONS:
                # bar height/label: new 5-ring mean; error bar: original 30-tunnel CI
                m = float(np.mean([r[cond] for r in fam_rows]))
                # Regular m+s+k: use the n=13 regular mean instead of the 3-ring mean.
                if fam_name == "Regular" and cond == "memory+state+knowledge":
                    m = regular_msk_means[llm_name]
                lo_off, hi_off = llm_ci[(llm_name, cond, fam_name)]
                means.append(m)
                ci_lo.append(lo_off)
                ci_hi.append(hi_off)

            offset = (llm_idx - 1) * bar_width
            x_positions = llm_x_centers + offset
            bars = ax.bar(
                x_positions,
                means,
                bar_width,
                yerr=[ci_lo, ci_hi],
                capsize=3,
                color=LLM_COLORS[llm_idx],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
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
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                    rotation=90,
                )

        fam_rows_ref = [r for r in all_data[ref_llm] if fam_pred(r)]
        rules_vals = np.array([r["rules"] for r in fam_rows_ref])
        rules_mean = np.mean(rules_vals)
        rules_lo_off, rules_hi_off = rules_ci[fam_name]
        rules_bar = ax.bar(
            p1,
            rules_mean,
            grp_rules,
            yerr=[[rules_lo_off], [rules_hi_off]],
            capsize=3,
            color=RULES_COLOR,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            error_kw={"linewidth": 0.8, "capthick": 0.8},
        )
        ax.text(
            p1,
            rules_mean + rules_hi_off + 0.006,
            f".{int(round(rules_mean * 1000)):03d}" if rules_mean < 1 else f"{rules_mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
            rotation=90,
        )
        if ax_idx == 0:
            llm_handles.append(rules_bar)
            llm_labels.append("no-LLM")

        baseline_val = baselines[fam_name]
        msk_vals = []
        for llm_name_inner, _ in LLMS:
            if fam_name == "Regular":
                msk_vals.append(regular_msk_means[llm_name_inner])
            else:
                fam_rows_inner = [r for r in all_data[llm_name_inner] if fam_pred(r)]
                msk_vals.append(np.mean([r["memory+state+knowledge"] for r in fam_rows_inner]))
        msk_mean = np.mean(msk_vals)

        ax.axhline(
            y=baseline_val,
            color="#888888",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"Baseline = {baseline_val:.3f}",
        )
        ax.axhline(
            y=msk_mean,
            color="#E8871E",
            linestyle="-.",
            linewidth=1.0,
            alpha=0.7,
            label=f"m+s+k mean = {msk_mean:.3f}",
        )
        ax.legend(fontsize=10, loc="upper left", framealpha=0.85, edgecolor="none")

        # Draft: keep the original 30-tunnel family sizes in the titles.
        n_fam = {"Regular": 13, "Complex": 17}[fam_name]
        ax.set_title(f"{fam_name} ($n={n_fam}$)", fontsize=13, fontweight="bold", pad=16)
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, fontsize=11)
        ax.set_ylabel("Mean mIoU" if ax_idx == 0 else "", fontsize=12)
        ax.set_xlabel("Ablation condition", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylim(0, 0.82)
    axes[1].set_ylim(0, 0.34)

    fig.legend(
        llm_handles,
        llm_labels,
        loc="upper center",
        ncol=4,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = REPO_ROOT / "methods" / "reviews" / "v7" / "figs" / "ablation_bar_revised.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")
    print(f"Baselines: Regular={baselines['Regular']:.3f}, Complex={baselines['Complex']:.3f}")
    plt.close(fig)


if __name__ == "__main__":
    main()
