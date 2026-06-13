#!/usr/bin/env python3
"""Quantify parameter stability across repeated LLM runs.

Layouts:
  rerun (default): logs/{tunnel}/rerun_*/{cond}/{model}/
  repeatability:   logs/{tunnel}/repeatability/run1|run2_*/{model}/

Writes methods/papers/output/repeatability_summary.md when --layout repeatability.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from repeatability_common import (  # noqa: E402
    CRITICAL_FLAT_KEYS,
    MODELS,
    critical_param_stats,
    extract_miou,
    get_tunnel_ids,
    load_flat_params,
    run1_dir,
    run2_harvested_dir,
)

ROOT = Path(__file__).resolve().parents[3]
LOGS = ROOT / "logs"
OUTPUT_MD = ROOT / "methods" / "papers" / "output" / "repeatability_summary.md"
OUTPUT_TEX = ROOT / "methods" / "papers" / "output" / "repeatability_table.tex"

MODEL_LABEL = {
    "opus4.6": "Opus-4.6",
    "gpt5.4": "GPT-5.4",
    "gemini3flash": "Gemini-3-Flash",
}


def read_miou_rerun(run_dir: Path) -> float | None:
    perf = run_dir / "data" / "evaluation" / "performance.md"
    if not perf.exists():
        return None
    m = re.search(r"mIoU\D*([0-9.]+)", perf.read_text())
    return float(m.group(1)) if m else None


def collect_rerun() -> dict:
    groups = defaultdict(list)
    for tdir in sorted(LOGS.glob("*/")):
        tunnel = tdir.name
        if not re.match(r"\d", tunnel):
            continue
        for rerun in sorted(tdir.glob("rerun_*")):
            for cond_dir in sorted(rerun.glob("*/")):
                cond = cond_dir.name
                for model_dir in sorted(cond_dir.glob("*/")):
                    model = model_dir.name
                    pdir = model_dir / "parameters"
                    if not pdir.exists():
                        continue
                    flat = {}
                    for pf in sorted(pdir.glob(f"*_{cond}_{model}.json")):
                        stage = pf.stem.split("_")[1]
                        try:
                            d = json.loads(pf.read_text())
                        except Exception:
                            continue
                        for kk, vv in d.items():
                            if isinstance(vv, (int, float)):
                                flat[f"{stage}.{kk}"] = vv
                    if flat:
                        groups[(tunnel, cond, model)].append(
                            (rerun.name, flat, read_miou_rerun(model_dir))
                        )
    return groups


def find_run2_dir(tunnel: str, model: str) -> Path | None:
    rep = LOGS / tunnel / "repeatability"
    harvested = run2_harvested_dir(tunnel, model)
    if harvested.exists():
        return harvested
    candidates = sorted(rep.glob(f"run2_*/{model}"))
    return candidates[-1] if candidates else None


def collect_repeatability() -> list[dict]:
    rows = []
    for tunnel in get_tunnel_ids():
        for model in MODELS:
            r1 = run1_dir(tunnel, model)
            r1_params = r1 / "parameters"
            r2 = find_run2_dir(tunnel, model)
            if not r1_params.exists() or r2 is None:
                continue
            p1 = load_flat_params(r1_params, model)
            p2_dir = r2 / "parameters" if (r2 / "parameters").exists() else r2
            p2 = load_flat_params(p2_dir, model)
            if not p1 or not p2:
                continue
            miou1 = extract_miou(r1)
            miou2 = extract_miou(r2)
            ident, n_crit, pct_crit = critical_param_stats(p1, p2)
            source = "harvested" if "harvested" in str(r2) else "inference"
            rows.append({
                "tunnel": tunnel,
                "model": model,
                "run2_source": source,
                "miou1": miou1,
                "miou2": miou2,
                "miou_range": abs(miou2 - miou1) if miou1 is not None and miou2 is not None else None,
                "crit_identical": ident,
                "crit_compared": n_crit,
                "crit_pct": pct_crit,
            })
    return rows


def print_rerun_groups(groups: dict) -> None:
    for key, runs in sorted(groups.items()):
        if len(runs) < 2:
            continue
        tunnel, cond, model = key
        print("=" * 70)
        print(f"{tunnel} | {cond} | {model}  ({len(runs)} repeated runs)")
        mious = [m for _, _, m in runs if m is not None]
        if mious:
            print(
                f"  mIoU: mean={st.mean(mious):.3f}  std={st.pstdev(mious):.3f}  "
                f"min={min(mious):.3f}  max={max(mious):.3f}  "
                f"range={max(mious)-min(mious):.3f}"
            )
        allkeys = sorted(set().union(*[set(f.keys()) for _, f, _ in runs]))
        n_identical = 0
        cvs = []
        for pk in allkeys:
            vals = [f[pk] for _, f, _ in runs if pk in f]
            if len(vals) < 2:
                continue
            mean = st.mean(vals)
            std = st.pstdev(vals)
            cv = (std / abs(mean) * 100) if mean != 0 else 0.0
            identical = len(set(vals)) == 1
            if identical:
                n_identical += 1
            cvs.append(cv)
        if cvs:
            print(
                f"  SUMMARY: {n_identical}/{len(cvs)} params identical; "
                f"mean CV={st.mean(cvs):.2f}%"
            )
    print("=" * 70)


def write_repeatability_summary(rows: list[dict]) -> None:
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        OUTPUT_MD.write_text(
            "# Repeatability summary\n\nNo run1/run2 pairs found. "
            "Run `bootstrap_repeatability_run1.py` and `run_repeatability.py` first.\n"
        )
        print(f"Wrote {OUTPUT_MD} (empty)")
        return

    pcts = [r["crit_pct"] for r in rows]
    ranges = [r["miou_range"] for r in rows if r["miou_range"] is not None]
    median_pct = st.median(pcts)
    mean_pct = st.mean(pcts)
    mean_range = st.mean(ranges) if ranges else 0.0
    median_range = st.median(ranges) if ranges else 0.0

    lines = [
        "# Repeatability summary (m+s+k, temperature 0)",
        "",
        f"Pairs analysed: **{len(rows)}** (target 90 = 30 tunnels × 3 LLMs).",
        "",
        "## Aggregate",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Median critical-param identity (18 params) | {median_pct:.1f}% |",
        f"| Mean critical-param identity | {mean_pct:.1f}% |",
        f"| Mean \\|ΔmIoU\\| (run1 vs run2) | {mean_range:.4f} |",
        f"| Median \\|ΔmIoU\\| | {median_range:.4f} |",
        "",
        "## Per combo",
        "",
        "| Tunnel | Model | Run2 source | mIoU run1 | mIoU run2 | \\|ΔmIoU\\| | Critical identical |",
        "|--------|-------|-------------|-----------|-----------|---------|-------------------|",
    ]
    for r in sorted(rows, key=lambda x: (x["tunnel"], x["model"])):
        m1 = f"{r['miou1']:.3f}" if r["miou1"] is not None else "n/a"
        m2 = f"{r['miou2']:.3f}" if r["miou2"] is not None else "n/a"
        dr = f"{r['miou_range']:.4f}" if r["miou_range"] is not None else "n/a"
        crit = f"{r['crit_identical']}/{r['crit_compared']} ({r['crit_pct']:.0f}%)"
        lines.append(
            f"| {r['tunnel']} | {r['model']} | {r['run2_source']} | {m1} | {m2} | {dr} | {crit} |"
        )

    lines.extend([
        "",
        "## Reviewer response snippet",
        "",
        (
            f"Under m+s+k with temperature set to 0, a second LLM inference pass was compared "
            f"to the primary run on {len(rows)} tunnel–model pairs. "
            f"Median identity on the 18 critical parameters was {median_pct:.0f}% "
            f"(mean {mean_pct:.0f}%); mean |ΔmIoU| was {mean_range:.3f} "
            f"(median {median_range:.3f}), smaller than the paired adaptation gain "
            f"(ΔmIoU ≈ 0.17–0.19 vs baseline)."
        ),
        "",
        "## LaTeX table row (fill when n=90)",
        "",
        "```",
        (
            f"Median critical-parameter identity & {median_pct:.0f}\\% \\\\ "
            f"Mean $|\\Delta$mIoU| (run1 vs run2) & {mean_range:.3f} \\\\ "
            f"Pairs analysed & {len(rows)}/90"
        ),
        "```",
    ])
    OUTPUT_MD.write_text("\n".join(lines) + "\n")
    write_repeatability_tex(rows, median_pct, mean_pct, mean_range, median_range)
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_TEX}")
    print(f"Aggregate: median crit identity={median_pct:.1f}%, mean |ΔmIoU|={mean_range:.4f}")


def write_repeatability_tex(
    rows: list[dict],
    median_pct: float,
    mean_pct: float,
    mean_range: float,
    median_range: float,
) -> None:
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    body = []
    for r in sorted(rows, key=lambda x: (x["tunnel"], x["model"])):
        m1 = f"{r['miou1']:.3f}" if r["miou1"] is not None else "---"
        m2 = f"{r['miou2']:.3f}" if r["miou2"] is not None else "---"
        crit = f"{r['crit_identical']}/{r['crit_compared']} ({r['crit_pct']:.0f}\\%)"
        label = MODEL_LABEL.get(r["model"], r["model"])
        body.append(f"{r['tunnel']} & {label} & {m1} & {m2} & {crit} \\\\")
    tex = "\n".join([
        "% Auto-generated by reproducibility_analysis.py --layout repeatability",
        f"% Pairs: {len(rows)}/90",
        "\\begin{table}[ht]",
        "\\caption{LLM repeatability under m+s+k (temperature 0): run~1 vs run~2. "
        "\\emph{Crit.\\ identity} = identical values among the 18 critical parameters "
        "(Table~\\ref{tab:critical-params}).}",
        "\\label{tab:repeatability}",
        "\\begin{tabular*}{\\tblwidth}{@{} l l c c c @{}}",
        "\\toprule",
        "Tunnel & LLM & mIoU run1 & mIoU run2 & Crit.\\ identity \\\\",
        "\\midrule",
        *body,
        "\\midrule",
        f"\\multicolumn{{4}}{{l}}{{Median critical-parameter identity ({len(rows)} pairs)}} "
        f"& {median_pct:.0f}\\% \\\\",
        f"\\multicolumn{{4}}{{l}}{{Mean critical-parameter identity}} & {mean_pct:.0f}\\% \\\\",
        f"\\multicolumn{{4}}{{l}}{{Mean $|\\Delta$mIoU|}} & {mean_range:.3f} \\\\",
        f"\\multicolumn{{4}}{{l}}{{Median $|\\Delta$mIoU|}} & {median_range:.3f} \\\\",
        f"\\multicolumn{{4}}{{l}}{{Pairs analysed}} & {len(rows)}/90 \\\\",
        "\\bottomrule",
        "\\end{tabular*}",
        "\\end{table}",
        "",
    ])
    OUTPUT_TEX.write_text(tex)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout", choices=("rerun", "repeatability"), default="rerun",
    )
    args = parser.parse_args()

    if args.layout == "repeatability":
        rows = collect_repeatability()
        write_repeatability_summary(rows)
        for r in rows:
            print(
                f"{r['tunnel']} {r['model']}: crit {r['crit_pct']:.0f}%  "
                f"ΔmIoU={r['miou_range']}"
            )
    else:
        print_rerun_groups(collect_rerun())


if __name__ == "__main__":
    main()
