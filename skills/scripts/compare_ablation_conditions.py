#!/usr/bin/env python3
"""
Compare mIoU across ablation conditions.

Reads performance.md from each condition's evaluation output, builds a
comparison table (rows=tunnels, columns=conditions), groups by tunnel
family, and computes paired deltas with statistics.

Usage:
    python skills/scripts/compare_ablation_conditions.py
    python skills/scripts/compare_ablation_conditions.py --conditions sam4tun m
    python skills/scripts/compare_ablation_conditions.py --baseline sam4tun --output results.md
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agents"))

from pipeline_data import ABLATION_CONDITIONS


FAMILY_MAP = {
    "1": "regularly_staggered",
    "2": "regularly_staggered",
    "3": "continuous",
    "4": "complex_staggered",
    "5": "complex_staggered",
}

MIOU_PATTERN = re.compile(r"Mean IoU \(mIoU\):\s*([\d.]+)")


def classify_family(tunnel_id: str) -> str:
    prefix = tunnel_id.split("-")[0]
    return FAMILY_MAP.get(prefix, "unknown")


def find_performance_md(condition_dir: Path, tunnel_id: str) -> Path | None:
    """Find performance.md in evaluation/ or evaluation_7/ subdirectories."""
    tunnel_dir = condition_dir / tunnel_id
    for eval_dir_name in ["evaluation", "evaluation_7", "evaluation_6"]:
        perf = tunnel_dir / eval_dir_name / "performance.md"
        if perf.is_file():
            return perf
    return None


def parse_miou(perf_path: Path) -> float | None:
    text = perf_path.read_text()
    m = MIOU_PATTERN.search(text)
    if m:
        return float(m.group(1))
    return None


def discover_tunnels(data_ablation: Path) -> set[str]:
    """Discover all tunnel IDs across all condition output dirs."""
    tunnels = set()
    for cond in ABLATION_CONDITIONS.values():
        cond_dir = data_ablation / cond["folder"].split("/")[-1]
        if not cond_dir.is_dir():
            # out_prefix might be nested; extract the last part
            prefix = cond["out_prefix"]
            cond_dir = REPO_ROOT / prefix
        if cond_dir.is_dir():
            for entry in cond_dir.iterdir():
                if entry.is_dir() and re.match(r"^\d+-", entry.name):
                    tunnels.add(entry.name)
    return tunnels


def collect_results(
    conditions: list[str],
) -> dict[str, dict[str, float | None]]:
    """
    Returns {tunnel_id: {condition_code: mIoU or None}}.
    """
    results: dict[str, dict[str, float | None]] = defaultdict(dict)

    for code in conditions:
        cond = ABLATION_CONDITIONS[code]
        cond_dir = REPO_ROOT / cond["out_prefix"]
        if not cond_dir.is_dir():
            continue
        for entry in sorted(cond_dir.iterdir()):
            if not entry.is_dir() or not re.match(r"^\d+-", entry.name):
                continue
            tid = entry.name
            perf = find_performance_md(cond_dir, tid)
            if perf:
                results[tid][code] = parse_miou(perf)
            else:
                results[tid][code] = None

    return dict(results)


def sort_tunnel_ids(tunnel_ids: list[str]) -> list[str]:
    def key(tid: str):
        parts = tid.split("-")
        return tuple(int(p) for p in parts)
    return sorted(tunnel_ids, key=key)


def format_table(
    results: dict[str, dict[str, float | None]],
    conditions: list[str],
    baseline: str | None,
) -> str:
    lines: list[str] = []
    tunnel_ids = sort_tunnel_ids(list(results.keys()))

    header = "| tunnel_id | family |"
    separator = "|-----------|--------|"
    for code in conditions:
        header += f" {code} |"
        separator += "------:|"
    if baseline and baseline in conditions:
        header += " delta |"
        separator += "------:|"
    lines.append(header)
    lines.append(separator)

    for tid in tunnel_ids:
        fam = classify_family(tid)[:3]
        row = f"| {tid} | {fam} |"
        for code in conditions:
            val = results[tid].get(code)
            row += f" {val:.3f} |" if val is not None else " — |"
        if baseline and baseline in conditions and len(conditions) >= 2:
            last_code = [c for c in conditions if c != baseline][-1]
            bval = results[tid].get(baseline)
            lval = results[tid].get(last_code)
            if bval is not None and lval is not None:
                delta = lval - bval
                sign = "+" if delta >= 0 else ""
                row += f" {sign}{delta:.3f} |"
            else:
                row += " — |"
        lines.append(row)

    return "\n".join(lines)


def compute_family_stats(
    results: dict[str, dict[str, float | None]],
    conditions: list[str],
    baseline: str,
) -> str:
    lines: list[str] = []

    non_baseline = [c for c in conditions if c != baseline]
    if not non_baseline:
        return "No non-baseline conditions to compare."

    families: dict[str, list[str]] = defaultdict(list)
    for tid in results:
        families[classify_family(tid)].append(tid)

    for fam_name in ["regularly_staggered", "continuous", "complex_staggered"]:
        tids = sort_tunnel_ids(families.get(fam_name, []))
        if not tids:
            continue

        lines.append(f"\n### {fam_name} (n={len(tids)})")
        lines.append("")

        header = "| condition |  mean_mIoU | mean_delta |    std |      p |"
        sep    = "|-----------|-----------|-----------|--------|--------|"
        lines.append(header)
        lines.append(sep)

        base_vals = [results[t].get(baseline) for t in tids]

        for code in non_baseline:
            code_vals = [results[t].get(code) for t in tids]
            pairs = [
                (b, c)
                for b, c in zip(base_vals, code_vals)
                if b is not None and c is not None
            ]
            if len(pairs) < 2:
                lines.append(f"| {code} | — | — | — | — |")
                continue

            import numpy as np

            b_arr = np.array([p[0] for p in pairs])
            c_arr = np.array([p[1] for p in pairs])
            deltas = c_arr - b_arr
            mean_miou = float(np.mean(c_arr))
            mean_d = float(np.mean(deltas))
            std_d = float(np.std(deltas, ddof=1))

            p_val = float("nan")
            test_name = "—"
            if len(pairs) >= 5:
                try:
                    from scipy.stats import wilcoxon
                    _, p_val = wilcoxon(deltas)
                    test_name = "wilcoxon"
                except ImportError:
                    try:
                        from scipy.stats import ttest_rel
                        _, p_val = ttest_rel(c_arr, b_arr)
                        test_name = "paired-t"
                    except ImportError:
                        pass
            elif len(pairs) >= 2:
                try:
                    from scipy.stats import ttest_rel
                    _, p_val = ttest_rel(c_arr, b_arr)
                    test_name = "paired-t"
                except ImportError:
                    pass

            sign = "+" if mean_d >= 0 else ""
            p_str = f"{p_val:.4f}" if not (p_val != p_val) else "—"
            lines.append(
                f"| {code} | {mean_miou:.3f} | {sign}{mean_d:.3f} | {std_d:.3f} | {p_str} ({test_name}) |"
            )

    return "\n".join(lines)


def main():
    all_codes = list(ABLATION_CONDITIONS.keys())

    parser = argparse.ArgumentParser(description="Compare mIoU across ablation conditions")
    parser.add_argument(
        "--conditions", "-c",
        nargs="+",
        default=all_codes,
        choices=all_codes,
        help=f"Condition codes to compare (default: all). Choices: {all_codes}",
    )
    parser.add_argument(
        "--baseline", "-b",
        default="sam4tun",
        choices=all_codes,
        help="Baseline condition for delta computation (default: sam4tun)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: stdout + data/ablation/comparison.md)",
    )
    args = parser.parse_args()

    conditions = args.conditions
    results = collect_results(conditions)

    if not results:
        print("No evaluation results found.")
        sys.exit(1)

    active = [c for c in conditions if any(results[t].get(c) is not None for t in results)]
    if not active:
        print("No conditions have evaluation results.")
        sys.exit(1)

    report_lines = [
        "# Ablation Comparison Report",
        "",
        f"Conditions: {', '.join(active)}",
        f"Baseline: {args.baseline}",
        f"Tunnels with data: {len(results)}",
        "",
        "## Per-tunnel mIoU",
        "",
        format_table(results, active, args.baseline),
        "",
        "## Family-level statistics",
        "",
        compute_family_stats(results, active, args.baseline),
        "",
    ]
    report = "\n".join(report_lines)

    print(report)

    out_path = args.output or str(REPO_ROOT / "data" / "ablation" / "comparison.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
