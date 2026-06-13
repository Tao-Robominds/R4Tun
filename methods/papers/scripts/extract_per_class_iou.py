#!/usr/bin/env python3
"""
Extract per-class IoU from existing evaluation/performance.md files across
ablation snapshot trees (GPT, Anthropic, Gemini).

Outputs:
  - methods/papers/output/per_class_iou_long.csv
  - methods/papers/output/per_class_iou_summary.md

Gemini tree has no sam4tun/; baseline per-class IoU for Gemini comparisons
falls back to data/ablation_gpt/sam4tun/ (same fixed pipeline as journals).
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

LLM_ROOTS = {
    "gpt5.4": REPO_ROOT / "data" / "ablation_gpt",
    "opus4.6": REPO_ROOT / "data" / "ablation_anthropic",
    "gemini3flash": REPO_ROOT / "data" / "ablation_gemini",
}

BASELINE_FALLBACK = REPO_ROOT / "data" / "ablation_gpt" / "sam4tun"

CONDITION_DIRS = {
    "sam4tun": "sam4tun",
    "memory": "memory",
    "memory+state": "memory+state",
    "memory+state+knowledge": "memory+state+knowledge",
}

PER_CLASS_LINE = re.compile(r"^- (.+):\s*([0-9.]+)\s*$", re.MULTILINE)
SCHEMA_LINE = re.compile(r"\*\*(\d+)-class\*\*")


def tunnel_family(tunnel_id: str) -> str:
    if tunnel_id.startswith(("1-", "2-")):
        return "alternated"
    if tunnel_id.startswith("3-"):
        return "continuous"
    if tunnel_id.startswith(("4-", "5-")):
        return "complex"
    return "unknown"


def tunnel_family_regular_all(tunnel_id: str) -> str:
    if tunnel_family(tunnel_id) in ("alternated", "continuous"):
        return "regular_all"
    return "complex"


def parse_performance_md(text: str) -> tuple[dict[str, float], str | None]:
    """Return (class_name -> iou, schema '6'|'7'|None). Only lines under ## Per-class IoU."""
    m = SCHEMA_LINE.search(text)
    schema = m.group(1) if m else None
    idx = text.find("## Per-class IoU")
    if idx < 0:
        return {}, schema
    rest = text[idx:]
    end = rest.find("\n## ", 1)
    block = rest if end < 0 else rest[:end]
    per_class: dict[str, float] = {}
    for line_m in PER_CLASS_LINE.finditer(block):
        name = line_m.group(1).strip()
        per_class[name] = float(line_m.group(2))
    return per_class, schema


def performance_path(root: Path, condition_key: str, tunnel_id: str) -> Path | None:
    sub = CONDITION_DIRS[condition_key]
    p = root / sub / tunnel_id / "evaluation" / "performance.md"
    return p if p.is_file() else None


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for llm_tag, root in LLM_ROOTS.items():
        for cond_key in CONDITION_DIRS:
            for tunnel_dir in sorted((root / CONDITION_DIRS[cond_key]).glob("*")):
                if not tunnel_dir.is_dir():
                    continue
                tid = tunnel_dir.name
                perf = tunnel_dir / "evaluation" / "performance.md"
                if not perf.is_file():
                    continue
                text = perf.read_text(encoding="utf-8", errors="replace")
                per_class, schema = parse_performance_md(text)
                if not per_class:
                    continue
                fam = tunnel_family(tid)
                fam_r = tunnel_family_regular_all(tid)
                for cname, iou in per_class.items():
                    rows.append(
                        {
                            "llm": llm_tag,
                            "condition": cond_key,
                            "tunnel_id": tid,
                            "family_alternated_continuous_complex": fam,
                            "family_regular_all_vs_complex": fam_r,
                            "schema": schema or "",
                            "class_name": cname,
                            "iou": iou,
                        }
                    )

        # Gemini (and any tree missing sam4tun): inject baseline from GPT sam4tun
        if not (root / "sam4tun").is_dir():
            gpt_sam = LLM_ROOTS["gpt5.4"] / "sam4tun"
            for tunnel_dir in sorted(gpt_sam.glob("*")):
                if not tunnel_dir.is_dir():
                    continue
                tid = tunnel_dir.name
                perf = tunnel_dir / "evaluation" / "performance.md"
                if not perf.is_file():
                    continue
                text = perf.read_text(encoding="utf-8", errors="replace")
                per_class, schema = parse_performance_md(text)
                for cname, iou in per_class.items():
                    rows.append(
                        {
                            "llm": llm_tag,
                            "condition": "sam4tun",
                            "tunnel_id": tid,
                            "family_alternated_continuous_complex": tunnel_family(tid),
                            "family_regular_all_vs_complex": tunnel_family_regular_all(tid),
                            "schema": schema or "",
                            "class_name": cname,
                            "iou": iou,
                            "note": "baseline_from_ablation_gpt",
                        }
                    )
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_class_iou_long.csv"
    fieldnames = [
        "llm",
        "condition",
        "tunnel_id",
        "family_alternated_continuous_complex",
        "family_regular_all_vs_complex",
        "schema",
        "class_name",
        "iou",
    ]
    if any("note" in r for r in rows):
        fieldnames.append("note")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregate: mean IoU per (llm, condition, family_regular_all_vs_complex, class_name)
    agg: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key = (r["llm"], r["condition"], r["family_regular_all_vs_complex"], r["class_name"])
        agg[key].append(float(r["iou"]))

    md_path = out_dir / "per_class_iou_summary.md"
    lines = [
        "# Per-class IoU summary (from existing performance.md)",
        "",
        "Mean IoU per class, aggregated over tunnels in each family.",
        "",
        "- **regular_all**: alternated (1-*, 2-*) ∪ continuous (3-*), n=13",
        "- **complex**: 4-*, 5-*, n=17",
        "",
        "Note: `sam4tun` rows for `gemini3flash` use the shared GPT snapshot baseline (same as comparison journals).",
        "",
    ]

    for llm in ("gpt5.4", "opus4.6", "gemini3flash"):
        lines.append(f"## LLM: {llm}")
        lines.append("")
        for fam in ("regular_all", "complex"):
            lines.append(f"### Family: {fam}")
            lines.append("")
            # Collect all class names seen for this llm+family across conditions
            classes = sorted(
                {
                    k[3]
                    for k in agg
                    if k[0] == llm and k[2] == fam
                },
                key=lambda x: (0 if x == "Background" else 1, x),
            )
            conds = ["sam4tun", "memory", "memory+state", "memory+state+knowledge"]
            header = "| class | " + " | ".join(conds) + " |"
            sep = "|" + "|".join(["---"] * (len(conds) + 1)) + "|"
            lines.append(header)
            lines.append(sep)
            for cname in classes:
                cells = [cname]
                for cond in conds:
                    v = agg.get((llm, cond, fam, cname))
                    cells.append(f"{mean(v):.3f}" if v else "—")
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "methods" / "papers" / "output",
        help="Output directory for CSV and markdown",
    )
    args = parser.parse_args()
    rows = collect_rows()
    write_outputs(rows, args.out_dir)


if __name__ == "__main__":
    main()
