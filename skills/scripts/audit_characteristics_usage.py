#!/usr/bin/env python
"""
Text-mine LLM reasoning traces to find which characteristic fields are referenced.

Reads reasoning .md files from a tunnel's analysis/ dir and checks every leaf
key (and its numeric value) from the corresponding characteristics JSONs.

Usage:
    ./venv/bin/python skills/scripts/audit_characteristics_usage.py 1-1
    ./venv/bin/python skills/scripts/audit_characteristics_usage.py 1-1 --chars-root data/ablation_anthropic/memory+state+knowledge
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STAGE_ORDER = ["unfolding", "denoising", "enhancing", "detecting", "sam"]

def reasoning_file(stage: str, model: str | None) -> str:
    if model:
        return f"{stage}_reasoning_{model}.md"
    return f"{stage}_reasoning.md"

CHARS_FILES_VISIBLE = {
    "unfolding": ["raw_characteristics.json"],
    "denoising": ["raw_characteristics.json", "unfolded_characteristics.json"],
    "enhancing": ["raw_characteristics.json", "unfolded_characteristics.json",
                  "denoised_characteristics.json"],
    "detecting": ["raw_characteristics.json", "unfolded_characteristics.json",
                  "denoised_characteristics.json", "enhanced_characteristics.json"],
    "sam":       ["raw_characteristics.json", "unfolded_characteristics.json",
                  "denoised_characteristics.json", "enhanced_characteristics.json",
                  "detected_characteristics.json"],
}


def flatten_json(obj, prefix=""):
    """Flatten nested JSON to {dotted_path: value} for leaf scalars."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else k
            out.update(flatten_json(v, new_key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten_json(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def check_field_referenced(key: str, value, text: str) -> bool:
    """Check if a characteristic field appears to be referenced in reasoning."""
    leaf = key.rsplit(".", 1)[-1] if "." in key else key
    leaf_clean = leaf.strip("[]0123456789")
    if not leaf_clean or len(leaf_clean) < 3:
        return False

    # Skip metadata/boilerplate keys
    skip_keys = {"tunnel_id", "input_file", "filtered_note", "source_file",
                 "analysis_timestamp", "description", "note", "method",
                 "units", "columns", "timestamp", "source_algorithm",
                 "output_directory", "analysis_type", "prompt_source",
                 "target_data", "workflow", "processing_metadata",
                 "diameter_discrepancy_note", "ring_thickness_note",
                 "coordinate_systems", "available_attributes"}
    if leaf_clean in skip_keys:
        return False

    text_lower = text.lower()

    # Check key name variants
    variants = [
        leaf_clean,
        leaf_clean.replace("_", " "),
        leaf_clean.replace("_", "-"),
    ]
    for v in variants:
        if v.lower() in text_lower:
            return True

    # Check if the numeric value appears
    if isinstance(value, (int, float)) and value != 0:
        val_str = str(value)
        if len(val_str) >= 3 and val_str in text:
            return True
        if isinstance(value, float):
            short = f"{value:.4f}"
            if short in text:
                return True
            short3 = f"{value:.3f}"
            if short3 in text:
                return True
            short2 = f"{value:.2f}"
            if len(short2) >= 4 and short2 in text:
                return True

    return False


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tunnel_ids", nargs="+")
    p.add_argument("--models", default="opus4.6,gpt5.4,gemini3flash",
                   help="Comma-separated model tags")
    p.add_argument("--analysis-root", default="data/ablation/memory+state+knowledge",
                   help="Root for {tid}/analysis/")
    p.add_argument("--chars-root", default="data/ablation/memory+state+knowledge",
                   help="Root for {tid}/characteristics/")
    args = p.parse_args()

    models = args.models.split(",")
    tunnel_ids = args.tunnel_ids

    print(f"Tunnels: {tunnel_ids}")
    print(f"Models : {models}")
    print(f"Chars  : {args.chars_root}")
    print()

    # Aggregate results: field -> stage -> hit_count / total_count
    agg: dict[str, dict[str, list[int]]] = {}  # field -> stage -> [hits, total]

    for tid in tunnel_ids:
        chars_dir = Path(args.chars_root) / tid / "characteristics"
        all_chars = {}
        for fname in ["raw_characteristics.json", "unfolded_characteristics.json",
                       "denoised_characteristics.json", "enhanced_characteristics.json",
                       "detected_characteristics.json"]:
            p_file = chars_dir / fname
            if p_file.exists():
                with open(p_file) as f:
                    all_chars[fname] = flatten_json(json.load(f))

        for model in models:
            analysis_dir = Path(args.analysis_root) / tid / "analysis"

            for stage in STAGE_ORDER:
                rpath = analysis_dir / reasoning_file(stage, model)
                if not rpath.exists():
                    continue
                text = rpath.read_text()
                visible = CHARS_FILES_VISIBLE[stage]

                for cf in visible:
                    if cf not in all_chars:
                        continue
                    for key, val in all_chars[cf].items():
                        full_key = f"{cf}::{key}"
                        agg.setdefault(full_key, {}).setdefault(stage, [0, 0])
                        agg[full_key][stage][1] += 1
                        if check_field_referenced(key, val, text):
                            agg[full_key][stage][0] += 1

    # Summary
    stages_present = STAGE_ORDER
    hdr = f"{'Field':<70} | " + " | ".join(f"{s:^11}" for s in stages_present) + " | total"
    print(hdr)
    print("-" * len(hdr))

    always, sometimes, never = [], [], []
    field_total_hits = {}
    for field in sorted(agg.keys()):
        row = agg[field]
        total_hits = sum(v[0] for v in row.values())
        total_possible = sum(v[1] for v in row.values())
        field_total_hits[field] = (total_hits, total_possible)

        cells = []
        for s in stages_present:
            if s not in row:
                cells.append(f"{'---':^11}")
            else:
                h, t = row[s]
                cells.append(f"{h}/{t}".center(11))
        print(f"{field:<70} | " + " | ".join(cells) + f" | {total_hits}/{total_possible}")

        if total_possible > 0 and total_hits == total_possible:
            always.append(field)
        elif total_hits > 0:
            sometimes.append(field)
        else:
            never.append(field)

    print(f"\n{'='*80}")
    print(f"ALWAYS referenced across all models/tunnels ({len(always)} fields):")
    for f in sorted(always):
        print(f"  {f}")
    print(f"\nSOMETIMES referenced ({len(sometimes)} fields):")
    for f in sorted(sometimes, key=lambda x: -field_total_hits[x][0]):
        h, t = field_total_hits[f]
        print(f"  {f}  ({h}/{t})")
    print(f"\nNEVER referenced ({len(never)} fields):")
    for f in sorted(never):
        print(f"  {f}")


if __name__ == "__main__":
    main()
