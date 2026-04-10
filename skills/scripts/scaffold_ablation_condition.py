#!/usr/bin/env python3
"""
Scaffold a new ablation condition from an existing one.

Copies the agents/ tree and creates empty parameter directories for each
tunnel_id found in the source. Optionally copies parameter JSONs with
renamed suffixes.

Usage:
    python skills/scripts/scaffold_ablation_condition.py --from m --to m_s
    python skills/scripts/scaffold_ablation_condition.py --from m --to m_s --copy-params
    python skills/scripts/scaffold_ablation_condition.py --from m --to m_s --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agents"))

from pipeline_data import ABLATION_CONDITIONS, DEFAULT_MODEL, _build_suffix


STAGES = ["unfolding", "denoising", "enhancing", "detecting", "sam"]


def scaffold(
    from_code: str,
    to_code: str,
    model: str,
    copy_params: bool,
    dry_run: bool,
) -> None:
    src_cond = ABLATION_CONDITIONS[from_code]
    dst_cond = ABLATION_CONDITIONS[to_code]

    ablation_base = REPO_ROOT / "agents" / "ablation"
    src_dir = ablation_base / src_cond["folder"]
    dst_dir = ablation_base / dst_cond["folder"]

    if not src_dir.is_dir():
        print(f"Source condition directory not found: {src_dir}")
        sys.exit(1)

    print(f"Scaffolding: {from_code} ({src_cond['folder']}) -> {to_code} ({dst_cond['folder']})")
    print(f"  Source: {src_dir}")
    print(f"  Dest:   {dst_dir}")
    print(f"  Model:  {model}")
    print(f"  Copy params: {copy_params}")
    print(f"  Dry run: {dry_run}")
    print()

    # 1. Copy agents/ tree
    src_agents = src_dir / "agents"
    dst_agents = dst_dir / "agents"
    if src_agents.is_dir():
        if dst_agents.exists():
            print(f"  [skip] agents/ already exists at {dst_agents}")
        else:
            print(f"  [copy] agents/ -> {dst_agents}")
            if not dry_run:
                shutil.copytree(
                    src_agents, dst_agents,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )

    # 2. Discover tunnel_ids from source parameters/
    src_params = src_dir / "parameters"
    if not src_params.is_dir():
        print(f"  No parameters/ in source — nothing to scaffold.")
        return

    tunnel_ids = sorted(
        [d.name for d in src_params.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda t: tuple(int(x) for x in t.split("-")),
    )
    print(f"  Tunnels found: {len(tunnel_ids)}")

    # 3. Create parameter directories and optionally copy/rename files
    dst_params = dst_dir / "parameters"
    src_suffix = _build_suffix(from_code, model)
    dst_suffix = _build_suffix(to_code, model)

    created = 0
    copied = 0
    skipped = 0

    for tid in tunnel_ids:
        tid_dir = dst_params / tid
        if not tid_dir.exists():
            print(f"  [mkdir] {tid_dir.relative_to(REPO_ROOT)}")
            if not dry_run:
                tid_dir.mkdir(parents=True, exist_ok=True)
            created += 1

        if copy_params:
            for stage in STAGES:
                src_file = src_params / tid / f"parameters_{stage}{src_suffix}.json"
                dst_file = tid_dir / f"parameters_{stage}{dst_suffix}.json"

                if dst_file.exists():
                    skipped += 1
                    continue

                if src_file.is_file():
                    print(f"  [copy] {src_file.name} -> {dst_file.name}")
                    if not dry_run:
                        shutil.copy2(src_file, dst_file)
                    copied += 1
                else:
                    print(f"  [warn] source not found: {src_file.relative_to(REPO_ROOT)}")

    # 4. Copy any non-parameter top-level files (process.md, etc.)
    for item in src_dir.iterdir():
        if item.name in ("agents", "parameters", "__pycache__"):
            continue
        dst_item = dst_dir / item.name
        if dst_item.exists():
            continue
        if item.is_file():
            print(f"  [copy] {item.name}")
            if not dry_run:
                shutil.copy2(item, dst_item)

    print()
    print(f"Done. Dirs created: {created}, files copied: {copied}, skipped: {skipped}")
    if dry_run:
        print("(dry run — no files were actually written)")


def main():
    all_codes = list(ABLATION_CONDITIONS.keys())

    parser = argparse.ArgumentParser(description="Scaffold a new ablation condition")
    parser.add_argument(
        "--from", dest="from_code", required=True,
        choices=all_codes,
        help="Source ablation code to scaffold from",
    )
    parser.add_argument(
        "--to", dest="to_code", required=True,
        choices=all_codes,
        help="Target ablation code to scaffold",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model tag for parameter file suffixes (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--copy-params", action="store_true",
        help="Copy parameter JSONs from source (renamed with target suffix). "
             "Use as a starting point to be re-inferred by the LLM.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without making changes",
    )
    args = parser.parse_args()

    if args.from_code == args.to_code:
        print("Source and target are the same.")
        sys.exit(1)

    scaffold(
        from_code=args.from_code,
        to_code=args.to_code,
        model=args.model,
        copy_params=args.copy_params,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
