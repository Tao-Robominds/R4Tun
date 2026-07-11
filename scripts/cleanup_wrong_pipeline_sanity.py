#!/usr/bin/env python3
"""Remove wrong-pipeline LLM ablation artifacts for sanity tunnels.

Keeps GLM runs (*_glm) and does not touch data/static or data/rules.

Removes:
  - data/ablation/{cond}/{tid}_{opus4.6,gpt5.4,gemini3flash,...} (not *_glm)
  - data/ablation/{cond}/{tid} scratch dirs without model suffix (wrong-pipeline in-place runs)
  - agents/ablation/{cond}/parameters/{tid}  (legacy wrong param path)
  - sam4tun/agents/parameters/{cond}/{tid}/parameters_*_{model}.json for non-glm models
  - sam4tun/data/{tid} scratch + symlink
  - logs/ablation/{tid}_* for non-glm models

Usage:
  ./venv/bin/python scripts/cleanup_wrong_pipeline_sanity.py
  ./venv/bin/python scripts/cleanup_wrong_pipeline_sanity.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SANITY = ["1-1", "2-1", "3-1-1", "4-1", "5-1"]
CONDS = ["memory", "memory+state", "memory+state+knowledge"]
KEEP_MODEL_SUFFIX = "_glm"
WRONG_MODEL_TAGS = ("opus4.6", "gpt5.4", "gemini3flash", "opus46")
WRONG_EXTRA_SUFFIXES = ("_sam4tunpipe", "_validate", "_rerun", "_failed", "_pre_", "_v2", "_v3", "_v4", "_k_align")


def is_glm_name(name: str) -> bool:
    return name.endswith(KEEP_MODEL_SUFFIX) or KEEP_MODEL_SUFFIX in name


def is_wrong_output_name(name: str, tid: str) -> bool:
    if not (name == tid or name.startswith(tid + "_")):
        return False
    if is_glm_name(name):
        return False
    if name == tid:
        return True  # in-place wrong-pipeline scratch archive
    rest = name[len(tid) + 1 :] if name.startswith(tid + "_") else ""
    if rest in WRONG_MODEL_TAGS:
        return True
    if any(rest.endswith(s) or s in rest for s in WRONG_EXTRA_SUFFIXES):
        return True
    if re.search(r"_(opus4\.6|gpt5\.4|gemini3flash)", name):
        return True
    return False


def rm(path: Path, dry_run: bool) -> None:
    if not path.exists():
        return
    print(f"{'[dry-run] ' if dry_run else ''}remove {path}")
    if dry_run:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    for cond in CONDS:
        base = ROOT / "data/ablation" / cond
        if not base.is_dir():
            continue
        for entry in base.iterdir():
            if any(is_wrong_output_name(entry.name, tid) for tid in SANITY):
                rm(entry, args.dry_run)

    for tid in SANITY:
        rm(ROOT / "sam4tun/data" / tid, args.dry_run)
        rm(ROOT / "sam4tun/data" / f"{tid}.txt", args.dry_run)

    for cond in CONDS:
        for tid in SANITY:
            rm(ROOT / "agents/ablation" / cond / "parameters" / tid, args.dry_run)
            param_dir = ROOT / "sam4tun/agents/parameters" / cond / tid
            if param_dir.is_dir():
                for f in param_dir.glob("parameters_*.json"):
                    if "_glm" in f.name:
                        continue
                    if any(f"_ {tag}" in f.name or f.name.endswith(f"_{tag}.json") for tag in WRONG_MODEL_TAGS):
                        rm(f, args.dry_run)
                    elif not f.name.endswith("_glm.json"):
                        rm(f, args.dry_run)

    log_dir = ROOT / "logs/ablation"
    if log_dir.is_dir():
        for f in log_dir.iterdir():
            for tid in SANITY:
                if not f.name.startswith(f"{tid}_"):
                    continue
                if is_glm_name(f.name):
                    break
                if any(tag in f.name for tag in WRONG_MODEL_TAGS):
                    rm(f, args.dry_run)
                    break
                if "m_s" in f.name or "m_opus" in f.name or "gemini" in f.name or "gpt" in f.name:
                    rm(f, args.dry_run)
                    break


if __name__ == "__main__":
    main()
