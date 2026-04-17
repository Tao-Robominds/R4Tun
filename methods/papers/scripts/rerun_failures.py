#!/usr/bin/env python3
"""
Rerun detecting+SAM stages for tunnels where rules beat LLM conditions.

For each (tunnel, condition, model) combo:
  1. Back up existing parameter JSONs
  2. Stage upstream artifacts from vendor-specific data dir to standard path
  3. Run orchestrator with --stages detecting sam (LLM re-inference + pipeline)
  4. Snapshot new outputs to logs/
  5. Restore backups so data/ and agents/ remain unchanged

Usage:
    ./venv/bin/python methods/papers/scripts/rerun_failures.py
    ./venv/bin/python methods/papers/scripts/rerun_failures.py --dry-run
    ./venv/bin/python methods/papers/scripts/rerun_failures.py --combo 4-4 m_s_k opus4.6
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
os.chdir(REPO_ROOT)

TUNNELS = ["4-4", "5-3", "5-4"]
CONDITIONS = ["m_s", "m_s_k"]
MODELS = ["opus4.6", "gpt5.4", "gemini3flash"]

COND_TO_FOLDER = {"m_s": "memory+state", "m_s_k": "memory+state+knowledge"}
MODEL_TO_VENDOR = {
    "opus4.6": "anthropic",
    "gpt5.4": "gpt",
    "gemini3flash": "gemini",
}
ORCHESTRATORS = {
    ("m_s", "opus4.6"): ["run_memory_state.py"],
    ("m_s", "gpt5.4"): ["run_memory_state_gpt.py"],
    ("m_s", "gemini3flash"): ["run_memory_state_gemini.py"],
    ("m_s_k", "opus4.6"): ["run_memory_state_knowledge.py"],
    ("m_s_k", "gpt5.4"): ["run_memory_state_knowledge_gpt.py"],
    ("m_s_k", "gemini3flash"): ["run_memory_state_knowledge_gemini.py"],
}

PYTHON = str(REPO_ROOT / "venv" / "bin" / "python")
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

OLD_MIOU = {
    ("4-4", "m_s_k", "opus4.6"): 0.047,
    ("4-4", "m_s_k", "gpt5.4"): 0.133,
    ("4-4", "m_s_k", "gemini3flash"): 0.072,
    ("5-3", "m_s_k", "opus4.6"): 0.089,
    ("5-3", "m_s_k", "gpt5.4"): 0.116,
    ("5-3", "m_s_k", "gemini3flash"): 0.080,
    ("5-4", "m_s_k", "opus4.6"): 0.068,
    ("5-4", "m_s_k", "gpt5.4"): 0.098,
    ("5-4", "m_s_k", "gemini3flash"): 0.122,
}

RULES_MIOU = {"4-4": 0.268, "5-3": 0.231, "5-4": 0.142}


def extract_miou(perf_md: Path) -> float | None:
    if not perf_md.exists():
        return None
    text = perf_md.read_text()
    m = re.search(r"Mean IoU \(mIoU\):\s*([\d.]+)", text)
    return float(m.group(1)) if m else None


def run_combo(tid: str, cond: str, model: str, log_base: Path, dry_run: bool) -> float | None:
    folder = COND_TO_FOLDER[cond]
    vendor = MODEL_TO_VENDOR[model]

    combo_dir = log_base / cond / model
    combo_dir.mkdir(parents=True, exist_ok=True)

    std_data = REPO_ROOT / "data" / "ablation" / folder / tid
    vendor_data = REPO_ROOT / "data" / f"ablation_{vendor}" / folder / tid
    param_dir = REPO_ROOT / "agents" / "ablation" / folder / "parameters" / tid

    print(f"\n{'='*70}")
    print(f"  COMBO: tunnel={tid}  cond={cond}  model={model}")
    print(f"  vendor_data: {vendor_data}")
    print(f"  std_data:    {std_data}")
    print(f"  param_dir:   {param_dir}")
    print(f"{'='*70}")

    if not vendor_data.exists():
        print(f"  ERROR: vendor data dir missing: {vendor_data}")
        return None

    # --- 1. Backup parameters ---
    param_backup = combo_dir / "parameters_backup"
    if param_dir.exists():
        shutil.copytree(param_dir, param_backup, dirs_exist_ok=True)
        print(f"  Backed up params -> {param_backup}")
    else:
        print(f"  No param dir to back up")

    # --- 2. Stage upstream artifacts ---
    std_data_backup = None
    if std_data.exists():
        std_data_backup = combo_dir / "std_data_backup"
        shutil.copytree(std_data, std_data_backup, dirs_exist_ok=True)
        shutil.rmtree(std_data)
        print(f"  Backed up existing std_data -> {std_data_backup}")

    shutil.copytree(vendor_data, std_data)
    print(f"  Staged {vendor_data} -> {std_data}")

    # --- 2a2. Stage memory condition raw characteristics for m_s combos ---
    # The m_s prompt builder reads raw_characteristics.json from the default
    # "memory" subroot (data/ablation/memory/{tid}/characteristics/).
    memory_char_dir = REPO_ROOT / "data" / "ablation" / "memory" / tid / "characteristics"
    memory_char_backup = None
    if not memory_char_dir.exists():
        # Find raw_characteristics.json from any vendor's memory dir
        for v in ("anthropic", "gpt", "gemini"):
            src = REPO_ROOT / "data" / f"ablation_{v}" / "memory" / tid / "characteristics" / "raw_characteristics.json"
            if src.exists():
                memory_char_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, memory_char_dir / "raw_characteristics.json")
                memory_char_backup = "created"
                print(f"  Staged memory raw_characteristics from {v}")
                break
        if not memory_char_backup:
            print(f"  WARNING: no memory raw_characteristics found for {tid}")

    # --- 2b. Delete all stale param files to force full pipeline re-run ---
    tag = "_m_s" if cond == "m_s" else "_m_s_k"
    for stage in ("unfolding", "denoising", "enhancing", "detecting", "sam"):
        stale = param_dir / f"parameters_{stage}{tag}_{model}.json"
        if stale.exists():
            stale.unlink()
            print(f"  Cleared stale: {stale.name}")

    new_miou = None
    try:
        if dry_run:
            print("  [dry-run] Skipping orchestrator call")
        else:
            # --- 3. Run orchestrator (full pipeline, all 5 stages) ---
            script = ORCHESTRATORS[(cond, model)][0]
            cmd = [PYTHON, script, tid]
            print(f"  Running: {' '.join(cmd)}")
            t0 = time.time()
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=1800,
            )
            elapsed = time.time() - t0
            print(f"  Orchestrator finished in {elapsed:.0f}s (exit={result.returncode})")

            log_file = combo_dir / "orchestrator.log"
            log_file.write_text(
                f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}\n"
            )

            if result.returncode != 0:
                print(f"  ERROR: orchestrator failed. See {log_file}")
                print(f"  Last stderr: {result.stderr[-500:]}")
            else:
                perf_md = std_data / "evaluation" / "performance.md"
                new_miou = extract_miou(perf_md)
                print(f"  New mIoU: {new_miou}")

        # --- 4. Snapshot fresh outputs ---
        snapshot_data = combo_dir / "data"
        if std_data.exists():
            shutil.copytree(std_data, snapshot_data, dirs_exist_ok=True)
            print(f"  Snapshot data -> {snapshot_data}")

        snapshot_params = combo_dir / "parameters"
        if param_dir.exists():
            shutil.copytree(param_dir, snapshot_params, dirs_exist_ok=True)
            print(f"  Snapshot params -> {snapshot_params}")

    finally:
        # --- 5. Restore backups ---
        if std_data.exists():
            shutil.rmtree(std_data)
        if std_data_backup:
            shutil.copytree(std_data_backup, std_data)
            print(f"  Restored std_data from backup")
        else:
            print(f"  Removed staged std_data (no prior backup)")

        if memory_char_backup == "created" and memory_char_dir.exists():
            shutil.rmtree(memory_char_dir.parent)
            print(f"  Removed staged memory characteristics")

        if param_backup.exists():
            if param_dir.exists():
                shutil.rmtree(param_dir)
            shutil.copytree(param_backup, param_dir)
            print(f"  Restored params from backup")

    return new_miou


def main():
    parser = argparse.ArgumentParser(description="Rerun detecting+SAM for failure tunnels")
    parser.add_argument("--dry-run", action="store_true", help="Skip orchestrator calls")
    parser.add_argument(
        "--combo", nargs=3, metavar=("TID", "COND", "MODEL"),
        help="Run a single combo, e.g. --combo 4-4 m_s_k opus4.6",
    )
    args = parser.parse_args()

    log_base_root = REPO_ROOT / "logs"

    if args.combo:
        combos = [(args.combo[0], args.combo[1], args.combo[2])]
    else:
        combos = [
            (tid, cond, model)
            for tid in TUNNELS
            for cond in CONDITIONS
            for model in MODELS
        ]

    summary_csv = log_base_root / f"rerun_{TS}_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    results = []
    t_total = time.time()

    for tid, cond, model in combos:
        log_base = log_base_root / tid / f"rerun_{TS}"
        new_miou = run_combo(tid, cond, model, log_base, args.dry_run)
        old = OLD_MIOU.get((tid, cond, model), "n/a")
        rules = RULES_MIOU.get(tid, "n/a")
        results.append({
            "tunnel": tid, "condition": cond, "model": model,
            "old_miou": old, "new_miou": new_miou if new_miou is not None else "FAIL",
            "rules_miou": rules,
        })
        with open(summary_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[-1].keys()))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(results[-1])

    total_elapsed = time.time() - t_total

    # Write markdown summary
    summary_md = log_base_root / f"rerun_{TS}_summary.md"
    lines = [
        f"# Rerun Summary ({TS})",
        f"\nTotal time: {total_elapsed:.0f}s\n",
        "| Tunnel | Condition | Model | Old mIoU | New mIoU | Rules mIoU | Delta |",
        "|--------|-----------|-------|----------|----------|------------|-------|",
    ]
    for r in results:
        old_v = r["old_miou"]
        new_v = r["new_miou"]
        rules_v = r["rules_miou"]
        if isinstance(new_v, (int, float)) and isinstance(old_v, (int, float)):
            delta = f"{new_v - old_v:+.3f}"
        else:
            delta = "n/a"
        lines.append(
            f"| {r['tunnel']} | {r['condition']} | {r['model']} | "
            f"{old_v} | {new_v} | {rules_v} | {delta} |"
        )
    summary_md.write_text("\n".join(lines) + "\n")
    print(f"\n{'='*70}")
    print(f"Summary written to {summary_csv} and {summary_md}")
    print(f"Total elapsed: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
