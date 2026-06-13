#!/usr/bin/env python3
"""
Run repeatability pass 2 for m+s+k: fresh LLM inference with run1 params seeded
so unchanged stages skip the GPU pipeline.

Usage:
    python3 methods/papers/scripts/run_repeatability.py --harvest-only
    python3 methods/papers/scripts/run_repeatability.py --dry-run
    python3 methods/papers/scripts/run_repeatability.py --tunnel 1-1 --model opus4.6
    python3 methods/papers/scripts/run_repeatability.py --skip-existing
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from repeatability_common import (  # noqa: E402
    ABLATION_FOLDER,
    HARVEST_TUNNELS,
    MODELS,
    ORCHESTRATORS,
    PARAM_BASE,
    copy_performance,
    extract_miou,
    get_tunnel_ids,
    has_run2,
    load_flat_params,
    param_json_name,
    params_identical,
    run1_dir,
    run2_harvested_dir,
    std_data_dir,
    vendor_data_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
TS = os.environ.get("REPEATABILITY_TS") or datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_venv_on_path() -> None:
    venv_site = (
        REPO_ROOT / "venv" / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    )
    if venv_site.is_dir():
        prefix = str(venv_site)
        cur = os.environ.get("PYTHONPATH", "")
        if prefix not in cur.split(os.pathsep):
            os.environ["PYTHONPATH"] = prefix + ((os.pathsep + cur) if cur else "")


def harvest_combo(tunnel: str, model: str) -> bool:
    """Copy latest logs/{tunnel}/rerun_*/m_s_k/{model}/ to run2_harvested."""
    tunnel_log = REPO_ROOT / "logs" / tunnel
    if not tunnel_log.exists():
        print(f"  harvest skip {tunnel} {model}: no logs")
        return False

    best = None
    for rerun in sorted(tunnel_log.glob("rerun_*")):
        combo = rerun / "m_s_k" / model
        params = combo / "parameters"
        if not params.exists():
            continue
        if not load_flat_params(params, model):
            continue
        best = combo

    if best is None:
        print(f"  harvest skip {tunnel} {model}: no valid rerun")
        return False

    dst = run2_harvested_dir(tunnel, model)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    src_params = best / "parameters"
    if src_params.exists():
        shutil.copytree(src_params, dst / "parameters")
    else:
        dst_params = dst / "parameters"
        dst_params.mkdir(parents=True)
        for stage in ("unfolding", "denoising", "enhancing", "detecting", "sam"):
            name = param_json_name(stage, model)
            if (best / name).exists():
                shutil.copy2(best / name, dst_params / name)
    copy_performance(best, dst)
    if extract_miou(dst) is None:
        copy_performance(best / "data", dst) if (best / "data").exists() else None
    print(f"  harvested {tunnel} {model} from {best.parent.parent.name}")
    return True


def seed_run1_params(tunnel: str, model: str, param_dir: Path) -> None:
    src = run1_dir(tunnel, model) / "parameters"
    param_dir.mkdir(parents=True, exist_ok=True)
    for stage in ("unfolding", "denoising", "enhancing", "detecting", "sam"):
        name = param_json_name(stage, model)
        if (src / name).exists():
            shutil.copy2(src / name, param_dir / name)


def run_combo(tunnel: str, model: str, log_dir: Path, dry_run: bool) -> dict:
    vendor = vendor_data_dir(tunnel, model)
    std_data = std_data_dir(tunnel)
    param_dir = PARAM_BASE / tunnel

    row = {
        "tunnel": tunnel,
        "model": model,
        "run2_source": "inference",
        "run1_miou": extract_miou(run1_dir(tunnel, model)),
        "run2_miou": None,
        "params_identical": None,
        "pipeline_ran": None,
        "status": "pending",
    }

    print(f"\n{'='*70}\n  REPEATABILITY run2: {tunnel} {model}\n{'='*70}")

    if not run1_dir(tunnel, model).exists():
        print("  ERROR: run1 snapshot missing; run bootstrap_repeatability_run1.py first")
        row["status"] = "no_run1"
        return row

    if not vendor.exists():
        print(f"  ERROR: vendor data missing: {vendor}")
        row["status"] = "no_vendor_data"
        return row

    param_backup = log_dir / "parameters_backup"
    std_backup = log_dir / "std_data_backup"
    memory_char_backup = None
    memory_char_dir = REPO_ROOT / "data" / "ablation" / "memory" / tunnel / "characteristics"

    if param_dir.exists():
        shutil.copytree(param_dir, param_backup, dirs_exist_ok=True)
    if std_data.exists():
        shutil.copytree(std_data, std_backup, dirs_exist_ok=True)
        shutil.rmtree(std_data)

    shutil.copytree(vendor, std_data)

    if not memory_char_dir.exists():
        for v in ("anthropic", "gpt", "gemini"):
            src = REPO_ROOT / "data" / f"ablation_{v}" / "memory" / tunnel / "characteristics" / "raw_characteristics.json"
            if src.exists():
                memory_char_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, memory_char_dir / "raw_characteristics.json")
                memory_char_backup = "created"
                break

    run1_flat = load_flat_params(run1_dir(tunnel, model) / "parameters", model)
    seed_run1_params(tunnel, model, param_dir)

    try:
        if dry_run:
            print("  [dry-run] would run orchestrator")
            row["status"] = "dry_run"
            return row

        script = ORCHESTRATORS[model]
        cmd = [PYTHON, str(REPO_ROOT / script), tunnel, "--model", model]
        print(f"  Running: {' '.join(cmd)}")
        t0 = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=1800,
        )
        elapsed = time.time() - t0
        (log_dir / "orchestrator.log").write_text(
            f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}\n"
        )
        print(f"  Orchestrator done in {elapsed:.0f}s (exit={result.returncode})")

        run2_flat = load_flat_params(param_dir, model)
        identical = params_identical(run1_flat, run2_flat)
        row["params_identical"] = identical

        if result.returncode != 0:
            row["status"] = "orchestrator_fail"
            return row

        if identical:
            row["pipeline_ran"] = False
            row["run2_miou"] = row["run1_miou"]
            row["status"] = "ok_identical"
            print("  All params identical to run1; reusing run1 mIoU")
        else:
            row["pipeline_ran"] = "upstream_pipeline_ran" in result.stdout
            row["run2_miou"] = extract_miou(std_data)
            row["status"] = "ok"
            print(f"  run2 mIoU: {row['run2_miou']}")

        snap_params = log_dir / "parameters"
        if param_dir.exists():
            shutil.copytree(param_dir, snap_params, dirs_exist_ok=True)
        if std_data.exists():
            shutil.copytree(std_data, log_dir / "data", dirs_exist_ok=True)
        elif not identical:
            copy_performance(run1_dir(tunnel, model), log_dir)

    finally:
        if std_data.exists():
            shutil.rmtree(std_data)
        if std_backup.exists():
            shutil.copytree(std_backup, std_data)
        if memory_char_backup == "created" and memory_char_dir.exists():
            shutil.rmtree(memory_char_dir.parent)
        if param_backup.exists():
            if param_dir.exists():
                shutil.rmtree(param_dir)
            shutil.copytree(param_backup, param_dir)

    return row


def append_summary(csv_path: Path, row: dict) -> None:
    fields = [
        "tunnel", "model", "run2_source", "run1_miou", "run2_miou",
        "params_identical", "pipeline_ran", "status",
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fields})


def main() -> None:
    os.chdir(REPO_ROOT)
    _ensure_venv_on_path()
    parser = argparse.ArgumentParser(description="Repeatability run 2 for m+s+k")
    parser.add_argument("--harvest-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--tunnel")
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()

    summary_csv = REPO_ROOT / "logs" / f"repeatability_{TS}_summary.csv"

    if args.harvest_only:
        n = 0
        for tunnel in HARVEST_TUNNELS:
            for model in MODELS:
                if harvest_combo(tunnel, model):
                    n += 1
                    append_summary(summary_csv, {
                        "tunnel": tunnel,
                        "model": model,
                        "run2_source": "harvested",
                        "run1_miou": extract_miou(run1_dir(tunnel, model)),
                        "run2_miou": extract_miou(run2_harvested_dir(tunnel, model)),
                        "params_identical": None,
                        "pipeline_ran": None,
                        "status": "harvested",
                    })
        print(f"\nHarvested {n} combos -> {summary_csv}")
        return

    tunnels = [args.tunnel] if args.tunnel else get_tunnel_ids()
    models = [args.model] if args.model else MODELS

    queue = []
    for tunnel in tunnels:
        for model in models:
            if tunnel in HARVEST_TUNNELS and run2_harvested_dir(tunnel, model).exists():
                continue
            if args.skip_existing and has_run2(tunnel, model):
                print(f"  skip existing: {tunnel} {model}")
                continue
            queue.append((tunnel, model))

    print(f"Queue: {len(queue)} combos (TS={TS})")
    if args.dry_run:
        for t, m in queue:
            print(f"  would run: {t} {m}")
        return

    for tunnel, model in queue:
        log_dir = REPO_ROOT / "logs" / tunnel / "repeatability" / f"run2_{TS}" / model
        log_dir.mkdir(parents=True, exist_ok=True)
        row = run_combo(tunnel, model, log_dir, dry_run=False)
        append_summary(summary_csv, row)

    print(f"\nSummary: {summary_csv}")


if __name__ == "__main__":
    main()
