#!/usr/bin/env python
"""
Replay saved LLM-generated ablation parameters on the sam4tun/agents pipeline.

No LLM calls — only runs the pipeline with pre-existing parameter JSONs staged under
sam4tun/agents/parameters/{condition}/{tunnel_id}/.

Usage:
    ./venv/bin/python run_replay_params.py --ablation m_s_k --model opus4.6 1-1
    ./venv/bin/python run_replay_params.py --ablation m_s_k --model opus4.6 --sanity
    ./venv/bin/python run_replay_params.py --all-conditions --all-models --sanity
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from agents.ablation import sam4tun_pipeline_runtime as spt

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

ABLATION_CONFIGS = {
    "m": {"folder": "memory", "suffix": "_m_"},
    "m_s": {"folder": "memory+state", "suffix": "_m_s_"},
    "m_s_k": {"folder": "memory+state+knowledge", "suffix": "_m_s_k_"},
}

MODELS = ("opus4.6", "gpt5.4", "gemini3flash")
T4T5_TUNNELS = ("4-1", "5-1")
SANITY_TUNNELS = ("1-1", "2-1", "3-1-1", "4-1", "5-1")
STAGES = [
    ("unfolding", "unfolding.py", "1-unfolded_characteriser.py"),
    ("denoising", "denoising.py", "2-denoised_characteriser.py"),
    ("enhancing", "enhancing.py", "3-enhanced_characteriser.py"),
    ("detecting", "detecting.py", "4-detected_characteriser.py"),
    ("sam", "sam.py", None),
]
STAGE_NAMES = [s[0] for s in STAGES]

SEED_ARTIFACTS = (
    "state.pkl",
    "depth_map.npy",
    "depth_map.png",
    "depth_map_outlier.npy",
    "pixel_to_point.pkl",
    "enhanced.csv",
    "denoised.csv",
    "initial_points.csv",
    "unwrapped.csv",
    "results.pkl",
)

DEFAULT_SEED_MAP: dict[str, dict[str, str]] = {
    "memory": {"4-1": "4-1_gpt5.4", "5-1": "5-1_gpt5.4"},
    "memory+state": {"4-1": "4-1_gpt5.4", "5-1": "5-1_opus4.6"},
    "memory+state+knowledge": {"4-1": "4-1_gemini3flash", "5-1": "5-1_opus4.6"},
}

SAMPLE_REF = {"OA": 0.9417, "F1": 0.9427, "mIoU": 0.8920}
METRIC_TOL = 0.005


def cfg(ablation: str) -> dict:
    if ablation not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation {ablation!r}")
    return ABLATION_CONFIGS[ablation]


def archive_key(tunnel_id: str, model: str) -> str:
    return f"{tunnel_id}_{model}"


def param_paths(tunnel_id: str, ablation: str, model: str) -> dict[str, Path]:
    folder = cfg(ablation)["folder"]
    suffix = cfg(ablation)["suffix"]
    base = REPO_ROOT / "sam4tun" / "agents" / "parameters" / folder / tunnel_id
    paths = {}
    for stage in STAGE_NAMES:
        paths[stage] = base / f"parameters_{stage}{suffix}{model}.json"
    return paths


def preflight_params(
    tunnel_id: str, ablation: str, model: str, stage_filter: list[str] | None = None
) -> list[Path]:
    """Return param paths; raise if any missing (would trigger sample fallback)."""
    stages = stage_filter if stage_filter else STAGE_NAMES
    paths = param_paths(tunnel_id, ablation, model)
    missing = [paths[s] for s in stages if s in paths and not paths[s].is_file()]
    if missing:
        lines = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            f"Missing {len(missing)} parameter file(s) for {tunnel_id} "
            f"ablation={ablation} model={model}:\n{lines}"
        )
    return [paths[s] for s in stages if s in paths]


def run_pipeline_stage_checked(
    tunnel_id: str,
    stage_script: str,
    stage_name: str,
    ablation: str,
    model: str,
    env: dict,
) -> None:
    cmd = [
        PYTHON,
        str(REPO_ROOT / "sam4tun" / "agents" / stage_script),
        tunnel_id,
        "--ablation",
        ablation,
        "--model",
        model,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess_run(cmd, env)
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.returncode != 0:
        print(stderr[-4000:] if stderr else "")
        raise RuntimeError(f"Stage {stage_script} failed (exit {result.returncode})")
    for line in (stdout + stderr).strip().split("\n")[-30:]:
        if line.strip():
            print(f"    {line}")
    loaded = [ln for ln in stdout.splitlines() if "Loaded" in ln and "parameters" in ln]
    if loaded:
        last_loaded = loaded[-1]
        if "ablation fallback" in last_loaded.lower() or "fallback for tunnel" in last_loaded.lower():
            raise RuntimeError(
                f"Stage {stage_name} used sample fallback instead of ablation params: {last_loaded}"
            )
        expected = f"(ablation {ablation} {model})"
        if expected not in last_loaded:
            raise RuntimeError(
                f"Stage {stage_name} did not load ablation-specific params.\n"
                f"  Expected marker: {expected}\n"
                f"  Got: {last_loaded}"
            )


def subprocess_run(cmd: list[str], env: dict):
    import subprocess

    return subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True)


def archive_replay_output(tunnel_id: str, ablation: str, model: str) -> Path:
    """Move work dir to data/ablation/{folder}/{tunnel_id}_{model}/."""
    folder = cfg(ablation)["folder"]
    src = spt.work_dir(tunnel_id)
    dest_name = archive_key(tunnel_id, model)
    dest = spt.out_root(folder) / dest_name
    if not src.is_dir():
        raise FileNotFoundError(f"Missing pipeline output: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    chars_src = spt.out_root(folder) / tunnel_id / "characteristics"
    analysis_backup = dest.parent / f".{dest_name}_analysis_tmp"
    if dest.is_dir() and (dest / "analysis").is_dir():
        if analysis_backup.exists():
            shutil.rmtree(analysis_backup)
        shutil.copytree(dest / "analysis", analysis_backup)

    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(src), str(dest))

    link = REPO_ROOT / "sam4tun" / "data" / f"{tunnel_id}.txt"
    if link.is_symlink():
        link.unlink()

    if chars_src.is_dir():
        shutil.copytree(chars_src, dest / "characteristics", dirs_exist_ok=True)
    if analysis_backup.is_dir():
        shutil.copytree(analysis_backup, dest / "analysis", dirs_exist_ok=True)
        shutil.rmtree(analysis_backup)

    print(f"  Archived -> {dest}")
    return dest


def extract_metrics(perf_path: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {"OA": None, "F1": None, "mIoU": None}
    if not perf_path.is_file():
        return out
    for line in perf_path.read_text().splitlines():
        for key, label in [
            ("OA", "Overall Accuracy (OA):"),
            ("F1", "F1 Score:"),
            ("mIoU", "Mean IoU (mIoU):"),
        ]:
            if label in line:
                try:
                    out[key] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
    return out


def run_evaluation(tunnel_id: str, ablation: str, model: str, env: dict) -> dict[str, float | None]:
    folder = cfg(ablation)["folder"]
    root = spt.out_root(folder)
    key = archive_key(tunnel_id, model)
    only_label = root / key / "only_label.csv"
    if not only_label.is_file():
        raise FileNotFoundError(f"Missing {only_label} for evaluation")

    cmd = [
        PYTHON,
        str(REPO_ROOT / "sam4tun" / "agents" / "evaluate_static.py"),
        key,
        "--data-root",
        str(root),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess_run(cmd, env)
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    if result.returncode != 0:
        print(result.stderr[-2000:] if result.stderr else "")
        raise RuntimeError("evaluation failed")
    return extract_metrics(root / key / "evaluation" / "performance.md")


def append_run_summary(
    tunnel_id: str,
    ablation: str,
    model: str,
    metrics: dict[str, float | None],
) -> None:
    folder = cfg(ablation)["folder"]
    summary_path = spt.out_root(folder) / "run_summary.csv"
    row = {
        "tunnel_id": tunnel_id,
        "archive_key": archive_key(tunnel_id, model),
        "ablation": ablation,
        "model": model,
        "mIoU": metrics.get("mIoU"),
        "OA": metrics.get("OA"),
        "F1": metrics.get("F1"),
    }
    fieldnames = list(row.keys())
    write_header = not summary_path.is_file()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  Appended to {summary_path}")


def resolve_seed_dir(
    tunnel_id: str, ablation: str, seed_from: str | None
) -> Path:
    folder = cfg(ablation)["folder"]
    if seed_from:
        p = Path(seed_from)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.is_dir():
            raise FileNotFoundError(f"Seed archive not found: {p}")
        return p
    default_key = DEFAULT_SEED_MAP.get(folder, {}).get(tunnel_id)
    if not default_key:
        raise ValueError(
            f"No default seed for {tunnel_id} under {folder}; pass --seed-from"
        )
    p = spt.out_root(folder) / default_key
    if not p.is_dir():
        raise FileNotFoundError(f"Default seed archive missing: {p}")
    return p


def seed_work_dir(tunnel_id: str, seed_dir: Path) -> None:
    wd = spt.work_dir(tunnel_id)
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    copied = []
    for name in SEED_ARTIFACTS:
        src = seed_dir / name
        if src.is_file():
            shutil.copy2(src, wd / name)
            copied.append(name)
    if "state.pkl" not in copied:
        raise FileNotFoundError(f"Seed {seed_dir} missing state.pkl (got {copied})")
    print(f"  Seeded work dir from {seed_dir} ({len(copied)} files)")


def replay_tunnel(
    tunnel_id: str,
    ablation: str,
    model: str,
    env: dict,
    skip_if_exists: bool = False,
    stage_filter: list[str] | None = None,
    seed_from: str | None = None,
    force: bool = False,
) -> dict[str, float | None]:
    print(f"\n{'='*60}")
    print(f"REPLAY {tunnel_id}  ablation={ablation}  model={model}")
    print(f"{'='*60}")

    folder = cfg(ablation)["folder"]
    key = archive_key(tunnel_id, model)
    perf_path = spt.out_root(folder) / key / "evaluation" / "performance.md"
    if skip_if_exists and not force and perf_path.is_file():
        metrics = extract_metrics(perf_path)
        print(f"  SKIP: existing {perf_path} (mIoU={metrics.get('mIoU')})")
        return metrics

    paths = preflight_params(tunnel_id, ablation, model, stage_filter)
    print(f"  Pre-flight OK: {len(paths)} parameter file(s) found")

    spt.out_root(folder).mkdir(parents=True, exist_ok=True)
    (spt.out_root(folder) / tunnel_id).mkdir(parents=True, exist_ok=True)

    sam_only = stage_filter == ["sam"]
    if sam_only:
        seed_dir = resolve_seed_dir(tunnel_id, ablation, seed_from)
        spt.symlink_input(tunnel_id)
        seed_work_dir(tunnel_id, seed_dir)
    else:
        spt.ensure_tunnel_characteristics(tunnel_id, folder)
        spt.symlink_input(tunnel_id)
        spt.prepare_work_dir(tunnel_id, stage_filter)
        if not stage_filter or "unfolding" in stage_filter:
            print("\n--- Raw characteristics ---")
            spt.run_raw_characteriser(tunnel_id, env)

    stages_to_run = [
        (n, script, char) for n, script, char in STAGES
        if not stage_filter or n in stage_filter
    ]
    for stage_name, stage_script, char_script in stages_to_run:
        print(f"\n--- Stage: {stage_name} ---")
        run_pipeline_stage_checked(
            tunnel_id, stage_script, stage_name, ablation, model, env
        )
        if char_script and not sam_only:
            spt.run_characteriser(tunnel_id, char_script, env)

    print("\n--- Archive + evaluation ---")
    archive_replay_output(tunnel_id, ablation, model)
    metrics = run_evaluation(tunnel_id, ablation, model, env)
    append_run_summary(tunnel_id, ablation, model, metrics)
    print(
        f"  Metrics: OA={metrics.get('OA')} F1={metrics.get('F1')} mIoU={metrics.get('mIoU')}"
    )
    return metrics


def write_gate_report(
    tunnel_id: str,
    ablation: str,
    model: str,
    metrics: dict[str, float | None],
    param_files: list[Path],
    archive_dest: Path,
    passed: bool,
) -> None:
    gate_path = REPO_ROOT / "data" / "ablation" / "replay_gate.md"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Replay single-instance gate",
        "",
        f"- **Case:** {tunnel_id}",
        f"- **Ablation:** {ablation}",
        f"- **Model:** {model}",
        f"- **Command:** `./venv/bin/python run_replay_params.py --ablation {ablation} --model {model} {tunnel_id}`",
        f"- **Finished:** {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"- **Status:** {'PASS' if passed else 'FAIL'}",
        "",
        "## Parameter files used",
        "",
    ]
    for p in param_files:
        lines.append(f"- `{p.relative_to(REPO_ROOT)}`")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- OA: {metrics.get('OA')}",
            f"- F1: {metrics.get('F1')}",
            f"- mIoU: {metrics.get('mIoU')}",
            "",
            "## Output",
            "",
            f"- `{archive_dest.relative_to(REPO_ROOT)}`",
            f"- `{archive_dest.relative_to(REPO_ROOT)}/evaluation/performance.md`",
            "",
            "## Sample validation (default params)",
            "",
            f"- Reference: OA={SAMPLE_REF['OA']}, F1={SAMPLE_REF['F1']}, mIoU={SAMPLE_REF['mIoU']}",
            f"- Rerun: see `sam4tun/data/sample/evaluation/performance.md`",
            f"- Tolerance: ±{METRIC_TOL}",
            "",
        ]
    )
    gate_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote gate report -> {gate_path}")


def write_consolidated_summary() -> None:
    summary_path = REPO_ROOT / "data" / "ablation" / "replay_summary.md"
    rows: list[dict] = []
    for ablation, conf in ABLATION_CONFIGS.items():
        root = REPO_ROOT / "data" / "ablation" / conf["folder"]
        if not root.is_dir():
            continue
        for perf in sorted(root.glob("*/evaluation/performance.md")):
            archive_key = perf.parent.parent.name
            if "_" not in archive_key:
                continue
            tunnel_id, _, model = archive_key.partition("_")
            if model not in MODELS:
                continue
            metrics = extract_metrics(perf)
            if metrics.get("mIoU") is None:
                continue
            rows.append({
                "ablation": ablation,
                "model": model,
                "tunnel_id": tunnel_id,
                "mIoU": metrics["mIoU"],
                "OA": metrics["OA"],
                "F1": metrics["F1"],
                "archive_key": archive_key,
            })

    lines = [
        "# Ablation parameter replay summary",
        "",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        "",
        "| ablation | model | tunnel | mIoU | OA | F1 | archive_key |",
        "|----------|-------|--------|------|----|----|-------------|",
    ]
    for row in sorted(rows, key=lambda r: (r.get("ablation", ""), r.get("model", ""), r.get("tunnel_id", ""))):
        lines.append(
            f"| {row.get('ablation', '')} | {row.get('model', '')} | {row.get('tunnel_id', '')} | "
            f"{row.get('mIoU', 'n/a')} | {row.get('OA', 'n/a')} | {row.get('F1', 'n/a')} | "
            f"{row.get('archive_key', '')} |"
        )
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote consolidated summary -> {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay saved ablation parameters (no LLM)")
    parser.add_argument("tunnel_ids", nargs="*", help="Tunnel IDs to replay")
    parser.add_argument("--ablation", "-a", choices=list(ABLATION_CONFIGS))
    parser.add_argument("--model", "-m", choices=MODELS)
    parser.add_argument("--all-conditions", action="store_true")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--sanity", action="store_true", help="Run sanity tunnels 1-1..5-1")
    parser.add_argument("--gate", action="store_true", help="Gate run: m_s_k/opus4.6/1-1 only")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if evaluation exists")
    parser.add_argument("--stages", nargs="+", choices=STAGE_NAMES)
    parser.add_argument("--seed-from", help="Archive dir to seed upstream artifacts (sam-only)")
    parser.add_argument("--t4t5-sam", action="store_true", help="SAM+eval on 4-1 and 5-1 all conditions/models")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()

    stage_filter = args.stages

    if args.t4t5_sam:
        tunnel_ids = list(T4T5_TUNNELS)
        ablations = list(ABLATION_CONFIGS) if args.all_conditions or not args.ablation else [args.ablation]
        models = list(MODELS) if args.all_models or not args.model else [args.model]
        stage_filter = ["sam"]
        args.force = True
    elif args.gate:
        ablations = ["m_s_k"]
        models = ["opus4.6"]
        tunnel_ids = ["1-1"]
    elif args.sanity:
        tunnel_ids = list(SANITY_TUNNELS)
        ablations = list(ABLATION_CONFIGS) if args.all_conditions else ([args.ablation] if args.ablation else [])
        models = list(MODELS) if args.all_models else ([args.model] if args.model else [])
    else:
        tunnel_ids = args.tunnel_ids
        ablations = [args.ablation] if args.ablation else []
        models = [args.model] if args.model else []

    if not tunnel_ids and not args.write_summary:
        parser.error("Provide tunnel IDs, --sanity, --gate, or --write-summary")
    if (ablations or models) and not ablations:
        parser.error("--ablation required (or use --all-conditions with --sanity)")
    if (ablations or models) and not models:
        parser.error("--model required (or use --all-models with --sanity)")

    log_dir = REPO_ROOT / "logs" / "ablation"
    log_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for ablation in ablations:
        env = spt.setup_env(cfg(ablation)["folder"])
        for model in models:
            for tid in tunnel_ids:
                log_file = log_dir / f"replay_{tid}_{ablation}_{model}.log"
                try:
                    metrics = replay_tunnel(
                        tid,
                        ablation,
                        model,
                        env,
                        skip_if_exists=args.skip_existing,
                        stage_filter=stage_filter,
                        seed_from=args.seed_from,
                        force=args.force,
                    )
                    if args.gate:
                        paths = preflight_params(tid, ablation, model, stage_filter)
                        dest = spt.out_root(cfg(ablation)["folder"]) / archive_key(tid, model)
                        passed = all(metrics.get(k) is not None for k in ("OA", "F1", "mIoU"))
                        write_gate_report(tid, ablation, model, metrics, paths, dest, passed)
                        if not passed:
                            sys.exit(1)
                except Exception as e:
                    msg = f"FAILED {tid}/{ablation}/{model}: {e}"
                    print(f"\n{msg}")
                    failures.append(msg)
                    log_file.write_text(msg + "\n", encoding="utf-8")
                    if args.gate or len(tunnel_ids) == 1:
                        sys.exit(1)

    if args.write_summary or args.sanity or args.all_conditions or args.t4t5_sam:
        write_consolidated_summary()

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
