#!/usr/bin/env python
"""
Memory+State ablation orchestrator.

Interleaves LLM inference (Anthropic Claude API) with pipeline stage execution
and characteristic extraction.  For each stage the LLM sees raw characteristics
plus *cumulative* processed characteristics from all prior stages — compared
against the sample tunnel.

Usage:
    ./venv/bin/python run_memory_state.py 1-1
    ./venv/bin/python run_memory_state.py 1-1 3-1-1 4-1
    ./venv/bin/python run_memory_state.py --all
    ./venv/bin/python run_memory_state.py 1-1 --dry-run   # prompts only, no API
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

from agents.ablation import sam4tun_pipeline_runtime as spt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABLATION_CODE = "m_s"
ABLATION_FOLDER = "memory+state"
PARAM_FILE_SUFFIX = "_m_s_"
DEFAULT_MODEL_TAG = "opus4.6"
CLAUDE_MODEL = "claude-opus-4-6"
MAX_TOKENS = 16384

STAGES = [
    ("unfolding", "unfolding.py", "1-unfolded_characteriser.py"),
    ("denoising", "denoising.py", "2-denoised_characteriser.py"),
    ("enhancing", "enhancing.py", "3-enhanced_characteriser.py"),
    ("detecting", "detecting.py", "4-detected_characteriser.py"),
    ("sam",       "sam.py",        None),
]

STAGE_TO_PARAM_NAME = {
    "unfolding": "unfolding",
    "denoising": "denoising",
    "enhancing": "enhancing",
    "detecting": "detecting",
    "sam":       "sam",
}

ANALYST_CLASSES = {
    "unfolding": ("unfolding.analyst", "UnfoldingAnalyser"),
    "denoising": ("denoising.analyst", "DenoisingAnalyser"),
    "enhancing": ("enhancing.analyst", "EnhancingAnalyser"),
    "detecting": ("detecting.analyst", "DetectingAnalyser"),
    "sam":       ("segmenting.analyst", "SegmentingAnalyser"),
}

PARAM_BASE = spt.param_base(ABLATION_FOLDER)
AGENTS_DIR = Path("agents/ablation") / ABLATION_FOLDER / "agents"
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Prompt building (via analyst classes)
# ---------------------------------------------------------------------------


def _import_analyst(stage_name: str):
    """Dynamically import the analyst class for a stage."""
    module_rel, class_name = ANALYST_CLASSES[stage_name]
    agents_abs = str(AGENTS_DIR.resolve())
    if agents_abs not in sys.path:
        sys.path.insert(0, agents_abs)

    import importlib
    mod = importlib.import_module(module_rel)
    return getattr(mod, class_name)


def build_prompt(tunnel_id: str, stage_name: str) -> str:
    cls = _import_analyst(stage_name)
    analyser = cls(tunnel_id)
    return analyser.build_llm_prompt_markdown()


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------


def call_claude(prompt: str, dry_run: bool = False) -> str:
    """Call Claude API and return the raw text response."""
    if dry_run:
        print("  [dry-run] Skipping API call")
        return ""

    client = anthropic.Anthropic()
    t0 = time.time()
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - t0
    text = message.content[0].text
    tokens_in = message.usage.input_tokens
    tokens_out = message.usage.output_tokens
    print(f"  API call: {elapsed:.1f}s, {tokens_in} in / {tokens_out} out")
    return text


def extract_json_from_response(response_text: str) -> dict:
    """Extract the JSON object from a markdown code fence in the response."""
    pattern = r"```json\s*\n(.*?)\n\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if not matches:
        raise ValueError("No ```json``` code fence found in Claude response")
    json_str = matches[-1].strip()
    return json.loads(json_str)


# ---------------------------------------------------------------------------
# Parameter save
# ---------------------------------------------------------------------------


def save_parameters(tunnel_id: str, stage_name: str, params: dict, model_tag: str) -> Path:
    return spt.save_parameters(
        tunnel_id, stage_name, params, ABLATION_FOLDER, PARAM_FILE_SUFFIX, STAGE_TO_PARAM_NAME, model_tag
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


STAGE_NAMES = [s[0] for s in STAGES]


def process_tunnel(tunnel_id: str, model_tag: str, dry_run: bool, env: dict,
                   stage_filter: list[str] | None = None) -> None:
    print(f"\n{'='*60}")
    print(f"TUNNEL: {tunnel_id}")
    if stage_filter:
        print(f"STAGES: {', '.join(stage_filter)}")
    print(f"{'='*60}")

    out_dir = spt.out_root(ABLATION_FOLDER) / tunnel_id
    spt.out_root(ABLATION_FOLDER).mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    spt.ensure_tunnel_characteristics(tunnel_id, ABLATION_FOLDER)

    if not dry_run:
        spt.symlink_input(tunnel_id)
        spt.prepare_work_dir(tunnel_id, stage_filter)

    needs_unfolding = not stage_filter or "unfolding" in stage_filter
    if needs_unfolding and not dry_run:
        print("\n--- Pre-stage: raw characteristics ---")
        spt.run_raw_characteriser(tunnel_id, env)

    upstream_pipeline_ran = False

    for stage_name, stage_script, characteriser in STAGES:
        if stage_filter and stage_name not in stage_filter:
            continue
        print(f"\n--- Stage: {stage_name} ---")

        param_name = STAGE_TO_PARAM_NAME[stage_name]
        param_path = PARAM_BASE / tunnel_id / f"parameters_{param_name}{PARAM_FILE_SUFFIX}{model_tag}.json"
        stale_path = PARAM_BASE / tunnel_id / f"parameters_{param_name}{PARAM_FILE_SUFFIX}{model_tag}.json"
        if stale_path.exists():
            stale_path.unlink()
            print(f"  Cleared stale: {stale_path}")
        old_params = None
        if param_path.exists():
            with open(param_path, encoding="utf-8") as f:
                old_params = json.load(f)

        print("  Building prompt...")
        prompt = build_prompt(tunnel_id, stage_name)
        print(f"  Prompt length: {len(prompt):,} chars")

        response_text = call_claude(prompt, dry_run=dry_run)

        if dry_run:
            continue

        analysis_dir = out_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / f"{stage_name}_reasoning_{model_tag}.md").write_text(response_text)

        try:
            params = extract_json_from_response(response_text)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  ERROR extracting JSON: {e}")
            (analysis_dir / f"{stage_name}_raw_response.md").write_text(response_text)
            print(f"  Raw response saved to {analysis_dir}/{stage_name}_raw_response.md")
            raise

        save_parameters(tunnel_id, stage_name, params, model_tag)

        params_match = old_params is not None and params == old_params
        skip_pipeline = spt.should_skip_pipeline(
            tunnel_id, ABLATION_FOLDER, stage_name, characteriser, params_match, upstream_pipeline_ran
        )
        artifact = spt.WORK_ARTIFACTS.get(stage_name)
        if skip_pipeline and artifact and not (spt.work_dir(tunnel_id) / artifact).exists():
            skip_pipeline = False
            print(f"  Re-run pipeline: missing {artifact}")
        if skip_pipeline:
            print(f"  SKIP pipeline: parameters unchanged for {stage_name}")
            continue

        spt.run_pipeline_stage(tunnel_id, stage_script, ABLATION_CODE, model_tag, env)

        if characteriser:
            spt.run_characteriser(tunnel_id, characteriser, env)

        upstream_pipeline_ran = True

    skip_eval = stage_filter and "sam" not in stage_filter
    if not dry_run and not skip_eval and upstream_pipeline_ran:
        print("\n--- Archive + evaluation ---")
        spt.archive_pipeline_output(tunnel_id, ABLATION_FOLDER)
        spt.run_evaluation(tunnel_id, ABLATION_FOLDER, env)

    print(f"\nDone: {tunnel_id}")


def get_all_tunnel_ids() -> list[str]:
    """List tunnel IDs from the parameters directory."""
    if not PARAM_BASE.exists():
        return []
    return sorted(
        d.name for d in PARAM_BASE.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def main():
    parser = argparse.ArgumentParser(
        description="Memory+State ablation orchestrator",
    )
    parser.add_argument(
        "tunnel_ids", nargs="*",
        help="Tunnel IDs to process (e.g. 1-1 3-1-1 4-1)",
    )
    parser.add_argument("--all", action="store_true", help="Process all tunnels")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_TAG,
        help=f"Model tag for parameter files (default: {DEFAULT_MODEL_TAG})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build prompts but skip API calls and pipeline execution",
    )
    parser.add_argument(
        "--stages", nargs="+", choices=STAGE_NAMES,
        help="Run only these stages (e.g. --stages unfolding denoising enhancing)",
    )
    args = parser.parse_args()

    if args.all:
        tunnel_ids = get_all_tunnel_ids()
        if not tunnel_ids:
            print("No tunnel IDs found under", PARAM_BASE)
            sys.exit(1)
        print(f"Processing {len(tunnel_ids)} tunnels: {', '.join(tunnel_ids)}")
    elif args.tunnel_ids:
        tunnel_ids = args.tunnel_ids
    else:
        parser.error("Provide tunnel IDs or --all")

    env = spt.setup_env(ABLATION_FOLDER)
    t_start = time.time()

    for tid in tunnel_ids:
        try:
            process_tunnel(tid, args.model, args.dry_run, env, args.stages)
        except Exception as e:
            print(f"\nFAILED on tunnel {tid}: {e}")
            if len(tunnel_ids) == 1:
                sys.exit(1)
            print("Continuing with next tunnel...\n")

    elapsed = time.time() - t_start
    print(f"\nAll done. Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
