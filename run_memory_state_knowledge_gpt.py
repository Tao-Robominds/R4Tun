#!/usr/bin/env python
"""
Memory+State+Knowledge ablation orchestrator (OpenAI GPT-5.4).

Interleaves LLM inference (OpenAI Responses API) with pipeline stage execution
and characteristic extraction.  For each stage the LLM sees raw characteristics,
cumulative processed characteristics from prior stages (vs sample), domain
knowledge, and reference parameters.

Usage:
    ./venv/bin/python run_memory_state_knowledge_gpt.py 1-1
    ./venv/bin/python run_memory_state_knowledge_gpt.py 1-1 3-1-1 4-1
    ./venv/bin/python run_memory_state_knowledge_gpt.py --all
    ./venv/bin/python run_memory_state_knowledge_gpt.py 1-1 --dry-run   # prompts only, no API

**Routine runs** omit ``--model``: default tag ``gpt5.4`` and parameter files ``*_m_s_k_gpt5.4.json``.
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

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABLATION_CODE = "m_s_k"
ABLATION_FOLDER = "memory+state+knowledge"
DEFAULT_MODEL_TAG = "gpt5.4"

MODEL_TAG_TO_OPENAI: dict[str, str] = {
    "gpt5.4": "gpt-5.4",
}


def openai_model_for_tag(model_tag: str) -> str:
    if model_tag not in MODEL_TAG_TO_OPENAI:
        known = ", ".join(sorted(MODEL_TAG_TO_OPENAI))
        raise ValueError(f"Unknown --model tag {model_tag!r}; known: {known}")
    return MODEL_TAG_TO_OPENAI[model_tag]


# Suffix for parameter filenames, e.g. parameters_unfolding_m_s_k_gpt5.4.json
PARAM_FILE_SUFFIX = "_m_s_k_"

STAGES = [
    ("unfolding", "configurable_unfolding.py", "1-unfolded_characteriser.py"),
    ("denoising", "configurable_denoising.py", "2-denoised_characteriser.py"),
    ("enhancing", "configurable_enhancing.py", "3-enhanced_characteriser.py"),
    ("detecting", "configurable_detecting.py", "4-detected_characteriser.py"),
    ("sam",       "configurable_sam.py",        None),
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

PARAM_BASE = Path("configurable/ablation") / ABLATION_FOLDER / "parameters"
AGENTS_DIR = Path("configurable/ablation") / ABLATION_FOLDER / "agents"
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Env setup
# ---------------------------------------------------------------------------


def _setup_env() -> dict[str, str]:
    env = os.environ.copy()
    env["R4TUN_PIPELINE_OUT_PREFIX"] = f"data/ablation/{ABLATION_FOLDER}"
    env["R4TUN_ABLATION_TUNNEL_SUBROOT"] = ABLATION_FOLDER
    env["PYTHONPATH"] = "."
    return env


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
# OpenAI API
# ---------------------------------------------------------------------------


MAX_RETRIES = 3
API_TIMEOUT = 300  # seconds


def call_gpt(prompt: str, model_tag: str, dry_run: bool = False) -> str:
    """Call OpenAI Responses API for the model mapped from ``model_tag``."""
    if dry_run:
        print("  [dry-run] Skipping API call")
        return ""

    api_model = openai_model_for_tag(model_tag)
    client = OpenAI(timeout=API_TIMEOUT)

    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        try:
            response = client.responses.create(
                model=api_model,
                reasoning={"effort": "high"},
                input=[{"role": "user", "content": prompt}],
            )
        except APIStatusError as e:
            body = e.body if isinstance(e.body, dict) else {}
            err = body.get("error") if isinstance(body.get("error"), dict) else body
            code = err.get("code") if isinstance(err, dict) else None
            msg = err.get("message") if isinstance(err, dict) else ""
            if e.status_code == 429 and code == "insufficient_quota":
                print(
                    "  OpenAI API: 429 insufficient_quota — no billable quota for this API key/project. "
                    "Confirm credits at https://platform.openai.com/account/billing and that .env uses a key from that org."
                )
                raise SystemExit(2) from e
            if e.status_code == 403 and code == "model_not_found":
                print(
                    "  OpenAI API: 403 model_not_found — this API key is tied to a project that does not allow "
                    f"{api_model} yet (or you edited a different project in the dashboard).\n"
                    "  Fix: OpenAI Platform → the same project as this key (sk-proj-…) → Allowed models → "
                    f"allow {api_model} → Save → create a new key from that project if needed."
                )
                if msg:
                    print(f"  Detail: {msg}")
                raise SystemExit(3) from e
            raise
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  API call attempt {attempt}/{MAX_RETRIES} failed after {elapsed:.0f}s: {type(e).__name__}: {e}")
            if attempt == MAX_RETRIES:
                raise
            wait = 10 * attempt
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)
            continue

        elapsed = time.time() - t0
        text = response.output_text
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "input_tokens", None) if usage else None
        tokens_out = getattr(usage, "output_tokens", None) if usage else None
        print(f"  API call: {elapsed:.1f}s, {tokens_in} in / {tokens_out} out")
        return text

    raise RuntimeError("Unreachable")


def extract_json_from_response(response_text: str) -> dict:
    """Extract the JSON object from a markdown code fence in the response."""
    pattern = r"```json\s*\n(.*?)\n\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if not matches:
        raise ValueError("No ```json``` code fence found in LLM response")
    json_str = matches[-1].strip()
    return json.loads(json_str)


# ---------------------------------------------------------------------------
# Parameter save
# ---------------------------------------------------------------------------


def save_parameters(tunnel_id: str, stage_name: str, params: dict, model_tag: str) -> Path:
    param_name = STAGE_TO_PARAM_NAME[stage_name]
    filename = f"parameters_{param_name}{PARAM_FILE_SUFFIX}{model_tag}.json"
    out_dir = PARAM_BASE / tunnel_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def run_pipeline_stage(tunnel_id: str, stage_script: str, model_tag: str, env: dict) -> None:
    cmd = [
        PYTHON, f"configurable/{stage_script}",
        tunnel_id,
        "--ablation", ABLATION_CODE,
        "--model", model_tag,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"Pipeline stage {stage_script} failed (exit {result.returncode})")


def run_characteriser(tunnel_id: str, characteriser_script: str, env: dict) -> None:
    cmd = [PYTHON, f"sam4tun/plugins/{characteriser_script}", tunnel_id]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"Characteriser {characteriser_script} failed (exit {result.returncode})")


def run_raw_characteriser(tunnel_id: str, env: dict) -> None:
    cmd = [
        PYTHON, "sam4tun/plugins/raw_characteristics.py",
        "--tunnel_id", tunnel_id,
        "--data_dir", "data/subsets",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"Raw characteriser failed (exit {result.returncode})")


def run_evaluation(tunnel_id: str, env: dict) -> None:
    only_label = Path(f"data/ablation/{ABLATION_FOLDER}/{tunnel_id}/only_label.csv")
    if not only_label.exists():
        print(f"  Skipping evaluation: {only_label} not found")
        return
    cmd = [
        PYTHON, "configurable/evaluation.py",
        tunnel_id,
        "--ablation", ABLATION_CODE,
        "--schema", "auto",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        print("  Warning: evaluation failed, continuing")


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

    out_dir = Path(f"data/ablation/{ABLATION_FOLDER}/{tunnel_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always regenerate raw characteristics before pipeline stages
    needs_unfolding = not stage_filter or "unfolding" in stage_filter
    if needs_unfolding:
        print("\n--- Pre-stage: raw characteristics ---")
        run_raw_characteriser(tunnel_id, env)

    # If an upstream stage re-ran the pipeline, downstream inputs changed — must re-run downstream too.
    upstream_pipeline_ran = False

    for stage_name, stage_script, characteriser in STAGES:
        if stage_filter and stage_name not in stage_filter:
            continue
        print(f"\n--- Stage: {stage_name} ---")

        param_name = STAGE_TO_PARAM_NAME[stage_name]
        param_path = PARAM_BASE / tunnel_id / f"parameters_{param_name}{PARAM_FILE_SUFFIX}{model_tag}.json"
        old_params = None
        if param_path.exists():
            with open(param_path, encoding="utf-8") as f:
                old_params = json.load(f)

        # 1. Build prompt
        print("  Building prompt...")
        prompt = build_prompt(tunnel_id, stage_name)
        prompt_len = len(prompt)
        print(f"  Prompt length: {prompt_len:,} chars")

        # 2. Call OpenAI API
        response_text = call_gpt(prompt, model_tag, dry_run=dry_run)

        if dry_run:
            continue

        # 3. Extract and save parameters
        try:
            params = extract_json_from_response(response_text)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  ERROR extracting JSON: {e}")
            analysis_dir = out_dir / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            (analysis_dir / f"{stage_name}_raw_response.md").write_text(response_text)
            print(f"  Raw response saved to {analysis_dir}/{stage_name}_raw_response.md")
            raise

        save_parameters(tunnel_id, stage_name, params, model_tag)

        params_match = old_params is not None and params == old_params
        skip_pipeline = params_match and not upstream_pipeline_ran
        if skip_pipeline:
            print(f"  SKIP pipeline: parameters unchanged for {stage_name}")
            continue

        # 4. Run pipeline stage
        run_pipeline_stage(tunnel_id, stage_script, model_tag, env)

        # 5. Extract characteristics
        if characteriser:
            run_characteriser(tunnel_id, characteriser, env)

        upstream_pipeline_ran = True

    skip_eval = stage_filter and "sam" not in stage_filter
    if not dry_run and not skip_eval and upstream_pipeline_ran:
        print(f"\n--- Evaluation ---")
        run_evaluation(tunnel_id, env)

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
        description="Memory+State+Knowledge ablation orchestrator (OpenAI GPT-5.4)",
    )
    parser.add_argument(
        "tunnel_ids", nargs="*",
        help="Tunnel IDs to process (e.g. 1-1 3-1-1 4-1)",
    )
    parser.add_argument("--all", action="store_true", help="Process all tunnels")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_TAG,
        choices=sorted(MODEL_TAG_TO_OPENAI.keys()),
        help=f"Parameter file tag; OpenAI model {MODEL_TAG_TO_OPENAI[DEFAULT_MODEL_TAG]!r} (default: {DEFAULT_MODEL_TAG})",
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

    env = _setup_env()
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
