#!/usr/bin/env python
"""
Memory ablation orchestrator (Google Gemini 3 Flash).

Interleaves LLM inference (Gemini API via google-genai) with pipeline stage execution
and characteristic extraction.  For each stage the LLM sees raw characteristics
(compared against the sample tunnel) and reference parameters — no prior-stage
state in the prompt (memory-only ablation).

Usage:
    ./venv/bin/python run_memory_gemini.py 1-1
    ./venv/bin/python run_memory_gemini.py 1-1 3-1-1 4-1
    ./venv/bin/python run_memory_gemini.py --all
    ./venv/bin/python run_memory_gemini.py 1-1 --dry-run   # prompts only, no API

**Routine runs** omit ``--model``: default tag ``gemini3flash`` and parameter files ``*_m_gemini3flash.json``.
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
from google import genai
from google.genai import errors as genai_errors

load_dotenv()

from agents.ablation import sam4tun_pipeline_runtime as spt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABLATION_CODE = "m"
ABLATION_FOLDER = "memory"
DEFAULT_MODEL_TAG = "gemini3flash"

# Parameter filename tag -> Gemini API model id
MODEL_TAG_TO_GEMINI: dict[str, str] = {
    "gemini3flash": "gemini-3-flash-preview",
}

# Suffix for parameter filenames, e.g. parameters_unfolding_m_gemini3flash.json
PARAM_FILE_SUFFIX = "_m_"


def gemini_model_for_tag(model_tag: str) -> str:
    if model_tag not in MODEL_TAG_TO_GEMINI:
        known = ", ".join(sorted(MODEL_TAG_TO_GEMINI))
        raise ValueError(f"Unknown --model tag {model_tag!r}; known: {known}")
    return MODEL_TAG_TO_GEMINI[model_tag]

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
# Env setup
# ---------------------------------------------------------------------------


def _setup_env() -> dict[str, str]:
    env = os.environ.copy()
    env["R4TUN_PIPELINE_OUT_PREFIX"] = f"data/ablation/{ABLATION_FOLDER}"
    env["R4TUN_PIPELINE_WORK_DIR"] = f"data/ablation/{ABLATION_FOLDER}"
    env["R4TUN_ABLATION_TUNNEL_SUBROOT"] = ABLATION_FOLDER
    root = Path(__file__).resolve().parent
    sam_sa = root / "sam4tun" / "segment-anything"
    pp = [str(root)]
    if sam_sa.is_dir():
        pp.append(str(sam_sa))
    env["PYTHONPATH"] = os.pathsep.join(pp)
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
# Gemini API
# ---------------------------------------------------------------------------


MAX_RETRIES = 3
API_TIMEOUT = 300  # seconds (google-genai http_options timeout is in milliseconds)


def call_gemini(prompt: str, model_tag: str, dry_run: bool = False) -> str:
    """Call Gemini API for the model mapped from ``model_tag``."""
    if dry_run:
        print("  [dry-run] Skipping API call")
        return ""

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "  GEMINI_API_KEY is not set. Add it to .env (Google AI Studio / Gemini API key)."
        )
        raise SystemExit(3)

    api_model = gemini_model_for_tag(model_tag)
    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": API_TIMEOUT * 1000},
    )

    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        try:
            response = client.models.generate_content(
                model=api_model,
                contents=prompt,
            )
        except genai_errors.APIError as e:
            code = getattr(e, "code", None)
            status = (getattr(e, "status", None) or "").upper()
            msg = (getattr(e, "message", None) or str(e)).lower()
            if code == 429 or status == "RESOURCE_EXHAUSTED" or "quota" in msg or "rate" in msg:
                print(
                    "  Gemini API: rate limit or quota — reduce concurrency or check "
                    "https://ai.google.dev/ usage and billing."
                )
                raise SystemExit(2) from e
            if code in (401, 403, 404) or "api key" in msg or "permission" in msg or "not found" in msg:
                print(
                    "  Gemini API: authentication, permission, or model error — check GEMINI_API_KEY in .env "
                    f"and that model {api_model!r} is available for your key."
                )
                if getattr(e, "message", None):
                    print(f"  Detail: {e.message}")
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
        text = (response.text or "").strip()
        if not text:
            pf = getattr(response, "prompt_feedback", None)
            raise ValueError(
                f"Empty Gemini text response; prompt_feedback={pf!r}, "
                f"candidates={getattr(response, 'candidates', None)!r}"
            )
        usage = getattr(response, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", None) if usage else None
        tokens_out = getattr(usage, "candidates_token_count", None) if usage else None
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
        old_params = None
        if param_path.exists():
            with open(param_path, encoding="utf-8") as f:
                old_params = json.load(f)

        print("  Building prompt...")
        prompt = build_prompt(tunnel_id, stage_name)
        print(f"  Prompt length: {len(prompt):,} chars")

        response_text = call_gemini(prompt, model_tag, dry_run=dry_run)

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
        description="Memory ablation orchestrator (Google Gemini 3 Flash)",
    )
    parser.add_argument(
        "tunnel_ids", nargs="*",
        help="Tunnel IDs to process (e.g. 1-1 3-1-1 4-1)",
    )
    parser.add_argument("--all", action="store_true", help="Process all tunnels")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_TAG,
        choices=sorted(MODEL_TAG_TO_GEMINI.keys()),
        help=(
            f"Parameter file tag; Gemini model {MODEL_TAG_TO_GEMINI[DEFAULT_MODEL_TAG]!r} "
            f"(default: {DEFAULT_MODEL_TAG})"
        ),
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
