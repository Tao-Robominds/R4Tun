"""Parameter loading for the sam4tun agents pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys

_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SAM4TUN_ROOT = os.path.dirname(_AGENTS_DIR)

STAGES = ("unfolding", "denoising", "enhancing", "detecting", "sam")

DEFAULT_MODEL = "glm"

ABLATION_CONDITIONS = {
    "m": {
        "folder": "memory",
        "ablation_tag": "_m",
        "out_prefix": "data/ablation/memory",
    },
    "m_s": {
        "folder": "memory+state",
        "ablation_tag": "_m_s",
        "out_prefix": "data/ablation/memory+state",
    },
    "m_s_k": {
        "folder": "memory+state+knowledge",
        "ablation_tag": "_m_s_k",
        "out_prefix": "data/ablation/memory+state+knowledge",
    },
}

ABLATION_CODES = list(ABLATION_CONDITIONS.keys())
_FALLBACK_PARAM_TUNNEL = "sample"


def parse_pipeline_args(stage_name: str) -> tuple[str, str | None, str | None]:
    """Parse CLI. Returns (tunnel_id, ablation_code, model_tag)."""
    p = argparse.ArgumentParser(description=f"SAM4Tun agents — {stage_name} stage")
    p.add_argument("tunnel_id", help="Tunnel identifier, e.g. sample, 1-1")
    p.add_argument(
        "--ablation", "-a",
        choices=ABLATION_CODES,
        default=None,
        help="LLM ablation condition: m, m_s, m_s_k",
    )
    p.add_argument(
        "--model",
        default=None,
        help=f"LLM model tag for parameter filenames (default: {DEFAULT_MODEL} when --ablation set)",
    )
    args = p.parse_args()

    if args.ablation:
        model = args.model or DEFAULT_MODEL
        cond = ABLATION_CONDITIONS[args.ablation]
        os.environ["R4TUN_PIPELINE_OUT_PREFIX"] = cond["out_prefix"]
        os.environ["R4TUN_ABLATION_TUNNEL_SUBROOT"] = cond["folder"]
        return args.tunnel_id, args.ablation, model

    return args.tunnel_id, None, None


def _build_suffix(ablation_code: str, model: str) -> str:
    tag = ABLATION_CONDITIONS[ablation_code]["ablation_tag"]
    return f"{tag}_{model}"


def resolve_ablation_param_file(
    tunnel_id: str,
    stage: str,
    ablation_code: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """sam4tun/agents/parameters/{folder}/{tunnel_id}/parameters_{stage}{suffix}.json"""
    folder = ABLATION_CONDITIONS[ablation_code]["folder"]
    suffix = _build_suffix(ablation_code, model)
    return os.path.join(
        _AGENTS_DIR,
        "parameters",
        folder,
        tunnel_id,
        f"parameters_{stage}{suffix}.json",
    )


def resolve_param_file(
    tunnel_id: str,
    stage: str,
    ablation_code: str | None = None,
    model: str | None = None,
) -> str:
    """Ablation GLM params, rules, tunnel-specific, or sample fallback."""
    if ablation_code and model:
        abl_path = resolve_ablation_param_file(tunnel_id, stage, ablation_code, model)
        if os.path.exists(abl_path):
            return abl_path
        return os.path.join(
            _AGENTS_DIR,
            "parameters",
            _FALLBACK_PARAM_TUNNEL,
            f"parameters_{stage}.json",
        )

    rules_path = os.path.join(
        _AGENTS_DIR,
        "parameters",
        "rules",
        tunnel_id,
        f"parameters_{stage}.json",
    )
    if os.path.exists(rules_path):
        return rules_path
    tunnel_path = os.path.join(
        _AGENTS_DIR,
        "parameters",
        tunnel_id,
        f"parameters_{stage}.json",
    )
    if os.path.exists(tunnel_path):
        return tunnel_path
    return os.path.join(
        _AGENTS_DIR,
        "parameters",
        _FALLBACK_PARAM_TUNNEL,
        f"parameters_{stage}.json",
    )


def load_stage_parameters(
    tunnel_id: str,
    stage: str,
    ablation_code: str | None = None,
    model: str | None = None,
) -> dict:
    """Load parameter dict; exit on missing file or parse error."""
    param_file = resolve_param_file(tunnel_id, stage, ablation_code, model)
    if not os.path.exists(param_file):
        print(f"❌ Parameter file not found: {param_file}")
        sys.exit(1)
    try:
        with open(param_file, "r") as f:
            params = json.load(f)
    except Exception as e:
        print(f"❌ Error loading parameters from {param_file}: {e}")
        sys.exit(1)

    rel = os.path.relpath(param_file, _SAM4TUN_ROOT)
    if ablation_code and model:
        abl_specific = resolve_ablation_param_file(tunnel_id, stage, ablation_code, model)
        if param_file == abl_specific:
            print(f"✅ Loaded {stage} parameters from {rel} (ablation {ablation_code} {model})")
        else:
            print(f"✅ Loaded {stage} parameters from {rel} (ablation fallback for {tunnel_id})")
    else:
        rules_specific = os.path.join(
            _AGENTS_DIR, "parameters", "rules", tunnel_id, f"parameters_{stage}.json"
        )
        tunnel_specific = os.path.join(
            _AGENTS_DIR, "parameters", tunnel_id, f"parameters_{stage}.json"
        )
        if os.path.isfile(rules_specific) and param_file == rules_specific:
            print(f"✅ Loaded {stage} parameters from {rel} (rules baseline)")
        elif tunnel_id != _FALLBACK_PARAM_TUNNEL and not os.path.isfile(tunnel_specific):
            print(f"✅ Loaded {stage} parameters from {rel} (fallback for tunnel {tunnel_id})")
        else:
            print(f"✅ Loaded {stage} parameters from {rel}")
    return params


def require_keys(params: dict, keys: list[str], param_file: str) -> None:
    for key in keys:
        if key not in params:
            print(f"❌ Error: Missing required parameter '{key}' in {param_file}")
            sys.exit(1)


def setup_sam4tun_path() -> None:
    """Ensure sam4tun root is on sys.path for helpers.* imports."""
    if _SAM4TUN_ROOT not in sys.path:
        sys.path.insert(0, _SAM4TUN_ROOT)
