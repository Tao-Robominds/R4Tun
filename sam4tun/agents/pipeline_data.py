"""Parameter loading for the sam4tun agents pipeline (no ablation machinery)."""

from __future__ import annotations

import argparse
import json
import os
import sys

_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SAM4TUN_ROOT = os.path.dirname(_AGENTS_DIR)

STAGES = ("unfolding", "denoising", "enhancing", "detecting", "sam")


def parse_pipeline_args(stage_name: str) -> str:
    """Parse CLI for a pipeline stage script. Returns tunnel_id."""
    p = argparse.ArgumentParser(description=f"SAM4Tun agents — {stage_name} stage")
    p.add_argument("tunnel_id", help="Tunnel identifier, e.g. sample")
    args = p.parse_args()
    return args.tunnel_id


_FALLBACK_PARAM_TUNNEL = "sample"


def resolve_param_file(tunnel_id: str, stage: str) -> str:
    """Tunnel-specific JSON, or frozen baseline under parameters/sample/."""
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


def load_stage_parameters(tunnel_id: str, stage: str) -> dict:
    """Load parameter dict; exit on missing file or parse error."""
    param_file = resolve_param_file(tunnel_id, stage)
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
    tunnel_specific = os.path.join(
        _AGENTS_DIR, "parameters", tunnel_id, f"parameters_{stage}.json"
    )
    if tunnel_id != _FALLBACK_PARAM_TUNNEL and not os.path.isfile(tunnel_specific):
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
