"""
Pipeline data helpers: argument parsing, ablation parameter resolution, output directories.

All configurable stage scripts import from here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# Ablation condition registry
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "opus4.6"

ABLATION_CONDITIONS = {
    "sam4tun": {
        "folder": "sam4tun",
        "ablation_tag": "",
        "per_tunnel": False,
        "out_prefix": "data/ablation/sam4tun",
    },
    "m": {
        "folder": "memory",
        "ablation_tag": "_m",
        "per_tunnel": True,
        "out_prefix": "data/ablation/memory",
    },
    "m_s": {
        "folder": "memory+state",
        "ablation_tag": "_m_s",
        "per_tunnel": True,
        "out_prefix": "data/ablation/memory+state",
    },
    "m_s_k": {
        "folder": "memory+state+knowledge",
        "ablation_tag": "_m_s_k",
        "per_tunnel": True,
        "out_prefix": "data/ablation/memory+state+knowledge",
    },
}

ABLATION_CODES = list(ABLATION_CONDITIONS.keys())


# ---------------------------------------------------------------------------
# Argument parsing (shared by all configurable stage scripts)
# ---------------------------------------------------------------------------

def parse_pipeline_args(stage_name: str) -> tuple[str, str, str]:
    """
    Parse CLI for a configurable stage script.

    Returns (tunnel_id, ablation_code, model).
    Side-effect: sets R4TUN_PIPELINE_OUT_PREFIX from the ablation code.
    """
    p = argparse.ArgumentParser(
        description=f"Configurable pipeline — {stage_name} stage",
    )
    p.add_argument("tunnel_id", help="Tunnel identifier, e.g. 1-1, 4-1")
    p.add_argument(
        "--ablation", "-a",
        required=True,
        choices=ABLATION_CODES,
        help=f"Ablation condition code: {', '.join(ABLATION_CODES)}",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model tag for parameter file suffix (default: {DEFAULT_MODEL})",
    )
    args = p.parse_args()

    cond = ABLATION_CONDITIONS[args.ablation]
    os.environ["R4TUN_PIPELINE_OUT_PREFIX"] = cond["out_prefix"]

    return args.tunnel_id, args.ablation, args.model


# ---------------------------------------------------------------------------
# Ablation parameter loading
# ---------------------------------------------------------------------------

def _repo_root() -> str:
    """Best-effort repo root: parent of configurable/."""
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(cfg_dir)


def _build_suffix(ablation_code: str, model: str) -> str:
    """Build the file suffix from ablation tag + model tag.

    sam4tun → '' (no suffix)
    m + opus4.6 → '_m_opus4.6'
    m_s + gemini2.5 → '_m_s_gemini2.5'
    """
    cond = ABLATION_CONDITIONS[ablation_code]
    tag = cond["ablation_tag"]
    if not tag:
        return ""
    return f"{tag}_{model}"


def resolve_ablation_param_file(
    tunnel_id: str, stage: str, ablation_code: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Return the absolute path to the parameter JSON for a given ablation condition.

    sam4tun (shared):  configurable/ablation/sam4tun/parameters_{stage}.json
    per-tunnel:        configurable/ablation/{folder}/parameters/{tunnel_id}/parameters_{stage}{tag}_{model}.json
    """
    cond = ABLATION_CONDITIONS[ablation_code]
    root = _repo_root()
    ablation_base = os.path.join(root, "configurable", "ablation")
    suffix = _build_suffix(ablation_code, model)

    if cond["per_tunnel"]:
        path = os.path.join(
            ablation_base,
            cond["folder"],
            "parameters",
            tunnel_id,
            f"parameters_{stage}{suffix}.json",
        )
    else:
        path = os.path.join(
            ablation_base,
            cond["folder"],
            f"parameters_{stage}.json",
        )
    return path


def load_stage_parameters(
    tunnel_id: str, stage: str, ablation_code: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Load and return the parameter dict; exit on missing file or parse error."""
    param_file = resolve_ablation_param_file(tunnel_id, stage, ablation_code, model)

    if not os.path.exists(param_file):
        print(f"❌ Parameter file not found: {param_file}")
        sys.exit(1)

    try:
        with open(param_file, "r") as f:
            params = json.load(f)
    except Exception as e:
        print(f"❌ Error loading parameters from {param_file}: {e}")
        sys.exit(1)

    cond = ABLATION_CONDITIONS[ablation_code]
    rel = os.path.relpath(param_file, _repo_root())
    print(f"✅ [{ablation_code}] Loaded {stage} parameters from {rel}")
    return params


# ---------------------------------------------------------------------------
# Output directory helpers (unchanged logic, env-var driven)
# ---------------------------------------------------------------------------

def resolve_tunnel_pointcloud_txt(tunnel_id: str) -> str:
    """
    Path to the 6-column raw cloud (cwd = repo root or ``configurable/``).

    Order: ``data/subsets/<tunnel_id>.txt``, then legacy ``data/<tunnel_id>.txt``.
    """
    rel_subsets = os.path.join("data", "subsets", f"{tunnel_id}.txt")
    rel_data = os.path.join("data", f"{tunnel_id}.txt")
    candidates = (
        rel_subsets,
        os.path.join("..", rel_subsets),
        rel_data,
        os.path.join("..", rel_data),
    )
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Point cloud not found for tunnel {tunnel_id!r}. "
        f"Expected data/subsets/{tunnel_id}.txt or data/{tunnel_id}.txt (from repo root)."
    )


def pipeline_out_prefix() -> str:
    p = (os.environ.get("R4TUN_PIPELINE_OUT_PREFIX") or "").strip().rstrip("/")
    if not p:
        raise RuntimeError(
            "R4TUN_PIPELINE_OUT_PREFIX is not set. "
            "Pass --ablation <code> to set it automatically, "
            "or export R4TUN_PIPELINE_OUT_PREFIX before running."
        )
    return p


def tunnel_output_relpath(tunnel_id: str) -> str:
    return f"{pipeline_out_prefix()}/{tunnel_id}"


def tunnel_output_dir(tunnel_id: str) -> str:
    rel = tunnel_output_relpath(tunnel_id)
    return os.path.normpath(rel).replace("\\", "/") + "/"


def resolve_output_base_dir(tunnel_id: str, marker: str | None = None) -> str:
    """
    Base directory for this tunnel's pipeline outputs (trailing slash).

    If marker is set (e.g. unwrapped.csv), require that file to exist.
    """
    rel = tunnel_output_relpath(tunnel_id)
    if not marker:
        return tunnel_output_dir(tunnel_id)

    candidates = (
        os.path.join(rel, marker),
        os.path.join("..", rel, marker),
    )
    for p in candidates:
        if os.path.exists(p):
            d = os.path.dirname(os.path.normpath(p))
            return d.replace("\\", "/") + "/"

    raise FileNotFoundError(
        f"Missing {marker} for tunnel {tunnel_id}. Tried: {candidates[0]}, {candidates[1]}. "
        f"Output prefix is {pipeline_out_prefix()!r}."
    )
