"""
Pipeline artefact directory (all stages except raw point cloud input).

Raw point cloud: prefer ``data/subsets/<tunnel_id>.txt`` (no copy/symlink under
``data/<tunnel_id>.txt`` required). If missing, falls back to ``data/<tunnel_id>.txt``
(e.g. ``data/sample.txt`` for tunnel_id ``sample``).

Outputs live under ``{R4TUN_PIPELINE_OUT_PREFIX}/{tunnel_id}/``. The variable must be set explicitly
(e.g. ``./run_agents.sh <id> --memory-ablation`` or ``--sam4tun-ablation``); there is no silent
fallback to ``data/<tunnel_id>/``.
"""

from __future__ import annotations

import os


def resolve_tunnel_pointcloud_txt(tunnel_id: str) -> str:
    """
    Path to the 6-column raw cloud (cwd = repo root or ``configurable/``).

    Order: ``data/subsets/<tunnel_id>.txt``, then legacy ``data/<tunnel_id>.txt``
    (e.g. ``sample`` at ``data/sample.txt``).
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
            "Use ./run_agents.sh <tunnel_id> --memory-ablation or --sam4tun-ablation, "
            "or export R4TUN_PIPELINE_OUT_PREFIX (e.g. data/ablation/memory) before running "
            "configurable stages or characteriser plugins."
        )
    return p


def tunnel_output_relpath(tunnel_id: str) -> str:
    """Path relative to repo root, no trailing slash."""
    return f"{pipeline_out_prefix()}/{tunnel_id}"


def tunnel_output_dir(tunnel_id: str) -> str:
    """Directory for CSV/NPY/PNG outputs; includes trailing slash."""
    rel = tunnel_output_relpath(tunnel_id)
    return os.path.normpath(rel).replace("\\", "/") + "/"


def resolve_output_base_dir(tunnel_id: str, marker: str | None = None) -> str:
    """
    Base directory for this tunnel's pipeline outputs (trailing slash).

    If marker is set (e.g. unwrapped.csv), require that file to exist under
    tunnel_output_relpath from project root or from configurable/.
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
        f"Output prefix is {pipeline_out_prefix()!r} (same R4TUN_PIPELINE_OUT_PREFIX for all stages)."
    )
