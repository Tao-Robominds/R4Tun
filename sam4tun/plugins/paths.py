"""
Layout convention — change **ABLATION_TUNNEL_SUBROOT** when experimenting with folder layout.

- Pipeline CSVs and artefacts: ``data/{tunnel_id}/...``
- Reference sample characterisation JSON: ``data/sample/characteristics/``
- Per-tunnel study (subsets) characterisation JSON:
  ``data/ablation/{ABLATION_TUNNEL_SUBROOT}/{tunnel_id}/characteristics/``
"""

import os

# Subfolder under data/ablation/ for per-tunnel characteristics (e.g. "memory", or "" for flat layout).
ABLATION_TUNNEL_SUBROOT = "memory"


def _ablation_tunnel_subroot() -> str:
    return os.environ.get("R4TUN_ABLATION_TUNNEL_SUBROOT", ABLATION_TUNNEL_SUBROOT)


def _pipeline_data_root() -> str:
    """Pipeline CSV workspace (sam4tun/data during sam4tun/agents runs)."""
    return os.environ.get("R4TUN_PIPELINE_WORK_DIR", "sam4tun/data")


def tunnel_pipeline_dir(tunnel_id: str) -> str:
    return os.path.join(_pipeline_data_root(), tunnel_id)


def tunnel_characteristics_parent_dir(tunnel_id: str) -> str:
    """Parent directory that contains the ``characteristics`` subfolder (plugin writers pass this as base_dir)."""
    if tunnel_id == "sample":
        return os.path.join("data", "sample")
    parts = ["data", "ablation"]
    subroot = _ablation_tunnel_subroot()
    if subroot:
        parts.append(subroot)
    parts.append(tunnel_id)
    return os.path.join(*parts)


def tunnel_characteristics_dir(tunnel_id: str) -> str:
    return os.path.join(tunnel_characteristics_parent_dir(tunnel_id), "characteristics")
