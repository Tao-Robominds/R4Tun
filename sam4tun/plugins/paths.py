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


def tunnel_pipeline_dir(tunnel_id: str) -> str:
    return os.path.join("data", tunnel_id)


def tunnel_characteristics_parent_dir(tunnel_id: str) -> str:
    """Parent directory that contains the ``characteristics`` subfolder (plugin writers pass this as base_dir)."""
    if tunnel_id == "sample":
        return os.path.join("data", "sample")
    parts = ["data", "ablation"]
    if ABLATION_TUNNEL_SUBROOT:
        parts.append(ABLATION_TUNNEL_SUBROOT)
    parts.append(tunnel_id)
    return os.path.join(*parts)


def tunnel_characteristics_dir(tunnel_id: str) -> str:
    return os.path.join(tunnel_characteristics_parent_dir(tunnel_id), "characteristics")
