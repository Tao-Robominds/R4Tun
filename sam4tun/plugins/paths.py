"""
Layout convention — see ``methods/plans/steps/00_methodology_chain.md``.

- Pipeline CSVs (default active run): ``data/{tunnel_id}/...``
- Reference sample JSON: ``data/sample/characteristics/``
- Raw / pre-pipeline subset JSON: ``data/ablation/{subroot}/{tunnel_id}/characteristics/``
  Default subroot is ``ABLATION_TUNNEL_SUBROOT`` (``memory`` for level 1 / ``-m``). Override with env
  ``R4TUN_ABLATION_TUNNEL_SUBROOT`` (e.g. ``memory+state``) when writing raw JSON for another ablation tree.

**Ablation full-run roots** (each ``{tunnel_id}/`` tree mirrors ``data/sample/``):

| Parameter archive suffix | Folder under ``data/ablation/`` |
|--------------------------|-----------------------------------|
| ``-m`` | ``memory`` |
| ``-m+s`` | ``memory+state`` |
| ``-m+s+k`` | ``memory+state+knowledge`` |
"""

import os

# Subfolder under data/ablation/ for raw + level-1 (-m) tunnel tree; must match first row in table above.
ABLATION_TUNNEL_SUBROOT = "memory"

# Canonical names for full pipeline output dirs (under data/ablation/<name>/{tunnel_id}/).
ABLATION_OUTPUT_MEMORY = "memory"
ABLATION_OUTPUT_MEMORY_STATE = "memory+state"
ABLATION_OUTPUT_MEMORY_STATE_KNOWLEDGE = "memory+state+knowledge"

# Maps archived parameter suffix (e.g. "-m+s") to output folder name under data/ablation/.
ABLATION_SUFFIX_TO_OUTPUT_ROOT = {
    "-m": ABLATION_OUTPUT_MEMORY,
    "-m+s": ABLATION_OUTPUT_MEMORY_STATE,
    "-m+s+k": ABLATION_OUTPUT_MEMORY_STATE_KNOWLEDGE,
}


def tunnel_pipeline_dir(tunnel_id: str) -> str:
    """
    Same artefact root as ``configurable.pipeline_data.tunnel_output_relpath`` (must match
    ``R4TUN_PIPELINE_OUT_PREFIX``). No default under ``data/<id>/`` without the env var.
    """
    p = (os.environ.get("R4TUN_PIPELINE_OUT_PREFIX") or "").strip().rstrip("/")
    if not p:
        raise RuntimeError(
            "R4TUN_PIPELINE_OUT_PREFIX is not set. "
            "Export it to match your E2E run (e.g. data/ablation/memory for memory ablation)."
        )
    return os.path.join(p, tunnel_id)


def ablation_run_data_dir(tunnel_id: str, output_root_name: str) -> str:
    """Full pipeline artefacts for one ablation condition, e.g. output_root_name=ABLATION_OUTPUT_MEMORY."""
    return os.path.join("data", "ablation", output_root_name, tunnel_id)


def tunnel_characteristics_subroot() -> str:
    """Folder under ``data/ablation/`` for per-tunnel ``characteristics/`` (default ``memory``)."""
    override = (os.environ.get("R4TUN_ABLATION_TUNNEL_SUBROOT") or "").strip()
    if override:
        return override
    return ABLATION_TUNNEL_SUBROOT


def tunnel_characteristics_parent_dir(tunnel_id: str) -> str:
    """Parent directory that contains the ``characteristics`` subfolder (plugin writers pass this as base_dir)."""
    if tunnel_id == "sample":
        return os.path.join("data", "sample")
    parts = ["data", "ablation"]
    subroot = tunnel_characteristics_subroot()
    if subroot:
        parts.append(subroot)
    parts.append(tunnel_id)
    return os.path.join(*parts)


def tunnel_characteristics_dir(tunnel_id: str) -> str:
    return os.path.join(tunnel_characteristics_parent_dir(tunnel_id), "characteristics")
