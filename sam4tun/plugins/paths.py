"""
Layout convention — see ``methods/plans/steps/00_methodology_chain.md``.

- Pipeline CSVs (default active run): ``data/{tunnel_id}/...``
- Reference sample JSON: ``data/sample/characteristics/``
- Raw / pre-pipeline subset JSON: ``data/ablation/{ABLATION_TUNNEL_SUBROOT}/{tunnel_id}/characteristics/``
  (``ABLATION_TUNNEL_SUBROOT`` must stay ``memory`` to match **level 1** / ``-m`` output root.)

**Ablation full-run roots** (each ``{tunnel_id}/`` tree mirrors ``data/sample/``):

| Parameter archive suffix | Folder under ``data/ablation/`` |
|--------------------------|-----------------------------------|
| ``-m`` | ``memory`` |
| ``-m+s`` | ``memory+state`` |
| ``-m+s+k`` | ``memory+state+knowledge`` |
| ``-m+s+k+r`` | ``reflection`` |
"""

import os

# Subfolder under data/ablation/ for raw + level-1 (-m) tunnel tree; must match first row in table above.
ABLATION_TUNNEL_SUBROOT = "memory"

# Canonical names for full pipeline output dirs (under data/ablation/<name>/{tunnel_id}/).
ABLATION_OUTPUT_MEMORY = "memory"
ABLATION_OUTPUT_MEMORY_STATE = "memory+state"
ABLATION_OUTPUT_MEMORY_STATE_KNOWLEDGE = "memory+state+knowledge"
ABLATION_OUTPUT_REFLECTION = "reflection"

# Maps archived parameter suffix (e.g. "-m+s") to output folder name under data/ablation/.
ABLATION_SUFFIX_TO_OUTPUT_ROOT = {
    "-m": ABLATION_OUTPUT_MEMORY,
    "-m+s": ABLATION_OUTPUT_MEMORY_STATE,
    "-m+s+k": ABLATION_OUTPUT_MEMORY_STATE_KNOWLEDGE,
    "-m+s+k+r": ABLATION_OUTPUT_REFLECTION,
}


def tunnel_pipeline_dir(tunnel_id: str) -> str:
    return os.path.join("data", tunnel_id)


def ablation_run_data_dir(tunnel_id: str, output_root_name: str) -> str:
    """Full pipeline artefacts for one ablation condition, e.g. output_root_name=ABLATION_OUTPUT_MEMORY."""
    return os.path.join("data", "ablation", output_root_name, tunnel_id)


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
