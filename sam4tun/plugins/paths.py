"""
Layout convention:
- Pipeline CSVs and artefacts: data/{tunnel_id}/...
- Characterisation JSON (plugins + analysts): data/ablation/{tunnel_id}/characteristics/
"""

import os


def tunnel_pipeline_dir(tunnel_id: str) -> str:
    return os.path.join("data", tunnel_id)


def tunnel_ablation_dir(tunnel_id: str) -> str:
    return os.path.join("data", "ablation", tunnel_id)


def tunnel_characteristics_dir(tunnel_id: str) -> str:
    return os.path.join(tunnel_ablation_dir(tunnel_id), "characteristics")
