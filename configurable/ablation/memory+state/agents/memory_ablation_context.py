"""
Helpers for **memory ablation analyst** prompts (not for ``run_agents.sh`` / configurable stages).

Loads sample vs tunnel ``raw_characteristics.json``, loads per-stage **reference** ``parameters_*.json``
from ``configurable/ablation/memory/parameters/<id>/`` or default ``configurable/sample/``, and builds
strict JSON-output instructions (leaf path → type) so LLM inference matches on-disk parameter shape.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from sam4tun.plugins.paths import tunnel_characteristics_dir

SAMPLE_RAW_PATH = Path("data/sample/characteristics/raw_characteristics.json")
RAW_FILENAME = "raw_characteristics.json"


def pipeline_tunnel_data_dir(tunnel_id: str) -> Path:
    """Same layout as configurable ``pipeline_data`` (memory ablation → data/ablation/memory/<id>)."""
    pfx = (os.environ.get("R4TUN_PIPELINE_OUT_PREFIX") or "data/ablation/memory").strip().rstrip("/") or "data/ablation/memory"
    return Path(f"{pfx}/{tunnel_id}")


def read_required_text(path: Path, description: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}")
    content = path.read_text()
    if not content.strip():
        raise ValueError(f"{description} at {path} is empty")
    return content


def read_required_json_pretty(path: Path, description: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{description} at {path} contains invalid JSON: {exc}") from exc
    return json.dumps(data, indent=2)


def load_raw_characteristics_pair(tunnel_id: str) -> tuple[str, str]:
    sample = read_required_json_pretty(SAMPLE_RAW_PATH, "Sample raw characteristics")
    tunnel_path = Path(tunnel_characteristics_dir(tunnel_id)) / RAW_FILENAME
    tunnel = read_required_json_pretty(tunnel_path, "Tunnel raw characteristics")
    return sample, tunnel


def load_stage_parameters_pretty(tunnel_id: str, archive_filename: str) -> tuple[str, str]:
    """
    Prefer ``configurable/ablation/memory/parameters/<tunnel_id>/<archive_filename>`` so the
    prompt + schema table match the on-disk archive; else ``configurable/sample/<archive_filename>``.

    Returns (pretty-printed JSON, header note for the prompt).
    """
    archive_path = Path("configurable/ablation/memory/parameters") / tunnel_id / archive_filename
    sample_path = Path("configurable/sample") / archive_filename
    if archive_path.exists():
        text = read_required_json_pretty(archive_path, f"Archived parameters at {archive_path}")
        note = f"Archived tunnel parameters (same file you will save as `{archive_path.as_posix()}`)."
        return text, note
    text = read_required_json_pretty(sample_path, f"Sample parameters at {sample_path}")
    note = f"Sample parameters (`{sample_path.as_posix()}`); no archive yet for tunnel `{tunnel_id}`."
    return text, note


def _scalar_type_name(v: object) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    return type(v).__name__


def _collect_leaf_paths(obj: object, prefix: str, rows: list[tuple[str, str]]) -> None:
    """Document every leaf with JSON path and type (matches archive JSON shape)."""
    if isinstance(obj, dict):
        for key in sorted(obj.keys()):
            path = f"{prefix}.{key}" if prefix else str(key)
            val = obj[key]
            if isinstance(val, dict):
                _collect_leaf_paths(val, path, rows)
            elif isinstance(val, list):
                if not val:
                    rows.append((path, "array (empty `[]`)"))
                elif isinstance(val[0], dict):
                    n = len(val)
                    rows.append((path, f"object array — **{n}** elements, each with same keys as below"))
                    _collect_leaf_paths(val[0], f"{path}[0]", rows)
                else:
                    el_t = _scalar_type_name(val[0])
                    rows.append((path, f"array[{len(val)}] of {el_t}"))
            else:
                rows.append((path, _scalar_type_name(val)))


def parameter_json_schema_contract_table(parameters_pretty: str) -> str:
    """Markdown table: leaf path → type, derived from sample parameters JSON."""
    data = json.loads(parameters_pretty)
    rows: list[tuple[str, str]] = []
    _collect_leaf_paths(data, "", rows)
    lines = [
        "| JSON path (must exist with this type) | Type |",
        "| --- | --- |",
    ]
    for path, typ in rows:
        lines.append(f"| `{path}` | {typ} |")
    return "\n".join(lines)


def strict_output_instructions(archive_filename: str, parameters_pretty: str) -> str:
    """
    Full block appended to analyst prompts so model output matches ``parameters_*.json`` on disk.
    ``archive_filename`` e.g. ``parameters_unfolding.json``.
    """
    table = parameter_json_schema_contract_table(parameters_pretty)
    return f"""
## Input scope
Use **only** the two raw characteristic JSON blobs, the **REFERENCE … PARAMETERS** JSON block above, and the pipeline code. Do not assume unfolded / denoised / enhanced / detected summaries.

## Required final output (must match `{archive_filename}`)
Your reply must end with **exactly one** markdown code fence labelled `json`, containing **one** JSON object and nothing else inside the fence.

That object must:
1. Parse with `json.loads` with **no** trailing commas or comments.
2. Have the **same tree of keys** as the **REFERENCE … PARAMETERS** JSON block above at every level — **no added keys, no removed keys, no renamed keys**.
3. Match **types** at every leaf path listed below (object vs array vs number vs integer vs boolean vs string). Preserve **array lengths** exactly.
4. Change **only** values where raw evidence justifies it; otherwise keep the reference numerics / booleans / strings unchanged.
5. For **string** leaves (e.g. segment codes in `segment_order`), keep the same literals unless a change is explicitly justified; **never** invent new keys under `processing`, `prompt_points`, or `template_mask`.

### Leaf paths and types (from reference JSON above)
{table}

### Before the code fence
At most a **short** prose note (optional); **no** CoT section headers. The fence must contain the full parameters object so it can be copied into `{archive_filename}`.
""".strip()


