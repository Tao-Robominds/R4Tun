"""
Helpers for **memory+state+knowledge ablation analyst** prompts.

Extends the memory ablation context with stage-wise processed characteristics:
after each pipeline stage, the characteriser plugin extracts a JSON summary which
is compared (sample vs tunnel) and injected into the *next* stage's LLM prompt.

Loads:
- Sample vs tunnel ``raw_characteristics.json`` (same as memory ablation)
- Sample vs tunnel stage characteristics (``unfolded_characteristics.json``, etc.)
- Per-stage **reference** ``parameters_*.json``
- Strict JSON-output instructions (leaf path -> type)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Repo root: .../configurable/ablation/memory+state+knowledge/agents/<this file> → 4 parents up
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sam4tun.plugins.paths import tunnel_characteristics_dir

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RAW_PATH = Path("data/sample/characteristics/raw_characteristics.json")
SAMPLE_CHAR_DIR = Path("data/sample/characteristics")
RAW_FILENAME = "raw_characteristics.json"

STAGE_ORDER = ["unfolding", "denoising", "enhancing", "detecting", "sam"]

STAGE_CHARS_MAP = {
    "unfolding": "unfolded_characteristics.json",
    "denoising": "denoised_characteristics.json",
    "enhancing": "enhanced_characteristics.json",
    "detecting": "detected_characteristics.json",
}

PRIOR_STAGES: dict[str, list[str]] = {
    "unfolding": [],
    "denoising": ["unfolding"],
    "enhancing": ["unfolding", "denoising"],
    "detecting": ["unfolding", "denoising", "enhancing"],
    "sam":       ["unfolding", "denoising", "enhancing", "detecting"],
}

ABLATION_FOLDER = "memory+state+knowledge"
PARAM_BASE = Path("configurable/ablation") / ABLATION_FOLDER / "parameters"

# ---------------------------------------------------------------------------
# File I/O helpers (same as memory ablation)
# ---------------------------------------------------------------------------


def pipeline_tunnel_data_dir(tunnel_id: str) -> Path:
    pfx = (
        os.environ.get("R4TUN_PIPELINE_OUT_PREFIX")
        or f"data/ablation/{ABLATION_FOLDER}"
    ).strip().rstrip("/") or f"data/ablation/{ABLATION_FOLDER}"
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


# ---------------------------------------------------------------------------
# Raw characteristics (memory layer)
# ---------------------------------------------------------------------------


def load_raw_characteristics_pair(tunnel_id: str) -> tuple[str, str]:
    sample = read_required_json_pretty(SAMPLE_RAW_PATH, "Sample raw characteristics")
    tunnel_path = Path(tunnel_characteristics_dir(tunnel_id)) / RAW_FILENAME
    tunnel = read_required_json_pretty(tunnel_path, "Tunnel raw characteristics")
    return sample, tunnel


# ---------------------------------------------------------------------------
# Stage characteristics (state layer)
# ---------------------------------------------------------------------------


def _tunnel_stage_char_path(tunnel_id: str, char_filename: str) -> Path:
    return Path(f"data/ablation/{ABLATION_FOLDER}/{tunnel_id}/characteristics/{char_filename}")


def load_stage_characteristics_pair(
    tunnel_id: str, stage_name: str
) -> tuple[str, str]:
    """Load sample and tunnel stage characteristics for a completed stage."""
    char_filename = STAGE_CHARS_MAP[stage_name]
    sample_path = SAMPLE_CHAR_DIR / char_filename
    tunnel_path = _tunnel_stage_char_path(tunnel_id, char_filename)
    sample = read_required_json_pretty(sample_path, f"Sample {stage_name} characteristics")
    tunnel = read_required_json_pretty(tunnel_path, f"Tunnel {stage_name} characteristics")
    return sample, tunnel


def build_state_comparison_block(tunnel_id: str, current_stage: str) -> str:
    """Build markdown comparing all prior completed stage characteristics (sample vs tunnel).

    Returns empty string for unfolding (no prior stages).
    """
    prior = PRIOR_STAGES.get(current_stage, [])
    if not prior:
        return ""

    sections: list[str] = []
    for stage_name in prior:
        try:
            sample_json, tunnel_json = load_stage_characteristics_pair(tunnel_id, stage_name)
        except FileNotFoundError as e:
            sections.append(f"## After {stage_name.title()}\n\n> Skipped: {e}\n")
            continue

        sections.append(
            f"## After {stage_name.title()}\n\n"
            f"### Sample {stage_name} characteristics\n```json\n{sample_json}\n```\n\n"
            f"### Target tunnel {stage_name} characteristics\n```json\n{tunnel_json}\n```"
        )

    if not sections:
        return ""

    header = "# STAGE CHARACTERISTICS — COMPARISON WITH SAMPLE\n\n"
    return header + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------


def load_stage_parameters_pretty(
    tunnel_id: str, archive_filename: str, model_tag: str = "opus4.6"
) -> tuple[str, str]:
    """
    Search order:
    1. ``configurable/ablation/memory+state+knowledge/parameters/<tunnel_id>/<archive_filename>``
    2. Same dir, with ablation+model suffix (e.g. ``parameters_unfolding_m_s_k_opus4.6.json``)
    3. ``configurable/ablation/sam4tun/<archive_filename>`` (baseline fallback)

    Returns (pretty-printed JSON, header note for the prompt).
    """
    tunnel_dir = PARAM_BASE / tunnel_id

    plain_path = tunnel_dir / archive_filename
    if plain_path.exists():
        text = read_required_json_pretty(plain_path, f"Parameters at {plain_path}")
        note = f"Archived tunnel parameters (`{plain_path.as_posix()}`)."
        return text, note

    stem = archive_filename.removesuffix(".json")
    suffixed_k = tunnel_dir / f"{stem}_m_s_k_{model_tag}.json"
    if suffixed_k.exists():
        text = read_required_json_pretty(suffixed_k, f"Parameters at {suffixed_k}")
        note = f"Archived tunnel parameters (`{suffixed_k.as_posix()}`)."
        return text, note

    suffixed_s = tunnel_dir / f"{stem}_m_s_{model_tag}.json"
    if suffixed_s.exists():
        text = read_required_json_pretty(suffixed_s, f"Parameters at {suffixed_s}")
        note = f"Archived tunnel parameters (`{suffixed_s.as_posix()}`)."
        return text, note

    baseline_path = Path("configurable/ablation/sam4tun") / archive_filename
    if baseline_path.exists():
        text = read_required_json_pretty(baseline_path, f"Baseline parameters at {baseline_path}")
        note = f"Baseline (sam4tun) parameters (`{baseline_path.as_posix()}`); no archive yet for tunnel `{tunnel_id}`."
        return text, note

    raise FileNotFoundError(
        f"No parameter file found for tunnel {tunnel_id}, stage {archive_filename}. "
        f"Tried: {plain_path}, {suffixed_k}, {suffixed_s}, {baseline_path}"
    )


# ---------------------------------------------------------------------------
# Schema contract table
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Strict output instructions
# ---------------------------------------------------------------------------


def strict_output_instructions(
    archive_filename: str,
    parameters_pretty: str,
    has_state: bool = False,
    has_knowledge: bool = False,
) -> str:
    """Full block appended to analyst prompts so model output matches ``parameters_*.json`` on disk."""
    table = parameter_json_schema_contract_table(parameters_pretty)

    knowledge_clause = (
        " the **DOMAIN KNOWLEDGE** section above,"
        if has_knowledge
        else ""
    )

    input_scope = (
        "Use the two raw characteristic JSON blobs, "
        "the **STAGE CHARACTERISTICS** comparison blocks,"
        f"{knowledge_clause} "
        "the **REFERENCE … PARAMETERS** JSON block above, and the pipeline code."
        if has_state
        else "Use **only** the two raw characteristic JSON blobs,"
        f"{knowledge_clause} "
        "the **REFERENCE … PARAMETERS** JSON block above, and the pipeline code. "
        "Do not assume unfolded / denoised / enhanced / detected summaries."
    )

    return f"""
## Input scope
{input_scope}

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
