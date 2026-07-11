"""
Helpers for **memory+state ablation analyst** prompts.

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

# Repo root: .../agents/ablation/memory+state/agents/<this file> → 4 parents up
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

ABLATION_FOLDER = "memory+state"
PARAM_BASE = Path("sam4tun/agents/parameters") / ABLATION_FOLDER
SAMPLE_PARAM_DIR = Path("sam4tun/agents/parameters/sample")
SHARED_SIMILAR_PATH = Path("agents/ablation/shared/similar_to_sample.md")
SHARED_T3_PATH = Path("agents/ablation/shared/t3_continuous.md")

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


def load_similar_to_sample_block(tunnel_id: str) -> str:
    """T1/T2 sample-like performance guidance (not 3-*)."""
    if not (tunnel_id.startswith("1-") or tunnel_id.startswith("2-")):
        return ""
    text = read_required_text(SHARED_SIMILAR_PATH, "SIMILAR_TO_SAMPLE guidance")
    return f"# SIMILAR_TO_SAMPLE REGIME (T1/T2 — target mIoU ≥ 0.70)\n\n{text}"


def load_t3_continuous_block(tunnel_id: str) -> str:
    """T3 continuous-joint guidance (3-*)."""
    if not tunnel_id.startswith("3-"):
        return ""
    text = read_required_text(SHARED_T3_PATH, "T3_CONTINUOUS guidance")
    return f"# T3_CONTINUOUS REGIME (3-* continuous joints)\n\n{text}"


def load_regime_blocks(tunnel_id: str) -> str:
    """Inject SIMILAR_TO_SAMPLE or T3_CONTINUOUS (mutually exclusive)."""
    parts = [b for b in (load_similar_to_sample_block(tunnel_id), load_t3_continuous_block(tunnel_id)) if b]
    return "\n\n".join(parts)


def t3_denoise_method_note(tunnel_id: str, archive_filename: str) -> str:
    """Percentile-driven denoise reasoning for 3-* (no external param anchors)."""
    if not tunnel_id.startswith("3-") or archive_filename != "parameters_denoising.json":
        return ""
    return (
        "**T3 denoise method:** if `p50(r) > d/2 + 0.15`, rules mask is too narrow — use "
        "`mask_r_low = p10 − 0.02`, `mask_r_high = p99 + 0.02`. "
        "Require `wall_pct ≥ 50%`. Never set `mask_r_high < p99`. "
        "`default_cutoff_z = mask_r_high + 0.02`.\n"
    )


def t3_enhancing_coverage_note(tunnel_id: str, archive_filename: str) -> str:
    """Depth-map coverage reasoning for 3-* enhancing (no external param anchors)."""
    if not tunnel_id.startswith("3-") or archive_filename != "parameters_enhancing.json":
        return ""
    return (
        "**T3 depth-map coverage:** run only when denoise retention ≥ 50%. "
        "Estimate `point_density = valid_points / ((h_span/0.005)×(θ_span/0.005))`. "
        "If < 0.08 or edge white > center: increase `window_size` (11–13), "
        "set upsampling stage1 ≈ 0.85×median_NN (typical range 0.05–0.08 on T3), lower `depth_threshold_low`, "
        "set `n_segment_end = ring_count − 1` when ring_count is known.\n"
    )


def t3_k_alignment_note(tunnel_id: str, archive_filename: str) -> str:
    """K uniform Y reasoning for 3-* detecting."""
    if not tunnel_id.startswith("3-") or archive_filename != "parameters_detecting.json":
        return ""
    return (
        "**T3 K uniformity:** one anchor defines Y* for all rings (one-K-knows-all). "
        "Tune horizontal Hough to detect the K seam; pipeline snaps **all** rings to median Y* "
        "from ≥1 anchor (`midpoint`/`horizontal`/slope). "
        "Target Y_std < 10 px, max |Y − Y*| = 0, assume < 10% pre-snap.\n"
    )


def t3_sam_k_uniform_note(tunnel_id: str, archive_filename: str) -> str:
    """K template uniformity reasoning for 3-* SAM."""
    if not tunnel_id.startswith("3-") or archive_filename != "parameters_sam.json":
        return ""
    return (
        "**T3 SAM K template:** K centre Y is identical every ring; only X shifts per column. "
        "Tune K_height, angle, crop_margin, y_bounds **once** — not per-ring Y. "
        "When K IoU < 0.65 with Y_std < 10, apply **K_HEIGHT_OVERSIZE** (cot.md): reduce K_height "
        "toward sample/band if mask is visually too tall — never from GT span measurement. "
        "State baseline (3-1-1): K IoU ~0.49, K_height 1137 px (above 1050–1100 band). "
        "Target K-block IoU > 0.65 when detecting Y_std < 10 px.\n"
    )


t3_denoise_anchor_note = t3_denoise_method_note


def load_stage_parameters_pretty(
    tunnel_id: str, archive_filename: str, model_tag: str = "glm"
) -> tuple[str, str]:
    tunnel_dir = PARAM_BASE / tunnel_id
    stem = archive_filename.removesuffix(".json")
    for name in (
        f"{stem}_m_s_{model_tag}.json",
        f"{stem}_m_{model_tag}.json",
        archive_filename,
    ):
        path = tunnel_dir / name
        if path.exists():
            text = read_required_json_pretty(path, f"Parameters at {path}")
            return text, f"Archived tunnel parameters (`{path.as_posix()}`)."
    baseline_path = SAMPLE_PARAM_DIR / archive_filename
    if baseline_path.exists():
        text = read_required_json_pretty(baseline_path, f"Sample parameters at {baseline_path}")
        note = (
            f"Frozen sam4tun sample parameters (`{baseline_path.as_posix()}`); "
            f"for T1/T2 retain these unless state shows a named failure."
        )
        return text, note
    raise FileNotFoundError(
        f"No parameter file for tunnel {tunnel_id}, stage {archive_filename}."
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


def strict_output_instructions(archive_filename: str, parameters_pretty: str, has_state: bool = False) -> str:
    """Full block appended to analyst prompts so model output matches ``parameters_*.json`` on disk."""
    table = parameter_json_schema_contract_table(parameters_pretty)

    input_scope = (
        "Use the two raw characteristic JSON blobs, "
        "the **STAGE CHARACTERISTICS** comparison blocks, "
        "the **REFERENCE … PARAMETERS** JSON block above, and the pipeline code."
        if has_state
        else "Use **only** the two raw characteristic JSON blobs, "
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
