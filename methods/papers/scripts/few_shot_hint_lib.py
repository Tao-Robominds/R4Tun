"""Few-shot parameter exemplars for continuous T3 hint experiments."""
from __future__ import annotations

import json
from pathlib import Path

from repeatability_common import (
    ABLATION_FOLDER,
    extract_miou,
    param_json_name,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_EXEMPLARS = ("1-5", "2-5")
BEST_EXEMPLARS = ("1-3", "2-2")

STAGE_ARCHIVE = {
    "unfolding": "parameters_unfolding.json",
    "denoising": "parameters_denoising.json",
    "enhancing": "parameters_enhancing.json",
    "detecting": "parameters_detecting.json",
    "sam": "parameters_sam.json",
}

EXEMPLAR_MIou: dict[str, float] = {
    "1-5": 0.630,
    "2-5": 0.669,
    "1-3": 0.658,
    "2-2": 0.685,
}


def exemplar_param_dir(tunnel: str, model: str = "opus4.6") -> Path:
  roots = [
      REPO_ROOT / "logs" / tunnel / "regular_hint" / model / "parameters",
      REPO_ROOT / "logs" / tunnel / "regular_hint_v3" / model / "parameters",
      REPO_ROOT / "data" / "regular_hint_loop" / "L0" / tunnel,
  ]
  for r in roots:
      if r.is_dir():
          return r
  return roots[0]


def load_exemplar_stage_params(
    exemplar_tunnel: str,
    stage: str,
    model: str = "opus4.6",
) -> dict | None:
    param_dir = exemplar_param_dir(exemplar_tunnel, model)
    name = param_json_name(stage, model)
    path = param_dir / name
    if not path.is_file():
        archive = STAGE_ARCHIVE.get(stage)
        if archive:
            alt = param_dir / archive
            if alt.is_file():
                path = alt
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def load_exemplar_params(
    stage: str,
    model: str = "opus4.6",
    exemplars: tuple[str, ...] = DEFAULT_EXEMPLARS,
) -> list[tuple[str, float | None, dict]]:
    out: list[tuple[str, float | None, dict]] = []
    for tid in exemplars:
        params = load_exemplar_stage_params(tid, stage, model)
        if params is None:
            continue
        miou = EXEMPLAR_MIou.get(tid)
        l0 = REPO_ROOT / "data" / "regular_hint_loop" / "L0" / tid
        measured = extract_miou(l0)
        if measured is not None:
            miou = measured
        out.append((tid, miou, params))
    return out


def build_few_shot_block(
    stage: str,
    model: str = "opus4.6",
    exemplars: tuple[str, ...] = DEFAULT_EXEMPLARS,
) -> str:
    rows = load_exemplar_params(stage, model, exemplars)
    if not rows:
        return ""

    lines = [
        "# FEW-SHOT EXEMPLARS — ADAPTED STAGGERED TUNNELS (GT-free)",
        "",
        "These are successful parameter sets from staggered regular tunnels (`1-*`, `2-*`).",
        "The target is a **continuous** tunnel (`3-*`): K-Y is **constant** across rings",
        "(no two-level zigzag). Adapt by analogy; keep SAM4Tun defaults unless target",
        "state evidence justifies a change similar to an exemplar.",
        "",
    ]
    for tid, miou, params in rows:
        fam = "T1" if tid.startswith("1-") else "T2"
        miou_s = f"{miou:.3f}" if miou is not None else "n/a"
        lines.append(f"## Exemplar {tid} ({fam} staggered, mIoU={miou_s})")
        lines.append(f"```json\n{json.dumps(params, indent=2)}\n```")
        lines.append("")
    return "\n".join(lines).strip()


def seed_exemplar_params_to_tunnel(
    tunnel: str,
    stages: list[str],
    exemplars: tuple[str, ...],
    model: str = "opus4.6",
    *,
    primary: str | None = None,
) -> None:
    """Copy exemplar stage JSONs into PARAM_BASE/tunnel (few-shot without LLM)."""
    pick = primary or (exemplars[-1] if exemplars else None)
    if not pick:
        return
    param_dir = REPO_ROOT / "agents" / "ablation" / "memory+state+knowledge" / "parameters" / tunnel
    param_dir.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        params = load_exemplar_stage_params(pick, stage, model)
        if params is None:
            continue
        out = param_dir / param_json_name(stage, model)
        out.write_text(json.dumps(params, indent=2) + "\n")


def exemplars_for_level(level: str) -> tuple[str, ...]:
    if level in ("T3", "T4", "T5"):
        return BEST_EXEMPLARS
    return DEFAULT_EXEMPLARS
