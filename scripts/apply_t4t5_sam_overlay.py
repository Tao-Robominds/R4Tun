#!/usr/bin/env python3
"""Overlay T4/T5 7-segment SAM fields onto staged LLM params (sam4tun/agents/parameters only)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONDITIONS = ("memory", "memory+state", "memory+state+knowledge")
TUNNELS = ("4-1", "5-1")
MODELS = ("opus4.6", "gpt5.4", "gemini3flash")
SUFFIX = {"memory": "_m_", "memory+state": "_m_s_", "memory+state+knowledge": "_m_s_k_"}
OVERLAY_KEYS = (
    "segment_per_ring",
    "segment_order",
    "segment_width",
    "K_height",
    "AB_height",
    "processing",
)


def main() -> None:
    manifest_lines = [
        "# T4/T5 SAM parameter overlay",
        "",
        "Applied 7-segment rules geometry to staged copies under `sam4tun/agents/parameters/`.",
        "Original LLM assets under `agents/ablation/` are untouched.",
        "",
        "| staged file | fields overlaid | angle kept |",
        "|-------------|-----------------|------------|",
    ]
    count = 0
    for cond in CONDITIONS:
        for tunnel in TUNNELS:
            rules = json.loads(
                (REPO / "sam4tun/agents/parameters/rules" / tunnel / "parameters_sam.json").read_text()
            )
            for model in MODELS:
                path = (
                    REPO / "sam4tun/agents/parameters" / cond / tunnel
                    / f"parameters_sam{SUFFIX[cond]}{model}.json"
                )
                if not path.is_file():
                    continue
                data = json.loads(path.read_text())
                angle = data.get("angle")
                for key in OVERLAY_KEYS:
                    if key in rules:
                        data[key] = rules[key]
                if angle is not None:
                    data["angle"] = angle
                path.write_text(json.dumps(data, indent=2) + "\n")
                count += 1
                manifest_lines.append(
                    f"| `{path.relative_to(REPO)}` | {', '.join(OVERLAY_KEYS)} | {angle} |"
                )
    manifest_lines.extend(["", f"Total files overlaid: {count}", ""])
    out = REPO / "data/ablation/t4_t5_sam_overlay.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(manifest_lines))
    print(f"Overlaid {count} SAM files -> {out}")


if __name__ == "__main__":
    main()
