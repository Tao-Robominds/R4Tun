#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Reflection SAM analyst: m_s_k outputs + intrinsic metrics → LLM prompt (markdown)."""

import sys
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent
_REFL_AGENTS = _AGENT_DIR.parent
_REPO_ROOT = Path(__file__).resolve().parents[5]
_MSK_AGENTS = _REPO_ROOT / "configurable/ablation/memory+state+knowledge/agents"

for p in (_REFL_AGENTS, _MSK_AGENTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from intrinsic_metrics import intrinsic_report_json  # noqa: E402
from memory_state_ablation_context import (  # noqa: E402
    build_state_comparison_block,
    load_raw_characteristics_pair,
    load_stage_parameters_pretty,
    read_required_text,
    strict_output_instructions,
)


class ReflectionSegmentingAnalyser:
    STAGE_NAME = "sam"

    def __init__(self, tunnel_id: str):
        self.tunnel_id = tunnel_id
        self._agent_dir = _AGENT_DIR

    def load_analysis_data(self):
        role_content = read_required_text(self._agent_dir / "role.md", "Role definition")
        cot_content = read_required_text(self._agent_dir / "cot.md", "Reflection CoT")
        knowledge_content = read_required_text(self._agent_dir / "knowledge.md", "Domain knowledge")
        sample_raw, tunnel_raw = load_raw_characteristics_pair(self.tunnel_id)
        code_path = _REPO_ROOT / "sam4tun/4-2_sam.py"
        code_content = read_required_text(code_path, "Sample SAM code")
        archive_name = "parameters_sam.json"
        params_json, params_source = load_stage_parameters_pretty(self.tunnel_id, archive_name)
        intrinsic = intrinsic_report_json(self.tunnel_id)
        return {
            "role": role_content,
            "cot": cot_content,
            "knowledge": knowledge_content,
            "sample_raw": sample_raw,
            "tunnel_raw": tunnel_raw,
            "sample_code": code_content,
            "parameters": params_json,
            "parameters_source": params_source,
            "intrinsic": intrinsic,
            "archive_filename": "parameters_sam_r_opus4.6.json",
        }

    def build_llm_prompt_markdown(self, state_context: str = "") -> str:
        ctx = self.load_analysis_data()
        if not state_context:
            state_context = build_state_comparison_block(self.tunnel_id, self.STAGE_NAME)
        has_state = bool(state_context.strip())
        parts = [
            f"# ROLE\n{ctx['role']}",
            f"# REFLECTION TASK\nAdjust **SAM** (`parameters_sam.json`) only. Upstream stages are fixed to the m_s_k run. "
            f"Use the intrinsic report (GT-free). Output JSON for `{ctx['archive_filename']}` "
            "(same schema as reference below).",
            f"# ANALYSIS METHODOLOGY\n{ctx['cot']}",
            f"# DOMAIN KNOWLEDGE\n{ctx['knowledge']}",
            f"# INTRINSIC QUALITY REPORT (m_s_k output, auto-computed)\n```json\n{ctx['intrinsic']}\n```",
            f"# SAMPLE TUNNEL — RAW CHARACTERISTICS (reference)\n```json\n{ctx['sample_raw']}\n```",
            f"# TARGET TUNNEL — RAW CHARACTERISTICS (tunnel_id={self.tunnel_id})\n```json\n{ctx['tunnel_raw']}\n```",
        ]
        if has_state:
            parts.append(state_context)
        parts += [
            f"# REFERENCE SAM PARAMETERS (m_s_k baseline — tune from here)\n{ctx['parameters_source']}\n\n```json\n{ctx['parameters']}\n```",
            f"# PIPELINE CODE (reference)\n```python\n{ctx['sample_code']}\n```",
            strict_output_instructions(
                ctx["archive_filename"],
                ctx["parameters"],
                has_state=has_state,
                has_knowledge=True,
            ),
        ]
        return "\n\n".join(parts)


def main():
    tunnel_id = sys.argv[1] if len(sys.argv) > 1 else "1-1"
    print(ReflectionSegmentingAnalyser(tunnel_id).build_llm_prompt_markdown())


if __name__ == "__main__":
    main()
