#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
from pathlib import Path

_agents_dir = Path(__file__).resolve().parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from memory_state_ablation_context import (
    load_raw_characteristics_pair,
    load_stage_parameters_pretty,
    pipeline_tunnel_data_dir,
    read_required_text,
    strict_output_instructions,
)


class UnfoldingAnalyser:
    def __init__(self, tunnel_id):
        self.tunnel_id = tunnel_id
        self.data_dir = pipeline_tunnel_data_dir(tunnel_id)
        self._agent_dir = Path(__file__).resolve().parent

    COT_PATH = Path("agents/unfolding/cot.md")

    def load_analysis_data(self):
        role_content = read_required_text(self._agent_dir / "role.md", "Role definition")
        cot_content = read_required_text(self.COT_PATH, "Chain-of-thought instructions")
        sample_raw, tunnel_raw = load_raw_characteristics_pair(self.tunnel_id)
        code_path = Path("sam4tun/1_upfolding.py")
        code_content = read_required_text(code_path, "Sample unfolding code")
        archive_name = "parameters_unfolding.json"
        params_json, params_source = load_stage_parameters_pretty(self.tunnel_id, archive_name)
        return {
            "role": role_content,
            "cot": cot_content,
            "sample_raw": sample_raw,
            "tunnel_raw": tunnel_raw,
            "sample_code": code_content,
            "parameters": params_json,
            "parameters_source": params_source,
            "archive_filename": archive_name,
        }

    def build_llm_prompt_markdown(self, state_context: str = "") -> str:
        ctx = self.load_analysis_data()
        parts = [
            f"# ROLE\n{ctx['role']}",
            f"# ANALYSIS METHODOLOGY\n{ctx['cot']}",
            f"# SAMPLE TUNNEL — RAW CHARACTERISTICS (reference)\n```json\n{ctx['sample_raw']}\n```",
            f"# TARGET TUNNEL — RAW CHARACTERISTICS (tunnel_id={self.tunnel_id})\n```json\n{ctx['tunnel_raw']}\n```",
            f"# REFERENCE UNFOLDING PARAMETERS\n{ctx['parameters_source']}\n\n```json\n{ctx['parameters']}\n```",
            f"# PIPELINE CODE (reference)\n```python\n{ctx['sample_code']}\n```",
            strict_output_instructions(ctx["archive_filename"], ctx["parameters"], has_state=False),
        ]
        return "\n\n".join(parts)


def main():
    tunnel_id = sys.argv[1] if len(sys.argv) > 1 else "1-1"
    analyser = UnfoldingAnalyser(tunnel_id)
    print(analyser.build_llm_prompt_markdown())


if __name__ == "__main__":
    main()
