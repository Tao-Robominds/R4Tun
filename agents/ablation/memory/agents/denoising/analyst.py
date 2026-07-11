#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import os
import sys
import requests
from pathlib import Path

_agents_dir = Path(__file__).resolve().parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from memory_ablation_context import (
    load_similar_to_sample_block,
    load_raw_characteristics_pair,
    load_stage_parameters_pretty,
    pipeline_tunnel_data_dir,
    read_required_text,
    strict_output_instructions,
)


class DenoisingAnalyser:
    def __init__(self, tunnel_id):
        self.tunnel_id = tunnel_id
        self.data_dir = pipeline_tunnel_data_dir(tunnel_id)
        self._agent_dir = Path(__file__).resolve().parent
        self.api_key = "app-2YyQbd7yv14XBQCf2DL3bifh"
        self.base_url = "https://api.dify.ai/v1"

    def load_analysis_data(self, model_tag: str = "glm"):
        role_content = read_required_text(self._agent_dir / "role.md", "Role definition")
        sample_raw, tunnel_raw = load_raw_characteristics_pair(self.tunnel_id)
        code_path = Path("sam4tun/agents/denoising.py")
        code_content = read_required_text(code_path, "Sample denoising code")
        archive_name = "parameters_denoising.json"
        params_json, params_source = load_stage_parameters_pretty(self.tunnel_id, archive_name, model_tag)
        return {
            "role": role_content,
            "sample_raw": sample_raw,
            "tunnel_raw": tunnel_raw,
            "sample_code": code_content,
            "parameters": params_json,
            "parameters_source": params_source,
            "archive_filename": archive_name,
        }

    def build_llm_prompt_markdown(self, model_tag: str = "glm") -> str:
        """Exact LLM user message; exported to parameters_denoising.md for copy-paste."""
        ctx = self.load_analysis_data(model_tag)
        similar_block = load_similar_to_sample_block(self.tunnel_id)
        similar_section = (similar_block + "\n\n") if similar_block else ""
        return f"""
# ROLE
{ctx["role"]}

{similar_section}
# SAMPLE TUNNEL — RAW CHARACTERISTICS (reference)
```json
{ctx["sample_raw"]}
```

# TARGET TUNNEL — RAW CHARACTERISTICS (tunnel_id={self.tunnel_id})
```json
{ctx["tunnel_raw"]}
```

# REFERENCE DENOISING PARAMETERS
{ctx["parameters_source"]}

```json
{ctx["parameters"]}
```

# PIPELINE CODE (reference)
```python
{ctx["sample_code"]}
```

{strict_output_instructions(ctx["archive_filename"], ctx["parameters"])}
""".strip()

    def get_denoising_recommendations(self):
        comprehensive_query = self.build_llm_prompt_markdown()

        response = requests.post(
            f"{self.base_url}/chat-messages",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "inputs": {"temperature": 0},
                "query": comprehensive_query,
                "response_mode": "streaming",
                "conversation_id": "",
                "user": f"denoising_analyser_{self.tunnel_id}",
                "files": [],
            },
        )

        result = ""
        for line in response.iter_lines():
            if line and line.decode("utf-8").startswith("data: "):
                try:
                    chunk = json.loads(line.decode("utf-8")[6:])
                    if chunk.get("event") == "agent_message":
                        result += chunk.get("answer", "")
                except Exception:
                    continue

        os.makedirs(self.data_dir / "analysis", exist_ok=True)
        output_file = self.data_dir / "analysis" / "denoising_analysis.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Denoising Analysis Recommendations - {self.tunnel_id}\n\n---\n\n{result}")

        print(f"Results saved to: {output_file}")
        return result


def main():
    import sys

    tunnel_id = sys.argv[1] if len(sys.argv) > 1 else "1-1"
    analyser = DenoisingAnalyser(tunnel_id)
    print(analyser.get_denoising_recommendations())


if __name__ == "__main__":
    main()
