#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import requests


class ReflectingAnalyser:
    """
    Analyst stage for the reflecting agent.

    Mirrors the structure of agents/segmenting/analyst.py but operates on:
    - final.csv coverage
    - tunnel characteristics
    - detected prompt points
    - tunnel-specific parameters_sam.json
    """

    def __init__(self, tunnel_id: str):
        self.tunnel_id = tunnel_id
        self.data_dir = Path(f"data/{tunnel_id}")
        self.analysis_dir = self.data_dir / "analysis"
        self.project_root = Path.cwd()

        # Dify configuration (prefer env var, fall back to existing key if needed)
        self.api_key = os.getenv("DIFY_API_KEY", "app-bKyUjJtUZhrkbsEkh5AvZpzE")
        self.base_url = "https://api.dify.ai/v1"

    # ---------- helpers ----------

    def _read_required_text(self, path: Path, description: str) -> str:
        if not path.exists():
            raise FileNotFoundError(f"{description} not found at {path}")
        content = path.read_text()
        if not content.strip():
            raise ValueError(f"{description} at {path} is empty")
        return content

    def _read_optional_json(self, path: Path, description: str):
        if not path.exists():
            return f"{description} not available at {path}"
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError as exc:
            return f"{description} at {path} contains invalid JSON: {exc}"

    # ---------- domain-specific loaders ----------

    def _load_coverage_summary(self):
        """Compute coverage metrics from final.csv, mirroring SAMEvolver.analyze_point_coverage."""
        final_csv_path = self.data_dir / "final.csv"
        if not final_csv_path.exists():
            raise FileNotFoundError(f"final.csv not found at {final_csv_path}")

        df = pd.read_csv(final_csv_path)

        class_names = {
            0: "Background",
            1: "K-block",
            2: "B1-block",
            3: "A1-block",
            4: "A2-block",
            5: "A3-block",
            6: "B2-block",
        }

        total_points = len(df)
        point_counts = {}
        point_percentages = {}

        for class_id, class_name in class_names.items():
            if class_id in df["pred"].values:
                count = len(df[df["pred"] == class_id])
                percentage = (count / total_points) * 100 if total_points > 0 else 0
            else:
                count = 0
                percentage = 0.0
            point_counts[class_name] = count
            point_percentages[class_name] = percentage

        # Exclude background for statistics
        block_counts = {k: v for k, v in point_counts.items() if k != "Background"}

        if block_counts:
            avg_points = np.mean(list(block_counts.values()))
            std_points = np.std(list(block_counts.values()))
            min_points = min(block_counts.values())
            max_points = max(block_counts.values())
            weakest_block = min(block_counts.keys(), key=lambda k: block_counts[k])
            critical_threshold = avg_points * 0.3
            critical_blocks = [k for k, v in block_counts.items() if v < critical_threshold]
            cv = (std_points / avg_points) * 100 if avg_points > 0 else 0
            if cv < 20:
                quality = "excellent"
            elif cv < 40:
                quality = "good"
            else:
                quality = "poor"
        else:
            avg_points = std_points = min_points = max_points = 0
            weakest_block = "None"
            critical_threshold = 0
            critical_blocks = []
            cv = 0
            quality = "poor"

        stats = {
            "average_points_per_block": avg_points,
            "std_points_per_block": std_points,
            "coefficient_of_variation": cv,
            "minimum_points": min_points,
            "maximum_points": max_points,
            "weakest_block": weakest_block,
            "critical_threshold": critical_threshold,
            "critical_blocks": critical_blocks,
            "coverage_quality": quality,
        }

        # Human-readable summary string
        summary_lines = [
            f"Tunnel ID: {self.tunnel_id}",
            f"Total points: {total_points:,}",
            "",
            "Per-block coverage:",
        ]
        for block, count in point_counts.items():
            if block == "Background":
                continue
            pct = point_percentages[block]
            status = "CRITICAL" if block in critical_blocks else "OK"
            summary_lines.append(f"- {block}: {count:,} points ({pct:.1f}%) [{status}]")

        summary_lines.extend(
            [
                "",
                "Global statistics:",
                f"- Average points per block: {avg_points:.0f}",
                f"- Weakest block: {weakest_block} ({min_points:,} points)",
                f"- Critical threshold: {critical_threshold:.0f} points",
                f"- Critical blocks: {critical_blocks}",
                f"- Coefficient of variation: {cv:.1f}% ({quality})",
            ]
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "point_counts": point_counts,
            "point_percentages": point_percentages,
            "statistics": stats,
            "text_summary": "\n".join(summary_lines),
        }

    def load_context(self):
        """Load all context needed for the reflecting analysis prompt."""
        # Role / instructions / knowledge
        role_path = Path("agents/reflecting/role.md")
        cot_path = Path("agents/reflecting/cot.md")
        knowledge_path = Path("agents/reflecting/knowledge.md")

        role_content = self._read_required_text(role_path, "Reflecting role definition")
        cot_content = self._read_required_text(cot_path, "Reflecting chain-of-thought instructions")
        knowledge_content = self._read_required_text(knowledge_path, "Reflecting knowledge base")

        # Coverage analysis
        coverage = self._load_coverage_summary()

        # Tunnel characteristics (algorithm 4) if available
        characteristics_path = (
            Path("data/ablation") / self.tunnel_id / "characteristics" / "algorithm4_characteristics.json"
        )
        characteristics_json = self._read_optional_json(
            characteristics_path, "Algorithm4 tunnel characteristics"
        )

        # Detected prompt points summary (optional, light-weight stats)
        detected_path = self.data_dir / "detected.csv"
        if detected_path.exists():
            try:
                df_det = pd.read_csv(detected_path)
                detected_summary = {
                    "total_prompt_points": int(len(df_det)),
                    "types": df_det["Type"].value_counts().to_dict() if "Type" in df_det.columns else {},
                }
                detected_json = json.dumps(detected_summary, indent=2)
            except Exception as exc:
                detected_json = f"Error summarising detected.csv: {exc}"
        else:
            detected_json = "No detected.csv available for this tunnel."

        # Current SAM parameters (baseline)
        params_path = Path(f"configurable/{self.tunnel_id}/parameters_sam.json")
        params_json = self._read_optional_json(
            params_path, "Tunnel-specific parameters_sam.json"
        )

        return {
            "role": role_content,
            "cot": cot_content,
            "knowledge": knowledge_content,
            "coverage": coverage,
            "characteristics": characteristics_json,
            "detected": detected_json,
            "parameters": params_json,
        }

    # ---------- main API ----------

    def get_reflecting_analysis(self) -> str:
        """Call Dify to produce reflecting analysis only (no JSON config)."""
        ctx = self.load_context()
        cov = ctx["coverage"]

        comprehensive_query = f"""
# ROLE
{ctx['role']}

# DOMAIN KNOWLEDGE
{ctx['knowledge']}

# COVERAGE ANALYSIS (AUTO-COMPUTED)
The following coverage statistics are computed from final.csv for tunnel {self.tunnel_id}:

{cov['text_summary']}

# TUNNEL CHARACTERISTICS (ALGORITHM 4)
```json
{ctx['characteristics']}
```

# DETECTED PROMPT POINTS SUMMARY
```json
{ctx['detected']}
```

# CURRENT SAM PARAMETERS (parameters_sam.json)
```json
{ctx['parameters']}
```

# ANALYSIS INSTRUCTIONS (CHAIN OF THOUGHT)
{ctx['cot']}

## Additional reflecting-specific guidance
- Focus especially on the weakest block {cov['statistics']['weakest_block']} and any CRITICAL blocks {cov['statistics']['critical_blocks']}.
- You MUST include a dedicated section titled exactly: `### Segment order decision`.
  - In that section, either propose a new `segment_order` that prioritises the weak/CRITICAL blocks,
    or explicitly justify keeping the existing `segment_order` unchanged.
- Do NOT output JSON; your output here should be pure analysis text only. A separate coder will turn your intent into a new parameters_sam.json.
"""

        response = requests.post(
            f"{self.base_url}/chat-messages",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "inputs": {"temperature": 0},
                "query": comprehensive_query,
                "response_mode": "streaming",
                "conversation_id": "",
                "user": f"reflecting_analyst_{self.tunnel_id}",
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

        # Save analysis markdown
        os.makedirs(self.analysis_dir, exist_ok=True)
        out_path = self.analysis_dir / "reflecting_analysis.md"
        with open(out_path, "w") as f:
            f.write(f"# Reflecting Analysis - {self.tunnel_id}\n\n---\n\n{result}")

        print(f"Reflecting analysis saved to: {out_path}")
        return result


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agents/reflecting/analyst.py <tunnel_id>")
        sys.exit(1)

    tunnel_id = sys.argv[1]
    analyser = ReflectingAnalyser(tunnel_id)
    analyser.get_reflecting_analysis()


if __name__ == "__main__":
    main()


