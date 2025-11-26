#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import os
from pathlib import Path

import requests


class ReflectingParameterEvolver:
    """
    Coder/evolver stage for reflecting, modelled after agents/segmenting/coder.py.

    - Reads the textual reflecting analysis.
    - Reads the current configurable/{tunnel_id}/parameters_sam.json.
    - Asks Dify to apply the analysis and produce a full, updated parameters_sam.json.
    - Backs up the old file and writes the new one.
    """

    def __init__(self, tunnel_id: str):
        self.tunnel_id = tunnel_id
        self.data_dir = Path(f"data/{tunnel_id}")
        self.analysis_dir = self.data_dir / "analysis"
        self.params_dir = Path(f"configurable/{tunnel_id}")

        self.api_key = os.getenv("DIFY_API_KEY", "app-bKyUjJtUZhrkbsEkh5AvZpzE")
        self.base_url = "https://api.dify.ai/v1"

    # ---------- helpers ----------

    def load_analysis(self) -> str:
        path = self.analysis_dir / "reflecting_analysis.md"
        if not path.exists():
            return (
                "No reflecting analysis recommendations available. "
                "Please run agents/reflecting/analyst.py first."
            )
        return path.read_text()

    def load_current_parameters(self) -> dict:
        params_path = self.params_dir / "parameters_sam.json"
        if not params_path.exists():
            raise FileNotFoundError(
                f"No parameters_sam.json found for tunnel {self.tunnel_id}.\n"
                f"Expected: {params_path}\n"
                "Ensure configurable parameters exist before running the reflecting coder."
            )
        with open(params_path, "r") as f:
            return json.load(f)

    # ---------- Dify interaction ----------

    def evolve_via_dify(self) -> str | None:
        analysis_text = self.load_analysis()
        if "No reflecting analysis recommendations" in analysis_text:
            print("❌ Reflecting analysis file not found. Please run analyst.py first.")
            return None

        current_params = self.load_current_parameters()

        prompt = f"""
# TASK: Apply Reflecting Analysis to Update parameters_sam.json

You are a SAM parameter evolution specialist. Your job is to read the reflecting analysis
and the current parameters_sam.json, then output a **complete** updated parameters_sam.json.

## REFLECTING ANALYSIS (FROM ANALYST)
{analysis_text}

## CURRENT parameters_sam.json (BASELINE)
```json
{json.dumps(current_params, indent=2)}
```

## STRICT INSTRUCTIONS
- Start from the current parameters as the baseline.
- Apply only the changes that are **clearly implied** by the analysis text.
- You MUST pay attention to any discussion of:
  - Weakest or CRITICAL blocks.
  - Segment geometry (segment_width, heights, angle).
  - Label distributions (use_original_label_distributions).
  - Processing parameters (resolution, padding, crop_margin, y_bounds, mask_eps).
  - Segment processing order (`segment_order`).
- When the analysis indicates that one or more blocks are CRITICAL, you must:
  - Either update `segment_order` as suggested,
  - Or keep `segment_order` unchanged if the analysis explicitly justifies that decision.
- You MUST preserve all keys and nested structures from the current JSON unless you have
  a very clear directive in the analysis to change or remove them.
- Do NOT introduce new top-level keys that are not present in the current parameters,
  unless the analysis explicitly asks for them.

## OUTPUT FORMAT (VERY IMPORTANT)
- Return **ONLY** a valid JSON object.
- No markdown, no backticks, no explanation, no comments.
- The JSON must be a **full** parameters_sam.json, not a patch.
"""

        try:
            response = requests.post(
                f"{self.base_url}/chat-messages",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": {},
                    "query": prompt,
                    "response_mode": "streaming",
                    "conversation_id": "",
                    "user": f"reflecting_coder_{self.tunnel_id}",
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

            return result

        except Exception as exc:
            print(f"❌ Error calling Dify API: {exc}")
            return None

    # ---------- parsing & saving ----------

    def parse_and_save(self, api_response: str) -> bool:
        try:
            json_start = api_response.find("{")
            json_end = api_response.rfind("}") + 1
            if json_start == -1 or json_end <= json_start:
                print("❌ No valid JSON object found in Dify response")
                print(f"Response (truncated): {api_response[:500]}...")
                return False

            json_text = api_response[json_start:json_end]
            new_params = json.loads(json_text)

            # Basic sanity: keep type as dict
            if not isinstance(new_params, dict):
                print("❌ Parsed JSON is not an object")
                return False

            # Backup and save
            os.makedirs(self.params_dir, exist_ok=True)
            params_path = self.params_dir / "parameters_sam.json"

            if params_path.exists():
                backup_path = self.params_dir / "parameters_sam_backup_reflecting.json"
                params_path.replace(backup_path)
                print(f"📋 Backed up previous parameters to: {backup_path}")

            with open(params_path, "w") as f:
                json.dump(new_params, f, indent=2)

            print(f"✅ Updated parameters_sam.json written to: {params_path}")
            return True

        except json.JSONDecodeError as exc:
            print(f"❌ Failed to decode JSON from Dify response: {exc}")
            print(f"Response (truncated): {api_response[:500]}...")
            return False
        except Exception as exc:
            print(f"❌ Error saving evolved parameters: {exc}")
            return False

    # ---------- main process ----------

    def process(self) -> bool:
        print(f"🔄 Reflecting coder started for tunnel {self.tunnel_id}")
        print("=" * 60)

        print("📊 Step 1: Loading reflecting analysis and current parameters...")
        try:
            _ = self.load_analysis()
            _ = self.load_current_parameters()
        except Exception as exc:
            print(f"❌ Pre-check failed: {exc}")
            return False

        print("🤖 Step 2: Requesting evolved parameters from Dify...")
        api_response = self.evolve_via_dify()
        if not api_response:
            print("❌ No response from Dify or request failed")
            return False

        print("💾 Step 3: Parsing and saving updated parameters_sam.json...")
        if not self.parse_and_save(api_response):
            print("❌ Failed to parse or save evolved parameters")
            return False

        print("\n" + "=" * 60)
        print(f"🎉 Reflecting evolution complete for tunnel {self.tunnel_id}")
        print("=" * 60)
        return True


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agents/reflecting/coder.py <tunnel_id>")
        sys.exit(1)

    tunnel_id = sys.argv[1]
    evolver = ReflectingParameterEvolver(tunnel_id)
    success = evolver.process()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()


