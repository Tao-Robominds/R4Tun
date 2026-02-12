#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
Preprocessing Analyst Agent

Analyzes tunnel characteristics and current intrinsics to recommend
preprocessing parameter adjustments. Uses Dify API for LLM reasoning.

Workflow:
1. Load knowledge files (raw.md, tuning.md, intrinsics.md)
2. Load tunnel characteristics and current intrinsics
3. Send structured prompt to Dify API for analysis
4. Save analysis results for coder.py to consume
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime


class PreprocessingAnalyst:
    def __init__(self, tunnel_id: str):
        self.tunnel_id = tunnel_id
        self.script_dir = Path(__file__).parent
        self.params_dir = self.script_dir / "parameters" / tunnel_id
        self.knowledge_dir = self.script_dir / "knowledge"
        self.memory_dir = self.script_dir / "memory"
        self.api_key = "app-l2Y5McAOreA3d2jWVNUhFjrG"
        self.base_url = "https://api.dify.ai/v1"
        
    def _read_required_text(self, path: Path, description: str) -> str:
        """Read required text file, raising error if not found or empty."""
        if not path.exists():
            raise FileNotFoundError(f"{description} not found at {path}")
        content = path.read_text()
        if not content.strip():
            raise ValueError(f"{description} at {path} is empty")
        return content
    
    def _read_optional_text(self, path: Path, description: str) -> str:
        """Read optional text file, returning empty string if not found."""
        if not path.exists():
            return f"[{description} not available]"
        content = path.read_text()
        return content if content.strip() else f"[{description} is empty]"
    
    def _read_required_json(self, path: Path, description: str) -> str:
        """Read required JSON file and return formatted string."""
        if not path.exists():
            raise FileNotFoundError(f"{description} not found at {path}")
        with open(path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{description} at {path} contains invalid JSON: {exc}") from exc
        return json.dumps(data, indent=2)
    
    def _read_optional_json(self, path: Path, description: str) -> str:
        """Read optional JSON file, returning placeholder if not found."""
        if not path.exists():
            return f"[{description} not available - first run]"
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError:
            return f"[{description} contains invalid JSON]"
    
    def load_analysis_data(self) -> dict:
        """Load all data needed for analysis."""
        # Load chain of thought instructions (includes role)
        cot_path = self.script_dir / "cot.md"
        cot_content = self._read_required_text(cot_path, "Chain-of-thought instructions")
        
        # Load knowledge files
        raw_knowledge = self._read_required_text(
            self.knowledge_dir / "raw.md", 
            "Raw characteristics knowledge"
        )
        tuning_knowledge = self._read_required_text(
            self.knowledge_dir / "tuning.md", 
            "Tuning guide knowledge"
        )
        intrinsics_knowledge = self._read_required_text(
            self.knowledge_dir / "intrinsics.md", 
            "Intrinsics metrics knowledge"
        )
        
        # Load tunnel-specific data
        characteristics = self._read_optional_json(
            self.params_dir / "characteristics.json",
            "Tunnel characteristics"
        )
        current_params = self._read_optional_json(
            self.params_dir / "parameters_preprocessing.json",
            "Current preprocessing parameters"
        )
        current_intrinsics = self._read_optional_json(
            self.params_dir / "intrinsics.json",
            "Current intrinsics"
        )
        
        # Load all past tuning experiences from memory (chronological by filename)
        memory_parts = []
        if self.memory_dir.exists():
            for path in sorted(self.memory_dir.glob("*.md")):
                if path.name.startswith("."):
                    continue
                try:
                    memory_parts.append(f"### {path.name}\n{path.read_text()}")
                except Exception:
                    pass
        memory_content = "\n\n---\n\n".join(memory_parts) if memory_parts else "[No past tuning experiences in memory yet.]"
        
        return {
            "cot": cot_content,
            "raw_knowledge": raw_knowledge,
            "tuning_knowledge": tuning_knowledge,
            "intrinsics_knowledge": intrinsics_knowledge,
            "characteristics": characteristics,
            "current_params": current_params,
            "current_intrinsics": current_intrinsics,
            "memory": memory_content,
        }
    
    def get_preprocessing_recommendations(self) -> str:
        """Use Dify API to get parameter recommendations."""
        # Load all context data
        context_data = self.load_analysis_data()
        
        # Construct comprehensive query
        comprehensive_query = f"""
# ROLE AND INSTRUCTIONS
{context_data['cot']}

---

# KNOWLEDGE BASE

## Raw Characteristics Guide
{context_data['raw_knowledge']}

## Tuning Guide
{context_data['tuning_knowledge']}

## Intrinsics Metrics Guide
{context_data['intrinsics_knowledge']}

---

# TUNNEL DATA (ID: {self.tunnel_id})

## Tunnel Characteristics (from raw point cloud)
```json
{context_data['characteristics']}
```

## Current Preprocessing Parameters
```json
{context_data['current_params']}
```

## Current Intrinsics (preprocessing output quality)
```json
{context_data['current_intrinsics']}
```

---

# PAST TUNING EXPERIENCE (memory)

Learn from these prior runs. Avoid changes that led to regression (e.g. point_retention or depth_map_valid_pixels dropping). Aim to achieve pre_ready_for_detection: true.

{context_data['memory']}

---

# TASK

Analyze the tunnel {self.tunnel_id} and recommend preprocessing parameter adjustments.

1. **Anchoring**: Compare characteristics against typical values and current parameters
2. **Classification**: Classify the tunnel's tuning regime
3. **Diagnostic Inspection**: Analyze current intrinsics for quality issues
4. **Parameter Adaptation**: Propose specific parameter changes with justification
5. **Validation**: Verify proposed parameters are valid and consistent

**OUTPUT REQUIREMENTS**: 
- Use flowing analysis text with natural section headers
- Always provide exact numerical values (never ranges)
- If intrinsics are good (all within thresholds), recommend minimal or no changes
- Conclude with clean JSON parameter block containing ALL parameters
"""

        # Use curl so the request shows in Dify logs (same as manual curl)
        url = f"{self.base_url}/chat-messages"
        body = json.dumps({
            "query": comprehensive_query,
            "user": f"preprocessing_analyst_{self.tunnel_id}",
            "response_mode": "streaming",
            "inputs": {"temperature": 0},
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(body)
            body_path = f.name
        try:
            proc = subprocess.run(
                [
                    "curl", "-s", "-S", "-N", "-w", "\nHTTP_CODE:%{http_code}", "-X", "POST", url,
                    "-H", f"Authorization: Bearer {self.api_key}",
                    "-H", "Content-Type: application/json",
                    "-d", f"@{body_path}",
                ],
                capture_output=True,
                timeout=120,
                text=True,
            )
        finally:
            os.unlink(body_path)
        if proc.returncode != 0:
            raise RuntimeError(f"Dify curl failed: returncode={proc.returncode} stderr={proc.stderr!r}")
        out = proc.stdout or ""
        # Last line may be HTTP_CODE:200
        if "HTTP_CODE:" in out:
            out, code_line = out.rsplit("\nHTTP_CODE:", 1)
            code = code_line.split("\n")[0].strip()
            if code and not code.startswith("2"):
                raise RuntimeError(f"Dify API error: HTTP {code}\n{out[:500]}")
        result = ""
        for line in out.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload in ("", "[DONE]"):
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("event") in ("message", "agent_message"):
                result += chunk.get("answer", "")
        
        # Create analysis directory
        analysis_dir = self.params_dir / "analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save as markdown file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = analysis_dir / f"preprocessing_analysis_{timestamp}.md"
        
        with open(output_file, 'w') as f:
            f.write(f"# Preprocessing Analysis - {self.tunnel_id}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(result)
        
        # Also save as latest analysis (for coder.py to consume)
        latest_file = analysis_dir / "latest_analysis.md"
        with open(latest_file, 'w') as f:
            f.write(f"# Preprocessing Analysis - {self.tunnel_id}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(result)
        
        print(f"Analysis saved to: {output_file}")
        print(f"Latest analysis: {latest_file}")
        return result


def main():
    import sys
    tunnel_id = sys.argv[1] if len(sys.argv) > 1 else "1-4"
    
    print(f"=" * 60)
    print(f"Preprocessing Analyst - Tunnel {tunnel_id}")
    print(f"=" * 60)
    
    analyst = PreprocessingAnalyst(tunnel_id)
    result = analyst.get_preprocessing_recommendations()
    
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
