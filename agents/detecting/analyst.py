#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import requests
import os
from pathlib import Path

class DetectingAnalyser:
    def __init__(self, tunnel_id):
        self.tunnel_id = tunnel_id
        self.data_dir = Path(f"data/{tunnel_id}")
        self.api_key = "app-2YyQbd7yv14XBQCf2DL3bifh"
        self.base_url = "https://api.dify.ai/v1"
        
    def _read_required_text(self, path: Path, description: str) -> str:
        if not path.exists():
            raise FileNotFoundError(f"{description} not found at {path}")
        content = path.read_text()
        if not content.strip():
            raise ValueError(f"{description} at {path} is empty")
        return content
    
    def _read_required_json(self, path: Path, description: str) -> str:
        if not path.exists():
            raise FileNotFoundError(f"{description} not found at {path}")
        with open(path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{description} at {path} contains invalid JSON: {exc}") from exc
        return json.dumps(data, indent=2)
    
    def load_analysis_data(self):
        # Load role definition
        role_path = Path("agents/detecting/role.md")
        role_content = self._read_required_text(role_path, "Role definition")
        
        # Load instructions
        instructions_path = Path("agents/detecting/cot.md")
        instructions_content = self._read_required_text(instructions_path, "Chain-of-thought instructions")
        
        # Load original sample characteristics (reference)
        sample_characteristics_path = Path("data/sample/characteristics/raw_characteristics.json")
        sample_characteristics = self._read_required_json(sample_characteristics_path, "Sample characteristics")
        
        # Load new tunnel characteristics
        new_characteristics_path = Path(f"data/{self.tunnel_id}/characteristics/raw_characteristics.json")
        new_characteristics = self._read_required_json(new_characteristics_path, "New tunnel characteristics")
        
        # Load original code with parameters
        code_path = Path("sam4tun/4-1_detection.py")
        code_content = self._read_required_text(code_path, "Sample detecting code")
        
        # Load current detecting parameters (default)
        detecting_params_path = Path("configurable/sample/parameters_detecting.json")
        detecting_parameters = self._read_required_json(detecting_params_path, "Sample detecting parameters")
        
        return {
            "role": role_content,
            "instructions": instructions_content,
            "sample_characteristics": sample_characteristics,
            "new_characteristics": new_characteristics,
            "sample_code": code_content,
            "detecting_parameters": detecting_parameters
        }
    
    def get_detecting_recommendations(self):
        # Load all context data
        context_data = self.load_analysis_data()
        
        # Construct comprehensive query with all context elements
        comprehensive_query = f"""
# ROLE
{context_data['role']}

# ORIGINAL SAMPLE TUNNEL CHARACTERISTICS (Reference)
The following are the characteristics of the original sample tunnel that the current parameters were tuned for:

```json
{context_data['sample_characteristics']}
```

# NEW TUNNEL CHARACTERISTICS (Target for Adaptation)
The following are the characteristics of the new tunnel (ID: {self.tunnel_id}) that needs parameter adaptation:

```json
{context_data['new_characteristics']}
```

# CURRENT DETECTING PARAMETERS (MINIMAL CHANGES PREFERRED)
The current parameters that work for most tunnels:

```json
{context_data['detecting_parameters']}
```

# SAMPLE CODE WITH PARAMETERS
```python
{context_data['sample_code']}
```

# ANALYSIS INSTRUCTIONS
{context_data['instructions']}

**ANALYSIS GUIDANCE**: Follow the structured reasoning process defined in the Chain of Thought instructions to systematically evaluate the tunnel characteristics and determine appropriate parameter adaptations.

**OUTPUT REQUIREMENTS**: 
- Use flowing analysis text with natural section headers (Anchoring:, Classification:, etc.)
- Always provide exact numerical values (never ranges)
- For similar tunnels: explicitly recommend keeping original parameters
- Conclude with clean JSON parameter block at the end
"""

        response = requests.post(
            f"{self.base_url}/chat-messages",
            headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
            json={
                "inputs": {
                    "temperature": 0
                }, 
                "query": comprehensive_query,
                "response_mode": "streaming", 
                "conversation_id": "",  # Empty for fresh conversation
                "user": f"detecting_analyser_{self.tunnel_id}", 
                "files": []
            }
        )
        
        result = ""
        for line in response.iter_lines():
            if line and line.decode('utf-8').startswith('data: '):
                try:
                    chunk = json.loads(line.decode('utf-8')[6:])
                    if chunk.get('event') == 'agent_message':
                        result += chunk.get('answer', '')
                except: continue
        
        # Create analysis directory
        os.makedirs(self.data_dir / "analysis", exist_ok=True)
        
        # Save as markdown file
        output_file = self.data_dir / "analysis" / "detecting_analysis.md"
        with open(output_file, 'w') as f:
            f.write(f"# Detecting Analysis Recommendations - {self.tunnel_id}\n\n---\n\n{result}")
        
        print(f"Results saved to: {output_file}")
        return result

def main():
    import sys
    tunnel_id = sys.argv[1] if len(sys.argv) > 1 else "1-1"
    analyser = DetectingAnalyser(tunnel_id)
    print(analyser.get_detecting_recommendations())

if __name__ == "__main__":
    main() 