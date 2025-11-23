#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import requests
import os
from pathlib import Path

class EnhancingAnalyser:
    def __init__(self, tunnel_id):
        self.tunnel_id = tunnel_id
        self.data_dir = Path(f"data/{tunnel_id}")
        self.api_key = "app-2YyQbd7yv14XBQCf2DL3bifh"
        self.base_url = "https://api.dify.ai/v1"
        
    def load_analysis_data(self):
        # Load role definition
        role_path = Path("ablation/enhancing/role.md")
        role_content = ""
        if role_path.exists():
            with open(role_path, 'r') as f:
                role_content = f.read()
        
        # Load instructions
        instructions_path = Path("ablation/enhancing/cot.md")
        instructions_content = ""
        if instructions_path.exists():
            with open(instructions_path, 'r') as f:
                instructions_content = f.read()
        
        # Load original sample denoised characteristics (reference)
        sample_characteristics_path = Path("data/sample/characteristics/denoised_characteristics.json")
        sample_characteristics = "No sample denoised characteristics available"
        if sample_characteristics_path.exists():
            with open(sample_characteristics_path, 'r') as f:
                sample_characteristics = json.dumps(json.load(f), indent=2)
        
        # Load new tunnel denoised characteristics
        new_characteristics_path = Path(f"data/{self.tunnel_id}/characteristics/denoised_characteristics.json")
        new_characteristics = "No new tunnel denoised characteristics available"
        if new_characteristics_path.exists():
            with open(new_characteristics_path, 'r') as f:
                new_characteristics = json.dumps(json.load(f), indent=2)
        
        # Load original code with parameters
        code_path = Path("mes/3_enhancing.py")
        code_content = ""
        if code_path.exists():
            with open(code_path, 'r') as f:
                code_content = f.read()
        
        # Load current enhancing parameters
        enhancing_params_path = Path("agents/configurable/parameters_enhancing.json")
        enhancing_parameters = "No enhancing parameters available"
        if enhancing_params_path.exists():
            with open(enhancing_params_path, 'r') as f:
                enhancing_parameters = json.dumps(json.load(f), indent=2)
        
        return {
            "role": role_content,
            "instructions": instructions_content,
            "sample_characteristics": sample_characteristics,
            "new_characteristics": new_characteristics,
            "sample_code": code_content,
            "enhancing_parameters": enhancing_parameters
        }
    
    def get_enhancing_recommendations(self):
        # Load all context data
        context_data = self.load_analysis_data()
        
        # Construct comprehensive query with structured analysis process
        comprehensive_query = f"""
# ROLE
{context_data['role']}

# ORIGINAL SAMPLE TUNNEL DENOISED CHARACTERISTICS (Reference Baseline)
The following are the denoised characteristics of the original sample tunnel that the current enhancing parameters were optimized for:

```json
{context_data['sample_characteristics']}
```

# NEW TUNNEL DENOISED CHARACTERISTICS (Target for Analysis)
The following are the denoised characteristics of the new tunnel (ID: {self.tunnel_id}) that needs parameter evaluation:

```json
{context_data['new_characteristics']}
```

# CURRENT ENHANCING PARAMETERS
The current parameters that work for the sample tunnel:

```json
{context_data['enhancing_parameters']}
```

# SAMPLE CODE WITH PARAMETERS
```python
{context_data['sample_code']}
```

# STRUCTURED ANALYSIS INSTRUCTIONS
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
                "user": f"enhancing_analyser_{self.tunnel_id}", 
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
        output_file = self.data_dir / "analysis" / "enhancing_analysis.md"
        with open(output_file, 'w') as f:
            f.write(f"# Enhancing Analysis Recommendations - {self.tunnel_id}\n\n---\n\n{result}")
        
        print(f"Results saved to: {output_file}")
        return result

def main():
    import sys
    tunnel_id = sys.argv[1] if len(sys.argv) > 1 else "1-1"
    analyser = EnhancingAnalyser(tunnel_id)
    print(analyser.get_enhancing_recommendations())

if __name__ == "__main__":
    main() 