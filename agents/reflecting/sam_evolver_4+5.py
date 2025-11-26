#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import requests
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from deepseek_json_parser import DeepSeekJSONParser

class SAMEvolver4Plus5:
    def __init__(self, tunnel_id):
        self.tunnel_id = tunnel_id
        self.data_dir = Path(f"data/{tunnel_id}")
        self.analysis_dir = self.data_dir / "analysis"
        self.api_key = "app-bKyUjJtUZhrkbsEkh5AvZpzE"
        self.base_url = "https://api.dify.ai/v1"
        self.json_parser = DeepSeekJSONParser(debug=True)
        
    def analyze_point_coverage(self):
        """Analyze current point distribution from final.csv for 7-segment tunnels"""
        try:
            final_csv_path = self.data_dir / "final.csv"
            if not final_csv_path.exists():
                return None
            
            df = pd.read_csv(final_csv_path)
            
            # Analyze point distribution by predicted labels (7-segment configuration)
            point_counts = {}
            point_percentages = {}
            class_names = {
                0: 'Background',
                1: 'K-block', 
                2: 'B1-block',
                3: 'A1-block', 
                4: 'A2-block',
                5: 'A3-block',
                6: 'A4-block',
                7: 'B2-block'
            }
            
            total_points = len(df)
            
            for class_id, class_name in class_names.items():
                if class_id in df['pred'].values:
                    count = len(df[df['pred'] == class_id])
                    percentage = (count / total_points) * 100 if total_points > 0 else 0
                else:
                    count = 0
                    percentage = 0.0
                
                point_counts[class_name] = count
                point_percentages[class_name] = percentage
            
            # Calculate statistics (excluding background)
            block_counts = {k: v for k, v in point_counts.items() if k != 'Background'}
            block_percentages = {k: v for k, v in point_percentages.items() if k != 'Background'}
            
            if block_counts:
                avg_points = np.mean(list(block_counts.values()))
                std_points = np.std(list(block_counts.values()))
                min_points = min(block_counts.values())
                max_points = max(block_counts.values())
                weakest_block = min(block_counts.keys(), key=lambda k: block_counts[k])
                critical_threshold = avg_points * 0.3
                critical_blocks = [k for k, v in block_counts.items() if v < critical_threshold]
            else:
                avg_points = std_points = min_points = max_points = 0
                weakest_block = "None"
                critical_blocks = []
            
            analysis_data = {
                'tunnel_id': self.tunnel_id,
                'timestamp': datetime.now().isoformat(),
                'total_points': total_points,
                'point_counts': point_counts,
                'point_percentages': point_percentages,
                'statistics': {
                    'average_points_per_block': avg_points,
                    'std_points_per_block': std_points,
                    'coefficient_of_variation': (std_points / avg_points) * 100 if avg_points > 0 else 0,
                    'minimum_points': min_points,
                    'maximum_points': max_points,
                    'weakest_block': weakest_block,
                    'strongest_block': max(block_counts.keys(), key=lambda k: block_counts[k]) if block_counts else "None",
                    'critical_threshold': critical_threshold,
                    'critical_blocks': critical_blocks,
                    'coverage_quality': 'excellent' if (std_points / avg_points) * 100 < 20 else 'good' if (std_points / avg_points) * 100 < 40 else 'poor'
                }
            }
            
            return analysis_data
            
        except Exception as e:
            print(f"Error analyzing point distribution: {e}")
            return None
    
    def load_sam_parameters(self):
        """Load current SAM parameters"""
        sam_params_path = self.data_dir / "sam_parameters.json"
        if sam_params_path.exists():
            with open(sam_params_path, 'r') as f:
                return json.load(f)
        return None
    
    def load_analysis_data(self):
        """Load and combine all analysis data for evolution"""
        # Get point coverage analysis
        coverage_analysis = self.analyze_point_coverage()
        if not coverage_analysis:
            return "Failed to analyze point coverage - please run SAM segmentation first"
        
        # Load algorithm4 characteristics
        characteristics_path = self.data_dir / "characteristics" / "algorithm4_characteristics.json"
        characteristics_data = "No algorithm4 characteristics data available"
        
        if characteristics_path.exists():
            try:
                with open(characteristics_path, 'r') as f:
                    characteristics_data = json.dumps(json.load(f), indent=2)
            except Exception as e:
                characteristics_data = f"Error reading algorithm4_characteristics.json: {str(e)}"
        
        # Load detected prompt points
        detected_path = self.data_dir / "detected.csv"
        detected_data = "No detected prompt points data available"
        
        if detected_path.exists():
            try:
                df_detected = pd.read_csv(detected_path)
                
                # Create summary statistics for the detected points
                detected_summary = {
                    "total_prompt_points": len(df_detected),
                    "point_types": df_detected['Type'].value_counts().to_dict(),
                    "x_range": {
                        "min": float(df_detected['X'].min()),
                        "max": float(df_detected['X'].max()),
                        "mean": float(df_detected['X'].mean())
                    },
                    "y_range": {
                        "min": float(df_detected['Y'].min()),
                        "max": float(df_detected['Y'].max()),
                        "mean": float(df_detected['Y'].mean())
                    }
                }
                
                detected_data = json.dumps(detected_summary, indent=2)
                
            except Exception as e:
                detected_data = f"Error reading detected.csv: {str(e)}"
        
        # Get current SAM parameters
        sam_params = self.load_sam_parameters()
        if not sam_params:
            sam_params_data = "No SAM parameters found - please run SAM analyser first"
        else:
            sam_params_data = json.dumps(sam_params, indent=2)
        
        # Format point coverage analysis with normalized block names
        weakest_block_original = coverage_analysis['statistics']['weakest_block']
        weakest_block = weakest_block_original.replace('-block', '') if weakest_block_original != 'Background' else weakest_block_original
        min_points = coverage_analysis['statistics']['minimum_points']
        avg_points = coverage_analysis['statistics']['average_points_per_block']
        cv = coverage_analysis['statistics']['coefficient_of_variation']
        critical_blocks_original = coverage_analysis['statistics']['critical_blocks']
        critical_blocks = [block.replace('-block', '') if block != 'Background' else block for block in critical_blocks_original]
        
        coverage_summary = f"""
Point Coverage Distribution (7-segment tunnel):
Total Points: {coverage_analysis['total_points']:,}

Block Distribution:
"""
        
        for block, count in coverage_analysis['point_counts'].items():
            if block != 'Background':
                percentage = coverage_analysis['point_percentages'][block]
                status = "🔴 CRITICAL" if count < coverage_analysis['statistics']['critical_threshold'] else "🟡 WEAK" if block == weakest_block_original else "✅"
                coverage_summary += f"- {status} {block}: {count:,} points ({percentage:.1f}%)\n"
        
        coverage_summary += f"""
Coverage Statistics:
- Average per block: {avg_points:.0f} points
- Weakest block: {weakest_block} ({min_points:,} points)
- Critical blocks (< 30% avg): {critical_blocks}
- Coefficient of variation: {cv:.1f}% ({coverage_analysis['statistics']['coverage_quality']})
"""
        
        # Combine all datasets for DIFY
        combined_data = f"""
You are an expert SAM parameter optimization agent for 7-segment point cloud segmentation (tunnels 4-1, 5-1).

# SAM Parameter Evolution - Tunnel {self.tunnel_id} (7-segment configuration)

## Current Point Coverage Analysis
{coverage_summary}

## Current SAM Parameters
{sam_params_data}

## Tunnel Characteristics Data
{characteristics_data}

## Detected Prompt Points Analysis  
{detected_data}

## Evolution Strategy for 7-Segment Tunnels

**Primary Objective**: Improve {weakest_block} coverage and overall point distribution balance across all 7 segments (K, B1, A1, A2, A3, A4, B2)

**7-Segment Specific Considerations**:
- This tunnel has 7 segments per ring: K-block, B1-block, A1-block, A2-block, A3-block, A4-block, B2-block
- More A-blocks (4 total: A1, A2, A3, A4) require balanced coverage optimization
- segment_per_ring should be 7 for proper processing
- segment_order should be ["K", "B1", "A1", "A2", "A3", "A4", "B2"]

**Optimization Approaches**:
1. **Segment Width**: Adjust segment_width to improve coverage (currently: {sam_params.get('segment_width', 'unknown')}mm)
2. **Processing Resolution**: Fine-tune processing resolution if available
3. **Label Distribution**: Toggle use_original_label_distributions for better coverage
4. **Geometric Parameters**: Adjust K_height, AB_height, and angle if needed
5. **Processing Parameters**: Optimize padding and crop_margin
6. **7-Segment Configuration**: Ensure segment_per_ring=7 and proper segment_order

**Guidelines**:
- ✅ Focus on improving {weakest_block} coverage by ≥10%
- ✅ Make reasonable parameter adjustments (avoid extreme values)
- ✅ Consider tunnel-specific characteristics in optimization
- ✅ Ensure proper 7-segment configuration

**Success Criteria**:
- Improve {weakest_block} coverage significantly
- Reduce coefficient of variation below 40% (currently {cv:.1f}%)
- Eliminate critical blocks (those below 30% of average coverage)
- Maintain balanced coverage across all 7 segments

## CRITICAL REQUIREMENT: JSON OUTPUT

You MUST analyze the data and provide your reasoning, but ALWAYS end your response with a valid JSON configuration block that contains ALL required SAM parameters. The JSON must be properly formatted and include:

**Required Fields (MANDATORY):**
- segment_per_ring: (integer, must be 7 for 7-segment tunnels)
- segment_order: (array of strings, must be ["K", "B1", "A1", "A2", "A3", "A4", "B2"])
- segment_width: (integer in mm)

**Additional Fields (include if present in current parameters):**
- K_height: (float in mm)
- AB_height: (float in mm) 
- angle: (float in degrees)
- use_original_label_distributions: (boolean)
- processing: (object with resolution, padding, crop_margin)

**Example JSON format for 7-segment tunnel:**
```json
{{
  "segment_per_ring": 7,
  "segment_order": ["K", "B1", "A1", "A2", "A3", "A4", "B2"],
  "segment_width": 1200,
  "K_height": 1079.92,
  "AB_height": 3239.77,
  "angle": 7.52,
  "use_original_label_distributions": false,
  "processing": {{
    "resolution": 0.005,
    "padding": 300,
    "crop_margin": 50
  }}
}}
```

Provide your analysis first, then ALWAYS conclude with the complete JSON configuration in the exact format above.
        """
        
        return combined_data
    
    def replace_sam_parameters(self, recommended_config):
        """Replace the existing sam_parameters.json with evolved configuration"""
        try:
            # Use the DeepSeekJSONParser plugin to extract and validate JSON
            required_fields = ['segment_per_ring', 'segment_order', 'segment_width']
            config_data = self.json_parser.extract_and_validate(recommended_config, required_fields)
            
            if not config_data:
                # Try manual JSON extraction as fallback
                print("🔍 Primary JSON extraction failed, attempting manual extraction...")
                config_data = self._manual_json_extraction(recommended_config)
                
                if not config_data:
                    raise ValueError("No valid SAM configuration found in response - manual extraction also failed")
            
            # Validate that all required fields are present
            missing_fields = [field for field in required_fields if field not in config_data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Ensure proper 7-segment configuration
            if config_data.get('segment_per_ring') != 7:
                print(f"⚠️ Warning: Setting segment_per_ring to 7 for tunnel {self.tunnel_id}")
                config_data['segment_per_ring'] = 7
            
            if 'segment_order' not in config_data or len(config_data['segment_order']) != 7:
                print(f"⚠️ Warning: Setting proper 7-segment order for tunnel {self.tunnel_id}")
                config_data['segment_order'] = ["K", "B1", "A1", "A2", "A3", "A4", "B2"]
            
            # Backup current parameters
            sam_params_path = self.data_dir / "sam_parameters.json"
            backup_path = None
            if sam_params_path.exists():
                backup_path = self.data_dir / f"sam_parameters_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                sam_params_path.rename(backup_path)
                print(f"📋 Backed up previous parameters to: {backup_path}")
            
            # Save evolved parameters
            os.makedirs(self.data_dir, exist_ok=True)
            with open(sam_params_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            print(f"✅ Successfully evolved SAM parameters for 7-segment tunnel {self.tunnel_id}")
            print(f"🔧 Key changes applied:")
            print(f"   - Segment per ring: {config_data.get('segment_per_ring', 'Not specified')}")
            print(f"   - Segment width: {config_data.get('segment_width', 'Not specified')}")
            print(f"   - Segment order: {config_data.get('segment_order', 'Not specified')}")
            print(f"   - Label distributions: {'Original' if config_data.get('use_original_label_distributions', True) else 'Enhanced'}")
            
            return True, backup_path
            
        except (ValueError, FileNotFoundError) as e:
            print(f"❌ Error updating SAM parameters: {e}")
            print(f"📝 Debug: Using parser to analyze response...")
            
            # Use the parser's debug functionality to show what went wrong
            if isinstance(recommended_config, str):
                self.json_parser.debug_response(recommended_config)
            
            # Save the problematic response for manual inspection
            debug_file = self.data_dir / "analysis" / f"failed_evolution_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            os.makedirs(self.data_dir / "analysis", exist_ok=True)
            with open(debug_file, 'w') as f:
                f.write(f"Failed SAM Evolution Response:\n\n{recommended_config}")
            print(f"📄 Saved debug response to: {debug_file}")
            
            return False, None
    
    def _manual_json_extraction(self, text):
        """Manual fallback JSON extraction method"""
        try:
            import re
            
            # Look for JSON blocks with various patterns
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
                r'```\s*(\{.*?\})\s*```',      # JSON in generic code blocks
                r'(\{[^{}]*"segment_per_ring"[^{}]*\})',  # Simple JSON with required field
                r'(\{(?:[^{}]|{[^{}]*})*\})'   # Complex nested JSON
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    try:
                        # Clean up the JSON string
                        json_str = match.strip()
                        if json_str.startswith('{') and json_str.endswith('}'):
                            config = json.loads(json_str)
                            
                            # Check if it has the required fields
                            required_fields = ['segment_per_ring', 'segment_order', 'segment_width']
                            if all(field in config for field in required_fields):
                                print(f"✅ Manual extraction successful using pattern: {pattern[:20]}...")
                                return config
                    except json.JSONDecodeError:
                        continue
            
            print("❌ Manual JSON extraction failed - no valid JSON found")
            return None
            
        except Exception as e:
            print(f"❌ Manual extraction error: {e}")
            return None
    
    def get_evolved_parameters(self):
        """Get evolved SAM parameters through DIFY agent"""
        
        response = requests.post(
            f"{self.base_url}/chat-messages",
            headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
            json={
                "inputs": {}, "query": self.load_analysis_data(),
                "response_mode": "streaming", "conversation_id": "",
                "user": f"sam_evolver_4+5_{self.tunnel_id}", "files": []
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
        
        # Save the evolution analysis to markdown file
        os.makedirs(self.data_dir / "analysis", exist_ok=True)
        output_file = self.data_dir / "analysis" / "sam_evolution_4+5.md"
        with open(output_file, 'w') as f:
            f.write(f"# SAM Parameter Evolution - {self.tunnel_id} (7-segment)\n\n")
            f.write(f"## AI Evolution Analysis\n\n{result}")
        
        # Try to replace the SAM parameters with evolved configuration
        success, backup_path = self.replace_sam_parameters(result)
        
        if not success:
            print(f"⚠️ First attempt failed. Requesting JSON-only response...")
            # Try a JSON-only follow-up request
            success, backup_path = self._request_json_only()
        
        if success:
            print(f"🚀 SAM parameters evolved for 7-segment tunnel {self.tunnel_id}")
        else:
            print(f"❌ Failed to apply evolved parameters for tunnel {self.tunnel_id}")
        
        return result
    
    def _request_json_only(self):
        """Request only JSON configuration as a follow-up"""
        try:
            # Load current parameters for context
            current_params = self.load_sam_parameters()
            current_params_str = json.dumps(current_params, indent=2) if current_params else "No parameters found"
            
            json_only_prompt = f"""
Based on the previous analysis for tunnel {self.tunnel_id} (7-segment configuration), provide ONLY the optimized SAM parameters in valid JSON format.

Current parameters:
{current_params_str}

Requirements for 7-segment tunnels:
- Must include: segment_per_ring (must be 7), segment_order (must be ["K", "B1", "A1", "A2", "A3", "A4", "B2"]), segment_width
- Include all other fields from current parameters
- Output ONLY valid JSON, no text before or after

Example format:
{{
  "segment_per_ring": 7,
  "segment_order": ["K", "B1", "A1", "A2", "A3", "A4", "B2"],
  "segment_width": 1200
}}
"""
            
            response = requests.post(
                f"{self.base_url}/chat-messages",
                headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
                json={
                    "inputs": {}, "query": json_only_prompt,
                    "response_mode": "streaming", "conversation_id": "",
                    "user": f"sam_evolver_json_4+5_{self.tunnel_id}", "files": []
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
            
            # Save JSON-only response
            json_output_file = self.data_dir / "analysis" / f"sam_evolution_json_only_4+5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(json_output_file, 'w') as f:
                f.write(f"JSON-only response for tunnel {self.tunnel_id} (7-segment):\n\n{result}")
            
            # Try to extract parameters from JSON-only response
            return self.replace_sam_parameters(result)
            
        except Exception as e:
            print(f"❌ JSON-only request failed: {e}")
            return False, None

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python sam_evolver_4+5.py <tunnel_id>")
        print("Example: python sam_evolver_4+5.py 4-1")
        sys.exit(1)
        
    tunnel_id = sys.argv[1]
    
    # Verify this is meant for 7-segment tunnels
    if tunnel_id not in ['4-1', '5-1']:
        print(f"⚠️ Warning: This script is designed for 7-segment tunnels (4-1, 5-1)")
        print(f"You provided: {tunnel_id}")
        proceed = input("Do you want to continue? (y/N): ")
        if proceed.lower() != 'y':
            sys.exit(1)
    
    # Verify final.csv exists (needed for point coverage analysis)
    final_csv_path = Path(f"data/{tunnel_id}/final.csv")
    if not final_csv_path.exists():
        print(f"❌ final.csv not found: {final_csv_path}")
        print("Please run SAM segmentation first:")
        print(f"  python ablation/5.self_reflecting/sam_4+5.py {tunnel_id}")
        sys.exit(1)
    
    # Verify sam_parameters.json exists
    sam_params_path = Path(f"data/{tunnel_id}/sam_parameters.json")
    if not sam_params_path.exists():
        print(f"❌ sam_parameters.json not found: {sam_params_path}")
        print("Please run SAM analyser first:")
        print(f"  python ablation/5.self_reflecting/sam_analyser.py {tunnel_id}")
        sys.exit(1)
    
    evolver = SAMEvolver4Plus5(tunnel_id)
    
    print("🔍 SAM parameter evolution starting for 7-segment tunnel...")
    
    # Analyze current coverage
    coverage = evolver.analyze_point_coverage()
    if coverage:
        weakest_block_original = coverage['statistics']['weakest_block']
        weakest_block = weakest_block_original.replace('-block', '') if weakest_block_original != 'Background' else weakest_block_original
        min_points = coverage['statistics']['minimum_points']
        cv = coverage['statistics']['coefficient_of_variation']
        
        print(f"📊 Point coverage analysis (7-segment):")
        print(f"   - Weakest block: {weakest_block} ({min_points:,} points)")
        print(f"   - Distribution quality: CV = {cv:.1f}%")
    
    print("\n🚀 Generating parameter evolution for 7-segment configuration...")
    result = evolver.get_evolved_parameters()
    
    if result:
        print("\n" + "="*70)
        print("SAM PARAMETER EVOLUTION COMPLETED (7-SEGMENT):")
        print("="*70)
        print("✅ AI Analysis Complete: Parameter optimization applied for 7-segment tunnel")
        print("🎯 Focus on improving point coverage distribution across all 7 segments")
        print("📋 Next steps:")
        print(f"   1. Test evolved parameters: python ablation/5.self_reflecting/sam_4+5.py {tunnel_id}")
        print(f"   2. Analyze results: python ablation/5.self_reflecting/sam_evolver_4+5.py {tunnel_id}")
        print(f"   3. Evaluate performance: python ablation/5.self_reflecting/evaluation_4+5.py {tunnel_id}")
        print("\n💡 Evolution Philosophy:")
        print("   - Data-driven optimization based on 7-segment point coverage analysis")
        print("   - Focus on improving weakest performing segments")
        print("   - Maintain system stability with reasonable parameter ranges")
        print("   - Ensure proper 7-segment configuration (K, B1, A1, A2, A3, A4, B2)")
    else:
        print("❌ Failed to generate evolved SAM parameters")
        print("💡 Check DIFY agent response and parameter constraints")

if __name__ == "__main__":
    main()
