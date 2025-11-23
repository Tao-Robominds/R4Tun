#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import requests
import os
import subprocess
from pathlib import Path
import time

class UnfoldingParameterExtractor:
    def __init__(self, tunnel_id):
        self.tunnel_id = tunnel_id
        self.data_dir = Path(f"data/{tunnel_id}")
        self.analysis_dir = self.data_dir / "analysis"
        self.params_dir = self.data_dir / "parameters"
        self.characteristics_dir = self.data_dir / "characteristics"
        self.api_key = "app-AwnQSxSdDfTN7Tez202ZcmxR"
        self.base_url = "https://api.dify.ai/v1"
        
    def load_analysis_data(self):
        """Load the analysis file for this tunnel"""
        analysis_path = self.analysis_dir / "unfolding_analysis.md"
        if analysis_path.exists():
            with open(analysis_path, 'r') as f:
                return f.read()
        return "No unfolding analysis recommendations available. Please run analyst.py first."
    
    def load_current_parameters(self):
        """Load current parameters_unfolding.json structure"""
        # Try tunnel-specific parameters first, then fall back to global
        params_path = self.params_dir / "parameters_unfolding.json"
        global_params_path = Path("agents/configurable/parameters_unfolding.json")
        
        if params_path.exists():
            with open(params_path, 'r') as f:
                return json.load(f)
        elif global_params_path.exists():
            with open(global_params_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"No parameter configuration found. Expected either:\n"
                                  f"- Tunnel-specific: {params_path}\n"
                                  f"- Global config: {global_params_path}")
    
    def extract_parameters_via_dify(self):
        """Use Dify API to extract parameter updates from analysis"""
        analysis_content = self.load_analysis_data()
        current_params = self.load_current_parameters()
        
        if "No unfolding analysis recommendations" in analysis_content:
            print("❌ Analysis file not found. Please run analyst.py first.")
            return None
        
        # Create extraction prompt for Dify API
        extraction_prompt = f"""
# TASK: Extract Parameter Values from Tunnel Analysis

You are a parameter extraction specialist. Extract specific parameter values from the tunnel analysis text and return them in the exact JSON format provided.

## ANALYSIS TEXT:
{analysis_content}

## CURRENT PARAMETERS (for reference):
{json.dumps(current_params, indent=2)}

## EXTRACTION INSTRUCTIONS:

1. **Find these specific parameters in the analysis:**
   - delta (slice thickness/half-thickness in meters)
   - slice_spacing_factor (spacing factor in meters, look for "reduce to X.X m")
   - vertical_filter_window (ellipse contour filter, look for "set to ~X.X m" or "≈X.X m")
   - ransac_threshold (RANSAC threshold, look for "from X.X → X.X m")
   - ransac_probability (RANSAC probability, look for "keep X.X" or "loosen P to X.X")
   - ransac_inlier_ratio (inlier ratio S, look for "from X.XX → X.XX")
   - ransac_sample_size (sample size N, look for "from X → X points")
   - polynomial_degree (polynomial degree, look for "reduce to X")
   - num_samples_factor (samples factor, look for "ring_count×XXXX")

2. **Extract only the numerical values mentioned in the analysis**
3. **If a parameter is not mentioned, keep the current value**
4. **Return ONLY a valid JSON object with the exact structure below**

## REQUIRED OUTPUT FORMAT:
```json
{{
  "delta": <extracted_value_or_current>,
  "slice_spacing_factor": <extracted_value_or_current>,
  "vertical_filter_window": <extracted_value_or_current>,
  "ransac_threshold": <extracted_value_or_current>,
  "ransac_probability": <extracted_value_or_current>,
  "ransac_inlier_ratio": <extracted_value_or_current>,
  "ransac_sample_size": <extracted_value_or_current>,
  "polynomial_degree": <extracted_value_or_current>,
  "num_samples_factor": <extracted_value_or_current>
}}
```

Return ONLY the JSON object, no explanations or markdown formatting.
"""

        try:
            response = requests.post(
                f"{self.base_url}/chat-messages",
                headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
                json={
                    "inputs": {},
                    "query": extraction_prompt,
                    "response_mode": "streaming",
                    "conversation_id": "",
                    "user": f"parameter_extractor_{self.tunnel_id}",
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
                    except:
                        continue
            
            return result
            
        except Exception as e:
            print(f"❌ Error calling Dify API: {e}")
            return None
    
    def parse_and_save_parameters(self, api_response):
        """Parse API response and save parameters to JSON file"""
        try:
            # Clean the response to extract JSON
            json_start = api_response.find('{')
            json_end = api_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                print("❌ No valid JSON found in API response")
                print(f"Response: {api_response[:500]}...")
                return False
                
            json_text = api_response[json_start:json_end]
            extracted_params = json.loads(json_text)
            
            # Load default parameters to replace null values
            try:
                current_params = self.load_current_parameters()
            except FileNotFoundError:
                # Fallback defaults if no config files exist
                current_params = {
                    "delta": 0.005,
                    "slice_spacing_factor": 1.2,
                    "vertical_filter_window": 4.5,
                    "ransac_threshold": 1.0,
                    "ransac_probability": 0.9,
                    "ransac_inlier_ratio": 0.75,
                    "ransac_sample_size": 5,
                    "polynomial_degree": 3,
                    "num_samples_factor": 1210
                }
            
            # Start with complete default parameters
            final_params = current_params.copy()
            
            # Update only the parameters that were extracted and are not null
            for key, value in extracted_params.items():
                if value is not None and value != "null" and key in final_params:
                    final_params[key] = value
                    print(f"  {key}: {value} (updated from analysis)")
                elif key in final_params:
                    print(f"  {key}: {final_params[key]} (keeping default)")
                else:
                    print(f"  {key}: ignored (not in parameter schema)")
            
            print(f"✅ Final parameters for tunnel {self.tunnel_id}:")
            for key, value in final_params.items():
                print(f"  {key}: {value}")
            
            # Save to parameters file in tunnel-specific directory
            os.makedirs(self.params_dir, exist_ok=True)
            param_file = self.params_dir / "parameters_unfolding.json"
            
            with open(param_file, 'w') as f:
                json.dump(final_params, f, indent=2)
            
            print(f"📁 Parameters saved to: {param_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON from API response: {e}")
            print(f"Response text: {api_response[:500]}...")
            return False
        except Exception as e:
            print(f"❌ Error saving parameters: {e}")
            return False
    
    def run_configurable_script(self):
        """Run the configurable unfolding script"""
        script_path = Path("agents/configurable/configurable_unfolding.py")
        
        if not script_path.exists():
            print(f"❌ Configurable script not found at {script_path}")
            return False, "Script not found"
        
        try:
            result = subprocess.run([
                'python', str(script_path), 
                self.tunnel_id
            ], capture_output=True, text=True, check=True, cwd=str(Path.cwd()))
            
            print(f"✅ Configurable unfolding completed successfully")
            if result.stdout: 
                print(f"Output: {result.stdout}")
            return True, result.stdout
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Configurable unfolding failed")
            if e.stderr: 
                print(f"Error: {e.stderr}")
            if e.stdout:
                print(f"Partial Output: {e.stdout}")
            return False, e.stderr
    
    def run_unfolded_characteriser(self):
        """Run the 1-unfolded_characteriser.py"""
        unwrapped_csv = self.data_dir / "unwrapped.csv"
        
        if not unwrapped_csv.exists():
            print(f"⚠️  unwrapped.csv not found at {unwrapped_csv}")
            return False, "unwrapped.csv not found"
        
        file_age = time.time() - unwrapped_csv.stat().st_mtime
        if file_age > 300:
            print(f"⚠️  unwrapped.csv is older than 5 minutes, may not be from current run")
        
        print(f"🔍 Running 1-unfolded_characteriser.py on {unwrapped_csv}")
        
        # Ensure characteristics directory exists
        os.makedirs(self.characteristics_dir, exist_ok=True)
        
        try:
            result = subprocess.run(['python', 'mes/plugins/1-unfolded_characteriser.py', self.tunnel_id], 
                                  capture_output=True, text=True, check=True, cwd=str(Path.cwd()))
            
            print(f"✅ Unfolding characteriser completed successfully")
            if result.stdout: print(f"Characteriser Output:\n{result.stdout}")
            
            # Verify that the characteristics file was created
            characteristics_file = self.characteristics_dir / "unfolded_characteristics.json"
            if characteristics_file.exists():
                print(f"📊 Characteristics saved to: {characteristics_file}")
            else:
                print(f"⚠️  Characteristics file not found at expected location: {characteristics_file}")
            
            return True, result.stdout
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Unfolding characteriser failed")
            if e.stderr: print(f"Error: {e.stderr}")
            return False, e.stderr
    
    def process(self):
        """Main processing function"""
        print(f"🔄 Processing parameterized unfolding for tunnel {self.tunnel_id}")
        print("="*60)
        
        # Step 1: Extract parameters from analysis using Dify API
        print("📊 Step 1: Extracting parameters from analysis using Dify API...")
        api_response = self.extract_parameters_via_dify()
        
        if not api_response:
            print("❌ Failed to get response from Dify API")
            return False
        
        # Step 2: Parse and save parameters
        print("💾 Step 2: Parsing and saving parameters...")
        if not self.parse_and_save_parameters(api_response):
            print("❌ Failed to parse and save parameters")
            return False
        
        # Step 3: Run configurable script
        print("🚀 Step 3: Running configurable unfolding script...")
        script_success, script_output = self.run_configurable_script()
        
        if not script_success:
            print("❌ Configurable script failed")
            return False
        
        # Step 4: Run unfolding characteriser
        print("🔍 Step 4: Running unfolding characteriser...")
        char_success, char_output = self.run_unfolded_characteriser()
        
        if char_success:
            print("\n" + "="*60)
            print("🎉 COMPLETE PIPELINE EXECUTED SUCCESSFULLY!")
            print("="*60)
            print(f"✅ Parameters extracted via Dify API and saved")
            print(f"📁 Parameters saved to: data/{self.tunnel_id}/parameters/")
            print(f"✅ Configurable unfolding completed for tunnel {self.tunnel_id}")
            print(f"✅ Unfolded characteriser completed")
            print(f"📁 Results saved to: data/{self.tunnel_id}/unwrapped.csv")
            print(f"📊 Characteristics saved to: data/{self.tunnel_id}/characteristics/")
            return True
        else:
            print("⚠️  Unfolding completed but characteriser failed")
            return False

def main():
    import sys
    tunnel_id = sys.argv[1] if len(sys.argv) > 1 else "3-1"
    
    extractor = UnfoldingParameterExtractor(tunnel_id)
    success = extractor.process()
    
    if success:
        print(f"\n✅ Parameterized processing complete for tunnel {tunnel_id}")
    else:
        print(f"\n❌ Processing failed for tunnel {tunnel_id}")

if __name__ == "__main__":
    main()