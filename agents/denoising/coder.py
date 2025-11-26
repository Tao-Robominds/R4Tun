#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import requests
import os
import subprocess
import sys
from pathlib import Path

class DenoisingParameterExtractor:
    def __init__(self, tunnel_id):
        self.tunnel_id = tunnel_id
        self.data_dir = Path(f"data/{tunnel_id}")
        self.analysis_dir = self.data_dir / "analysis"
        self.params_dir = Path(f"configurable/{tunnel_id}")  # Save under configurable/{tunnel_id}/
        self.api_key = "app-AwnQSxSdDfTN7Tez202ZcmxR"
        self.base_url = "https://api.dify.ai/v1"
        
    def load_analysis_data(self):
        """Load the analysis file for this tunnel"""
        analysis_path = self.analysis_dir / "denoising_analysis.md"
        if analysis_path.exists():
            with open(analysis_path, 'r') as f:
                return f.read()
        return "No denoising analysis recommendations available. Please run analyst.py first."
    
    def load_current_parameters(self):
        """Load tunnel-specific parameters_denoising.json structure"""
        params_path = self.params_dir / "parameters_denoising.json"
        
        if not params_path.exists():
            raise FileNotFoundError(
                f"No parameter configuration found for tunnel {self.tunnel_id}.\n"
                f"Expected file: {params_path}\n"
                "Please ensure tunnel-specific parameters exist before running the coder."
            )
        
        with open(params_path, 'r') as f:
            return json.load(f)
    
    def extract_parameters_via_dify(self):
        """Use Dify API to extract parameter updates from analysis"""
        analysis_content = self.load_analysis_data()
        current_params = self.load_current_parameters()
        
        if "No denoising analysis recommendations" in analysis_content:
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
   - mask_r_low (lower radius mask threshold in meters)
   - mask_r_high (upper radius mask threshold in meters)  
   - y_step (angular step size in radians)
   - z_step (height step size in meters)
   - grad_threshold (gradient threshold for denoising)
   - smoothing_window_size (smoothing filter window size)
   - smoothing_offset (smoothing offset in meters)
   - default_cutoff_z (z height cutoff to discard noisy data, in meters)

2. **Extract only the numerical values mentioned in the analysis**
3. **If a parameter is not mentioned, keep the current value**
4. **Return ONLY a valid JSON object with the exact structure below**

## REQUIRED OUTPUT FORMAT:
```json
{{
  "mask_r_low": <extracted_value_or_current>,
  "mask_r_high": <extracted_value_or_current>,
  "y_step": <extracted_value_or_current>,
  "z_step": <extracted_value_or_current>,
  "grad_threshold": <extracted_value_or_current>,
  "smoothing_window_size": <extracted_value_or_current>,
  "smoothing_offset": <extracted_value_or_current>,
  "default_cutoff_z": <extracted_value_or_current>
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
            
            # Load current parameters to provide defaults when values are missing
            current_params = self.load_current_parameters()
            
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
            param_file = self.params_dir / "parameters_denoising.json"
            
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
        """Run the configurable denoising script"""
        script_path = Path("configurable/configurable_denoising.py")
        
        if not script_path.exists():
            print(f"❌ Configurable script not found at {script_path}")
            return False, "Script not found"
        
        try:
            result = subprocess.run([
                sys.executable, str(script_path), 
                self.tunnel_id
            ], capture_output=True, text=True, check=True, cwd=str(Path.cwd()))
            
            print(f"✅ Configurable denoising completed successfully")
            if result.stdout: 
                print(f"Output: {result.stdout}")
            return True, result.stdout
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Configurable denoising failed")
            if e.stderr: 
                print(f"Error: {e.stderr}")
            if e.stdout:
                print(f"Partial Output: {e.stdout}")
            return False, e.stderr
    
    def run_characteriser_plugin(self):
        """Run the denoised characteriser plugin to generate characteristics for next stage"""
        plugin_path = Path("sam4tun/plugins/2-denoised_characteriser.py")
        
        if not plugin_path.exists():
            print(f"⚠️  Characteriser plugin not found at {plugin_path}")
            return False, "Plugin not found"
        
        # Check if denoised.csv exists
        denoised_file = self.data_dir / "denoised.csv"
        if not denoised_file.exists():
            print(f"⚠️  Denoised data not found at {denoised_file}")
            return False, "Denoised data not found"
        
        try:
            result = subprocess.run([
                sys.executable, str(plugin_path), 
                self.tunnel_id
            ], capture_output=True, text=True, check=True, cwd=str(Path.cwd()))
            
            print(f"✅ Denoised characteriser completed successfully")
            if result.stdout: 
                print(f"Output: {result.stdout}")
            return True, result.stdout
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Denoised characteriser failed")
            if e.stderr: 
                print(f"Error: {e.stderr}")
            if e.stdout:
                print(f"Partial Output: {e.stdout}")
            return False, e.stderr
    
    def process(self):
        """Main processing function"""
        print(f"🔄 Processing parameterized denoising for tunnel {self.tunnel_id}")
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
        print("🚀 Step 3: Running configurable denoising script...")
        script_success, script_output = self.run_configurable_script()
        
        if not script_success:
            print("❌ Configurable script failed")
            return False
        
        # Step 4: Run characteriser plugin to generate characteristics for next stage
        print("📊 Step 4: Running denoised characteriser plugin...")
        char_success, char_output = self.run_characteriser_plugin()
        
        if not char_success:
            print("⚠️  Characteriser plugin failed (non-critical)")
            print("   Next stage analysis may not have required characteristics")
        
        print("\n" + "="*60)
        print("🎉 COMPLETE PIPELINE EXECUTED SUCCESSFULLY!")
        print("="*60)
        print(f"✅ Parameters extracted via Dify API and saved")
        print(f"📁 Parameters saved to: configurable/{self.tunnel_id}/")
        print(f"✅ Configurable denoising completed for tunnel {self.tunnel_id}")
        print(f"📁 Results saved to: data/{self.tunnel_id}/denoised.csv")
        if char_success:
            print(f"📁 Characteristics saved to: data/{self.tunnel_id}/characteristics/denoised_characteristics.json")
        return True

def main():
    import sys
    tunnel_id = sys.argv[1] if len(sys.argv) > 1 else "3-1"
    
    extractor = DenoisingParameterExtractor(tunnel_id)
    success = extractor.process()
    
    if success:
        print(f"\n✅ Parameterized processing complete for tunnel {tunnel_id}")
    else:
        print(f"\n❌ Processing failed for tunnel {tunnel_id}")

if __name__ == "__main__":
    main()
