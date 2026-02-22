#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
Preprocessing Coder Agent

Extracts parameter recommendations from analyst output, applies changes,
runs preprocessing pipeline, and records experiences to memory.

Workflow:
1. Load analyst's latest_analysis.md
2. Extract parameter JSON via Dify API
3. Save new parameters to parameters_preprocessing.json
4. Run 1_preprocessing.py
5. Extract new intrinsics using extract_intrinsics.py
6. Compare old vs new intrinsics
7. Record experience to memory folder with timestamp
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


class PreprocessingCoder:
    def __init__(self, tunnel_id: str):
        self.tunnel_id = tunnel_id
        self.script_dir = Path(__file__).parent
        self.params_dir = self.script_dir / "parameters" / tunnel_id
        self.memory_dir = self.script_dir / "memory"
        self.knowledge_dir = self.script_dir / "knowledge"
        self.data_dir = Path("data") / tunnel_id
        self.api_key = "app-zKyylPpAA26g5zuAmxyuWCJZ"  # Same as denoising coder
        self.base_url = "https://api.dify.ai/v1"
        
    def load_analysis(self) -> str:
        """Load the latest analysis file."""
        analysis_path = self.params_dir / "analysis" / "latest_analysis.md"
        if analysis_path.exists():
            with open(analysis_path, 'r') as f:
                return f.read()
        return "No analysis available. Please run analyst.py first."
    
    def load_current_parameters(self) -> Dict[str, Any]:
        """Load current preprocessing parameters."""
        params_path = self.params_dir / "parameters_preprocessing.json"
        
        if not params_path.exists():
            # Try sample parameters as fallback
            sample_path = self.script_dir / "parameters" / "sample" / "parameters_preprocessing.json"
            if sample_path.exists():
                print(f"Using sample parameters as base: {sample_path}")
                with open(sample_path, 'r') as f:
                    return json.load(f)
            raise FileNotFoundError(f"No parameter configuration found for tunnel {self.tunnel_id}")
        
        with open(params_path, 'r') as f:
            return json.load(f)
    
    def load_current_intrinsics(self) -> Optional[Dict[str, Any]]:
        """Load current intrinsics if available."""
        intrinsics_path = self.params_dir / "intrinsics.json"
        if intrinsics_path.exists():
            with open(intrinsics_path, 'r') as f:
                return json.load(f)
        return None
    
    def extract_parameters_via_dify(self) -> Optional[str]:
        """Use Dify API to extract parameter values from analysis."""
        analysis_content = self.load_analysis()
        current_params = self.load_current_parameters()
        
        if "No analysis available" in analysis_content:
            print("❌ Analysis file not found. Please run analyst.py first.")
            return None
        
        extraction_prompt = f"""
# TASK: Extract Parameter Values from Preprocessing Analysis

You are a parameter extraction specialist. Extract specific parameter values from the analysis text and return them in the exact JSON format provided.

## ANALYSIS TEXT:
{analysis_content}

## CURRENT PARAMETERS (for reference):
{json.dumps(current_params, indent=2)}

## EXTRACTION INSTRUCTIONS:

1. **Find these specific parameters in the analysis:**
   - ring_spacing (ring width in meters - physical constant)
   - tunnel_diameter (tunnel diameter in meters - physical constant)
   - radius_min (lower radius bound in meters)
   - radius_max (upper radius bound in meters)
   - gradient_threshold (noise detection sensitivity, 0.1-0.4)
   - target_distances (array of upsampling distances, e.g., [0.08, 0.04, 0.02])
   - curvature_neighbors (neighbors for curvature estimation, 15-30)
   - depth_map_resolution (resolution in meters, 0.003-0.008)
   - interpolation_window (gap filling window size, 3-9)

2. **Extract only the numerical values mentioned in the analysis**
3. **If a parameter is not mentioned, keep the current value**
4. **CRITICAL: Ensure radius_min < radius_max**
5. **Return ONLY a valid JSON object with the exact structure below**

## REQUIRED OUTPUT FORMAT:
```json
{{
  "ring_spacing": <extracted_value_or_current>,
  "tunnel_diameter": <extracted_value_or_current>,
  "radius_min": <extracted_value_or_current>,
  "radius_max": <extracted_value_or_current>,
  "gradient_threshold": <extracted_value_or_current>,
  "target_distances": <extracted_value_or_current>,
  "curvature_neighbors": <extracted_value_or_current>,
  "depth_map_resolution": <extracted_value_or_current>,
  "interpolation_window": <extracted_value_or_current>
}}
```

Return ONLY the JSON object, no explanations or markdown formatting.
"""

        try:
            url = f"{self.base_url}/chat-messages"
            body = json.dumps({
                "query": extraction_prompt,
                "user": f"parameter_extractor_{self.tunnel_id}",
                "response_mode": "streaming",
                "inputs": {},
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
                print(f"❌ Dify curl failed: returncode={proc.returncode} stderr={proc.stderr!r}")
                return None
            out = proc.stdout or ""
            if "HTTP_CODE:" in out:
                out, code_line = out.rsplit("\nHTTP_CODE:", 1)
                code = code_line.split("\n")[0].strip()
                if code and not code.startswith("2"):
                    print(f"❌ Dify API error: HTTP {code}")
                    return None
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
            return result
        except Exception as e:
            print(f"❌ Error calling Dify API: {e}")
            return None
    
    def parse_and_save_parameters(self, api_response: str) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """Parse API response and save parameters to JSON file."""
        try:
            # Clean the response to extract JSON
            json_start = api_response.find('{')
            json_end = api_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                print("❌ No valid JSON found in API response")
                print(f"Response: {api_response[:500]}...")
                return False, {}, {}
            
            json_text = api_response[json_start:json_end]
            extracted_params = json.loads(json_text)
            
            # Load current parameters
            current_params = self.load_current_parameters()
            old_params = current_params.copy()
            
            # Start with complete current parameters
            final_params = current_params.copy()
            
            # Update parameters from extraction
            changes = []
            for key, value in extracted_params.items():
                if value is not None and value != "null" and key in final_params:
                    if final_params[key] != value:
                        changes.append(f"  {key}: {final_params[key]} → {value}")
                    final_params[key] = value
            
            # Validate critical constraint
            if final_params.get('radius_min', 0) >= final_params.get('radius_max', 1):
                print("❌ CRITICAL: radius_min >= radius_max! Fixing...")
                radius_mid = (final_params['radius_min'] + final_params['radius_max']) / 2
                final_params['radius_min'] = radius_mid - 0.05
                final_params['radius_max'] = radius_mid + 0.05
                changes.append(f"  [AUTO-FIX] radius bounds adjusted to [{final_params['radius_min']}, {final_params['radius_max']}]")
            
            if changes:
                print("📊 Parameter changes:")
                for change in changes:
                    print(change)
            else:
                print("📊 No parameter changes detected")
            
            # Save parameters
            os.makedirs(self.params_dir, exist_ok=True)
            param_file = self.params_dir / "parameters_preprocessing.json"
            
            with open(param_file, 'w') as f:
                json.dump(final_params, f, indent=4)
            
            print(f"📁 Parameters saved to: {param_file}")
            return True, old_params, final_params
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON from API response: {e}")
            print(f"Response text: {api_response[:500]}...")
            return False, {}, {}
        except Exception as e:
            print(f"❌ Error saving parameters: {e}")
            return False, {}, {}
    
    def run_preprocessing(self) -> Tuple[bool, str]:
        """Run the preprocessing pipeline."""
        script_path = self.script_dir / "1_preprocessing.py"
        
        if not script_path.exists():
            print(f"❌ Preprocessing script not found at {script_path}")
            return False, "Script not found"
        
        try:
            print(f"🚀 Running preprocessing for tunnel {self.tunnel_id}...")
            result = subprocess.run(
                [sys.executable, str(script_path), self.tunnel_id],
                capture_output=True,
                text=True,
                check=True,
                cwd=str(Path.cwd())
            )
            
            print("✅ Preprocessing completed successfully")
            return True, result.stdout
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Preprocessing failed")
            if e.stderr:
                print(f"Error: {e.stderr}")
            return False, e.stderr
    
    def extract_intrinsics(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Extract intrinsics from preprocessing output."""
        script_path = self.script_dir / "scripts" / "extract_intrinsics.py"
        
        if not script_path.exists():
            print(f"⚠️ Intrinsics extraction script not found at {script_path}")
            return False, None
        
        # Ensure params directory exists
        os.makedirs(self.params_dir, exist_ok=True)
        intrinsics_path = self.params_dir / "intrinsics.json"
        
        try:
            print(f"📊 Extracting intrinsics for tunnel {self.tunnel_id}...")
            result = subprocess.run(
                [
                    sys.executable, str(script_path), 
                    self.tunnel_id,
                    "--output", str(intrinsics_path)
                ],
                capture_output=True,
                text=True,
                check=True,
                cwd=str(Path.cwd())
            )
            
            # Load the generated intrinsics
            if intrinsics_path.exists():
                with open(intrinsics_path, 'r') as f:
                    return True, json.load(f)
            
            return False, None
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Intrinsics extraction failed: {e.stderr}")
            return False, None
    
    def analyze_intrinsics_change(
        self, 
        old_intrinsics: Optional[Dict], 
        new_intrinsics: Optional[Dict]
    ) -> Dict[str, Any]:
        """Analyze the change in intrinsics and determine if improved or worsened."""
        analysis = {
            "improved": [],
            "worsened": [],
            "unchanged": [],
            "overall_assessment": "unknown"
        }
        
        if old_intrinsics is None or new_intrinsics is None:
            analysis["overall_assessment"] = "insufficient_data"
            return analysis
        
        # Define thresholds and ideal ranges
        metrics = {
            "pre_theta_coverage_pct": {"ideal": (99.5, 100.5), "acceptable": (98.0, 102.0)},
            "pre_point_retention_pct": {"ideal": (70.0, 98.0), "acceptable": (60.0, 99.0)},
            "pre_depth_map_valid_pixels": {"ideal": (8000, 35000), "acceptable": (5000, 50000)}
        }
        
        def in_range(value, range_tuple):
            return range_tuple[0] <= value <= range_tuple[1]
        
        def distance_to_ideal(value, ideal_range):
            if in_range(value, ideal_range):
                return 0
            if value < ideal_range[0]:
                return ideal_range[0] - value
            return value - ideal_range[1]
        
        improvements = 0
        regressions = 0
        
        for metric, ranges in metrics.items():
            old_val = old_intrinsics.get(metric)
            new_val = new_intrinsics.get(metric)
            
            if old_val is None or new_val is None:
                continue
            
            old_dist = distance_to_ideal(old_val, ranges["ideal"])
            new_dist = distance_to_ideal(new_val, ranges["ideal"])
            
            change_info = {
                "metric": metric,
                "old": old_val,
                "new": new_val,
                "old_in_ideal": in_range(old_val, ranges["ideal"]),
                "new_in_ideal": in_range(new_val, ranges["ideal"])
            }
            
            if new_dist < old_dist:
                analysis["improved"].append(change_info)
                improvements += 1
            elif new_dist > old_dist:
                analysis["worsened"].append(change_info)
                regressions += 1
            else:
                analysis["unchanged"].append(change_info)
        
        # Determine overall assessment
        if improvements > regressions:
            analysis["overall_assessment"] = "improved"
        elif regressions > improvements:
            analysis["overall_assessment"] = "worsened"
        else:
            analysis["overall_assessment"] = "neutral"
        
        return analysis
    
    def record_experience(
        self,
        old_params: Dict[str, Any],
        new_params: Dict[str, Any],
        old_intrinsics: Optional[Dict[str, Any]],
        new_intrinsics: Optional[Dict[str, Any]],
        intrinsics_analysis: Dict[str, Any]
    ) -> str:
        """Record the experience to memory folder."""
        os.makedirs(self.memory_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate meaningful name based on outcome
        outcome = intrinsics_analysis.get("overall_assessment", "unknown")
        if outcome == "improved":
            outcome_name = "success"
        elif outcome == "worsened":
            outcome_name = "regression"
        else:
            outcome_name = "neutral"
        
        # Identify what changed
        changed_params = []
        for key in new_params:
            if old_params.get(key) != new_params.get(key):
                changed_params.append(key)
        
        if changed_params:
            param_summary = "_".join(changed_params[:3])  # First 3 changed params
        else:
            param_summary = "no_change"
        
        filename = f"{timestamp}_{self.tunnel_id}_{outcome_name}_{param_summary}.md"
        filepath = self.memory_dir / filename
        
        # Build the experience report
        content = f"""# Preprocessing Tuning Experience - {self.tunnel_id}

**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Outcome**: {outcome.upper()}

---

## Parameter Changes

### Old Parameters
```json
{json.dumps(old_params, indent=2)}
```

### New Parameters
```json
{json.dumps(new_params, indent=2)}
```

### Changed Parameters
"""
        
        if changed_params:
            for param in changed_params:
                old_val = old_params.get(param, "N/A")
                new_val = new_params.get(param, "N/A")
                content += f"- **{param}**: {old_val} → {new_val}\n"
        else:
            content += "No parameters were changed.\n"
        
        content += """
---

## Intrinsics Comparison

### Old Intrinsics
"""
        
        if old_intrinsics:
            content += f"```json\n{json.dumps(old_intrinsics, indent=2)}\n```\n"
        else:
            content += "*No previous intrinsics available (first run)*\n"
        
        content += """
### New Intrinsics
"""
        
        if new_intrinsics:
            content += f"```json\n{json.dumps(new_intrinsics, indent=2)}\n```\n"
        else:
            content += "*Intrinsics extraction failed*\n"
        
        content += """
---

## Analysis

### Metrics Assessment
"""
        
        if intrinsics_analysis["improved"]:
            content += "\n**Improved:**\n"
            for item in intrinsics_analysis["improved"]:
                content += f"- {item['metric']}: {item['old']:.4f} → {item['new']:.4f}\n"
        
        if intrinsics_analysis["worsened"]:
            content += "\n**Worsened:**\n"
            for item in intrinsics_analysis["worsened"]:
                content += f"- {item['metric']}: {item['old']:.4f} → {item['new']:.4f}\n"
        
        if intrinsics_analysis["unchanged"]:
            content += "\n**Unchanged:**\n"
            for item in intrinsics_analysis["unchanged"]:
                content += f"- {item['metric']}: {item['old']:.4f} → {item['new']:.4f}\n"
        
        content += f"""
### Overall Assessment

**{intrinsics_analysis['overall_assessment'].upper()}**

---

## Lessons Learned

"""
        
        # Add automated insights
        if intrinsics_analysis["overall_assessment"] == "worsened":
            content += """⚠️ **Regression detected.** Consider:
- Reverting to previous parameters
- The hypothesis about parameter effect may be incorrect
- External factors may have influenced results
"""
        elif intrinsics_analysis["overall_assessment"] == "improved":
            content += """✅ **Improvement achieved.** Key insights:
- The parameter adjustment was effective
- Consider this configuration as a new baseline
"""
        else:
            content += """➡️ **Neutral outcome.** The parameter change had minimal effect on intrinsics.
"""
        
        # Write the experience file
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"📝 Experience recorded: {filepath}")
        return str(filepath)
    
    def process(self) -> bool:
        """Main processing function."""
        print(f"🔄 Processing preprocessing parameters for tunnel {self.tunnel_id}")
        print("=" * 60)
        
        # Step 0: Load current intrinsics (before changes)
        old_intrinsics = self.load_current_intrinsics()
        if old_intrinsics:
            print(f"📊 Loaded previous intrinsics")
        else:
            print(f"📊 No previous intrinsics found (first run)")
        
        # Step 1: Extract parameters from analysis using Dify API
        print("\n📊 Step 1: Extracting parameters from analysis using Dify API...")
        api_response = self.extract_parameters_via_dify()
        
        if not api_response:
            print("❌ Failed to get response from Dify API")
            return False
        
        # Step 2: Parse and save parameters
        print("\n💾 Step 2: Parsing and saving parameters...")
        success, old_params, new_params = self.parse_and_save_parameters(api_response)
        
        if not success:
            print("❌ Failed to parse and save parameters")
            return False
        
        # Step 3: Run preprocessing pipeline
        print("\n🚀 Step 3: Running preprocessing pipeline...")
        preprocess_success, preprocess_output = self.run_preprocessing()
        
        if not preprocess_success:
            print("❌ Preprocessing failed")
            # Still record the failure
            self.record_experience(
                old_params, new_params, old_intrinsics, None,
                {"improved": [], "worsened": [], "unchanged": [], "overall_assessment": "failed"}
            )
            return False
        
        # Step 4: Extract new intrinsics
        print("\n📊 Step 4: Extracting new intrinsics...")
        intrinsics_success, new_intrinsics = self.extract_intrinsics()
        
        if not intrinsics_success:
            print("⚠️ Intrinsics extraction failed")
            new_intrinsics = None
        
        # Step 5: Analyze intrinsics change
        print("\n📈 Step 5: Analyzing intrinsics change...")
        intrinsics_analysis = self.analyze_intrinsics_change(old_intrinsics, new_intrinsics)
        
        # Step 6: Record experience
        print("\n📝 Step 6: Recording experience...")
        experience_path = self.record_experience(
            old_params, new_params, old_intrinsics, new_intrinsics, intrinsics_analysis
        )
        
        print("\n" + "=" * 60)
        print("🎉 PREPROCESSING PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"✅ Parameters updated: {self.params_dir / 'parameters_preprocessing.json'}")
        print(f"📁 Preprocessing outputs: {self.data_dir}")
        if new_intrinsics:
            print(f"📊 New intrinsics: {self.params_dir / 'intrinsics.json'}")
        print(f"📝 Experience recorded: {experience_path}")
        print(f"\n📊 Overall assessment: {intrinsics_analysis['overall_assessment'].upper()}")
        
        return True


def main():
    import sys
    tunnel_id = sys.argv[1] if len(sys.argv) > 1 else "1-4"
    
    coder = PreprocessingCoder(tunnel_id)
    success = coder.process()
    
    if success:
        print(f"\n✅ Processing complete for tunnel {tunnel_id}")
    else:
        print(f"\n❌ Processing failed for tunnel {tunnel_id}")


if __name__ == "__main__":
    main()
