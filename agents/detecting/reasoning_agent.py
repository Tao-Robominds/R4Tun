"""
Reasoning Agent for Tunnel Segment Detection

This agent uses LLM reasoning + domain knowledge to:
1. Analyze detected.csv for missing/default values
2. Use design patterns and neighboring rings to infer missing positions
3. Validate detected values against expected patterns
4. Output an enhanced detected.csv with reasoning-based inferences

The agent can call tools to:
- Read/write CSV files
- Load design patterns
- Query tunnel engineering knowledge
- Analyze depth map images
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple

class TunnelReasoningAgent:
    """
    Agent that uses reasoning to enhance detection results.
    """
    
    def __init__(self, tunnel_id: str, base_dir: str = "data"):
        self.tunnel_id = tunnel_id
        self.base_dir = base_dir
        self.detected_path = f"{base_dir}/{tunnel_id}/detected.csv"
        self.pattern_path = f"configurable/{tunnel_id}/design_pattern.json"
        
        # Load data
        self.detected_df = None
        self.design_pattern = None
        self.load_data()
        
    def load_data(self):
        """Load detected.csv and design pattern if available."""
        if os.path.exists(self.detected_path):
            self.detected_df = pd.read_csv(self.detected_path)
            print(f"✅ Loaded {self.detected_path}")
        else:
            raise FileNotFoundError(f"detected.csv not found at {self.detected_path}")
            
        if os.path.exists(self.pattern_path):
            with open(self.pattern_path, 'r') as f:
                self.design_pattern = json.load(f)
            print(f"✅ Loaded design pattern from {self.pattern_path}")
        else:
            print(f"ℹ️ No design pattern found at {self.pattern_path}")
    
    def analyze_detections(self) -> Dict:
        """
        Analyze the current detections to identify:
        - Which rings have valid detections
        - Which rings have default/missing values
        - Patterns in the y-coordinates
        """
        df = self.detected_df
        
        analysis = {
            "total_rings": len(df),
            "detected_rings": len(df[df['Type'] != 'default']),
            "default_rings": len(df[df['Type'] == 'default']),
            "default_indices": df[df['Type'] == 'default'].index.tolist(),
            "y_range": (df['Y'].min(), df['Y'].max()),
            "detection_types": df['Type'].value_counts().to_dict(),
        }
        
        # Identify y-coordinate clusters (potential K-block bands)
        detected_y = df[df['Type'] != 'default']['Y'].values
        if len(detected_y) > 0:
            # Simple clustering: find gaps > 500 pixels
            sorted_y = sorted(detected_y)
            bands = []
            current_band = [sorted_y[0]]
            for y in sorted_y[1:]:
                if y - current_band[-1] > 500:
                    bands.append({
                        'center': np.mean(current_band),
                        'range': (min(current_band), max(current_band)),
                        'count': len(current_band)
                    })
                    current_band = [y]
                else:
                    current_band.append(y)
            bands.append({
                'center': np.mean(current_band),
                'range': (min(current_band), max(current_band)),
                'count': len(current_band)
            })
            analysis['y_bands'] = bands
        
        return analysis
    
    def infer_missing_positions(self) -> List[Dict]:
        """
        Use reasoning to infer missing ring positions.
        
        Reasoning strategy:
        1. Look at neighboring detected rings
        2. Check if design pattern provides guidance
        3. Analyze the alternating pattern of K-block positions
        4. Make educated inference with confidence score
        """
        df = self.detected_df
        analysis = self.analyze_detections()
        inferences = []
        
        for idx in analysis['default_indices']:
            row = df.iloc[idx]
            inference = {
                'ring_index': idx,
                'original_x': row['X'],
                'original_y': row['Y'],
                'original_type': row['Type'],
            }
            
            # Strategy 1: Use design pattern if available
            if self.design_pattern:
                ring_mapping = self.design_pattern.get('ring_mapping', {})
                precise_y = self.design_pattern.get('precise_k_block_y', {}).get('by_ring_id', {})
                first_ring_id = ring_mapping.get('first_ring_id', 0)
                
                ring_id = str(first_ring_id + idx)
                if ring_id in precise_y:
                    inference['inferred_y'] = precise_y[ring_id]
                    inference['method'] = 'design_pattern_precise'
                    inference['confidence'] = 0.95
                    inference['reasoning'] = f"Used precise y-coordinate from design pattern for ring {ring_id}"
                    inferences.append(inference)
                    continue
            
            # Strategy 2: Analyze neighbors
            neighbors = []
            if idx > 0:
                prev = df.iloc[idx-1]
                if prev['Type'] != 'default':
                    neighbors.append(('prev', prev['Y'], prev['Type']))
            if idx < len(df) - 1:
                next_r = df.iloc[idx+1]
                if next_r['Type'] != 'default':
                    neighbors.append(('next', next_r['Y'], next_r['Type']))
            
            if len(neighbors) == 2:
                # Both neighbors detected - check for alternating pattern
                prev_y, next_y = neighbors[0][1], neighbors[1][1]
                
                # If neighbors are in same band, this ring might be in different band
                if abs(prev_y - next_y) < 500:
                    # Neighbors in same band - look for other band
                    if 'y_bands' in analysis and len(analysis['y_bands']) > 1:
                        current_band_center = np.mean([prev_y, next_y])
                        other_bands = [b for b in analysis['y_bands'] 
                                      if abs(b['center'] - current_band_center) > 500]
                        if other_bands:
                            # Use the other band
                            inference['inferred_y'] = other_bands[0]['center']
                            inference['method'] = 'alternating_band_inference'
                            inference['confidence'] = 0.7
                            inference['reasoning'] = (
                                f"Neighbors at y≈{prev_y:.0f} and y≈{next_y:.0f} are in same band. "
                                f"Inferred this ring is in alternate band at y≈{other_bands[0]['center']:.0f}"
                            )
                            inferences.append(inference)
                            continue
                else:
                    # Neighbors in different bands - interpolate
                    inference['inferred_y'] = np.mean([prev_y, next_y])
                    inference['method'] = 'neighbor_interpolation'
                    inference['confidence'] = 0.6
                    inference['reasoning'] = (
                        f"Neighbors at y={prev_y:.0f} and y={next_y:.0f} are in different bands. "
                        f"Using midpoint as estimate."
                    )
                    inferences.append(inference)
                    continue
            
            elif len(neighbors) == 1:
                # Only one neighbor - use it with lower confidence
                neighbor_y = neighbors[0][1]
                inference['inferred_y'] = neighbor_y
                inference['method'] = 'single_neighbor'
                inference['confidence'] = 0.5
                inference['reasoning'] = f"Only one neighbor detected at y={neighbor_y:.0f}. Using same value."
                inferences.append(inference)
                continue
            
            # Fallback: no good inference possible
            inference['inferred_y'] = row['Y']  # Keep default
            inference['method'] = 'no_inference'
            inference['confidence'] = 0.0
            inference['reasoning'] = "No neighbors or pattern available for inference"
            inferences.append(inference)
        
        return inferences
    
    def apply_inferences(self, inferences: List[Dict], min_confidence: float = 0.5) -> pd.DataFrame:
        """
        Apply inferences to create enhanced detected.csv
        """
        df = self.detected_df.copy()
        
        for inf in inferences:
            if inf['confidence'] >= min_confidence and 'inferred_y' in inf:
                idx = inf['ring_index']
                df.at[idx, 'Y'] = inf['inferred_y']
                df.at[idx, 'Type'] = 'inferred'
                print(f"  Ring {idx}: y = {inf['inferred_y']:.1f} ({inf['method']}, conf={inf['confidence']:.2f})")
                print(f"    Reasoning: {inf['reasoning']}")
        
        return df
    
    def enhance_detections(self, output_path: Optional[str] = None, min_confidence: float = 0.5):
        """
        Main method to enhance detections using reasoning.
        """
        print(f"\n{'='*60}")
        print(f"Reasoning Agent Analysis for Tunnel {self.tunnel_id}")
        print('='*60)
        
        # Analyze current state
        analysis = self.analyze_detections()
        print(f"\nCurrent state:")
        print(f"  Total rings: {analysis['total_rings']}")
        print(f"  Detected: {analysis['detected_rings']}")
        print(f"  Missing/default: {analysis['default_rings']}")
        
        if analysis['default_rings'] == 0:
            print("\n✅ No missing detections - nothing to infer")
            return self.detected_df
        
        if 'y_bands' in analysis:
            print(f"\nDetected Y-bands:")
            for i, band in enumerate(analysis['y_bands']):
                print(f"  Band {i+1}: center={band['center']:.0f}, range={band['range'][0]:.0f}-{band['range'][1]:.0f}, count={band['count']}")
        
        # Generate inferences
        print(f"\nGenerating inferences for {analysis['default_rings']} missing rings...")
        inferences = self.infer_missing_positions()
        
        # Apply inferences
        print(f"\nApplying inferences (min_confidence={min_confidence}):")
        enhanced_df = self.apply_inferences(inferences, min_confidence)
        
        # Save if output path provided
        if output_path:
            enhanced_df.to_csv(output_path, index=False)
            print(f"\n✅ Saved enhanced detections to {output_path}")
        
        return enhanced_df


def main():
    """Demo the reasoning agent."""
    import sys
    
    tunnel_id = sys.argv[1] if len(sys.argv) > 1 else "4-1"
    
    agent = TunnelReasoningAgent(tunnel_id)
    # Save enhanced version back to detected.csv to be used by SAM
    enhanced_df = agent.enhance_detections(
        output_path=f"data/{tunnel_id}/detected.csv",
        min_confidence=0.5
    )
    
    print("\n" + "="*60)
    print("Enhanced detected.csv:")
    print(enhanced_df.to_string())


if __name__ == "__main__":
    main()

