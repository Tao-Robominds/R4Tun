"""
Pattern Discovery Module for Tunnel Segmentation

This module discovers design patterns from raw point cloud and depth map data
WITHOUT requiring ground truth labels. It uses:
1. Depth map image analysis (edge detection, Hough lines)
2. Point cloud geometric features (curvature, intensity)
3. Domain knowledge about tunnel construction

The discovered patterns can be used by a reasoning model to:
- Detect wrap-around scenarios
- Infer segment positions
- Generate all_segments.csv for SAM processing
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.cluster.hierarchy import fclusterdata
from typing import Dict, List, Tuple, Optional


# Domain knowledge constants (standard shield tunnel)
DOMAIN_KNOWLEDGE = {
    "K_height_mm": 1079.92,
    "AB_height_mm": 3239.77,
    "segment_width_mm": 1200,
    "oblique_angle_degrees": 7.52,
    "segments_per_ring": 7,
    "segment_order": ["K", "B1", "A1", "A2", "A3", "A4", "B2"],
    "resolution_mm_per_pixel": 5.0  # Default resolution
}


class PatternDiscovery:
    """Discover segment patterns from raw tunnel data."""
    
    def __init__(self, tunnel_id: str, base_dir: str = "data"):
        self.tunnel_id = tunnel_id
        self.base_dir = base_dir
        
        # Try different data directory locations
        possible_dirs = [
            os.path.join(base_dir, tunnel_id),
            os.path.join(base_dir, "configurable", tunnel_id),
        ]
        self.data_dir = None
        for d in possible_dirs:
            if os.path.exists(os.path.join(d, "depth_map.png")):
                self.data_dir = d
                break
        
        if self.data_dir is None:
            raise FileNotFoundError(f"Could not find data directory for tunnel {tunnel_id}")
        
        # Load data
        self.depth_map = self._load_depth_map()
        self.point_cloud = self._load_point_cloud()
        self.detected_points = self._load_detected_points()
        
        # Image dimensions
        self.height, self.width = self.depth_map.shape[:2]
        
        # Analysis results
        self.analysis = {}
        
    def _load_depth_map(self) -> np.ndarray:
        """Load depth map image."""
        path = os.path.join(self.data_dir, "depth_map.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load depth map from {path}")
        return img
    
    def _load_point_cloud(self) -> Optional[pd.DataFrame]:
        """Load enhanced point cloud if available."""
        path = os.path.join(self.data_dir, "enhanced.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        return None
    
    def _load_detected_points(self) -> Optional[pd.DataFrame]:
        """Load detected prompt points if available."""
        path = os.path.join(self.data_dir, "detected.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        return None
    
    def analyze_ring_boundaries(self) -> Dict:
        """Detect ring boundaries from vertical lines in depth map."""
        # Column-wise intensity variance
        col_std = np.std(self.depth_map, axis=0)
        col_std_smooth = gaussian_filter1d(col_std, sigma=5)
        
        # Find peaks (vertical lines = ring boundaries)
        peaks, properties = find_peaks(col_std_smooth, distance=100, prominence=5)
        
        # Calculate ring spacing
        if len(peaks) >= 2:
            spacings = np.diff(peaks)
            avg_spacing = np.mean(spacings)
        else:
            avg_spacing = self.width / 9  # Default assumption
        
        result = {
            "vertical_line_positions": peaks.tolist(),
            "ring_count_estimate": len(peaks) + 1,
            "average_ring_spacing_pixels": float(avg_spacing),
            "ring_width_mm": float(avg_spacing * DOMAIN_KNOWLEDGE["resolution_mm_per_pixel"])
        }
        
        self.analysis["ring_boundaries"] = result
        return result
    
    def analyze_horizontal_joints(self) -> Dict:
        """Detect horizontal joint lines from depth map."""
        # Take middle section for cleaner detection
        mid_section = self.depth_map[:, self.width//3:2*self.width//3]
        
        # Row-wise intensity variance
        row_std = np.std(mid_section, axis=1)
        row_std_smooth = gaussian_filter1d(row_std, sigma=10)
        
        # Find peaks
        peaks, _ = find_peaks(row_std_smooth, distance=200, prominence=3)
        
        result = {
            "horizontal_line_positions": peaks.tolist(),
            "joint_count": len(peaks),
            "positions_normalized": [float(p / self.height) for p in peaks]
        }
        
        self.analysis["horizontal_joints"] = result
        return result
    
    def analyze_k_block_distribution(self) -> Dict:
        """Analyze K-block Y-position distribution to detect wrap-around."""
        if self.detected_points is None:
            return {"error": "No detected points available"}
        
        # Get K-block positions (or any detected positions)
        k_blocks = self.detected_points[
            self.detected_points['Type'].isin(['K-block', 'midpoint', 'positive_slope', 'negative_slope'])
        ]
        
        if len(k_blocks) == 0:
            return {"error": "No K-block positions detected"}
        
        y_values = k_blocks['Y'].values
        y_min, y_max = y_values.min(), y_values.max()
        y_range = y_max - y_min
        coverage_percent = (y_range / self.height) * 100
        
        # Key insight: > 40% coverage indicates wrap-around
        wrap_around_detected = coverage_percent > 40
        
        # Estimate rotation positions for each ring
        segment_pixel_size = self.height / DOMAIN_KNOWLEDGE["segments_per_ring"]
        rotation_positions = []
        
        for y in y_values:
            position = int((y / self.height) * DOMAIN_KNOWLEDGE["segments_per_ring"]) % 7 + 1
            rotation_positions.append(position)
        
        result = {
            "y_min": float(y_min),
            "y_max": float(y_max),
            "y_range_pixels": float(y_range),
            "coverage_percent": float(coverage_percent),
            "wrap_around_detected": wrap_around_detected,
            "estimated_rotation_positions": rotation_positions,
            "segment_pixel_size": float(segment_pixel_size)
        }
        
        self.analysis["k_block_distribution"] = result
        return result
    
    def analyze_oblique_lines(self) -> Dict:
        """Detect oblique joint lines at top and bottom of image."""
        def count_oblique_lines(section):
            edges = cv2.Canny(section, 30, 100)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
            if lines is None:
                return 0, []
            
            oblique_count = 0
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                if 3 < abs(angle) < 15:
                    oblique_count += 1
                    angles.append(angle)
            return oblique_count, angles
        
        # Analyze top and bottom sections
        section_height = 500
        top_count, top_angles = count_oblique_lines(self.depth_map[:section_height, :])
        bottom_count, bottom_angles = count_oblique_lines(self.depth_map[-section_height:, :])
        
        # If both have significant oblique lines, segments wrap around
        wrap_around_evidence = top_count > 3 and bottom_count > 3
        
        result = {
            "top_oblique_lines": top_count,
            "bottom_oblique_lines": bottom_count,
            "top_angles": top_angles[:5] if top_angles else [],
            "bottom_angles": bottom_angles[:5] if bottom_angles else [],
            "wrap_around_evidence": wrap_around_evidence,
            "average_angle": float(np.mean(top_angles + bottom_angles)) if (top_angles + bottom_angles) else 0
        }
        
        self.analysis["oblique_lines"] = result
        return result
    
    def analyze_point_cloud_features(self) -> Dict:
        """Analyze point cloud curvature and intensity for joint detection."""
        if self.point_cloud is None:
            return {"error": "No point cloud data available"}
        
        df = self.point_cloud
        
        result = {
            "total_points": len(df),
            "rings": sorted(df['ring'].unique().tolist()) if 'ring' in df.columns else [],
            "ring_count": len(df['ring'].unique()) if 'ring' in df.columns else 0,
        }
        
        # Curvature analysis
        if 'curvature' in df.columns:
            curvature = df['curvature']
            high_curvature_threshold = curvature.mean() + 3 * curvature.std()
            result["curvature_stats"] = {
                "mean": float(curvature.mean()),
                "std": float(curvature.std()),
                "high_curvature_points": int((curvature > high_curvature_threshold).sum())
            }
        
        # Intensity analysis
        if 'intensity' in df.columns:
            intensity = df['intensity']
            result["intensity_stats"] = {
                "mean": float(intensity.mean()),
                "std": float(intensity.std()),
                "range": [float(intensity.min()), float(intensity.max())]
            }
        
        # Theta (angular) coverage analysis
        if 'theta' in df.columns:
            theta = df['theta']
            theta_range = theta.max() - theta.min()
            result["angular_coverage"] = {
                "theta_min": float(theta.min()),
                "theta_max": float(theta.max()),
                "theta_range_radians": float(theta_range),
                "theta_range_degrees": float(np.degrees(theta_range)),
                "full_coverage": theta_range > 6.0  # > ~343 degrees
            }
        
        self.analysis["point_cloud_features"] = result
        return result
    
    def infer_segment_positions(self) -> List[Dict]:
        """Infer all segment positions based on K-block detections and domain knowledge."""
        if self.detected_points is None:
            return []
        
        k_dist = self.analysis.get("k_block_distribution", {})
        if not k_dist or "error" in k_dist:
            return []
        
        segments = []
        segment_order = DOMAIN_KNOWLEDGE["segment_order"]
        segment_heights = {
            "K": DOMAIN_KNOWLEDGE["K_height_mm"],
            "B1": DOMAIN_KNOWLEDGE["AB_height_mm"],
            "A1": DOMAIN_KNOWLEDGE["AB_height_mm"],
            "A2": DOMAIN_KNOWLEDGE["AB_height_mm"],
            "A3": DOMAIN_KNOWLEDGE["AB_height_mm"],
            "A4": DOMAIN_KNOWLEDGE["AB_height_mm"],
            "B2": DOMAIN_KNOWLEDGE["AB_height_mm"],
        }
        
        resolution = DOMAIN_KNOWLEDGE["resolution_mm_per_pixel"]
        
        # For each detected K-block, calculate all segment positions
        k_blocks = self.detected_points[
            self.detected_points['Type'].isin(['K-block', 'midpoint', 'positive_slope', 'negative_slope'])
        ]
        
        for idx, row in k_blocks.iterrows():
            ring_x = row['X']
            k_y = row['Y']
            ring_id = idx + 1  # Estimated ring ID
            
            # Start from K-block and calculate other positions
            current_y = k_y
            
            for block in segment_order:
                block_height_px = segment_heights[block] / resolution
                
                # Handle wrap-around (Y goes 0 -> height)
                segment_y = current_y % self.height
                
                segments.append({
                    "Ring": ring_id,
                    "Block": block,
                    "X": float(ring_x),
                    "Y": float(segment_y),
                    "inferred": True
                })
                
                # Move to next segment (going up in Y)
                current_y += block_height_px
        
        return segments
    
    def discover_pattern(self) -> Dict:
        """Run full pattern discovery and return comprehensive results."""
        print(f"Discovering patterns for tunnel {self.tunnel_id}...")
        
        # Run all analyses
        ring_analysis = self.analyze_ring_boundaries()
        joint_analysis = self.analyze_horizontal_joints()
        k_block_analysis = self.analyze_k_block_distribution()
        oblique_analysis = self.analyze_oblique_lines()
        pc_analysis = self.analyze_point_cloud_features()
        
        # Determine wrap-around status
        wrap_around = False
        wrap_around_confidence = 0.0
        
        if k_block_analysis.get("wrap_around_detected", False):
            wrap_around = True
            wrap_around_confidence = min(k_block_analysis.get("coverage_percent", 0) / 60, 1.0)
        
        if oblique_analysis.get("wrap_around_evidence", False):
            wrap_around = True
            wrap_around_confidence = max(wrap_around_confidence, 0.7)
        
        # Compile discovered pattern
        pattern = {
            "tunnel_id": self.tunnel_id,
            "image_dimensions": {
                "width": self.width,
                "height": self.height
            },
            "domain_knowledge": DOMAIN_KNOWLEDGE,
            "ring_analysis": ring_analysis,
            "k_block_distribution": k_block_analysis,
            "oblique_line_analysis": oblique_analysis,
            "wrap_around": {
                "detected": wrap_around,
                "confidence": wrap_around_confidence,
                "requires_individual_segment_processing": wrap_around
            },
            "recommendations": []
        }
        
        # Add recommendations
        if wrap_around:
            pattern["recommendations"].append(
                "Use individual segment processing (all_segments.csv) instead of row-based processing"
            )
            pattern["recommendations"].append(
                "Infer segment positions from K-block + domain knowledge"
            )
        else:
            pattern["recommendations"].append(
                "Standard row-based SAM processing should work"
            )
        
        # Infer segment positions if wrap-around detected
        if wrap_around:
            inferred_segments = self.infer_segment_positions()
            pattern["inferred_segments"] = inferred_segments
            pattern["inferred_segment_count"] = len(inferred_segments)
        
        return pattern
    
    def save_pattern(self, output_path: Optional[str] = None):
        """Save discovered pattern to JSON file."""
        pattern = self.discover_pattern()
        
        if output_path is None:
            output_path = os.path.join(
                "configurable", self.tunnel_id, "discovered_pattern.json"
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            else:
                return obj
        
        pattern = convert_to_serializable(pattern)
        
        with open(output_path, 'w') as f:
            json.dump(pattern, f, indent=2)
        
        print(f"Pattern saved to {output_path}")
        return pattern


def compare_tunnels(tunnel_ids: List[str]) -> pd.DataFrame:
    """Compare pattern characteristics across multiple tunnels."""
    results = []
    
    for tid in tunnel_ids:
        try:
            pd_obj = PatternDiscovery(tid)
            pattern = pd_obj.discover_pattern()
            
            results.append({
                "tunnel_id": tid,
                "ring_count": pattern["ring_analysis"].get("ring_count_estimate", 0),
                "k_block_coverage_%": pattern["k_block_distribution"].get("coverage_percent", 0),
                "wrap_around": pattern["wrap_around"]["detected"],
                "wrap_confidence": pattern["wrap_around"]["confidence"],
                "image_height": pattern["image_dimensions"]["height"],
                "image_width": pattern["image_dimensions"]["width"]
            })
        except Exception as e:
            print(f"Error analyzing {tid}: {e}")
            results.append({
                "tunnel_id": tid,
                "error": str(e)
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pattern_discovery.py <tunnel_id>")
        print("       python pattern_discovery.py --compare <tunnel_id1> <tunnel_id2> ...")
        sys.exit(1)
    
    if sys.argv[1] == "--compare":
        tunnel_ids = sys.argv[2:]
        comparison = compare_tunnels(tunnel_ids)
        print("\nTunnel Comparison:")
        print(comparison.to_string(index=False))
    else:
        tunnel_id = sys.argv[1]
        discovery = PatternDiscovery(tunnel_id)
        pattern = discovery.save_pattern()
        
        print("\n" + "="*70)
        print("PATTERN DISCOVERY SUMMARY")
        print("="*70)
        print(f"Tunnel: {tunnel_id}")
        print(f"Image size: {pattern['image_dimensions']['width']} x {pattern['image_dimensions']['height']}")
        print(f"Estimated rings: {pattern['ring_analysis'].get('ring_count_estimate', 'N/A')}")
        print(f"K-block Y coverage: {pattern['k_block_distribution'].get('coverage_percent', 0):.1f}%")
        print(f"Wrap-around detected: {pattern['wrap_around']['detected']} (confidence: {pattern['wrap_around']['confidence']:.2f})")
        print("\nRecommendations:")
        for rec in pattern.get("recommendations", []):
            print(f"  - {rec}")

