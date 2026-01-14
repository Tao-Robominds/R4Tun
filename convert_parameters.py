#!/usr/bin/env python3
"""
Convert old configurable parameter format to new p4tun format.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Mapping functions for each stage
def convert_unfolding(old_params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert unfolding parameters from old to new format."""
    new = {
        "description": "Tunnel centerline extraction and point cloud unfolding parameters",
        "physical_constants": {
            "ring_spacing": old_params.get("slice_spacing_factor", 1.2),
            "tunnel_diameter": old_params.get("diameter", 5.5)
        },
        "slicing": {
            "slice_half_thickness": old_params.get("delta", 0.005),
            "max_distance_from_top": old_params.get("vertical_filter_window", 4.5)
        },
        "curve_fitting": {
            "polynomial_degree": old_params.get("polynomial_degree", 3)
        },
        "ransac_ellipse": {
            "inlier_ratio": old_params.get("ransac_inlier_ratio", 0.75),
            "confidence": old_params.get("ransac_probability", 0.9),
            "min_samples": old_params.get("ransac_sample_size", 5),
            # Calculate inlier_threshold: ransac_threshold * ransac_inlier_threshold_multiplier
            "inlier_threshold": old_params.get("ransac_threshold", 1.0) * old_params.get("ransac_inlier_threshold_multiplier", 0.8)
        },
        "arc_length": {
            "samples_per_ring": old_params.get("num_samples_factor", 1210)
        },
        "performance": {
            "batch_size": old_params.get("batch_size", 1000000),
            "num_jobs": old_params.get("n_jobs", 12)
        }
    }
    return new

def convert_denoising(old_params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert denoising parameters from old to new format."""
    new = {
        "description": "Local density-based point cloud denoising parameters",
        "physical_constants": {
            "tunnel_diameter": old_params.get("diameter", 5.5)  # Not in old format, use default
        },
        "radius_filtering": {
            "radius_min": old_params.get("mask_r_low", 2.7),
            "radius_max": old_params.get("mask_r_high", 2.8)
        },
        "grid_resolution": {
            "theta_step": old_params.get("y_step", 0.5),
            "radial_step": old_params.get("z_step", 0.001)
        },
        "gradient_detection": {
            "gradient_threshold": old_params.get("grad_threshold", 0.2),
            "gradient_epsilon": 1e-6  # Not in old format, use default
        },
        "cutoff_smoothing": {
            "smoothing_window": old_params.get("smoothing_window_size", 3),
            "smoothing_offset": old_params.get("smoothing_offset", -0.003)
        }
    }
    return new

def convert_enhancing(old_params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert enhancing parameters from old to new format."""
    new = {
        "description": "Geometry-guided point cloud enhancement parameters",
        "physical_constants": {
            "ring_spacing": 1.2  # Not in old format, use default
        },
        "curvature": {
            "curvature_neighbors": old_params.get("num_neighbors", 20)
        },
        "upsampling": {
            "target_distances": [
                old_params.get("upsampling_stage1_target_distance", 0.08),
                old_params.get("upsampling_stage2_target_distance", 0.04),
                old_params.get("upsampling_stage3_target_distance", 0.02)
            ],
            "curvature_threshold": old_params.get("curvature_threshold", 0.0005),
            "upsampling_neighbors": old_params.get("num_neighbors", 20),
            "distance_tolerance_low": 0.9,  # Not in old format, use default
            "distance_tolerance_high": 2.0,  # Not in old format, use default
            "radius_filter_factor": 0.15,  # Not in old format, use default
            "min_new_point_distance_factor": 0.2  # Not in old format, use default
        },
        "outlier_detection": {
            "depth_threshold_low": old_params.get("depth_threshold_low", 0.003),
            "depth_threshold_high": old_params.get("depth_threshold_high", 0.008),
            "high_density_ring_start": old_params.get("n_segment_start", 0),
            "high_density_ring_end": old_params.get("n_segment_end", 5),
            "outlier_neighbors": old_params.get("num_neighbors", 20)
        },
        "outlier_interpolation": {
            "interpolation_radius": old_params.get("inter_radius", 0.06),
            "num_interpolations": old_params.get("num_interpolations", 2),
            "duplicate_threshold": old_params.get("duplicate_threshold", 0.02),
            "max_outlier_points": 5000  # Not in old format, use default
        },
        "depth_map": {
            "resolution": old_params.get("resolution", 0.005),
            "interpolation_window": old_params.get("window_size", 9)
        }
    }
    return new

def convert_detection(old_params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert detection parameters from old to new format."""
    # Handle morphological_kernel_size (can be list or single value)
    kernel_size = old_params.get("morphological_kernel_size", [3, 3])
    if isinstance(kernel_size, list):
        dilation_kernel_size = kernel_size[0] if len(kernel_size) > 0 else 3
    else:
        dilation_kernel_size = kernel_size
    
    # Handle angle ranges
    angle_pos = old_params.get("angle_range_oblique_positive", [6, 9])
    angle_neg = old_params.get("angle_range_oblique_negative", [-9, -6])
    
    new = {
        "description": "Hough line detection and segment pattern inference parameters",
        "physical_constants": {
            "ring_spacing_m": old_params.get("ring_spacing_constant", 1.2),
            "k_height_mm": 1079.92,  # Not in old format, use default
            "ab_height_mm": 3239.77,  # Not in old format, use default
            "segment_width_mm": 1200,  # Not in old format, use default
            "oblique_angle_deg": 7.52,  # Not in old format, use default
            "resolution": old_params.get("resolution", 0.005)
        },
        "preprocessing": {
            "binary_threshold": old_params.get("binary_threshold", 127),
            "dilation_kernel_size": dilation_kernel_size,
            "dilation_iterations": old_params.get("dilation_iterations", 1)
        },
        "hough_oblique": {
            "rho": 1,  # Not in old format, use default
            "theta_deg": 1.0,  # Not in old format, use default
            "threshold": old_params.get("hough_threshold_oblique", 50),
            "min_length": old_params.get("minLineLength_oblique", 100),
            "max_gap": old_params.get("maxLineGap_oblique", 40),
            "angle_positive_min": angle_pos[0] if isinstance(angle_pos, list) else angle_pos,
            "angle_positive_max": angle_pos[1] if isinstance(angle_pos, list) and len(angle_pos) > 1 else 9,
            "angle_negative_min": angle_neg[0] if isinstance(angle_neg, list) else angle_neg,
            "angle_negative_max": angle_neg[1] if isinstance(angle_neg, list) and len(angle_neg) > 1 else -6
        },
        "hough_horizontal": {
            "threshold": old_params.get("hough_threshold_horizontal", 50),
            "min_length": old_params.get("minLineLength_horizontal", 100),
            "max_gap": old_params.get("maxLineGap_horizontal", 10),
            "angle_tolerance": 1  # Not in old format, use default
        },
        "hough_vertical": {
            "threshold": old_params.get("hough_threshold_vertical", 500),
            "angle_tolerance": 0.5,  # Not in old format, use default
            "filter_rings": 5  # Not in old format, use default
        },
        "line_processing": {
            "merge_distance_threshold": old_params.get("merge_distance", 3),
            "intersection_merge_threshold": 6,  # Not in old format, use default
            "pattern_tolerance": 10,  # Not in old format, use default
            "horizontal_pattern_tolerance": 50  # Not in old format, use default
        }
    }
    return new

def convert_sam(old_params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert SAM parameters from old to new format (simplified)."""
    new = {
        "description": "SAM-based tunnel segment segmentation parameters",
        "segment_geometry": {
            "segment_width": old_params.get("segment_width", 1200),
            "k_height": old_params.get("K_height", 1079.92),
            "ab_height": old_params.get("AB_height", 3239.77),
            "angle_deg": old_params.get("angle", 7.52)
        },
        "image": {
            "resolution": old_params.get("processing", {}).get("resolution", 0.005) if isinstance(old_params.get("processing"), dict) else 0.005
        },
        "processing": {
            "mode": "auto"
        }
    }
    return new

def convert_tunnel(tunnel_id: str):
    """Convert all parameter files for a tunnel."""
    configurable_dir = Path("configurable") / tunnel_id
    output_dir = Path("p4tun") / "parameters" / tunnel_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conversions = {
        "parameters_unfolding.json": convert_unfolding,
        "parameters_denoising.json": convert_denoising,
        "parameters_enhancing.json": convert_enhancing,
        "parameters_sam.json": convert_sam
    }
    
    # Detection file has different name in old format
    detecting_file = configurable_dir / "parameters_detecting.json"
    if detecting_file.exists():
        with open(detecting_file, 'r') as f:
            old_detection = json.load(f)
        new_detection = convert_detection(old_detection)
        output_file = output_dir / "parameters_detection.json"
        with open(output_file, 'w') as f:
            json.dump(new_detection, f, indent=4)
        print(f"✓ Converted detection parameters for {tunnel_id}")
    
    # Convert other files
    for old_filename, convert_func in conversions.items():
        old_file = configurable_dir / old_filename
        if old_file.exists():
            with open(old_file, 'r') as f:
                old_params = json.load(f)
            new_params = convert_func(old_params)
            output_file = output_dir / old_filename
            with open(output_file, 'w') as f:
                json.dump(new_params, f, indent=4)
            print(f"✓ Converted {old_filename} for {tunnel_id}")
        else:
            print(f"⚠ Missing {old_filename} for {tunnel_id}")

if __name__ == "__main__":
    tunnels = ["1-4", "2-2", "3-1", "4-1", "5-1"]
    for tunnel_id in tunnels:
        print(f"\nConverting {tunnel_id}...")
        convert_tunnel(tunnel_id)
    print("\n✓ All conversions complete!")

