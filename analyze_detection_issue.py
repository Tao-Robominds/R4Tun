"""
Analyze why p4tun/4-1_detection.py finds wrong K-block positions.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import from p4tun module
import importlib.util
spec = importlib.util.spec_from_file_location("detection", "p4tun/4-1_detection.py")
detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detection)

# Use functions from detection module
detect_oblique_lines = detection.detect_oblique_lines
detect_horizontal_lines = detection.detect_horizontal_lines
detect_vertical_lines = detection.detect_vertical_lines
preprocess_depth_map = detection.preprocess_depth_map
find_intersection = detection.find_intersection
generate_center_lines = detection.generate_center_lines
generate_fallback_center_lines = detection.generate_fallback_center_lines

tunnel_id = "4-1"
tunnel_dir = f"data/{tunnel_id}"

# Load depth map
depth_map = cv2.imread(f'{tunnel_dir}/depth_map.png', cv2.IMREAD_GRAYSCALE)
if depth_map is None:
    depth_map = np.load(f'{tunnel_dir}/depth_map_outlier.npy')

height, width = depth_map.shape

# Load parameters
import json
with open(f'p4tun/parameters/{tunnel_id}/parameters_detection.json', 'r') as f:
    params = json.load(f)

# Preprocess
edge_image = preprocess_depth_map(
    depth_map,
    params['preprocessing']['binary_threshold'],
    params['preprocessing']['dilation_kernel_size'],
    params['preprocessing']['dilation_iterations']
)

# Detect lines
positive_lines, negative_lines = detect_oblique_lines(
    edge_image,
    params['hough_oblique']['rho'],
    np.pi / 180 * params['hough_oblique']['theta_deg'],
    params['hough_oblique']['threshold'],
    params['hough_oblique']['min_length'],
    params['hough_oblique']['max_gap'],
    params['hough_oblique']['angle_positive_min'],
    params['hough_oblique']['angle_positive_max'],
    params['hough_oblique']['angle_negative_min'],
    params['hough_oblique']['angle_negative_max']
)

horizontal_lines = detect_horizontal_lines(
    edge_image,
    params['hough_horizontal']['threshold'],
    params['hough_horizontal']['min_length'],
    params['hough_horizontal']['max_gap'],
    params['hough_horizontal']['angle_tolerance']
)

with open(f'{tunnel_dir}/ring_count.txt', 'r') as f:
    ring_count = int(f.read().strip())

vertical_lines = detect_vertical_lines(
    edge_image,
    params['physical_constants']['resolution'],
    ring_count,
    params['hough_vertical']['threshold'],
    params['hough_vertical']['angle_tolerance'],
    params['hough_vertical']['filter_rings']
)

# Generate center lines
if vertical_lines:
    center_lines = generate_center_lines(vertical_lines, width, height, ring_count)
else:
    center_lines = generate_fallback_center_lines(width, ring_count)

print("=" * 70)
print("DETECTION ANALYSIS FOR TUNNEL 4-1")
print("=" * 70)
print(f"Image: {width} x {height}")
print(f"Positive oblique lines: {len(positive_lines)}")
print(f"Negative oblique lines: {len(negative_lines)}")
print(f"Horizontal lines: {len(horizontal_lines)}")
print(f"Vertical lines: {len(vertical_lines)}")
print(f"Center lines: {len(center_lines)}")
print()

# Analyze first ring center
cx = center_lines[0]
center_line = (cx, 0, cx, height)

print(f"Analyzing Ring 1 (center X = {cx:.1f})")
print("-" * 70)

# Find all intersections
positive_intersections = []
for line in positive_lines:
    pt = find_intersection(center_line, line)
    if pt and 0 <= pt[1] <= height:
        positive_intersections.append(pt)

negative_intersections = []
for line in negative_lines:
    pt = find_intersection(center_line, line)
    if pt and 0 <= pt[1] <= height:
        negative_intersections.append(pt)

horizontal_intersections = []
for line in horizontal_lines:
    pt = find_intersection(center_line, line)
    if pt and 0 <= pt[1] <= height:
        horizontal_intersections.append(pt)

print(f"Positive intersections: {len(positive_intersections)}")
if positive_intersections:
    pos_ys = sorted([p[1] for p in positive_intersections])
    print(f"  Y values: {pos_ys[:5]} ... {pos_ys[-5:] if len(pos_ys) > 5 else pos_ys}")
    print(f"  First Y: {pos_ys[0]:.1f}")

print(f"Negative intersections: {len(negative_intersections)}")
if negative_intersections:
    neg_ys = sorted([n[1] for n in negative_intersections])
    print(f"  Y values: {neg_ys[:5]} ... {neg_ys[-5:] if len(neg_ys) > 5 else neg_ys}")
    print(f"  First Y: {neg_ys[0]:.1f}")

print(f"Horizontal intersections: {len(horizontal_intersections)}")
if horizontal_intersections:
    h_ys = sorted([h[1] for h in horizontal_intersections])
    print(f"  Y values: {h_ys[:5]} ... {h_ys[-5:] if len(h_ys) > 5 else h_ys}")

# Current method (wrong)
if positive_intersections and negative_intersections:
    pos_y = positive_intersections[0][1]
    neg_y = negative_intersections[0][1]
    mid_y = (pos_y + neg_y) / 2
    print()
    print("CURRENT METHOD (WRONG):")
    print(f"  Using first positive Y: {pos_y:.1f}")
    print(f"  Using first negative Y: {neg_y:.1f}")
    print(f"  Midpoint Y: {mid_y:.1f}")
    print(f"  This is WRONG! (should be ~3338)")

# Expected method (should find Y ~3338)
print()
print("EXPECTED RESULT:")
print(f"  Old detected Y: 3338.4")
print(f"  This is at {3338.4/height*100:.1f}% of image height")
print()
print("PROBLEM:")
print("  The current method takes the FIRST intersection from unsorted lists.")
print("  It should filter/select intersections that are near the expected K-block position.")
print("  Expected K-block Y is around 3338 (71% of image height).")

