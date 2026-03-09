"""
Test: Run fyr's detection (from data/fyr/6.self-reflection/5-1/code/detecting_modification.py)
then our agents SAM with fyr params. Goal: see if detection is the bottleneck for fyr mIoU 0.429.
"""

import os
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd

# Run fyr detection (headless - no plt.show)
def run_fyr_detection(tunnel_id="5-1", data_dir="data"):
    base_dir = os.path.join(data_dir, tunnel_id)
    depth_map_outlier = np.load(os.path.join(base_dir, "depth_map_outlier.npy"))
    resolution = 0.005
    with open(os.path.join(base_dir, "ring_count.txt")) as f:
        ring_count = int(f.read())

    binary_map = np.where(np.isnan(depth_map_outlier), 0, 255).astype(np.uint8)
    ret, binary_image = cv2.threshold(binary_map, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(binary_image, kernel, iterations=2)

    L, W = binary_map.shape
    lines_oblique = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, 40, minLineLength=150, maxLineGap=50)
    lines_horizontal = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, 40, minLineLength=150, maxLineGap=15)
    lines_vertical = cv2.HoughLines(dilated_edges, 1, np.pi/180, 300)
    if lines_vertical is not None:
        lines_vertical = lines_vertical[lines_vertical[:, 0, 0] <= (5 * 1200 / (resolution * 1000))]

    joint_oblique_positive, joint_oblique_negtive, joint_horizontal = [], [], []
    if lines_oblique is not None:
        for line in lines_oblique:
            x1, y1, x2, y2 = line[0]
            x1, x2, y1, y2 = (x2, x1, y2, y1) if x1 > x2 else (x1, x2, y1, y2)
            angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
            if 6 <= angle <= 9:
                joint_oblique_positive.append(line)
            elif -9 <= angle <= -6:
                joint_oblique_negtive.append(line)
    if lines_horizontal is not None:
        for line in lines_horizontal:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -1 <= angle <= 1:
                joint_horizontal.append(line)

    merged_lines = []
    if lines_vertical is not None:
        lines_vertical = lines_vertical[:, 0]
        for (rho1, theta1) in lines_vertical:
            if -0.5 * np.pi / 180 <= abs(theta1) <= 0.5 * np.pi / 180:
                x1, y1 = rho1 * np.cos(theta1), rho1 * np.sin(theta1)
                is_merged = False
                for j, (rho2, theta2) in enumerate(merged_lines):
                    x2, y2 = rho2 * np.cos(theta2), rho2 * np.sin(theta2)
                    if np.hypot(x1 - x2, y1 - y2) < 10:
                        merged_lines[j] = ((rho1 + rho2) / 2, (theta1 + theta2) / 2)
                        is_merged = True
                        break
                if not is_merged:
                    merged_lines.append((rho1, theta1))
        merged_lines.sort(key=lambda line: line[0])

    mid_lines = []
    if len(merged_lines) >= 2:
        for i in range(len(merged_lines) - 1):
            rho1, theta1 = merged_lines[i]
            rho2, theta2 = merged_lines[i + 1]
            mid_lines.append(((rho1 + rho2) / 2, (theta1 + theta2) / 2))
        distances = [np.hypot(
            mid_lines[i][0]*np.cos(mid_lines[i][1]) - mid_lines[i+1][0]*np.cos(mid_lines[i+1][1]),
            mid_lines[i][0]*np.sin(mid_lines[i][1]) - mid_lines[i+1][0]*np.sin(mid_lines[i+1][1])
        ) for i in range(len(mid_lines)-1)]
        avg_distance_detected = np.mean(distances) if distances else 0
        avg_distance_designed = W / ring_count
        avg_distance = avg_distance_detected if abs(avg_distance_detected - (1.2/resolution)) <= abs(avg_distance_designed - (1.2/resolution)) else avg_distance_designed
        all_mid_lines = list(mid_lines)
        if mid_lines:
            leftmost_rho, leftmost_theta = mid_lines[0]
            a, b = np.cos(leftmost_theta), np.sin(leftmost_theta)
            x0 = a * leftmost_rho
            while x0 >= 0:
                all_mid_lines.append((x0, leftmost_theta))
                x0 -= avg_distance
            rightmost_rho, rightmost_theta = mid_lines[-1]
            a, b = np.cos(rightmost_theta), np.sin(rightmost_theta)
            x0 = a * rightmost_rho
            while x0 <= W:
                all_mid_lines.append((x0, rightmost_theta))
                x0 += avg_distance
            all_mid_lines = sorted(list(set(all_mid_lines)), key=lambda t: t[0])
    else:
        all_mid_lines = []
        block_width = W / ring_count
        for i in range(ring_count):
            all_mid_lines.append(((i + 0.5) * block_width, 0))

    def line_segment_vertical_intersection(vertical_x, segment):
        x1, y1, x2, y2 = segment
        if x1 == x2:
            return None
        if min(x1, x2) <= vertical_x <= max(x1, x2):
            t = (vertical_x - x1) / (x2 - x1)
            return (vertical_x, y1 + t * (y2 - y1))
        return None

    def merge_close_points(points, threshold=6):
        points = np.array(points)
        if len(points) < 2:
            return points
        merged = []
        while len(points):
            p = points[0]
            close = np.linalg.norm(points - p, axis=1) < threshold
            merged.append(np.mean(points[close], axis=0))
            points = points[~close]
        return np.array(merged)

    def compute_midpoint(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def check_distance_pattern(points, k, ab, tolerance=10):
        points = sorted(points, key=lambda p: p[0])
        for i in range(len(points) - 1):
            for j in range(i+1, len(points)):
                dist = np.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
                if any(abs(dist - (k + m * ab)) < tolerance for m in [2, 4]):
                    return compute_midpoint(points[i], points[j])
        return None

    K_height_pixel = 1079.92 / (1000 * resolution)
    AB_height_pixel = 3239.77 / (1000 * resolution)
    adjusted_points = []

    for vertical_x, _ in all_mid_lines:
        # vertical_x is rho (x-position for theta≈0) or x_pos for fallback
        vertical_x_val = float(vertical_x)
        pos_pts = [line_segment_vertical_intersection(vertical_x_val, s[0]) for s in joint_oblique_positive]
        neg_pts = [line_segment_vertical_intersection(vertical_x_val, s[0]) for s in joint_oblique_negtive]
        pos_pts = [p for p in pos_pts if p is not None]
        neg_pts = [p for p in neg_pts if p is not None]
        mp = merge_close_points(pos_pts)
        mn = merge_close_points(neg_pts)

        if len(mp) > 0 and len(mn) > 0:
            adjusted_points.append(('midpoint', compute_midpoint(mp[0], mn[0])))
        elif len(mp) > 0:
            adjusted_points.append(('positive_slope', (mp[0][0], mp[0][1] - 0.5 * K_height_pixel)))
        elif len(mn) > 0:
            adjusted_points.append(('negative_slope', (mn[0][0], mn[0][1] + 0.5 * K_height_pixel)))
        else:
            hor_pts = [line_segment_vertical_intersection(vertical_x_val, s[0]) for s in joint_horizontal]
            hor_pts = [p for p in hor_pts if p is not None]
            mh = merge_close_points(hor_pts)
            pat_mid = check_distance_pattern(mh, K_height_pixel, AB_height_pixel, tolerance=50)
            if pat_mid:
                adjusted_points.append(('horizontal', pat_mid))
            else:
                if adjusted_points:
                    last_y = adjusted_points[-1][1][1]
                    if 1035 <= last_y <= 1265:
                        assumed_y = last_y + 431.87
                    elif 1422 <= last_y <= 1738:
                        assumed_y = last_y - 431.87
                    elif len(adjusted_points) > 1:
                        sec_last_y = adjusted_points[-2][1][1]
                        assumed_y = sec_last_y if (1035 <= sec_last_y <= 1265 or 1422 <= sec_last_y <= 1738) else None
                    else:
                        assumed_y = None
                else:
                    assumed_y = None
                if assumed_y is not None:
                    adjusted_points.append(('assume', (vertical_x_val, assumed_y)))
                else:
                    adjusted_points.append(('default', (vertical_x_val, L / 2)))

    df_loc = pd.DataFrame(adjusted_points, columns=['Type', 'Coordinates'])
    df_loc['X'] = df_loc['Coordinates'].apply(lambda c: c[0])
    df_loc['Y'] = df_loc['Coordinates'].apply(lambda c: c[1])
    df_loc = df_loc.drop(columns=['Coordinates']).sort_values(by='X').reset_index(drop=True)
    df_loc['Confidence'] = 1.0  # for compatibility with agents
    return df_loc


def main():
    tunnel_id = "5-1"
    data_dir = "data"
    tunnel_dir = os.path.join(data_dir, tunnel_id)
    detected_csv = os.path.join(tunnel_dir, "detected.csv")
    backup_csv = os.path.join(tunnel_dir, "detected_backup_agents.csv")

    # Backup agents detection
    shutil.copy(detected_csv, backup_csv)
    print(f"Backed up agents detected.csv to {backup_csv}")

    # Run fyr detection
    print("Running fyr-style detection...")
    df_fyr = run_fyr_detection(tunnel_id, data_dir)
    df_fyr.to_csv(detected_csv, index=False)
    print(f"  Detected {len(df_fyr)} K positions")
    print(df_fyr[['Type', 'X', 'Y']].to_string())

    # Set fyr SAM params
    params_file = PROJECT_ROOT / "agents" / "complex_staggered" / "3_segmentation" / "parameters" / tunnel_id / "parameters_sam.json"
    fyr_params = {
        "resolution": 0.005, "k_height": 1079.92, "ab_height": 3239.77,
        "b1_height_top": 1500, "b1_height_bottom_pos": 1540.69, "b1_height_bottom_neg": 1699.08,
        "b2_height_top_pos": 1540.69, "b2_height_top_neg": 1699.08, "b2_height_bottom": 1500,
        "segment_width": 1300, "angle_deg": 7.52,
        "k_mask_width": 625, "k_mask_height_pos": 620, "k_mask_height_neg": 460,
        "ab_mask_width": 625, "ab_mask_height": 1620,
        "padding": 300, "crop_margin": 50,
        "min_quality_threshold": 0.3, "use_quality_weighting": True,
        "walk_order": [["K", 0], ["B1", 1], ["A1", 1], ["A2", 1], ["A3", 1], ["A4", 1], ["B2", -1]]
    }
    import json
    with open(params_file, 'w') as f:
        json.dump(fyr_params, f, indent=2)
    print(f"Set fyr SAM params (padding=300, segment_width=1300)")

    # Run SAM
    import importlib.util
    sam_dir = PROJECT_ROOT / 'agents' / 'complex_staggered' / '3_segmentation'
    spec = importlib.util.spec_from_file_location("sam", sam_dir / "3_sam.py")
    sam_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sam_module)
    print("Running agents SAM...")
    sam_module.run_sam(tunnel_id, data_dir)

    # Evaluate
    eval_dir = PROJECT_ROOT / 'agents' / 'complex_staggered'
    spec = importlib.util.spec_from_file_location("evaluation", eval_dir / "evaluation.py")
    eval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_module)
    segment_count = eval_module.detect_segment_count(tunnel_dir, default=7)
    final_csv = os.path.join(tunnel_dir, "final.csv")
    df = pd.read_csv(final_csv)
    gt = np.nan_to_num(df["segment"].values, nan=-1).astype(int)
    pred = np.nan_to_num(df["pred"].values, nan=-1).astype(int)
    results = eval_module.calculate_metrics(gt, pred, eval_module.get_class_names(segment_count), segment_count)
    miou = results["mIoU"]
    print(f"\n{'='*60}")
    print(f"fyr detection + fyr SAM params: mIoU = {miou:.4f} (fyr target: 0.429)")
    print(f"{'='*60}")

    # Restore agents detection
    shutil.copy(backup_csv, detected_csv)
    print(f"\nRestored agents detected.csv")


if __name__ == "__main__":
    main()
