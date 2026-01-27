import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Specify the tunnel ID
tunnel_id = "5-1"
base_dir = f"data/{tunnel_id}"

# Load data
depth_map_outlier = np.load(os.path.join(base_dir, "depth_map_outlier.npy"))
resolution = 0.005
ring_count = int(open(f'data/{tunnel_id}/ring_count.txt', 'r').read())

# Cell 4: pre-processing
binary_map = np.where(np.isnan(depth_map_outlier), 0, 255).astype(np.uint8)

# Lower binarization threshold to preserve fainter joint edges
ret, binary_image = cv2.threshold(binary_map, 120, 255, cv2.THRESH_BINARY)

# Use a larger morphological kernel and increase dilation to close wider gaps
kernel = np.ones((5, 5), np.uint8)
dilated_edges = cv2.dilate(binary_image, kernel, iterations=2)

# Cell 5: detection
L, W = binary_map.shape

# Adjusted Hough parameters for oblique, horizontal, and vertical detection
lines_oblique = cv2.HoughLinesP(
    dilated_edges, 1, np.pi / 180,
    40,             # lower vote threshold
    minLineLength=150,
    maxLineGap=50
)
lines_horizontal = cv2.HoughLinesP(
    dilated_edges, 1, np.pi / 180,
    40,
    minLineLength=150,
    maxLineGap=15
)
lines_vertical = cv2.HoughLines(dilated_edges, 1, np.pi / 180, 300)
if lines_vertical is not None:
    lines_vertical = lines_vertical[lines_vertical[:, 0, 0] <= (5 * 1200 / (resolution * 1000))]

# Prepare output image
output_image = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)

# Define drawing parameters
color_angle1 = (255, 0, 0)
color_angle2 = (0, 255, 0)
color_horizontal = (0, 0, 255)
color_vertical = (255, 165, 0)
color_mid_lines = (255, 0, 255)
line_thickness = 3

# Detect and draw oblique lines
joint_oblique_positive = []
joint_oblique_negtive = []
joint_horizontal = []

if lines_oblique is not None:
    for line in lines_oblique:
        x1, y1, x2, y2 = line[0]
        x1, x2, y1, y2 = (x2, x1, y2, y1) if x1 > x2 else (x1, x2, y1, y2)
        angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
        if 6 <= angle <= 9:
            joint_oblique_positive.append(line)
            cv2.line(output_image, (x1, y1), (x2, y2), color_angle1, line_thickness)
        elif -9 <= angle <= -6:
            joint_oblique_negtive.append(line)
            cv2.line(output_image, (x1, y1), (x2, y2), color_angle2, line_thickness)

# Detect and draw horizontal lines
if lines_horizontal is not None:
    for line in lines_horizontal:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -1 <= angle <= 1:
            joint_horizontal.append(line)
            cv2.line(output_image, (x1, y1), (x2, y2), color_horizontal, line_thickness)

# Merge close vertical lines
merged_lines = []
all_mid_lines = []
threshold_distance = 10  # increased to 10 pixels

if lines_vertical is not None:
    lines_vertical = lines_vertical[:, 0]
    for i, (rho1, theta1) in enumerate(lines_vertical):
        if -0.5 * np.pi / 180 <= abs(theta1) <= 0.5 * np.pi / 180:
            x1, y1 = rho1 * np.cos(theta1), rho1 * np.sin(theta1)
            is_merged = False
            for j, (rho2, theta2) in enumerate(merged_lines):
                x2, y2 = rho2 * np.cos(theta2), rho2 * np.sin(theta2)
                if np.hypot(x1 - x2, y1 - y2) < threshold_distance:
                    merged_lines[j] = ((rho1 + rho2) / 2, (theta1 + theta2) / 2)
                    is_merged = True
                    break
            if not is_merged:
                merged_lines.append((rho1, theta1))
    merged_lines.sort(key=lambda line: line[0])

    for rho, theta in merged_lines:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1 = int(x0 + 2677 * (-b))
        y1 = int(y0 + 2677 * (a))
        x2 = int(x0 - 2677 * (-b))
        y2 = int(y0 - 2677 * (a))
        cv2.line(output_image, (x1, y1), (x2, y2), color_vertical, line_thickness)

    mid_lines = []
    for i in range(len(merged_lines) - 1):
        rho1, theta1 = merged_lines[i]
        rho2, theta2 = merged_lines[i + 1]
        new_rho = (rho1 + rho2) / 2
        new_theta = (theta1 + theta2) / 2
        mid_lines.append((new_rho, new_theta))
        a, b = np.cos(new_theta), np.sin(new_theta)
        x0, y0 = a * new_rho, b * new_rho
        x1 = int(x0 + L * (-b))
        y1 = int(y0 + L * (a))
        x2 = int(x0 - L * (-b))
        y2 = int(y0 - L * (a))
        cv2.line(output_image, (x1, y1), (x2, y2), color_mid_lines, line_thickness)

    distances = []
    for i in range(len(mid_lines) - 1):
        rho1, theta1 = mid_lines[i]
        rho2, theta2 = mid_lines[i + 1]
        x1, y1 = rho1 * np.cos(theta1), rho1 * np.sin(theta1)
        x2, y2 = rho2 * np.cos(theta2), rho2 * np.sin(theta2)
        distances.append(np.hypot(x1 - x2, y1 - y2))
    avg_distance_detected = np.mean(distances) if distances else 0
    avg_distance_designed = W / ring_count

    if abs(avg_distance_detected - (1.2 / resolution)) <= abs(avg_distance_designed - (1.2 / resolution)):
        avg_distance = avg_distance_detected
    else:
        avg_distance = avg_distance_designed

    all_mid_lines = mid_lines.copy()

    if mid_lines:
        leftmost_rho, leftmost_theta = mid_lines[0]
        a, b = np.cos(leftmost_theta), np.sin(leftmost_theta)
        x0, y0 = a * leftmost_rho, b * leftmost_rho
        while x0 >= 0:
            x1 = int(x0 + L * (-b))
            y1 = int(y0 + L * (a))
            x2 = int(x0 - L * (-b))
            y2 = int(y0 - L * (a))
            cv2.line(output_image, (x1, y1), (x2, y2), color_mid_lines, line_thickness)
            all_mid_lines.append((x0, leftmost_theta))
            x0 -= avg_distance

        rightmost_rho, rightmost_theta = mid_lines[-1]
        a, b = np.cos(rightmost_theta), np.sin(rightmost_theta)
        x0, y0 = a * rightmost_rho, b * rightmost_rho
        while x0 <= output_image.shape[1]:
            x1 = int(x0 + L * (-b))
            y1 = int(y0 + L * (a))
            x2 = int(x0 - L * (-b))
            y2 = int(y0 - L * (a))
            cv2.line(output_image, (x1, y1), (x2, y2), color_mid_lines, line_thickness)
            all_mid_lines.append((x0, rightmost_theta))
            x0 += avg_distance

    all_mid_lines = sorted(list(set(all_mid_lines)), key=lambda line: line[0])

# Fallback if no vertical lines detected
if lines_vertical is None or len(all_mid_lines) == 0:
    print("No vertical lines detected. Using fallback evenly spaced method.")
    all_mid_lines = []
    block_width = W / ring_count
    for i in range(ring_count):
        x_pos = (i + 0.5) * block_width
        all_mid_lines.append((x_pos, 0))
        x1, y1, x2, y2 = int(x_pos), 0, int(x_pos), L
        cv2.line(output_image, (x1, y1), (x2, y2), color_mid_lines, line_thickness)
    print(f"Generated {len(all_mid_lines)} synthetic vertical lines at ring centers")

# Save and display detected lines
plt.figure(figsize=(12, 12))
plt.imshow(output_image)
os.makedirs(base_dir, exist_ok=True)
plt.savefig(f'{base_dir}/detected_lines.png', dpi=300, bbox_inches='tight')

# Cell 6: Intersection and point adjustments
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

vertical_lines = all_mid_lines
horizontal_lines = joint_horizontal
positive_slope_lines = joint_oblique_positive
negative_slope_lines = joint_oblique_negtive

adjusted_points = []
K_height_pixel = 1079.92 / (1000 * resolution)
AB_height_pixel = 3239.77 / (1000 * resolution)

for vertical_x, _ in vertical_lines:
    pos_pts, neg_pts, hor_pts = [], [], []
    for seg in positive_slope_lines:
        ip = line_segment_vertical_intersection(vertical_x, seg[0])
        if ip: pos_pts.append(ip)
    for seg in negative_slope_lines:
        ip = line_segment_vertical_intersection(vertical_x, seg[0])
        if ip: neg_pts.append(ip)
    mp = merge_close_points(pos_pts)
    mn = merge_close_points(neg_pts)

    if len(mp) > 0 and len(mn) > 0:
        adjusted_points.append(('midpoint', compute_midpoint(mp[0], mn[0])))
    elif len(mp) > 0:
        p = mp[0]
        adjusted_points.append(('positive_slope', (p[0], p[1] - 0.5 * K_height_pixel)))
    elif len(mn) > 0:
        p = mn[0]
        adjusted_points.append(('negative_slope', (p[0], p[1] + 0.5 * K_height_pixel)))
    else:
        for seg in horizontal_lines:
            ip = line_segment_vertical_intersection(vertical_x, seg[0])
            if ip: hor_pts.append(ip)
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
                    if 1035 <= sec_last_y <= 1265 or 1422 <= sec_last_y <= 1738:
                        assumed_y = sec_last_y
                    else:
                        assumed_y = None
                else:
                    assumed_y = None
            else:
                assumed_y = None

            if assumed_y is not None:
                adjusted_points.append(('assume', (vertical_x, assumed_y)))
            else:
                default_y = L / 2
                adjusted_points.append(('default', (vertical_x, default_y)))
                print(f"Warning: Using default y-coordinate ({default_y}) for x = {vertical_x}")

# Record and save
df_loc = pd.DataFrame(adjusted_points, columns=['Type', 'Coordinates'])
df_loc['X'] = df_loc['Coordinates'].apply(lambda c: c[0])
df_loc['Y'] = df_loc['Coordinates'].apply(lambda c: c[1])
df_loc = df_loc.drop(columns=['Coordinates']).sort_values(by='X').reset_index(drop=True)

print(f"Number of vertical lines: {len(vertical_lines)}")
print(f"Number of adjusted points: {len(adjusted_points)}")
print(df_loc)

# Cell 7: Visualization (optional)
plt.figure(figsize=(16, 16))
ax = plt.gca()
colors = {'horizontal': 'b', 'positive_slope': 'r', 'negative_slope': 'c',
          'midpoint': 'm', 'assume': 'g', 'default': 'orange'}
markers = {'horizontal': 'o', 'positive_slope': '^', 'negative_slope': 's',
           'midpoint': '*', 'assume': 'd', 'default': 'x'}
for lbl, (x, y) in adjusted_points:
    ax.plot(x, y, color=colors[lbl], marker=markers[lbl], markersize=10, label=lbl)

handles, labels = ax.get_legend_handles_labels()
ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc='lower right')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Intersection Points')
ax.set_aspect('equal', adjustable='box')
ax.invert_yaxis()

x_min, x_max = df_loc['X'].min(), df_loc['X'].max()
y_min, y_max = df_loc['Y'].min(), df_loc['Y'].max()
margin = 0.1
ax.set_xlim(x_min - margin*(x_max - x_min), x_max + margin*(x_max - x_min))
ax.set_ylim(y_max + margin*(y_max - y_min), y_min - margin*(y_max - y_min))
plt.grid(True)
plt.tight_layout()
plt.show()

# Cell 8: Save detected points
df_loc.to_csv(f'{base_dir}/detected.csv', index=False)