import os
import numpy as np
import pandas as pd
import torch
import cv2
import math
from tqdm import tqdm
import pickle
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from matplotlib.path import Path
import matplotlib
matplotlib.use('Agg')  # headless: save figures to files instead of plt.show()
import matplotlib.pyplot as plt
import sys


# --- Visualization helpers (mirrors notebook cell 64) ---
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
               s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def save_instance_checks(results, filename, ring_index=0):
    """Per-instance check (mirrors notebook cell 74): crop + mask + prompt points for
    every block of one ring, saved as a grid."""
    if not results or ring_index >= len(results) or not results[ring_index]:
        return
    ring = results[ring_index]
    n = len(ring)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False)
    axes = axes.ravel()
    for idx, item in enumerate(ring):
        ax = axes[idx]
        ax.imshow(item['cropped_image'])
        show_mask(item['mask'], ax)
        show_points(item['points'], item['labels'], ax, marker_size=120)
        score = float(np.ravel(item['score'])[0])
        ax.set_title(f"ring{ring_index} {item['block']} score={score:.3f}")
        ax.axis('off')
    for j in range(n, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close(fig)


def visualize_combined_results(image, result_image, ring_image, filename):
    """Combined semantic + ring-instance overlays (mirrors notebook cell 77)."""
    cmap_result = plt.colormaps['tab10']
    cmap_ring = plt.get_cmap('tab20')

    colored_result = np.zeros((*result_image.shape, 3))
    colored_ring = np.zeros((*ring_image.shape, 3))

    for label in range(7):  # 0-6: background + 6 blocks
        colored_result[result_image == label] = cmap_result(label)[:3]

    unique_rings = np.unique(ring_image)
    for ring in unique_rings:
        colored_ring[ring_image == ring] = cmap_ring(int(ring) % 20)[:3]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18),
                                        gridspec_kw={'height_ratios': [1, 1, 1]})
    plt.subplots_adjust(hspace=0.1)

    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Depth Map', pad=10)
    ax1.axis('off')

    ax2.imshow(image, cmap='gray', alpha=0.5)
    ax2.imshow(colored_result, alpha=0.5)
    ax2.set_title('Labeled Mask Results', pad=10)
    ax2.axis('off')

    ax3.imshow(image, cmap='gray', alpha=0.5)
    ax3.imshow(colored_ring, alpha=0.5)
    ax3.set_title('Ring Instance Segmentation Results', pad=10)
    ax3.axis('off')

    legend_elements_result = [
        plt.Line2D([0], [0], color=cmap_result(label), lw=4, label=f'{block} (Label {label})')
        for label, block in {v: k for k, v in block_to_label.items()}.items()
    ]
    ax2.legend(handles=legend_elements_result, loc='center left', bbox_to_anchor=(1, 0.5))

    legend_elements_ring = [
        plt.Line2D([0], [0], color=cmap_ring(int(ring) % 20), lw=4, label=f'Ring {ring}')
        for ring in unique_rings
    ]
    ax3.legend(handles=legend_elements_ring, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

# Check if tunnel_id is provided
if len(sys.argv) != 2:
    print("Usage: python 4-2_sam.py <tunnel_id>")
    print("Example: python 4-2_sam.py 1-4")
    sys.exit(1)

tunnel_id = sys.argv[1]
base_dir = f"data/{tunnel_id}/"
initial_prompt_points = pd.read_csv(os.path.join(base_dir, "detected.csv"))
pixel_to_point = pickle.load(open(os.path.join(base_dir, "pixel_to_point.pkl"), "rb"))
df_point_cloud = pd.read_csv(os.path.join(base_dir, "enhanced.csv"))
ring_count = int(open(f'data/{tunnel_id}/ring_count.txt', 'r').read())

print(f"Processing tunnel: {tunnel_id}")

sam_checkpoint = "sam4tun/segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

image = cv2.imread(f'{base_dir}/depth_map.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def fill_polygon(mask, vertices):
    path = Path(vertices)
    y_coords, x_coords = np.mgrid[:mask.shape[0], :mask.shape[1]]
    points = np.vstack((x_coords.flatten(), y_coords.flatten())).T
    mask_inside = path.contains_points(points).reshape(mask.shape)
    mask[mask_inside] = 1

def generate_template_mask(height, width, prompt_centre, block, resolution=0.005, K_height=1079.92):
    mask = np.zeros((height, width), dtype=np.uint8)
    prompt_centre_x, prompt_centre_y = prompt_centre
    x = prompt_centre_x * (resolution*1000)
    y = prompt_centre_y * (resolution*1000)
    
    if block == 'K':
        # Tighter K parallelogram: scale template to nominal K_height (was ~1238 mm tall).
        ky = K_height / (2 * 619.16)
        vertices_real = np.array([
            [x-625, y-619.16*ky], [x-625, y+619.16*ky],
            [x+625, y+460.77*ky], [x+625, y-460.77*ky],
        ])
    elif block == 'B1':
        vertices_real = np.array([[x-625,y-1619.89],[x-625,y+1540.69],[x+625,y+1699.08],[x+625,y-1619.89]])
    elif block == 'B2':
        vertices_real = np.array([[x-625,y-1540.69],[x-625,y+1619.89],[x+625,y+1619.89],[x+625,y-1699.08]])
    else:
        vertices_real = np.array([[x-625,y-1619.89],[x-625,y+1619.89],[x+625,y+1619.89],[x+625,y-1619.89]])
        
    vertices = vertices_real / (resolution*1000)
    fill_polygon(mask, vertices)
    return mask

def generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution=0.005,
                           segment_width=1200, K_height=1079.92, AB_height=3239.77):
    prompt_centre_x, prompt_centre_y = prompt_centre
    x = prompt_centre_x * (resolution*1000)
    y = prompt_centre_y * (resolution*1000)
    map_y = map_y * (resolution*1000)
    
    if block == 'K':
        points_real = np.array([
            [x-700,y-732.35],[x-700,y-505.96],[x-700,y-310.91],[x-700,y],[x-700,y+310.91],[x-700,y+505.96],[x-700,y+732.35],
            [x-500,y-705.96],[x-500,y+705.96],
            [x-348.16,y-685.91],[x-348.16,y-310.91],[x-325,y],[x-348.16,y+310.91],[x-348.16,y+685.91],
            [x,y-639.96],[x,y],[x,y+639.96],
            [x+348.16,y-594.01],[x+348.16,y-219.01],[x+325,y],[x+348.16,y+219.01],[x+348.16,y+594.01],
            [x+500,y-573.96],[x+500,y+573.96],
            [x+700,y-547.57],[x+700,y-373.96],[x+700,y-219.01],[x+700,y],[x+700,y+219.01],[x+700,y+373.96],[x+700,y+547.57],
            [x-500,y-505.96],[x-511.06,y-310.91],[x-500,y],[x-511.06,y+310.91],[x-500,y+505.96],
            [x-348.16,y-485.91],[x-348.16,y+485.91],
            [x,y-439.96],[x,y+439.96],
            [x+348.16,y-394.01],[x+348.16,y+394.01],
            [x+500,y-373.96],[x+511.06,y-219.01],[x+500,y],[x+511.06,y+219.01],[x+500,y+373.96]
        ])
        labels = np.repeat([0, 1], [31, 16])
    elif block == 'B1':
        points_real = np.array([
            [x-700,y-1719.89],[x-511.06,y-1719.89],[x-348.16,y-1719.89],[x,y-1719.89],[x+348.16,y-1719.89],[x+511.06,y-1719.89],[x+700,y-1719.89],
            [x-700,y-1519.89],[x+700,y-1519.89],
            [x-700,y-1344.89],[x-348.16,y-1344.89],[x+348.16,y-1344.89],[x+700,y-1344.89],
            [x-700,y-1090.09],[x-325,y-1090.09],[x+325,y-1090.09],[x+700,y-1090.09],
            [x-700,y-817.57],[x+700,y-817.57],
            [x-700,y-545.05],[x+700,y-545.05],
            [x-700,y-272.52],[x+700,y-272.52],
            [x-700,y],[x-325,y],[x,y],[x+325,y],[x+700,y],
            [x-700,y+272.52],[x+700,y+272.52],
            [x-700,y+545.05],[x+700,y+545.05],
            [x-700,y+817.57],[x+700,y+817.57],
            [x-700,y+1090.09],[x-325,y+1090.09],[x+325,y+1090.09],[x+700,y+1090.09],
            [x-700,y+1298.93],[x-350,y+1298.93],[x+350,y+1390.84],[x+700,y+1390.84],
            [x-700,y+1427.43],[x+700,y+1612.28],
            [x-700,y+1627.49],[x-511.06,y+1652.43],[x-350,y+1673.69],[x,y+1719.89],[x+350,y+1766.08],[x+511.06,y+1787.34],[x+700,y+1812.28],
            [x-511.06,y-1519.89],[x-348.16,y-1519.89],[x,y-1519.89],[x+348.16,y-1519.89],[x+511.06,y-1519.89],
            [x-511.06,y-1344.89],[x,y-1344.89],[x+511.06,y-1344.89],
            [x-500,y-1090.09],[x,y-1090.09],[x+500,y-1090.09],
            [x-500,y-817.57],[x-250,y-817.57],[x,y-817.57],[x+250,y-817.57],[x+500,y-817.57],
            [x-500,y-545.05],[x-250,y-545.05],[x,y-545.05],[x+250,y-545.05],[x+500,y-545.05],
            [x-500,y-272.52],[x-250,y-272.52],[x,y-272.52],[x+250,y-272.52],[x+500,y-272.52],
            [x-500,y],[x-162.5,y],[x+162.5,y],[x+500,y],
            [x-500,y+272.52],[x-250,y+272.52],[x,y+272.52],[x+250,y+272.52],[x+500,y+272.52],
            [x-500,y+545.05],[x-250,y+545.05],[x,y+545.05],[x+250,y+545.05],[x+500,y+545.05],
            [x-500,y+817.57],[x-250,y+817.57],[x,y+817.57],[x+250,y+817.57],[x+500,y+817.57],
            [x-500,y+1090.09],[x,y+1090.09],[x+500,y+1090.09],
            [x-511.06,y+1298.93],[x,y+1345.01],[x+511.06,y+1390.84],
            [x-511.06,y+1452.43],[x-350,y+1473.69],[x,y+1519.89],[x+350,y+1566.08],[x+511.06,y+1587.34]      
        ])
        labels = np.repeat([0,1],[51,56])
    elif block == 'B2':
        points_real = np.array([
            [x-700,y-1627.49],[x-511.06,y-1652.43],[x-350,y-1673.69],[x,y-1719.89],[x+350,y-1766.08],[x+511.06,y-1787.34],[x+700,y-1812.28],
            [x-700,y-1427.43],[x+700,y-1612.28],
            [x-700,y-1298.93],[x-350,y-1298.93],[x+350,y-1390.84],[x+700,y-1390.84],            
            [x-700,y-1090.09],[x-325,y-1090.09],[x+325,y-1090.09],[x+700,y-1090.09],
            [x-700,y-817.57],[x+700,y-817.57],
            [x-700,y-545.05],[x+700,y-545.05],
            [x-700,y-272.52],[x+700,y-272.52],
            [x-700,y],[x-325,y],[x,y],[x+325,y],[x+700,y],
            [x-700,y+272.52],[x+700,y+272.52],
            [x-700,y+545.05],[x+700,y+545.05],
            [x-700,y+817.57],[x+700,y+817.57],
            [x-700,y+1090.09],[x-325,y+1090.09],[x+325,y+1090.09],[x+700,y+1090.09],
            [x-700,y+1344.89],[x-348.16,y+1344.89],[x+348.16,y+1344.89],[x+700,y+1344.89],
            [x-700,y+1519.89],[x+700,y+1519.89],
            [x-700,y+1719.89],[x-511.06,y+1719.89],[x-348.16,y+1719.89],[x,y+1719.89],[x+348.16,y+1719.89],[x+511.06,y+1719.89],[x+700,y+1719.89],
            [x-511.06,y-1452.43],[x-350,y-1473.69],[x,y-1519.89],[x+350,y-1566.08],[x+511.06,y-1587.34],     
            [x-511.06,y-1298.93],[x,y-1345.01],[x+511.06,y-1390.84],
            [x-500,y-1090.09],[x,y-1090.09],[x+500,y-1090.09],
            [x-500,y-817.57],[x-250,y-817.57],[x,y-817.57],[x+250,y-817.57],[x+500,y-817.57],
            [x-500,y-545.05],[x-250,y-545.05],[x,y-545.05],[x+250,y-545.05],[x+500,y-545.05],
            [x-500,y-272.52],[x-250,y-272.52],[x,y-272.52],[x+250,y-272.52],[x+500,y-272.52],
            [x-500,y],[x-162.5,y],[x+162.5,y],[x+500,y],
            [x-500,y+272.52],[x-250,y+272.52],[x,y+272.52],[x+250,y+272.52],[x+500,y+272.52],
            [x-500,y+545.05],[x-250,y+545.05],[x,y+545.05],[x+250,y+545.05],[x+500,y+545.05],
            [x-500,y+817.57],[x-250,y+817.57],[x,y+817.57],[x+250,y+817.57],[x+500,y+817.57],
            [x-500,y+1090.09],[x,y+1090.09],[x+500,y+1090.09],
            [x-511.06,y+1344.89],[x,y+1344.89],[x+511.06,y+1344.89],
            [x-511.06,y+1519.89],[x-348.16,y+1519.89],[x,y+1519.89],[x+348.16,y+1519.89],[x+511.06,y+1519.89],
        ])
        labels = np.repeat([0,1],[51,56])
    else:
        points_real = np.array([
            [x-700,y-1719.89],[x-511.06,y-1719.89],[x-348.16,y-1719.89],[x,y-1719.89],[x+348.16,y-1719.89],[x+511.06,y-1719.89],[x+700,y-1719.89],
            [x-700,y-1519.89],[x+700,y-1519.89],
            [x-700,y-1344.89],[x-348.16,y-1344.89],[x+348.16,y-1344.89],[x+700,y-1344.89],
            [x-700,y-1090.09],[x-325,y-1090.09],[x+325,y-1090.09],[x+700,y-1090.09],
            [x-700,y-817.57],[x+700,y-817.57],
            [x-700,y-545.05],[x+700,y-545.05],
            [x-700,y-272.52],[x+700,y-272.52],
            [x-700,y],[x-325,y],[x,y],[x+325,y],[x+700,y],
            [x-700,y+272.52],[x+700,y+272.52],
            [x-700,y+545.05],[x+700,y+545.05],
            [x-700,y+817.57],[x+700,y+817.57],
            [x-700,y+1090.09],[x-325,y+1090.09],[x+325,y+1090.09],[x+700,y+1090.09],
            [x-700,y+1344.89],[x-348.16,y+1344.89],[x+348.16,y+1344.89],[x+700,y+1344.89],
            [x-700,y+1519.89],[x+700,y+1519.89],
            [x-700,y+1719.89],[x-511.06,y+1719.89],[x-348.16,y+1719.89],[x,y+1719.89],[x+348.16,y+1719.89],[x+511.06,y+1719.89],[x+700,y+1719.89],
            [x-511.06,y-1519.89],[x-348.16,y-1519.89],[x,y-1519.89],[x+348.16,y-1519.89],[x+511.06,y-1519.89],
            [x-511.06,y-1344.89],[x,y-1344.89],[x+511.06,y-1344.89],
            [x-500,y-1090.09],[x,y-1090.09],[x+500,y-1090.09],
            [x-500,y-817.57],[x-250,y-817.57],[x,y-817.57],[x+250,y-817.57],[x+500,y-817.57],
            [x-500,y-545.05],[x-250,y-545.05],[x,y-545.05],[x+250,y-545.05],[x+500,y-545.05],
            [x-500,y-272.52],[x-250,y-272.52],[x,y-272.52],[x+250,y-272.52],[x+500,y-272.52],
            [x-500,y],[x-162.5,y],[x+162.5,y],[x+500,y],
            [x-500,y+272.52],[x-250,y+272.52],[x,y+272.52],[x+250,y+272.52],[x+500,y+272.52],
            [x-500,y+545.05],[x-250,y+545.05],[x,y+545.05],[x+250,y+545.05],[x+500,y+545.05],
            [x-500,y+817.57],[x-250,y+817.57],[x,y+817.57],[x+250,y+817.57],[x+500,y+817.57],
            [x-500,y+1090.09],[x,y+1090.09],[x+500,y+1090.09],
            [x-511.06,y+1344.89],[x,y+1344.89],[x+511.06,y+1344.89],
            [x-511.06,y+1519.89],[x-348.16,y+1519.89],[x,y+1519.89],[x+348.16,y+1519.89],[x+511.06,y+1519.89],
        ])
        labels = np.repeat([0,1],[51,56])

    keep_mask = np.ones(len(labels), dtype=bool)
    for i in range(len(labels)):
        if labels[i] == 0:
            y_cond = points_real[i, 1] + map_y < 4200 or points_real[i, 1] + map_y > 13100
            x_cond = abs(points_real[i, 0] - x) <= segment_width * 0.5
            y_limit = K_height if block == 'K' else AB_height
            y_cond2 = abs(points_real[i, 1] - y) <= y_limit * 0.5
            
            if y_cond and x_cond and y_cond2:
                keep_mask[i] = False
            
    points_real = points_real[keep_mask]
    labels = labels[keep_mask]
    
    points = points_real / (resolution*1000)

    within_bounds = (points[:, 0] >= 0) & ((points[:, 0] + initial_x - (segment_width*0.5+150)/(resolution*1000)) <= image.shape[1])
    points = points[within_bounds]
    labels = labels[within_bounds]
        
    return points, labels

def convert_to_pixel_coords(real_dist, resolution=0.005):
    return int(real_dist / (resolution*1000))

def k_priority_rect(cx, cy, resolution=0.005, K_height=1079.92, segment_width=1200):
    """Axis-aligned K-block footprint: B1/B2 must not overwrite pixels inside."""
    half_h = convert_to_pixel_coords(0.5 * K_height, resolution)
    half_w = convert_to_pixel_coords(0.5 * segment_width, resolution)
    return int(cx - half_w), int(cx + half_w), int(cy - half_h), int(cy + half_h)

def k_template_mask_global(height, width, cx, cy, resolution=0.005, K_height=1079.92, scale=1.0):
    """Full-image K parallelogram (same geometry as generate_template_mask)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    ky = (K_height / (2 * 619.16)) * scale
    x = cx * (resolution * 1000)
    y = cy * (resolution * 1000)
    vertices_real = np.array([
        [x - 625, y - 619.16 * ky], [x - 625, y + 619.16 * ky],
        [x + 625, y + 460.77 * ky], [x + 625, y - 460.77 * ky],
    ])
    vertices = vertices_real / (resolution * 1000)
    fill_polygon(mask, vertices)
    return mask.astype(bool)

def crop_image_and_mask_logits(image, cx, cy, crop_width, crop_height, block, resolution):
    img_height, img_width, _ = image.shape
    x1 = max(cx - crop_width // 2, 0)
    y1 = max(cy - crop_height // 2, 0)
    x2 = min(cx + crop_width // 2, img_width)
    y2 = min(cy + crop_height // 2, img_height)

    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    prompt_centre_x = cx - x1
    prompt_centre_y = cy - y1
    prompt_centre = (prompt_centre_x,prompt_centre_y)
    
    cropped_template_mask = generate_template_mask(cropped_image.shape[0],cropped_image.shape[1],prompt_centre,block,resolution)
    template_mask_logits = compute_logits_from_mask(cropped_template_mask)

    return cropped_image, template_mask_logits, prompt_centre

def compute_logits_from_mask(mask, eps=1e-3):
    def inv_sigmoid(x):
        return np.log(x / (1 - x))

    logits = np.zeros(mask.shape, dtype="float32")
    logits[mask == 1] = 1 - eps
    logits[mask == 0] = eps
    logits = inv_sigmoid(logits)

    assert logits.ndim == 2
    expected_shape = (256, 256)

    if logits.shape == expected_shape:
        pass
    elif logits.shape[0] == logits.shape[1]:
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image(logits[..., None])
    else:
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image(logits[..., None])
        h, w = logits.shape
        padh = expected_shape[0] - h
        padw = expected_shape[1] - w
        pad_width = ((0, padh), (0, padw))
        logits = np.pad(logits, pad_width, mode="constant", constant_values=0)

    logits = logits[None]
    assert logits.shape == (1, 256, 256)
    return logits

def restore_sam_logits(logits, original_shape):
    orig_h, orig_w = original_shape
    trafo = ResizeLongestSide(max(orig_h, orig_w))
    resized_logits = trafo.apply_image(logits[..., None])
    resized_logits = resized_logits.squeeze()
    resized_logits = resized_logits[:orig_h, :orig_w]
    return resized_logits

def compute_block_label(segment_per_ring):
    block_labels = ['K','B1']
    num_a_labels = segment_per_ring - 3
    block_labels += [f'A{i+1}' for i in range(num_a_labels)]
    block_labels += ['B2']
    return block_labels

def sam_prediction(cropped_image, points, labels, template_mask_logit):
    predictor.set_image(cropped_image)
    mask, score, logit = predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=template_mask_logit,
            multimask_output=False,
    )
    return mask, score, logit[0]

def process_row(df_row, image, resolution=0.005, segment_per_ring=6, segment_width=1200, 
                K_height=1079.92, angle=7.52, AB_height=3239.77):
    initial_x, initial_y = df_row['X'], df_row['Y']
    block_labels = compute_block_label(segment_per_ring)

    delta_x = convert_to_pixel_coords(0.5*segment_width + 150, resolution)
    delta_y = 0

    reverse = False
    stop = False
    map_y = 0
    block_label_index = 0

    results = []
    for i in range(segment_per_ring):
        if reverse == False:
            block = block_labels[block_label_index]
            if block_label_index == 0:
                delta_y = convert_to_pixel_coords(0.5*K_height + math.tan(math.radians(angle))*700+100 + 50, resolution)
                map_y = initial_y
            else:
                delta_y = convert_to_pixel_coords(0.5*AB_height + math.tan(math.radians(angle))*700+100 + 50, resolution)
                if block_label_index == 1:
                    map_y = initial_y - convert_to_pixel_coords(0.5 * K_height + 0.5 * AB_height, resolution)
                else:
                    map_y = map_y - convert_to_pixel_coords(AB_height, resolution)

            cropped_image, template_mask_logit, prompt_centre = crop_image_and_mask_logits(
                image, initial_x, map_y,2 * delta_x, 2 * delta_y, block, resolution)
            points, labels = generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution)
        
            if np.any(points[:, 1] < 0):
                within_bounds = (points[:, 1] >= 0)
                points = points[within_bounds]
                labels = labels[within_bounds]
                reverse = True
                
            mask, score, logit = sam_prediction(cropped_image, points, labels, template_mask_logit)
        
            results.append({
                'left_top': (initial_x-prompt_centre[0], map_y-prompt_centre[1]),
                'block': block,
                'cropped_image': cropped_image,
                'mask': mask,
                'points':points,
                'labels':labels,
                'score': score,
                'logit': logit
            })
            
            if reverse:
                block_label_index = -1
                continue

            block_label_index = block_label_index + 1
            
        if reverse:
            block = block_labels[block_label_index]
            delta_y = convert_to_pixel_coords(0.5*AB_height + math.tan(math.radians(angle))*700+100 + 50, resolution)
            if block_label_index == -1:
                map_y = initial_y + convert_to_pixel_coords(0.5 * K_height + 0.5 * AB_height, resolution)
            else:
                map_y = map_y + convert_to_pixel_coords(AB_height, resolution)

            cropped_image, template_mask_logit, prompt_centre = crop_image_and_mask_logits(image, initial_x, map_y, 
                                                                                            2 * delta_x, 2 * delta_y, block, resolution)
            points, labels = generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution)

            if np.any((points[:, 1]+map_y-delta_y) > image.shape[0]):
                within_bounds = ((points[:, 1]+map_y-delta_y) <= image.shape[0])
                points = points[within_bounds]
                labels = labels[within_bounds]
                stop = True

            mask, score, logit = sam_prediction(cropped_image, points, labels, template_mask_logit)

            results.append({
                'left_top': (initial_x-prompt_centre[0], map_y-prompt_centre[1]),
                'block': block,
                'cropped_image': cropped_image,
                'mask': mask,
                'points':points,
                'labels':labels,
                'score': score,
                'logit': logit
            })

            if stop:
                break

            block_label_index = block_label_index - 1
             
    return results

def sam_segment(df, image, resolution=0.005, segment_per_ring=6):
    all_results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        result = process_row(row, image, resolution, segment_per_ring)
        all_results.append(result)
    return all_results

results = sam_segment(initial_prompt_points, image)

block_to_label = {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'B2': 6}

K_LOGIT_BOOST = 2.0
K_TEMPLATE_CLAIM_SCALE = 1.14

k_priority_rects = [
    k_priority_rect(row['X'], row['Y']) for _, row in initial_prompt_points.iterrows()
]
k_template_masks = [
    k_template_mask_global(image.shape[0], image.shape[1], row['X'], row['Y'], scale=K_TEMPLATE_CLAIM_SCALE)
    for _, row in initial_prompt_points.iterrows()
]

logits_map = np.full(image.shape[:2], -np.inf, dtype=float)
label_map = np.zeros(image.shape[:2], dtype=int)
ring_map = np.zeros(image.shape[:2], dtype=int)

for ring_index, ring in enumerate(results, start=0):
    for item in ring:
        mask = item['mask'][0]
        logits = item['logit']
        block = item['block']
        start_x, start_y = map(int, item['left_top'])

        end_y, end_x = start_y + mask.shape[0], start_x + mask.shape[1]
        start_y, start_x = max(0, start_y), max(0, start_x)
        end_y, end_x = min(image.shape[0], end_y), min(image.shape[1], end_x)
        
        valid_slice_y = slice(start_y, end_y)
        valid_slice_x = slice(start_x, end_x)

        new_logits = restore_sam_logits(logits, mask.shape)
        current_logits = logits_map[valid_slice_y, valid_slice_x]

        if mask.shape != current_logits.shape or new_logits.shape != current_logits.shape:
            raise ValueError(f"Shape mismatch after resizing: mask {mask.shape}, new_logits {new_logits.shape}, current_logits {current_logits.shape}")

        yy = np.arange(start_y, end_y)[:, None]
        xx = np.arange(start_x, end_x)[None, :]

        if block == 'K':
            k_tmpl = k_template_masks[ring_index][valid_slice_y, valid_slice_x]
            boosted = new_logits + K_LOGIT_BOOST
            update_mask = (boosted > current_logits) & mask & k_tmpl
        else:
            update_mask = (new_logits > current_logits) & mask

        # K priority: B1/B2 must not overwrite any pixel inside the K footprint.
        if block in ('B1', 'B2'):
            x1k, x2k, y1k, y2k = k_priority_rects[ring_index]
            in_k_rect = (xx >= x1k) & (xx < x2k) & (yy >= y1k) & (yy < y2k)
            update_mask = update_mask & ~in_k_rect
        
        logits_map[valid_slice_y, valid_slice_x][update_mask] = (
            boosted[update_mask] if block == 'K' else new_logits[update_mask]
        )
        label_map[valid_slice_y, valid_slice_x][update_mask] = block_to_label[block]
        ring_map[valid_slice_y, valid_slice_x][update_mask] = ring_index

result_image = label_map
ring_image = ring_map

# Save visualizations (mirrors notebook cells 74 & 77; written to file in this script).
save_instance_checks(results, f'{base_dir}/sam_instances.png')
visualize_combined_results(image, result_image, ring_image, f'{base_dir}/segmentation_overlay.png')

fix_ring = np.where((ring_image >= 1) & (ring_image <= (ring_count-1)), ring_count - ring_image, ring_image)

def project_back_to_point_cloud(segmented_map, instance_map, pixel_to_point, df):
    df_copy = df.copy()
    pred = df_copy['pred'].values
    pred_ring = np.full(len(df_copy), -1, dtype=int)

    pixel_to_point_df = pd.DataFrame(pixel_to_point)
    y = pixel_to_point_df['pixel_y'].values
    x = pixel_to_point_df['pixel_x'].values
    point_indices = pixel_to_point_df['index'].values

    valid_point_mask = np.isin(point_indices, df_copy.index.values)
    valid_update_mask = (pred[point_indices[valid_point_mask]] == 7)

    pred[point_indices[valid_point_mask][valid_update_mask]] = segmented_map[y[valid_point_mask][valid_update_mask], x[valid_point_mask][valid_update_mask]]
    pred_ring[point_indices[valid_point_mask][valid_update_mask]] = instance_map[y[valid_point_mask][valid_update_mask], x[valid_point_mask][valid_update_mask]]

    df_copy['pred'] = pred
    df_copy['pred_ring'] = pred_ring

    return df_copy

updated_df = project_back_to_point_cloud(result_image, fix_ring, pixel_to_point, df_point_cloud)

os.makedirs(base_dir, exist_ok=True)
updated_df.to_csv(f'{base_dir}/final.csv', index=False)

df_pred = pd.DataFrame()
df_pred['gt_labels'] = updated_df['segment']
df_pred['gt_rings'] = updated_df['ring']
df_pred['pred_labels'] = updated_df['pred']
df_pred['pred_rings'] = updated_df['pred_ring']
df_pred.to_csv(f'{base_dir}/only_label.csv', index=False) 