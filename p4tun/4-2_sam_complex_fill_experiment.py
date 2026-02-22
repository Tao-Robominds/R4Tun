"""
SAM Template-Fill Experiment

Quantifies the mIoU gap caused by SAM's conservative masking.
Runs SAM inference ONCE, then tests three aggregation modes:

  Mode A (baseline):  (logits > current) & sam_mask
  Mode B (template):  (logits > current) & template_mask
  Mode C (hybrid):    SAM-first pass, then template fallback for unlabeled pixels

Outputs go to data/<tunnel_id>/experiments/sam_fill/  (never overwrites existing data).
"""

import os
import sys
import json
import math
import pickle

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from matplotlib.path import Path
from sklearn.metrics import jaccard_score, accuracy_score, f1_score

if len(sys.argv) != 2:
    print("Usage: python 4-2_sam_complex_fill_experiment.py <tunnel_id>")
    sys.exit(1)

tunnel_id = sys.argv[1]
base_dir = f"data/{tunnel_id}/"
out_dir = os.path.join(base_dir, "experiments", "sam_fill")
os.makedirs(out_dir, exist_ok=True)

# ── Load SAM parameters ────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, 'parameters', tunnel_id, 'parameters_sam.json')

SAM_PARAMS = {
    'segment_width': 1200,
    'K_height': 1079.92,
    'AB_height': 3239.77,
    'angle': 7.52,
    'resolution': 0.005,
    'k_mask_width': 625,
    'k_mask_height_pos': 619.16,
    'k_mask_height_neg': 460.77,
    'ab_mask_width': 625,
    'ab_mask_height': 1619.89,
}

if os.path.exists(params_path):
    try:
        with open(params_path, 'r') as f:
            loaded_params = json.load(f)
        if 'segment_geometry' in loaded_params:
            sg = loaded_params['segment_geometry']
            SAM_PARAMS['segment_width'] = sg.get('segment_width', SAM_PARAMS['segment_width'])
            SAM_PARAMS['K_height'] = sg.get('k_height', SAM_PARAMS['K_height'])
            SAM_PARAMS['AB_height'] = sg.get('ab_height', SAM_PARAMS['AB_height'])
            SAM_PARAMS['angle'] = sg.get('angle_deg', SAM_PARAMS['angle'])
        if 'prompt_points' in loaded_params and 'template_mask' in loaded_params['prompt_points']:
            tm = loaded_params['prompt_points']['template_mask']
            if 'k_block' in tm:
                SAM_PARAMS['k_mask_width'] = tm['k_block'].get('width', SAM_PARAMS['k_mask_width'])
                SAM_PARAMS['k_mask_height_pos'] = tm['k_block'].get('height_pos', SAM_PARAMS['k_mask_height_pos'])
                SAM_PARAMS['k_mask_height_neg'] = tm['k_block'].get('height_neg', SAM_PARAMS['k_mask_height_neg'])
            if 'a_blocks' in tm:
                SAM_PARAMS['ab_mask_width'] = tm['a_blocks'].get('width', SAM_PARAMS['ab_mask_width'])
                SAM_PARAMS['ab_mask_height'] = tm['a_blocks'].get('height', SAM_PARAMS['ab_mask_height'])
        print(f"Loaded SAM parameters from {params_path}")
    except Exception as e:
        print(f"Warning: Could not load SAM parameters: {e}")
else:
    print(f"Using default SAM parameters (no {params_path})")

# ── Load data ──────────────────────────────────────────────────────────────
all_segments_path = os.path.join(base_dir, "all_segments.csv")
if not os.path.exists(all_segments_path):
    print(f"ERROR: {all_segments_path} not found!")
    sys.exit(1)

all_segments_df = pd.read_csv(all_segments_path)
if 'ring' in all_segments_df.columns and 'Ring' not in all_segments_df.columns:
    all_segments_df = all_segments_df.rename(columns={'ring': 'Ring'})
if 'segment_name' in all_segments_df.columns and 'Block' not in all_segments_df.columns:
    all_segments_df = all_segments_df.rename(columns={'segment_name': 'Block'})

pixel_to_point = pickle.load(open(os.path.join(base_dir, "pixel_to_point.pkl"), "rb"))
df_point_cloud = pd.read_csv(os.path.join(base_dir, "enhanced.csv"))
ring_count = int(open(os.path.join(base_dir, "ring_count.txt"), 'r').read())

print(f"Processing tunnel: {tunnel_id}")
print(f"Total segments: {len(all_segments_df)}")

# ── SAM model ──────────────────────────────────────────────────────────────
sam_checkpoint = "skills/segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

image = cv2.imread(os.path.join(base_dir, 'depth_map.png'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ── Helper functions (unchanged from original) ─────────────────────────────

def fill_polygon(mask, vertices):
    path = Path(vertices)
    y_coords, x_coords = np.mgrid[:mask.shape[0], :mask.shape[1]]
    points = np.vstack((x_coords.flatten(), y_coords.flatten())).T
    mask_inside = path.contains_points(points).reshape(mask.shape)
    mask[mask_inside] = 1


def generate_template_mask(height, width, prompt_centre, block, resolution=0.005):
    mask = np.zeros((height, width), dtype=np.uint8)
    prompt_centre_x, prompt_centre_y = prompt_centre
    x = prompt_centre_x * (resolution * 1000)
    y = prompt_centre_y * (resolution * 1000)

    k_w = SAM_PARAMS['k_mask_width']
    k_hp = SAM_PARAMS['k_mask_height_pos']
    k_hn = SAM_PARAMS['k_mask_height_neg']
    ab_w = SAM_PARAMS['ab_mask_width']
    ab_h = SAM_PARAMS['ab_mask_height']

    if block == 'K':
        vertices_real = np.array([[x-k_w,y-k_hp],[x-k_w,y+k_hp],[x+k_w,y+k_hn],[x+k_w,y-k_hn]])
    elif block == 'B1':
        vertices_real = np.array([[x-ab_w,y-ab_h],[x-ab_w,y+1540.69],[x+ab_w,y+1699.08],[x+ab_w,y-ab_h]])
    elif block == 'B2':
        vertices_real = np.array([[x-ab_w,y-1540.69],[x-ab_w,y+ab_h],[x+ab_w,y+ab_h],[x+ab_w,y-1699.08]])
    else:
        vertices_real = np.array([[x-ab_w,y-ab_h],[x-ab_w,y+ab_h],[x+ab_w,y+ab_h],[x+ab_w,y-ab_h]])

    vertices = vertices_real / (resolution * 1000)
    fill_polygon(mask, vertices)
    return mask


def generate_prompt_points(prompt_centre, map_y, block, crop_shape, resolution=0.005,
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
        labels = np.repeat([0, 1], [51, 56])
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
            [x-500,y-817.57],[x-250,y-817.57],[x,y-817.57],[x+250,y-817.57],[x+500,y+817.57],
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
        labels = np.repeat([0, 1], [51, 56])
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
        labels = np.repeat([0, 1], [51, 56])

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
    points = points_real / (resolution * 1000)

    crop_height_px, crop_width_px = crop_shape
    within_bounds = (
        (points[:, 0] >= 0) & (points[:, 0] < crop_width_px) &
        (points[:, 1] >= 0) & (points[:, 1] < crop_height_px)
    )
    points = points[within_bounds]
    labels = labels[within_bounds]
    return points, labels


def convert_to_pixel_coords(real_dist, resolution=0.005):
    return int(real_dist / (resolution * 1000))


def crop_image_and_mask_logits(image, cx, cy, crop_width, crop_height, block, resolution):
    """Returns (cropped_image, template_mask_logits, prompt_centre, crop_info, template_mask)."""
    img_height, img_width, _ = image.shape
    x1 = int(cx - crop_width // 2)
    x2 = int(cx + crop_width // 2)
    y1 = max(int(cy - crop_height // 2), 0)
    y2 = min(int(cy + crop_height // 2), img_height)

    wraparound = x1 < 0 or x2 > img_width
    crop_mappings = []

    if not wraparound:
        cropped_image = image[y1:y2, x1:x2]
        prompt_centre_x = cx - x1
        crop_mappings.append({"crop_x": (0, x2 - x1), "img_x": (x1, x2)})
    elif x1 < 0:
        right_start = img_width + x1
        right_part = image[y1:y2, right_start:img_width]
        left_part = image[y1:y2, 0:x2]
        cropped_image = np.concatenate([right_part, left_part], axis=1)
        right_width = right_part.shape[1]
        crop_mappings.append({"crop_x": (0, right_width), "img_x": (right_start, img_width)})
        crop_mappings.append({"crop_x": (right_width, right_width + left_part.shape[1]), "img_x": (0, x2)})
        prompt_centre_x = right_width + cx
    else:
        right_part = image[y1:y2, x1:img_width]
        left_part = image[y1:y2, 0:x2 - img_width]
        cropped_image = np.concatenate([right_part, left_part], axis=1)
        right_width = right_part.shape[1]
        crop_mappings.append({"crop_x": (0, right_width), "img_x": (x1, img_width)})
        crop_mappings.append({"crop_x": (right_width, right_width + left_part.shape[1]), "img_x": (0, x2 - img_width)})
        prompt_centre_x = cx - x1

    prompt_centre_y = cy - y1
    prompt_centre = (prompt_centre_x, prompt_centre_y)

    cropped_template_mask = generate_template_mask(
        cropped_image.shape[0], cropped_image.shape[1],
        prompt_centre, block, resolution,
    )
    template_mask_logits = compute_logits_from_mask(cropped_template_mask)

    crop_info = {"y1": y1, "y2": y2, "wraparound": wraparound, "mappings": crop_mappings}
    return cropped_image, template_mask_logits, prompt_centre, crop_info, cropped_template_mask


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
        logits = np.pad(logits, ((0, padh), (0, padw)), mode="constant", constant_values=0)

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


def sam_prediction(cropped_image, points, labels, template_mask_logit):
    predictor.set_image(cropped_image)
    mask, score, logit = predictor.predict(
        point_coords=points,
        point_labels=labels,
        mask_input=template_mask_logit,
        multimask_output=False,
    )
    return mask, score, logit[0]


# ── SAM inference (run once) ───────────────────────────────────────────────

def sam_segment_all_segments(all_segments_df, image, resolution=0.005):
    all_results = []
    segment_width = SAM_PARAMS['segment_width']
    K_height = SAM_PARAMS['K_height']
    AB_height = SAM_PARAMS['AB_height']
    angle = SAM_PARAMS['angle']

    for _, segment_row in tqdm(all_segments_df.iterrows(), total=len(all_segments_df), desc="SAM inference"):
        initial_x, initial_y = segment_row['X'], segment_row['Y']
        block = segment_row['Block']
        ring_id = segment_row['Ring']

        delta_x_pixels = convert_to_pixel_coords(0.5 * segment_width + 150, resolution)
        if block == 'K':
            delta_y_pixels = convert_to_pixel_coords(0.5 * K_height + math.tan(math.radians(angle)) * 700 + 100 + 50, resolution)
        else:
            delta_y_pixels = convert_to_pixel_coords(0.5 * AB_height + math.tan(math.radians(angle)) * 700 + 100 + 50, resolution)

        try:
            cropped_image, template_mask_logit, prompt_centre, crop_info, template_mask = \
                crop_image_and_mask_logits(
                    image, initial_x, initial_y,
                    2 * delta_x_pixels, 2 * delta_y_pixels,
                    block, resolution,
                )

            if cropped_image.size == 0:
                print(f"Warning: Empty crop for Ring {ring_id} Block {block}")
                continue

            points, labels = generate_prompt_points(
                prompt_centre, initial_y, block, cropped_image.shape[:2],
                resolution, segment_width=segment_width,
                K_height=K_height, AB_height=AB_height,
            )
            if len(points) == 0:
                print(f"Warning: No valid prompt points for Ring {ring_id} Block {block}")
                continue

            mask, score, logit = sam_prediction(cropped_image, points, labels, template_mask_logit)

            all_results.append({
                'block': block,
                'ring_id': ring_id,
                'mask': mask,
                'score': score,
                'logit': logit,
                'crop_info': crop_info,
                'template_mask': template_mask,
            })
        except Exception as e:
            print(f"Error processing Ring {ring_id} Block {block}: {e}")
            continue

    return all_results


# ── Aggregation (parameterised by mode) ────────────────────────────────────

BLOCK_TO_LABEL = {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'A4': 6, 'B2': 7}
FALLBACK_LOGIT = -5.0


def aggregate(results, img_shape, mode="A"):
    """
    mode A: SAM mask gate        (original behaviour)
    mode B: template mask gate   (fill all pixels within geometric boundary)
    mode C: SAM first, then template fallback for still-unlabeled pixels
    """
    h, w = img_shape[:2]
    logits_map = np.full((h, w), -np.inf, dtype=float)
    label_map = np.zeros((h, w), dtype=int)
    ring_map = np.zeros((h, w), dtype=int)

    for item in results:
        sam_mask = item['mask'][0]
        logits_raw = item['logit']
        block = item['block']
        ring_id = item['ring_id']
        crop_info = item['crop_info']
        tpl_mask = item['template_mask']
        start_y, end_y = crop_info['y1'], crop_info['y2']

        valid_slice_y = slice(start_y, end_y)
        new_logits = restore_sam_logits(logits_raw, sam_mask.shape)

        for mapping in crop_info['mappings']:
            crop_x_start, crop_x_end = mapping['crop_x']
            img_x_start, img_x_end = mapping['img_x']
            valid_slice_x = slice(img_x_start, img_x_end)

            sam_slice = sam_mask[:, crop_x_start:crop_x_end]
            logits_slice = new_logits[:, crop_x_start:crop_x_end]
            tpl_slice = tpl_mask[:, crop_x_start:crop_x_end].astype(bool)
            current_logits = logits_map[valid_slice_y, valid_slice_x]

            if sam_slice.shape != current_logits.shape or logits_slice.shape != current_logits.shape:
                continue

            if mode == "A":
                update = (logits_slice > current_logits) & sam_slice
            elif mode == "B":
                update = (logits_slice > current_logits) & tpl_slice
            elif mode == "C":
                update = (logits_slice > current_logits) & sam_slice
            else:
                raise ValueError(f"Unknown mode: {mode}")

            logits_map[valid_slice_y, valid_slice_x][update] = logits_slice[update]
            label_map[valid_slice_y, valid_slice_x][update] = BLOCK_TO_LABEL.get(block, 0)
            ring_map[valid_slice_y, valid_slice_x][update] = ring_id

    if mode == "C":
        for item in results:
            sam_mask = item['mask'][0]
            crop_info = item['crop_info']
            tpl_mask = item['template_mask']
            block = item['block']
            ring_id = item['ring_id']
            start_y, end_y = crop_info['y1'], crop_info['y2']
            valid_slice_y = slice(start_y, end_y)

            for mapping in crop_info['mappings']:
                crop_x_start, crop_x_end = mapping['crop_x']
                img_x_start, img_x_end = mapping['img_x']
                valid_slice_x = slice(img_x_start, img_x_end)

                tpl_slice = tpl_mask[:, crop_x_start:crop_x_end].astype(bool)
                region_labels = label_map[valid_slice_y, valid_slice_x]

                if tpl_slice.shape != region_labels.shape:
                    continue

                fill = tpl_slice & (region_labels == 0)
                label_map[valid_slice_y, valid_slice_x][fill] = BLOCK_TO_LABEL.get(block, 0)
                ring_map[valid_slice_y, valid_slice_x][fill] = ring_id
                logits_map[valid_slice_y, valid_slice_x][fill] = FALLBACK_LOGIT

    return label_map, ring_map


# ── Point cloud projection ─────────────────────────────────────────────────

def project_back_to_point_cloud(segmented_map, instance_map, pixel_to_point, df):
    df_copy = df.copy()
    pred = df_copy['pred'].values
    pred_ring = np.full(len(df_copy), -1, dtype=int)

    pixel_to_point_df = pd.DataFrame(pixel_to_point)
    y = pixel_to_point_df['pixel_y'].values
    x = pixel_to_point_df['pixel_x'].values
    point_indices = pixel_to_point_df['index'].values

    img_height, img_width = segmented_map.shape
    valid_point_mask = np.isin(point_indices, df_copy.index.values)
    valid_update_mask = np.isin(pred[point_indices[valid_point_mask]], [0, 7])

    y_valid = y[valid_point_mask][valid_update_mask]
    x_valid = x[valid_point_mask][valid_update_mask]
    bounds_mask = (y_valid >= 0) & (y_valid < img_height) & (x_valid >= 0) & (x_valid < img_width)

    final_point_indices = point_indices[valid_point_mask][valid_update_mask][bounds_mask]
    final_y = y_valid[bounds_mask]
    final_x = x_valid[bounds_mask]

    pred[final_point_indices] = segmented_map[final_y, final_x]
    pred_ring[final_point_indices] = instance_map[final_y, final_x]

    df_copy['pred'] = pred
    df_copy['pred_ring'] = pred_ring
    return df_copy


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(df, label=""):
    gt = df['segment'].values
    pr = df['pred'].values
    mask = np.isfinite(gt) & np.isfinite(pr)
    gt = gt[mask].astype(int)
    pr = pr[mask].astype(int)

    classes = np.arange(0, 8)
    class_names = ['Bg', 'K', 'B1', 'A1', 'A2', 'A3', 'A4', 'B2']

    valid = (gt <= 7) & (pr <= 7)
    gt_v, pr_v = gt[valid], pr[valid]

    oa = accuracy_score(gt_v, pr_v)
    f1 = f1_score(gt_v, pr_v, average='macro', labels=classes, zero_division=0)
    iou = jaccard_score(gt_v, pr_v, average=None, labels=classes, zero_division=0)
    miou = iou.mean()

    block_mask = (gt >= 1) & (gt <= 7)
    gt_b, pr_b = gt[block_mask], pr[block_mask]
    iou_blocks = jaccard_score(gt_b, pr_b, average=None, labels=np.arange(1, 8), zero_division=0)
    miou_blocks = iou_blocks.mean()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  OA={oa:.4f}  F1={f1:.4f}  mIoU(all)={miou:.4f}  mIoU(blocks)={miou_blocks:.4f}")
    for name, v in zip(class_names, iou):
        print(f"    {name:4s}: {v:.4f}")

    return {
        'oa': oa, 'f1': f1, 'miou_all': miou, 'miou_blocks': miou_blocks,
        'iou_per_class': dict(zip(class_names, iou.tolist())),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Running SAM inference (once) ──")
results = sam_segment_all_segments(all_segments_df, image)
print(f"Processed {len(results)} segments")

# Prepare point cloud base (reset pred for clean evaluation each mode)
if 'pred' not in df_point_cloud.columns:
    df_point_cloud['pred'] = 0
else:
    df_point_cloud['pred'] = np.where(
        np.isin(df_point_cloud['pred'].values, [0, 7]),
        df_point_cloud['pred'].values, 0,
    )

all_metrics = {}
for mode, label in [("A", "Mode A: SAM mask only (baseline)"),
                     ("B", "Mode B: Template mask gate"),
                     ("C", "Mode C: SAM + template fallback")]:
    label_map, ring_map = aggregate(results, image.shape, mode=mode)
    updated_df = project_back_to_point_cloud(label_map, ring_map, pixel_to_point, df_point_cloud)

    only_label = pd.DataFrame({
        'gt_labels': updated_df['segment'],
        'gt_rings': updated_df['ring'],
        'pred_labels': updated_df['pred'],
        'pred_rings': updated_df['pred_ring'],
    })
    only_label.to_csv(os.path.join(out_dir, f"only_label_mode_{mode.lower()}.csv"), index=False)

    metrics = evaluate(updated_df, label)
    all_metrics[mode] = metrics

# ── Summary ────────────────────────────────────────────────────────────────
class_names = ['Bg', 'K', 'B1', 'A1', 'A2', 'A3', 'A4', 'B2']

summary_lines = [
    f"# SAM Template-Fill Experiment — Tunnel {tunnel_id}\n",
    "## Overall Metrics\n",
    "| Mode | OA | F1 | mIoU (all) | mIoU (blocks 1-7) |",
    "|------|----|----|------------|-------------------|",
]
for mode, label in [("A", "SAM mask only"), ("B", "Template mask gate"), ("C", "SAM + template fallback")]:
    m = all_metrics[mode]
    summary_lines.append(
        f"| {label} | {m['oa']:.4f} | {m['f1']:.4f} | {m['miou_all']:.4f} | {m['miou_blocks']:.4f} |"
    )

summary_lines += ["\n## Per-Class IoU\n",
                   "| Class | Mode A | Mode B | Mode C |",
                   "|-------|--------|--------|--------|"]
for cn in class_names:
    a = all_metrics["A"]['iou_per_class'].get(cn, 0)
    b = all_metrics["B"]['iou_per_class'].get(cn, 0)
    c = all_metrics["C"]['iou_per_class'].get(cn, 0)
    summary_lines.append(f"| {cn} | {a:.4f} | {b:.4f} | {c:.4f} |")

summary_text = "\n".join(summary_lines) + "\n"
summary_path = os.path.join(out_dir, "performance_summary.md")
with open(summary_path, 'w') as f:
    f.write(summary_text)

print(f"\n\nSummary saved to {summary_path}")
print(summary_text)
