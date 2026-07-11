#!/usr/bin/env python3
# Parameterized sam — sam4tun/5_sam.py + JSON top-level/processing params
# Deferred JSON: prompt_points, segment_order, use_original_label_distributions, processing.mask_eps
import sys
import os
import matplotlib
matplotlib.use("Agg")

_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SAM4TUN_ROOT = os.path.dirname(_AGENTS_DIR)
for _p in (_AGENTS_DIR, _SAM4TUN_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline_data import (
    parse_pipeline_args, load_stage_parameters, require_keys, resolve_param_file, setup_sam4tun_path,
)
setup_sam4tun_path()
from helpers.pipeline_io import ensure_dir
from helpers.pipeline_state import load_state, save_state

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
import math
import pickle
from tqdm.auto import tqdm
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from matplotlib.path import Path

tunnel_id, ablation, model = parse_pipeline_args("sam")
config = load_stage_parameters(tunnel_id, "sam", ablation, model)
param_file = resolve_param_file(tunnel_id, "sam", ablation, model)
expected_keys = [
    "segment_per_ring", "segment_width", "K_height", "AB_height", "angle", "processing",
]
require_keys(config, expected_keys, param_file)
segment_per_ring = config["segment_per_ring"]
segment_width = config["segment_width"]
K_height = config["K_height"]
AB_height = config["AB_height"]
angle = config["angle"]
processing = config["processing"]
resolution = processing["resolution"]
padding = processing["padding"]
crop_margin = processing["crop_margin"]
y_bounds = processing["y_bounds"]

_tunnel_prefix = tunnel_id.split("-")[0]
_is_complex_tunnel = _tunnel_prefix in ("4", "5")
_use_geometric_fallback = _is_complex_tunnel and segment_per_ring == 7

_denoise_params = load_stage_parameters(tunnel_id, "denoising", ablation, model)
_depth_vmin = float(_denoise_params.get("mask_r_low", 2.70))
_depth_vmax = float(_denoise_params.get("mask_r_high", 2.80))

paths = ensure_dir(tunnel_id)
state = load_state(paths["state"])
df_point_cloud = state["df_point_cloud"]
df_loc = state["df_loc"]
pixel_to_point = state["pixel_to_point"]
ring_count = state["ring_count"]

# Regenerate depth_map.png from stored float array (I/O only; same save_depth_map_exact as monolith)
_depth_npy = os.path.join(os.path.dirname(paths["depth_map"]), "depth_map.npy")
if os.path.exists(_depth_npy):
    from matplotlib import pyplot as _plt

    def save_depth_map_exact(depth_map, resolution, filename="depth_map.png"):
        height, width = depth_map.shape
        dpi = 1.0 / resolution
        fig = _plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.imshow(depth_map, cmap="viridis", vmin=_depth_vmin, vmax=_depth_vmax)
        _plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
        _plt.close()

    save_depth_map_exact(np.load(_depth_npy), resolution, filename=paths["depth_map"])

import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.path import Path
import cv2
import math
from tqdm import tqdm
import pickle

import sys
sys.path.append(paths["segment_anything"]) # you should prepare SAM before you use the next step
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
sam_checkpoint = paths["sam_checkpoint"]
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
# read projection map, here you also can change the channel combination of input image
image = cv2.imread(paths["depth_map"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('on')
plt.savefig(os.path.join(os.path.dirname(paths["state"]), "sam_depth_input.png"), dpi=150, bbox_inches='tight')
# plt.show()
# template prompt generation, including coarse mask and point
def fill_polygon(mask, vertices):

    path = Path(vertices)
    
    y_coords, x_coords = np.mgrid[:mask.shape[0], :mask.shape[1]]
    points = np.vstack((x_coords.flatten(), y_coords.flatten())).T
    mask_inside = path.contains_points(points).reshape(mask.shape)

    mask[mask_inside] = 1
_mirror_k_geometry = _tunnel_prefix == "3"


def _mirror_real_x_about_center(coords_mm: np.ndarray, center_x_mm: float) -> np.ndarray:
    out = coords_mm.copy()
    out[:, 0] = 2 * center_x_mm - out[:, 0]
    return out

             
def generate_template_mask(height, width, prompt_centre, block, resolution=0.005):
    mask = np.zeros((height, width), dtype=np.uint8)

    prompt_centre_x, prompt_centre_y = prompt_centre
    x = prompt_centre_x * (resolution*1000)
    y = prompt_centre_y * (resolution*1000)
    
    if block == 'K':
        vertices_real = np.array([[x-625,y-619.16],[x-625,y+619.16],[x+625,y+460.77],[x+625,y-460.77]])
        if _mirror_k_geometry:
            vertices_real = _mirror_real_x_about_center(vertices_real, x)
    elif block == 'B1':
        vertices_real = np.array([[x-625,y-1619.89],[x-625,y+1540.69],[x+625,y+1699.08],[x+625,y-1619.89]])
    elif block == 'B2':
        vertices_real = np.array([[x-625,y-1540.69],[x-625,y+1619.89],[x+625,y+1619.89],[x+625,y-1699.08]])
    else:
        vertices_real = np.array([[x-625,y-1619.89],[x-625,y+1619.89],[x+625,y+1619.89],[x+625,y-1619.89]])
        
    vertices = vertices_real / (resolution*1000)
        
    fill_polygon(mask, vertices)
    return mask

def generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution,
                           segment_width, K_height, AB_height, y_bounds):
    """
    prompt_centre is prompt centre of this sub-image, pixel coordinate
    (initial_x, map_y) pixel coordinate of prompt centre in orignal depth map
    block is type of block of segment
    resolution is pixel/real ratio
    other parameters are design params
    """
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
            [x+700,y-547.57],[x+700,y-373.96],[x+700,y-219.01],[x+700,y],[x+700,y+219.01],[x+700,y+373.96],[x+700,y+547.57],  # above are negative

            [x-500,y-505.96],[x-511.06,y-310.91],[x-500,y],[x-511.06,y+310.91],[x-500,y+505.96],
            [x-348.16,y-485.91],[x-348.16,y+485.91],
            [x,y-439.96],[x,y+439.96],
            [x+348.16,y-394.01],[x+348.16,y+394.01],
            [x+500,y-373.96],[x+511.06,y-219.01],[x+500,y],[x+511.06,y+219.01],[x+500,y+373.96]
        ])
        if _mirror_k_geometry:
            points_real = _mirror_real_x_about_center(points_real, x)
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

    # adjust y based on map_y, the bolt holes within this range are sealed.
    # for i in range(len(labels)):
    #     if labels[i] == 0:
    #         y_cond = points_real[i, 1] + map_y < 4100 or points_real[i, 1] + map_y > 13000
    #         x_cond = abs(points_real[i, 0] - x) <= segment_width*0.5
    #         y_limit = K_height if block == 'K' else AB_height
    #         y_cond2 = abs(points_real[i, 1] - y) <= y_limit*0.5
            
    #         if y_cond and x_cond and y_cond2:
    #             labels[i] = 1

    keep_mask = np.ones(len(labels), dtype=bool)
    for i in range(len(labels)):
        if labels[i] == 0:
            y_cond = points_real[i, 1] + map_y < y_bounds[0] or points_real[i, 1] + map_y > y_bounds[1]
            x_cond = abs(points_real[i, 0] - x) <= segment_width * 0.5
            y_limit = K_height if block == 'K' else AB_height
            y_cond2 = abs(points_real[i, 1] - y) <= y_limit * 0.5
            
            if y_cond and x_cond and y_cond2:
                keep_mask[i] = False
            
    points_real = points_real[keep_mask]
    labels = labels[keep_mask]
    
    points = points_real / (resolution*1000)

    # outside the image x direction
    within_bounds = (points[:, 0] >= 0) & ((points[:, 0] + initial_x - (segment_width*0.5+padding)/(resolution*1000)) <= image.shape[1])
    points = points[within_bounds]
    labels = labels[within_bounds]
        
    return points, labels
# define all auxiliary functions
def convert_to_pixel_coords(real_dist, resolution=0.005):
    return int(real_dist / (resolution*1000))

def crop_image_and_mask_logits(image, cx, cy, crop_width, crop_height, block, resolution):
    img_height, img_width, _ = image.shape

    # crop box calculation
    x1 = max(cx - crop_width // 2, 0)
    y1 = max(cy - crop_height // 2, 0)
    x2 = min(cx + crop_width // 2, img_width)
    y2 = min(cy + crop_height // 2, img_height)

    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    
    # prompt centre of this sub-image, pixel coordinate
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

    h, w = logits.shape
    if h == expected_shape[0] and w == expected_shape[1]:
        pass
    else:
        scale = expected_shape[0] / max(h, w)
        new_h = int(h * scale + 0.5)
        new_w = int(w * scale + 0.5)
        logits = cv2.resize(logits, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # pad to 256x256
        h, w = logits.shape
        padh = expected_shape[0] - h
        padw = expected_shape[1] - w
        logits = np.pad(logits, ((0, padh), (0, padw)), mode="constant", constant_values=0)

    logits = logits[None]
    assert logits.shape == (1, 256, 256), f"{logits.shape}"
    return logits

def restore_sam_logits(logits, original_shape):
    """
    reproject logits to same size of mask
    
    params:
        logits: 256x256 logits
        original_shape: original mask shape (height, width)
    
    return:
        logits with same shape of mask
    """
    
    orig_h, orig_w = original_shape
    
    trafo = ResizeLongestSide(max(orig_h, orig_w))
    resized_logits = trafo.apply_image(logits[..., None])
    resized_logits = resized_logits.squeeze()
    
    resized_logits = resized_logits[:orig_h, :orig_w]
    
    return resized_logits

def compute_block_label(segment_per_ring):
    """
    obtain the block name
    """
    block_labels = ['K','B1']
    num_a_labels = segment_per_ring - 3
    block_labels += [f'A{i+1}' for i in range(num_a_labels)]
    block_labels += ['B2']
    
    return block_labels


def geometric_segment(detected_df, image_shape, ring_count, K_height, AB_height,
                      segment_per_ring, segment_order, resolution):
    """Complex-tunnel (7-seg) fallback: label from detected K Y and circular tiling."""
    H, W = image_shape[:2]
    scale = resolution * 1000.0
    K_px = int(round(K_height / scale))
    AB_px = int(round(AB_height / scale))
    ring_w = W / float(ring_count)

    if segment_per_ring != 7 or len(segment_order) != 7:
        raise ValueError(
            "geometric_segment requires segment_per_ring==7 and len(segment_order)==7"
        )
    if segment_order[0] != "K":
        raise ValueError("geometric_segment requires segment_order[0] == 'K'")

    block_to_label = {name: i + 1 for i, name in enumerate(segment_order)}
    downward_blocks = list(reversed(segment_order[1:]))

    label_map = np.zeros((H, W), dtype=int)
    ring_map = np.zeros((H, W), dtype=int)
    y_coords = np.arange(H, dtype=np.float64)

    if len(detected_df) != ring_count:
        print(
            f"  geometric_segment: aligning {len(detected_df)} detected rows "
            f"to ring_count={ring_count} (sort by X, take first {ring_count})"
        )
        detected_df = detected_df.sort_values("X").head(ring_count).reset_index(drop=True)

    for ring_idx, (_, row) in enumerate(detected_df.iterrows()):
        x0 = int(math.floor(ring_idx * ring_w))
        x1 = int(math.ceil(min((ring_idx + 1) * ring_w, W)))
        ky = float(row["Y"])
        pos = (y_coords - (ky + K_px / 2.0)) % H

        for i, block in enumerate(downward_blocks):
            mask = (pos >= i * AB_px) & (pos < (i + 1) * AB_px)
            label_map[mask, x0:x1] = block_to_label[block]
            ring_map[mask, x0:x1] = ring_idx

        k_start = len(downward_blocks) * AB_px
        k_mask = (pos >= k_start) & (pos < k_start + K_px)
        label_map[k_mask, x0:x1] = block_to_label["K"]
        ring_map[k_mask, x0:x1] = ring_idx

    return label_map, ring_map


def sam_prediction(cropped_image, points, labels, template_mask_logit):
    predictor.set_image(cropped_image)
            
    mask, score, logit = predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=template_mask_logit,
            multimask_output=False,
    )
        
    return mask, score, logit[0]
# define main processing functions
def process_row(df_row, image, resolution, segment_per_ring, segment_width,
                K_height, angle, AB_height, padding, crop_margin, y_bounds):

    initial_x, initial_y = df_row['X'], df_row['Y']
    block_labels = compute_block_label(segment_per_ring)  # start from "K, B1, A1, A2, A3,..., B2" depends on number of segment per ring

    delta_x = convert_to_pixel_coords(0.5*segment_width + padding, resolution)
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
                delta_y = convert_to_pixel_coords(0.5*K_height + math.tan(math.radians(angle))*700+100 + crop_margin, resolution) # K-block
                map_y = initial_y
            else:
                delta_y = convert_to_pixel_coords(0.5*AB_height + math.tan(math.radians(angle))*700+100 + crop_margin, resolution) # other block
                if block_label_index == 1:
                    map_y = initial_y - convert_to_pixel_coords(0.5 * K_height + 0.5 * AB_height, resolution)
                else:
                    map_y = map_y - convert_to_pixel_coords(AB_height, resolution)

            cropped_image, template_mask_logit, prompt_centre = crop_image_and_mask_logits(
                image, initial_x, map_y,2 * delta_x, 2 * delta_y, block, resolution)
            points, labels = generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution, segment_width, K_height, AB_height, y_bounds)
        
            # check if outside the image in y-direction
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

            if segment_per_ring == 7 and block_label_index >= 4 and not reverse:
                reverse = True
                block_label_index = -1
                continue
            
        if reverse:
            block = block_labels[block_label_index]
            if block_label_index == -1:
                map_y = initial_y + convert_to_pixel_coords(0.5 * K_height + 0.5 * AB_height, resolution)
            else:
                map_y = map_y + convert_to_pixel_coords(AB_height, resolution)

            cropped_image, template_mask_logit, prompt_centre = crop_image_and_mask_logits(image, initial_x, map_y, 
                                                                                            2 * delta_x, 2 * delta_y, block, resolution)
            points, labels = generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution, segment_width, K_height, AB_height, y_bounds)

            # check if outside the image in y-direction
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
# Define execution functions
def sam_segment(df, image, resolution, segment_per_ring, segment_width, K_height, AB_height, angle, padding, crop_margin, y_bounds):
    all_results = []
 
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        result = process_row(row, image, resolution, segment_per_ring, segment_width, K_height, angle, AB_height, padding, crop_margin, y_bounds)
        all_results.append(result)
    
    return all_results

if _use_geometric_fallback:
    segment_order_geom = config.get("segment_order")
    if not segment_order_geom:
        segment_order_geom = compute_block_label(7)
    print(f"Using geometric fallback for complex tunnel {tunnel_id}")
    print(f"  segment_order={segment_order_geom}, K_px≈{int(round(K_height / (resolution * 1000)))}, "
          f"AB_px≈{int(round(AB_height / (resolution * 1000)))}")
    result_image, ring_image = geometric_segment(
        df_loc, image.shape, ring_count, K_height, AB_height,
        segment_per_ring, segment_order_geom, resolution,
    )
    fix_ring = np.where(
        (ring_image >= 1) & (ring_image <= (ring_count - 1)),
        ring_count - ring_image,
        ring_image,
    )
    if "segment_order" in config and config.get("use_original_label_distributions"):
        block_to_label = {
            name: i + 1 for i, name in enumerate(config["segment_order"])
        }
    else:
        block_to_label = {name: i + 1 for i, name in enumerate(segment_order_geom)}
    results = []
else:
    results = sam_segment(df_loc, image, resolution, segment_per_ring, segment_width, K_height, AB_height, angle, padding, crop_margin, y_bounds)
    # for saving results
    with open(paths["results_pkl"], "wb") as file:
        pickle.dump(results, file)
    # for loading results
    with open(paths["results_pkl"], "rb") as file:
        results = pickle.load(file)

    test_sample = results[0][0]
    input_point = test_sample['points']
    input_label = test_sample['labels']
    cropped_image = test_sample['cropped_image']
    masks = test_sample['mask']
    logits = test_sample['logit']
    print(test_sample['block'])

    height, width, _ = cropped_image.shape
    display_dpi = 72
    display_figsize = (width / display_dpi * 2, height / display_dpi * 2)
    plt.figure(figsize=display_figsize, dpi=display_dpi)
    plt.imshow(cropped_image)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.savefig(os.path.join(os.path.dirname(paths["state"]), "sam_sample_block.png"), dpi=120, bbox_inches='tight')

    if "segment_order" in config and config.get("use_original_label_distributions"):
        block_to_label = {}
        for i, block_name in enumerate(config["segment_order"], start=1):
            block_to_label[block_name] = i
        print(f"Using configured segment order: {config['segment_order']}")
        print(f"Block to label mapping: {block_to_label}")
    else:
        block_to_label = {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'B2': 6}
        if segment_per_ring == 7:
            block_to_label = {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'A4': 6, 'B2': 7}
        print(f"Using default block mapping: {block_to_label}")

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

            update_mask = (new_logits > current_logits) & mask
            logits_map[valid_slice_y, valid_slice_x][update_mask] = new_logits[update_mask]
            label_map[valid_slice_y, valid_slice_x][update_mask] = block_to_label[block]
            ring_map[valid_slice_y, valid_slice_x][update_mask] = ring_index

    result_image = label_map
    ring_image = ring_map
    fix_ring = (ring_image + 1) % ring_count

## 3. Project back to point cloud
import numpy as np

def visualize_combined_results(image, result_image, ring_image, assigned_mask):
    # Use new method to get color maps
    cmap_result = plt.colormaps['tab10']  # Use 'tab10' color map, you can choose others like 'Set1', 'Set2', 'Set3', etc.
    cmap_ring = plt.get_cmap('tab20')  # Use 'tab20' color map, which provides 20 different colors

    # Create RGB images for labeled and ring results
    colored_result = np.zeros((*result_image.shape, 3))
    colored_ring = np.zeros((*ring_image.shape, 3))
    
    # Assign colors to each label for result_image
    for label in range(max(block_to_label.values()) + 1):
        mask = result_image == label
        colored_result[mask] = cmap_result(label)[:3]
    
    # Show GT ring ids (1..9,0 left-to-right); ring_image 0 is SAM column index, not GT ring 0.
    display_ring = (ring_image + 1) % ring_count
    unique_rings = np.unique(display_ring[assigned_mask])
    for ring in unique_rings:
        mask = assigned_mask & (display_ring == ring)
        color_index = int(ring) % 20
        colored_ring[mask] = cmap_ring(color_index)[:3]
    
    # Create figure with three vertically stacked subplots, reduce height
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [1, 1, 1]})
    
    # Reduce space between subplots
    plt.subplots_adjust(hspace=0.1)
    
    # Display original depth map
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Depth Map', pad=10)
    ax1.axis('off')
    
    # Display results with color labels
    ax2.imshow(image, cmap='gray', alpha=0.5)  # Show original depth map as background
    ax2.imshow(colored_result, alpha=0.5)  # Overlay color labels
    ax2.set_title('Labeled Mask Results', pad=10)
    ax2.axis('off')
    
    # Display ring instance segmentation results
    ax3.imshow(image, cmap='gray', alpha=0.5)  # Show original depth map as background
    ax3.imshow(colored_ring, alpha=0.5)  # Overlay color labels
    ax3.set_title('Ring Instance Segmentation Results', pad=10)
    ax3.axis('off')
    
    # Add color legend for labeled results
    legend_elements_result = [
        plt.Line2D([0], [0], color=cmap_result(label), lw=4, label=f'{block} (Label {label})')
        for label, block in {v: k for k, v in block_to_label.items()}.items()
    ]
    ax2.legend(handles=legend_elements_result, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add color legend for ring results
    legend_elements_ring = [
        plt.Line2D([0], [0], color=cmap_ring(int(ring) % 20), lw=4, label=f'Ring {ring}')
        for ring in unique_rings
    ]
    ax3.legend(handles=legend_elements_ring, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(paths["state"]), "segmentation_results.png"), dpi=150, bbox_inches='tight')
    # plt.show()

# Call the visualization function
assigned_mask = result_image > 0
visualize_combined_results(image, result_image, ring_image, assigned_mask)
# SAM column index 0..9 (left->right) -> GT ring labels 1..9,0 along tunnel axis
import numpy as np
import pandas as pd

def project_back_to_point_cloud(segmented_map, instance_map, pixel_to_point, df):
    """
    Project the segmented depth map back to the point cloud.

    Parameters:
    segmented_map: Segmented depth map
    instance_map: Instance segmentation depth map
    pixel_to_point: Dict, mapping pixels to points
    df: Original point cloud data

    Returns:
    Updated point cloud DataFrame, including new segmentation labels.
    """

    # Create a copy of df to avoid modifying the original df
    df_copy = df.copy()

    # Initialize new label arrays
    pred = df_copy['pred'].values  # Keep the existing 'pred' values
    pred_ring = np.full(len(df_copy), -1, dtype=int)  # Initialize to -1

    # Get pixel coordinates and point cloud indices
    pixel_to_point_df = pd.DataFrame(pixel_to_point)
    y = pixel_to_point_df['pixel_y'].values
    x = pixel_to_point_df['pixel_x'].values
    point_indices = pixel_to_point_df['index'].values

    # Find indices of rows that need updating
    valid_point_mask = np.isin(point_indices, df_copy.index.values)

    # Update only rows where 'pred' value is 7
    valid_update_mask = (pred[point_indices[valid_point_mask]] == 7)

    h, w = segmented_map.shape[:2]
    yy = y[valid_point_mask][valid_update_mask]
    xx = x[valid_point_mask][valid_update_mask]
    yy = np.clip(yy, 0, h - 1)
    xx = np.clip(xx, 0, w - 1)

    # Use point_indices to index and update pred and pred_ring
    pred[point_indices[valid_point_mask][valid_update_mask]] = segmented_map[yy, xx]
    pred_ring[point_indices[valid_point_mask][valid_update_mask]] = instance_map[yy, xx]

    # Update df_copy
    df_copy['pred'] = pred
    df_copy['pred_ring'] = pred_ring

    return df_copy
updated_df = project_back_to_point_cloud(result_image, fix_ring, pixel_to_point, df_point_cloud)
updated_df.head()
updated_df.to_csv(paths["final_csv"],index=False)
df_pred = pd.DataFrame()
df_pred['gt_labels'] = updated_df['segment']
df_pred['gt_rings'] = updated_df['ring']
df_pred['pred_labels'] = updated_df['pred']
df_pred['pred_rings'] = updated_df['pred_ring']
df_pred.to_csv(paths["only_label"],index=False)
# =============for single station============
import pandas as pd


import pickle

with open(paths["results_pkl"], "wb") as file:
    pickle.dump(results, file)
with open(paths["results_pkl"], "rb") as file:
    results = pickle.load(file)

updated_df.to_csv(paths["final_csv"], index=False)
df_pred.to_csv(paths["only_label"], index=False)
print(f"SAM complete -> {paths['only_label']}")

