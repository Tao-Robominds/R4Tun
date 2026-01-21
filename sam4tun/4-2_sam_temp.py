"""
SAM Segmentation with GT-Learned Parameters for Tunnel 4-1
Modified version with optimized template parameters.
"""

import sys
sys.path.insert(0, "/home/boringtao/Projects/P4Tun_Off/sam4tun/segment-anything")

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

if len(sys.argv) != 2:
    print("Usage: python 4-2_sam_gt_learned.py <tunnel_id>")
    sys.exit(1)

tunnel_id = sys.argv[1]
base_dir = f"data/{tunnel_id}/"
initial_prompt_points = pd.read_csv(os.path.join(base_dir, "detected.csv"))
pixel_to_point = pickle.load(open(os.path.join(base_dir, "pixel_to_point.pkl"), "rb"))
df_point_cloud = pd.read_csv(os.path.join(base_dir, "enhanced.csv"))
ring_count = int(open(f'data/{tunnel_id}/ring_count.txt', 'r').read())

print(f"Processing tunnel: {tunnel_id}")

# GT-LEARNED PARAMETERS
K_HEIGHT = 1300
AB_HEIGHT = 3600
SEGMENT_WIDTH = 1400.0
ANGLE = 7.52
K_TEMPLATE_HW = 700
K_TEMPLATE_HH_LEFT = 680.0
K_TEMPLATE_HH_RIGHT = 520.0
AB_TEMPLATE_HW = 700.0
AB_TEMPLATE_HH = 2000

print(f"Using GT-learned parameters: K_HEIGHT={K_HEIGHT}, AB_HEIGHT={AB_HEIGHT}")

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

def generate_template_mask(height, width, prompt_centre, block, resolution=0.005):
    mask = np.zeros((height, width), dtype=np.uint8)
    x = prompt_centre[0] * (resolution*1000)
    y = prompt_centre[1] * (resolution*1000)
    
    if block == 'K':
        vertices_real = np.array([
            [x-K_TEMPLATE_HW, y-K_TEMPLATE_HH_LEFT],
            [x-K_TEMPLATE_HW, y+K_TEMPLATE_HH_LEFT],
            [x+K_TEMPLATE_HW, y+K_TEMPLATE_HH_RIGHT],
            [x+K_TEMPLATE_HW, y-K_TEMPLATE_HH_RIGHT]
        ])
    elif block == 'B1':
        vertices_real = np.array([
            [x-AB_TEMPLATE_HW, y-AB_TEMPLATE_HH],
            [x-AB_TEMPLATE_HW, y+AB_TEMPLATE_HH*0.95],
            [x+AB_TEMPLATE_HW, y+AB_TEMPLATE_HH*1.05],
            [x+AB_TEMPLATE_HW, y-AB_TEMPLATE_HH]
        ])
    elif block == 'B2':
        vertices_real = np.array([
            [x-AB_TEMPLATE_HW, y-AB_TEMPLATE_HH*0.95],
            [x-AB_TEMPLATE_HW, y+AB_TEMPLATE_HH],
            [x+AB_TEMPLATE_HW, y+AB_TEMPLATE_HH],
            [x+AB_TEMPLATE_HW, y-AB_TEMPLATE_HH*1.05]
        ])
    else:
        vertices_real = np.array([
            [x-AB_TEMPLATE_HW, y-AB_TEMPLATE_HH],
            [x-AB_TEMPLATE_HW, y+AB_TEMPLATE_HH],
            [x+AB_TEMPLATE_HW, y+AB_TEMPLATE_HH],
            [x+AB_TEMPLATE_HW, y-AB_TEMPLATE_HH]
        ])
    vertices = vertices_real / (resolution*1000)
    fill_polygon(mask, vertices)
    return mask

def generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution=0.005):
    x = prompt_centre[0] * (resolution*1000)
    y = prompt_centre[1] * (resolution*1000)
    map_y_mm = map_y * (resolution*1000)
    
    if block == 'K':
        hh = K_TEMPLATE_HH_LEFT * 0.95
        hw = K_TEMPLATE_HW * 0.95
        points_real = np.array([
            [x-hw, y-hh*0.9], [x-hw, y], [x-hw, y+hh*0.9],
            [x, y-hh*0.8], [x, y], [x, y+hh*0.8],
            [x+hw, y-hh*0.65], [x+hw, y], [x+hw, y+hh*0.65],
            [x-hw*0.5, y], [x+hw*0.5, y],
        ])
        labels = np.repeat([0, 1], [9, 2])
    else:
        hh = AB_TEMPLATE_HH * 0.95
        hw = AB_TEMPLATE_HW * 0.95
        points_real = np.array([
            [x-hw, y-hh], [x, y-hh], [x+hw, y-hh],
            [x-hw, y], [x, y], [x+hw, y],
            [x-hw, y+hh], [x, y+hh], [x+hw, y+hh],
            [x-hw*0.5, y], [x+hw*0.5, y],
        ])
        labels = np.repeat([0, 1], [9, 2])

    keep_mask = np.ones(len(labels), dtype=bool)
    for i in range(len(labels)):
        if labels[i] == 0:
            y_cond = points_real[i, 1] + map_y_mm < 4200 or points_real[i, 1] + map_y_mm > 13100
            x_cond = abs(points_real[i, 0] - x) <= SEGMENT_WIDTH * 0.5
            y_limit = K_HEIGHT if block == 'K' else AB_HEIGHT
            y_cond2 = abs(points_real[i, 1] - y) <= y_limit * 0.5
            if y_cond and x_cond and y_cond2:
                keep_mask[i] = False
    points_real = points_real[keep_mask]
    labels = labels[keep_mask]
    points = points_real / (resolution*1000)
    within_bounds = (points[:, 0] >= 0) & ((points[:, 0] + initial_x - (SEGMENT_WIDTH*0.5+150)/(resolution*1000)) <= image.shape[1])
    points = points[within_bounds]
    labels = labels[within_bounds]
    return points, labels

def convert_to_pixel_coords(real_dist, resolution=0.005):
    return int(real_dist / (resolution*1000))

def crop_image_and_mask_logits(image, cx, cy, crop_width, crop_height, block, resolution):
    img_height, img_width, _ = image.shape
    x1 = max(cx - crop_width // 2, 0)
    y1 = max(cy - crop_height // 2, 0)
    x2 = min(cx + crop_width // 2, img_width)
    y2 = min(cy + crop_height // 2, img_height)
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    prompt_centre = (cx - x1, cy - y1)
    cropped_template_mask = generate_template_mask(cropped_image.shape[0], cropped_image.shape[1], prompt_centre, block, resolution)
    template_mask_logits = compute_logits_from_mask(cropped_template_mask)
    return cropped_image, template_mask_logits, prompt_centre

def compute_logits_from_mask(mask, eps=1e-3):
    def inv_sigmoid(x):
        return np.log(x / (1 - x))
    logits = np.zeros(mask.shape, dtype="float32")
    logits[mask == 1] = 1 - eps
    logits[mask == 0] = eps
    logits = inv_sigmoid(logits)
    expected_shape = (256, 256)
    if logits.shape != expected_shape:
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image(logits[..., None])
        if logits.shape[0] != expected_shape[0] or logits.shape[1] != expected_shape[1]:
            h, w = logits.shape[:2]
            padh = expected_shape[0] - h
            padw = expected_shape[1] - w
            logits = np.pad(logits.squeeze() if logits.ndim > 2 else logits, ((0, padh), (0, padw)), mode="constant", constant_values=0)
    logits = logits.reshape(1, 256, 256) if logits.ndim == 2 else logits[None] if logits.ndim == 2 else logits
    if logits.shape != (1, 256, 256):
        logits = logits.reshape(1, 256, 256)
    return logits

def restore_sam_logits(logits, original_shape):
    orig_h, orig_w = original_shape
    trafo = ResizeLongestSide(max(orig_h, orig_w))
    resized_logits = trafo.apply_image(logits[..., None])
    resized_logits = resized_logits.squeeze()
    return resized_logits[:orig_h, :orig_w]

def compute_block_label(segment_per_ring):
    block_labels = ['K', 'B1']
    block_labels += [f'A{i+1}' for i in range(segment_per_ring - 3)]
    block_labels += ['B2']
    return block_labels

def sam_prediction(cropped_image, points, labels, template_mask_logit):
    predictor.set_image(cropped_image)
    mask, score, logit = predictor.predict(
        point_coords=points, point_labels=labels,
        mask_input=template_mask_logit, multimask_output=False)
    return mask, score, logit[0]

def process_row(df_row, image, resolution=0.005, segment_per_ring=7):
    initial_x, initial_y = df_row['X'], df_row['Y']
    block_labels = compute_block_label(segment_per_ring)
    delta_x = convert_to_pixel_coords(0.5*SEGMENT_WIDTH + 150, resolution)
    reverse, stop, map_y, block_label_index = False, False, 0, 0
    results = []
    
    for i in range(segment_per_ring):
        if not reverse:
            block = block_labels[block_label_index]
            if block_label_index == 0:
                delta_y = convert_to_pixel_coords(0.5*K_HEIGHT + math.tan(math.radians(ANGLE))*700+150, resolution)
                map_y = initial_y
            else:
                delta_y = convert_to_pixel_coords(0.5*AB_HEIGHT + math.tan(math.radians(ANGLE))*700+150, resolution)
                if block_label_index == 1:
                    map_y = initial_y - convert_to_pixel_coords(0.5 * K_HEIGHT + 0.5 * AB_HEIGHT, resolution)
                else:
                    map_y = map_y - convert_to_pixel_coords(AB_HEIGHT, resolution)
            
            cropped_image, template_mask_logit, prompt_centre = crop_image_and_mask_logits(
                image, initial_x, map_y, 2 * delta_x, 2 * delta_y, block, resolution)
            points, labels = generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution)
            
            if np.any(points[:, 1] < 0):
                points = points[points[:, 1] >= 0]
                labels = labels[:len(points)]
                reverse = True
            
            mask, score, logit = sam_prediction(cropped_image, points, labels, template_mask_logit)
            results.append({
                'left_top': (initial_x-prompt_centre[0], map_y-prompt_centre[1]),
                'block': block, 'mask': mask, 'logit': logit
            })
            
            if reverse:
                block_label_index = -1
                continue
            block_label_index += 1
            
        if reverse:
            block = block_labels[block_label_index]
            if block_label_index == -1:
                map_y = initial_y + convert_to_pixel_coords(0.5 * K_HEIGHT + 0.5 * AB_HEIGHT, resolution)
            else:
                map_y = map_y + convert_to_pixel_coords(AB_HEIGHT, resolution)
            
            cropped_image, template_mask_logit, prompt_centre = crop_image_and_mask_logits(
                image, initial_x, map_y, 2 * delta_x, 2 * delta_y, block, resolution)
            points, labels = generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution)
            
            if np.any((points[:, 1]+map_y-delta_y) > image.shape[0]):
                valid = (points[:, 1]+map_y-delta_y) <= image.shape[0]
                points, labels = points[valid], labels[:sum(valid)]
                stop = True
            
            mask, score, logit = sam_prediction(cropped_image, points, labels, template_mask_logit)
            results.append({
                'left_top': (initial_x-prompt_centre[0], map_y-prompt_centre[1]),
                'block': block, 'mask': mask, 'logit': logit
            })
            
            if stop:
                break
            block_label_index -= 1
    return results

all_results = []
for _, row in tqdm(initial_prompt_points.iterrows(), total=len(initial_prompt_points), desc="Processing rows"):
    all_results.append(process_row(row, image))

block_to_label = {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'A4': 6, 'B2': 7}
logits_map = np.full(image.shape[:2], -np.inf, dtype=float)
label_map = np.zeros(image.shape[:2], dtype=int)
ring_map = np.zeros(image.shape[:2], dtype=int)

for ring_index, ring in enumerate(all_results):
    for item in ring:
        mask = item['mask'][0]
        logits = item['logit']
        block = item['block']
        start_x, start_y = map(int, item['left_top'])
        end_y, end_x = start_y + mask.shape[0], start_x + mask.shape[1]
        start_y, start_x = max(0, start_y), max(0, start_x)
        end_y, end_x = min(image.shape[0], end_y), min(image.shape[1], end_x)
        
        new_logits = restore_sam_logits(logits, mask.shape)
        current_logits = logits_map[start_y:end_y, start_x:end_x]
        
        if mask.shape == current_logits.shape and new_logits.shape == current_logits.shape:
            update_mask = (new_logits > current_logits) & mask
            logits_map[start_y:end_y, start_x:end_x][update_mask] = new_logits[update_mask]
            label_map[start_y:end_y, start_x:end_x][update_mask] = block_to_label[block]
            ring_map[start_y:end_y, start_x:end_x][update_mask] = ring_index

fix_ring = np.where((ring_map >= 1) & (ring_map <= (ring_count-1)), ring_count - ring_map, ring_map)

def project_back_to_point_cloud(segmented_map, instance_map, pixel_to_point, df):
    df_copy = df.copy()
    pred = df_copy['pred'].values
    pred_ring = np.full(len(df_copy), -1, dtype=int)
    pixel_to_point_df = pd.DataFrame(pixel_to_point)
    y, x = pixel_to_point_df['pixel_y'].values, pixel_to_point_df['pixel_x'].values
    point_indices = pixel_to_point_df['index'].values
    img_height, img_width = segmented_map.shape
    
    valid_point_mask = np.isin(point_indices, df_copy.index.values)
    valid_update_mask = np.isin(pred[point_indices[valid_point_mask]], [0, 7])
    y_valid, x_valid = y[valid_point_mask][valid_update_mask], x[valid_point_mask][valid_update_mask]
    bounds_mask = (y_valid >= 0) & (y_valid < img_height) & (x_valid >= 0) & (x_valid < img_width)
    
    final_indices = point_indices[valid_point_mask][valid_update_mask][bounds_mask]
    final_y, final_x = y_valid[bounds_mask], x_valid[bounds_mask]
    pred[final_indices] = segmented_map[final_y, final_x]
    pred_ring[final_indices] = instance_map[final_y, final_x]
    df_copy['pred'], df_copy['pred_ring'] = pred, pred_ring
    return df_copy

updated_df = project_back_to_point_cloud(label_map, fix_ring, pixel_to_point, df_point_cloud)
updated_df.to_csv(f'{base_dir}/final.csv', index=False)

df_pred = pd.DataFrame({
    'gt_labels': updated_df['segment'],
    'gt_rings': updated_df['ring'],
    'pred_labels': updated_df['pred'],
    'pred_rings': updated_df['pred_ring']
})
df_pred.to_csv(f'{base_dir}/only_label.csv', index=False)
print(f"Saved results to {base_dir}")
