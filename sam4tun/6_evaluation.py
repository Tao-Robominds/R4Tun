#!/usr/bin/env python3
# AUTO-GENERATED from SAM4Tun.py — do not edit body; re-run generate_modules.py

import sys
import os
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from helpers.pipeline_io import ensure_dir

tunnel_id = sys.argv[1]
paths = ensure_dir(tunnel_id)

test = pd.read_csv(paths["only_label"])
test
# =================  start evaluation  =================
gt_rings = test['gt_rings'].values.astype(int)
# global_gt_rings = test['global_gt_rings'].values.astype(int)
gt_labels = test['gt_labels'].values.astype(int)

pred_rings = test['pred_rings'].values.astype(int)
# global_pred_rings = test['global_pred_rings'].values.astype(int)
pred_labels = test['pred_labels'].values.astype(int)
def compute_iou(pred_points, gt_points):
    """Compute the IoU between predicted points and ground truth points."""
    intersection = len(np.intersect1d(pred_points, gt_points))
    union = len(np.union1d(pred_points, gt_points))
    if union == 0:
        return 0
    return intersection / union

def compute_iou_matrix(pred_rings, gt_rings, pred_labels, gt_labels, category):
    """Compute the IoU matrix between predicted and ground truth instances for a given category."""
    pred_instances = np.unique(pred_rings[pred_labels == category])
    gt_instances = np.unique(gt_rings[gt_labels == category])

    iou_matrix = np.zeros((len(pred_instances), len(gt_instances)))

    for i, pred_ring in enumerate(pred_instances):
        pred_points = np.where((pred_labels == category) & (pred_rings == pred_ring))[0]

        for j, gt_ring in enumerate(gt_instances):
            gt_points = np.where((gt_labels == category) & (gt_rings == gt_ring))[0]
            iou_matrix[i, j] = compute_iou(pred_points, gt_points)

    return iou_matrix, pred_instances, gt_instances
# For one station
import numpy as np
from tqdm import tqdm

iou_thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2) 
categories = [1, 2, 3, 4, 5, 6]  # 0 is background
results = {cat: {'TP': [], 'FP': [], 'FN': []} for cat in categories}

for cat in tqdm(categories, desc='Processing categories', unit='category'):
    # calculate iou matrix
    # iou_matrix, pred_instances, gt_instances = compute_iou_matrix(global_pred_rings, global_gt_rings, pred_labels, gt_labels, cat)
    iou_matrix, pred_instances, gt_instances = compute_iou_matrix(pred_rings, gt_rings, pred_labels, gt_labels, cat)

    for iou_thresh in iou_thresholds:
        TP = 0
        FP = 0
        FN = 0

        # matching results
        matched_pred = set()
        matched_gt = set()

        for i in range(len(pred_instances)):
            for j in range(len(gt_instances)):
                if iou_matrix[i, j] >= iou_thresh:
                    TP += 1
                    matched_pred.add(i)
                    matched_gt.add(j)

        # FP
        FP = len(pred_instances) - len(matched_pred)

        # FN
        FN = len(gt_instances) - len(matched_gt)

        results[cat]['TP'].append(TP)
        results[cat]['FP'].append(FP)
        results[cat]['FN'].append(FN)
for cat in categories:
    print(f"Category: {cat}")
    for idx, iou_thresh in enumerate(iou_thresholds):
        print(f"IoU Threshold: {iou_thresh:.2f}, TP: {results[cat]['TP'][idx]}, "
              f"FP: {results[cat]['FP'][idx]}, FN: {results[cat]['FN'][idx]}")
# Initialize dictionaries to store the aggregated results across all categories
total_results = {'TP': [], 'FP': [], 'FN': []}

# Iterate over all IoU thresholds
for idx, iou_thresh in enumerate(iou_thresholds):
    total_TP, total_FP, total_FN = 0, 0, 0
    
    # Aggregate TP, FP, FN for all categories at this IoU threshold
    for cat in categories:
        total_TP += results[cat]['TP'][idx]
        total_FP += results[cat]['FP'][idx]
        total_FN += results[cat]['FN'][idx]
    
    # Store the aggregated results
    total_results['TP'].append(total_TP)
    total_results['FP'].append(total_FP)
    total_results['FN'].append(total_FN)

# Output the total results
for idx, iou_thresh in enumerate(iou_thresholds):
    print(f"IoU Threshold: {iou_thresh:.2f}, Total TP: {total_results['TP'][idx]}, "
          f"Total FP: {total_results['FP'][idx]}, Total FN: {total_results['FN'][idx]}")
from collections import defaultdict

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales). this part is from coco"""
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap

def calculate_metrics(results):
    class_aps = defaultdict(list)
    ap_per_class_iou = defaultdict(list)
    iou_thresholds = iou_thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)  # List of IoU thresholds

    # Iterate over all IoU thresholds
    for idx, iou_thresh in enumerate(iou_thresholds):
        for cat in results.keys():
            tp = results[cat]['TP'][idx]
            fp = results[cat]['FP'][idx]
            fn = results[cat]['FN'][idx]

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            class_aps[cat].append((recall, precision))

            # Calculate AP for this specific class and IoU threshold
            ap = average_precision(np.array([[recall]]), np.array([[precision]]))
            ap_per_class_iou[cat].append((iou_thresh, ap[0]))
    
    # Calculate mAP, mAP50, mAP75, mAP90, and class_mAP
    all_aps = [ap for aps in ap_per_class_iou.values() for _, ap in aps]
    mAP = np.mean(all_aps)
    
    mAP50 = np.mean([ap for cat in ap_per_class_iou.values() for iou, ap in cat if iou == 0.5])
    mAP75 = np.mean([ap for cat in ap_per_class_iou.values() for iou, ap in cat if iou == 0.75])
    mAP90 = np.mean([ap for cat in ap_per_class_iou.values() for iou, ap in cat if iou == 0.9])
    
    class_mAP = {cat: np.mean([ap for _, ap in aps]) for cat, aps in ap_per_class_iou.items()}
    
    return class_aps, ap_per_class_iou, mAP, mAP50, mAP75, mAP90, class_mAP
class_aps, ap_per_class_iou, mAP, mAP50, mAP75, mAP90, class_mAP = calculate_metrics(results)

print(f"mAP: {mAP:.4f}")
print(f"mAP@50: {mAP50:.4f}")
print(f"mAP@75: {mAP75:.4f}")
print(f"mAP@90: {mAP90:.4f}")
print("\nClass-wise mAP:")
for class_id, ap in class_mAP.items():
    print(f"Class {class_id}: {ap:.4f}")
import numpy as np
gt_binary = np.where(gt_labels == 0, 0, 1)
pred_binary = np.where(pred_labels == 0, 0, 1)
# sematic segmentation
def calculate_semantic_metrics(gt_labels, pred_labels):
    num_classes = int(max(max(gt_labels), max(pred_labels)) + 1)
    class_counts = np.zeros(num_classes)
    class_correct = np.zeros(num_classes)
    ious = np.zeros(num_classes)

    for gt, pred in zip(gt_labels.astype(int), pred_labels):
        class_counts[gt] += 1
        if gt == pred:
            class_correct[gt] += 1

    for cls in range(num_classes):
        tp = class_correct[cls]
        fp = np.sum(pred_labels == cls) - tp
        fn = class_counts[cls] - tp
        ious[cls] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    overall_accuracy = np.sum(class_correct) / np.sum(class_counts)
    mean_iou = np.mean(ious)
    class_accuracy = class_correct / class_counts
    mean_class_acc = class_accuracy.mean()
    per_class_f1_scores = 2 * (class_accuracy * ious) / (class_accuracy + ious)
    f1_scores = per_class_f1_scores.mean()

    return overall_accuracy, mean_iou, class_accuracy, mean_class_acc, ious, per_class_f1_scores, f1_scores
overall_accuracy, mean_iou, class_accuracy, mean_class_acc, ious, per_class_f1_scores, f1_scores = calculate_semantic_metrics(gt_labels, pred_labels)
overall_accuracy, mean_iou, class_accuracy, mean_class_acc, ious, per_class_f1_scores, f1_scores = calculate_semantic_metrics(gt_binary, pred_binary)
# print(f"overall_accuracy: {overall_accuracy:.4f}")
# print(f"mean_class_accuracy: {mean_class_acc:.4f}")
# print(f"mean_iou: {mean_iou:.4f}")
# print(f"f1_score: {f1_scores:.4f}")

# print("\nclass_accuracy:")
# for idx, acc in enumerate(class_accuracy):
#     print(f"Class {idx}: {acc:.4f}")

# print("\niou_per_class:")
# for idx, iou in enumerate(ious):
#     print(f"Class {idx}: {iou:.4f}")

# print("\nper_class_f1_scores:")
# for idx, f1_score in enumerate(per_class_f1_scores):
#     print(f"Class {idx}: {f1_score:.4f}")

print(f"{overall_accuracy:.4f}")
print(f"{mean_class_acc:.4f}")
print(f"{mean_iou:.4f}")
print(f"{f1_scores:.4f}")

for idx, acc in enumerate(class_accuracy):
    print(f"{acc:.4f}")

for idx, iou in enumerate(ious):
    print(f"{iou:.4f}")

for idx, f1_score in enumerate(per_class_f1_scores):
    print(f"{f1_score:.4f}")


