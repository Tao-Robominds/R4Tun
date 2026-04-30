## Reasoning for Tunnels

R4Tun: a reasoning-based multi-agent framework for segmental tunnel analysis in point clouds.

## SAM4Tun Pipeline – High-Level Overview

The `sam4tun/` folder implements a 5‑stage pipeline that takes **raw tunnel point clouds** to **segment‑wise SAM predictions**:

- **Stage 1 – Upfolding (`sam4tun/1_upfolding.py`)**  
  - Input: raw 3D point cloud `data/{tunnel_id}.txt` (`x, y, z, intensity, segment, ring`).  
  - Estimate tunnel direction via convex hull + minimum rotated rectangle.  
  - Slice tunnel into many cross‑sections and fit an ellipse per slice to get centre points.  
  - Fit a smooth 3D centreline curve and convert all points to cylindrical coordinates `(h, θ, r)`.  
  - Output: `data/{tunnel_id}/unwrapped.csv` + `ring_count.txt`.

- **Stage 2 – Denoising (`sam4tun/2_denoising.py`)**  
  - Input: `unwrapped.csv`, `ring_count.txt`.  
  - Initialize a label `pred = 7` for all points.  
  - Radius filter removes points far from the tunnel wall (`pred = 0` → background).  
  - For each ring‑aligned slice, build a 2D density map in `(θ, r)` and estimate a radial cutoff curve; remove low‑density outside points.  
  - Output: `data/{tunnel_id}/denoised.csv`.

- **Stage 3 – Enhancing (`sam4tun/3_enhancing.py`)**  
  - Input: `denoised.csv`.  
  - Compute local curvature for remaining (non‑background) points.  
  - Surface enhancement: iteratively insert midpoint samples between suitable neighbors to densify structural surfaces (new points have `pred = 8`).  
  - Outlier enhancement: detect depth outliers (likely joints) and interpolate additional points along them.  
  - Project enhanced points to a dense depth map in `(h, θ)` space and record pixel→point mappings.  
  - Output: `depth_map.png`, `pixel_to_point.pkl`, `depth_map_outlier.npy`, updated `enhanced.csv`.

- **Stage 4 – Joint Detection (`sam4tun/4-1_detection.py`)**  
  - Input: `depth_map_outlier.npy`, `ring_count.txt`.  
  - Threshold and dilate the outlier depth map to highlight joint structures.  
  - Use Hough transforms to detect oblique, horizontal, and vertical lines; merge verticals into centred “ring lines”.  
  - For each ring line, intersect with oblique/horizontal lines and apply geometric rules to locate joint centres in image space.  
  - Output: joint centre positions `data/{tunnel_id}/detected.csv` and diagnostic `detected_lines.png`.

- **Stage 5 – SAM Segmentation (`sam4tun/4-2_sam.py`)**  
  - Input: `detected.csv`, `pixel_to_point.pkl`, `enhanced.csv`, `ring_count.txt`, SAM checkpoint.  
  - Load `depth_map.png` and SAM; define geometric templates and dense prompt point patterns for each block type (K, B1, A1–A3, B2).  
  - For each detected joint centre and block, crop the image, build a template mask, generate prompt points, and run SAM.  
  - Fuse all crop‑level masks into global label and ring maps in image space.  
  - Project labels back to 3D via the pixel→point mapping, updating `pred` (class ID) and `pred_ring` (instance ID).  
  - Output: `final.csv` (full point cloud with predictions) and `only_label.csv` (GT vs. predicted labels/rings for evaluation).

## Evaluation Metrics

We report three standard segmentation metrics computed from `only_label.csv`:

- **Overall Accuracy (OA)**:  
  - **Definition**: the fraction of correctly classified points over all points.  
  - **Formula**: \( \text{OA} = \frac{\sum_i \mathbf{1}[\text{pred\_labels}_i = \text{gt\_labels}_i]}{N} \).

- **F1 Score (macro / mean F1 over classes)**:  
  - **Per-class F1**: for each class \(c\),  
    \( \text{F1}_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c} \),  
    with \( \text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c} \) and \( \text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c} \).  
  - **Reported F1**: mean of \(\text{F1}_c\) over all semantic classes.

- **Mean Intersection over Union (mIoU)**:  
  - **Per-class IoU**: for each class \(c\),  
    \( \text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c} \).  
  - **Reported mIoU**: mean of \(\text{IoU}_c\) over all semantic classes.