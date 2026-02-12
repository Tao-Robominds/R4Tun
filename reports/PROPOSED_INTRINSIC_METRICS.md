# Proposed Intrinsic Metrics Design

**Date:** 2026-02-02  
**Purpose:** Design output-based intrinsic metrics for all pipeline stages, enabling quality assessment without ground truth for reflection/rerun decisions.

---

## Design Principles

1. **Computable without GT**: All metrics must be derived from pipeline outputs only
2. **Correlate with mIoU**: Metrics should predict final segmentation quality
3. **Actionable**: Metrics should guide what to fix when quality is low
4. **Stage-specific**: Each stage should have metrics relevant to its function

---

## 1. Preprocessing Metrics (Guardrails)

These have low mIoU impact (+0.1%) but catch catastrophic failures.

### 1.1 Unfolding Stage

| Metric | Formula | Good Range | What It Detects |
|--------|---------|------------|-----------------|
| `pre_theta_coverage` | (θ_max - θ_min) / 2π × 100 | 98-102% | Wraparound issues, incomplete unfolding |
| `pre_theta_gap_max` | max(Δθ between consecutive points) | < 5° | Large gaps in angular coverage |
| `pre_ring_count_match` | 1 if detected == expected | 1.0 | Missing or extra rings |
| `pre_centerline_smoothness` | std(centerline curvature) | < 0.01 | Poor centerline fitting |
| `pre_point_density_cv` | CV of points per ring | < 0.20 | Uneven point distribution |

**Data Source:** `unwrapped.csv`

### 1.2 Denoising Stage

| Metric | Formula | Good Range | What It Detects |
|--------|---------|------------|-----------------|
| `pre_point_retention_ratio` | points_after / points_before × 100 | 85-98% | Over/under denoising |
| `pre_outlier_cluster_count` | number of isolated outlier clusters | < 5 | Noise patterns not removed |
| `pre_boundary_sharpness` | gradient at inner/outer boundary | > 0.5 | Blurred boundaries |

**Data Source:** `denoised.csv` vs `unwrapped.csv`

### 1.3 Enhancing Stage

| Metric | Formula | Good Range | What It Detects |
|--------|---------|------------|-----------------|
| `pre_interpolation_coverage` | valid_pixels / total_pixels × 100 | > 95% | Sparse depth map |
| `pre_depth_range_ratio` | (max-min) / expected_range | 0.8-1.2 | Depth map scaling issues |
| `pre_depth_smoothness` | mean(local variance) | < 0.01 | Noisy depth map |
| `pre_hole_count` | number of NaN regions > 10 pixels | < 3 | Unfilled gaps |

**Data Source:** `depth_map_outlier.npy`

---

## 2. Detection Metrics

### 2.1 Common Detection Metrics (Both Simple and Complex)

| Metric | Formula | Good Range | What It Detects |
|--------|---------|------------|-----------------|
| `det_k_count` | number of detected K-positions | expected ± 2 | Wrong detection count |
| `det_k_count_match` | 1 if count == expected | 1.0 | Exact count match |
| `det_x_spacing_cv` | CV of X-position differences | < 0.15 (simple), < 0.60 (complex) | Irregular spacing |
| `det_y_range` | max(Y) - min(Y) | 200-1500 px | Y-position spread |
| `det_y_std` | std(Y positions) | varies | Y-position consistency |

**Data Source:** `detected.csv`

### 2.2 Simple Pattern Detection Metrics (1-4, 2-2, 3-1)

| Metric | Formula | Good Range | What It Detects |
|--------|---------|------------|-----------------|
| `det_midpoint_ratio` | midpoint_detections / total_detections | > 0.60 | Detection method quality |
| `det_real_detection_ratio` | non_fallback / total | > 0.70 | Actual vs assumed detections |
| `det_assume_default_ratio` | fallback / total | < 0.30 | Reliance on defaults |
| `det_line_confidence_mean` | mean(Hough vote counts) | > 50 | Line detection confidence |
| `det_intersection_quality` | % intersections with both lines | > 0.80 | Line intersection quality |

**Data Source:** `detected.csv`, `detection_results.json`

### 2.3 Complex Pattern Detection Metrics (4-1, 5-1) - NEW

| Metric | Formula | Good Range | What It Detects |
|--------|---------|------------|-----------------|
| `det_cluster_compactness` | mean(cluster_std) / mean(cluster_distance) | < 0.20 | Cluster quality |
| `det_cluster_separation` | min(inter_cluster_distance) / mean(cluster_size) | > 2.0 | Cluster distinctness |
| `det_stagger_pattern_score` | correlation with expected stagger pattern | > 0.70 | Pattern match |
| `det_vertical_alignment` | max(X deviation within ring) | < 30 px | Vertical K-block alignment |
| `det_ring_completeness` | rings_with_detection / total_rings | > 0.90 | Coverage across rings |
| `det_confidence_intersection` | mean confidence of intersection clusters | > 0.70 | Detection confidence |

**Data Source:** `detected.csv`, compute from cluster analysis

---

## 3. SAM Metrics

### 3.1 Common SAM Metrics (Both Simple and Complex)

| Metric | Formula | Good Range | What It Detects |
|--------|---------|------------|-----------------|
| `sam_mask_fill_rate` | non_background_pixels / total_pixels | 0.30-0.90 | Over/under segmentation |
| `sam_segment_count` | unique segments (excluding background) | expected ± 1 | Segment count |
| `sam_segment_count_match` | 1 if count >= expected - 1 | 1.0 | Approximate count match |
| `sam_prompt_count` | prompts sent to SAM | matches det_k_count | Prompt count |

**Data Source:** `final.csv`

### 3.2 Simple Pattern SAM Metrics (1-4, 2-2, 3-1)

| Metric | Formula | Good Range | What It Detects |
|--------|---------|------------|-----------------|
| `sam_k_block_coverage` | K-block pixels / expected_K_area | 0.80-1.20 | K-block size |
| `sam_ab_block_balance` | std(A/B block sizes) / mean | < 0.15 | Block size uniformity |
| `sam_segment_size_cv` | CV of segment sizes | < 0.20 | Size consistency |

**Data Source:** `final.csv`, compute from segment analysis

### 3.3 Complex Pattern SAM Metrics (4-1, 5-1) - NEW

These are the key metrics needed to assess complex pattern quality without GT.

#### 3.3.1 Geometry Metrics

| Metric | Formula | Good Range | What It Detects |
|--------|---------|------------|-----------------|
| `sam_segment_width_cv` | CV of segment widths across rings | < 0.10 | Width inconsistency |
| `sam_segment_height_cv` | CV of segment heights | < 0.15 | Height inconsistency |
| `sam_aspect_ratio_mean` | mean(width/height) per segment type | 0.3-0.5 | Shape correctness |
| `sam_aspect_ratio_cv` | CV of aspect ratios | < 0.10 | Shape consistency |

#### 3.3.2 Coverage Metrics

| Metric | Formula | Good Range | What It Detects |
|--------|---------|------------|-----------------|
| `sam_overlap_ratio` | pixels_with_multiple_segments / total | < 0.05 | Over-segmentation |
| `sam_gap_ratio` | unassigned_pixels_in_roi / roi_area | < 0.10 | Under-segmentation |
| `sam_coverage_uniformity` | 1 - CV(segment_areas) | > 0.80 | Uneven coverage |
| `sam_ring_coverage_cv` | CV of coverage per ring | < 0.15 | Ring-to-ring consistency |

#### 3.3.3 Boundary Metrics

| Metric | Formula | Good Range | What It Detects |
|--------|---------|------------|-----------------|
| `sam_boundary_straightness` | 1 - mean(boundary_curvature) | > 0.85 | Jagged boundaries |
| `sam_boundary_angle_match` | correlation(boundary_angle, expected_angle) | > 0.80 | Angle alignment |
| `sam_boundary_gradient` | mean(gradient_at_boundary) | > 0.70 | Boundary sharpness |
| `sam_inter_segment_gap` | mean(gap_between_adjacent_segments) | 0-10 px | Gap consistency |

#### 3.3.4 Pattern Metrics

| Metric | Formula | Good Range | What It Detects |
|--------|---------|------------|-----------------|
| `sam_stagger_alignment` | correlation with expected stagger pattern | > 0.80 | Stagger correctness |
| `sam_k_block_position_error` | mean(|detected_K - expected_K|) | < 20 px | K-block positioning |
| `sam_ring_consistency` | 1 - CV(per_ring_metrics) | > 0.85 | Cross-ring consistency |
| `sam_symmetry_score` | similarity(left_half, right_half) | > 0.80 | Pattern symmetry |

---

## 4. Implementation Priority

### Phase 1: High Priority (Implement First)

| Metric | Stage | Why Priority |
|--------|-------|--------------|
| `sam_overlap_ratio` | SAM | Direct indicator of over-segmentation |
| `sam_gap_ratio` | SAM | Direct indicator of under-segmentation |
| `sam_segment_width_cv` | SAM | Geometry consistency (key for complex) |
| `det_cluster_compactness` | Detection | Quality of complex detection |
| `det_ring_completeness` | Detection | Coverage indicator |

### Phase 2: Medium Priority

| Metric | Stage | Why Priority |
|--------|-------|--------------|
| `sam_boundary_straightness` | SAM | Boundary quality |
| `sam_stagger_alignment` | SAM | Pattern correctness |
| `det_stagger_pattern_score` | Detection | Pattern match |
| `pre_hole_count` | Preprocessing | Gap detection |

### Phase 3: Lower Priority

- Remaining preprocessing metrics
- Confidence-based metrics
- Symmetry metrics

---

## 5. Expected Correlation Targets

For these metrics to be useful, they should correlate with mIoU:

| Metric Category | Target Spearman | Minimum Useful |
|-----------------|-----------------|----------------|
| SAM geometry (complex) | > 0.70 | > 0.50 |
| SAM coverage | > 0.60 | > 0.40 |
| Detection quality | > 0.50 | > 0.30 |
| Preprocessing | > 0.30 | > 0.20 |

---

## 6. Validation Plan

### Step 1: Implement Metrics

Add to `bo4tun/intrinsic_metrics.py`:
- `compute_complex_detection_metrics()`
- `compute_complex_sam_metrics()`

### Step 2: Collect Training Data

For each metric:
1. Run pipeline on 50+ configurations
2. Compute metric value
3. Record GT mIoU

### Step 3: Correlation Analysis

```python
for metric in new_metrics:
    correlation = spearmanr(metric_values, miou_values)
    if correlation > 0.50:
        print(f"{metric}: USEFUL (r={correlation:.2f})")
    else:
        print(f"{metric}: NOT USEFUL (r={correlation:.2f})")
```

### Step 4: Build Predictor

Train Ridge regression with useful metrics:
```python
useful_metrics = [m for m in metrics if correlation[m] > 0.50]
model = Ridge().fit(X[useful_metrics], y_miou)
```

### Step 5: Validate Reflection/Rerun

Test the full loop:
1. Run pipeline
2. Compute intrinsic metrics
3. Predict mIoU
4. If low → adjust params → rerun
5. Verify improvement

---

## 7. Reflection/Rerun Decision Tree

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPLEX PATTERN REFLECTION FLOW                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   After SAM completes:                                              │
│                                                                     │
│   1. Compute intrinsic metrics:                                     │
│      - sam_overlap_ratio                                            │
│      - sam_gap_ratio                                                │
│      - sam_segment_width_cv                                         │
│      - sam_boundary_straightness                                    │
│                                                                     │
│   2. Check guardrails:                                              │
│      ┌─────────────────────────────────────────────────────────┐    │
│      │ sam_overlap_ratio > 0.10?                               │    │
│      │   → RERUN with smaller segment_width                    │    │
│      │                                                         │    │
│      │ sam_gap_ratio > 0.15?                                   │    │
│      │   → RERUN with larger segment_width                     │    │
│      │                                                         │    │
│      │ sam_segment_width_cv > 0.15?                            │    │
│      │   → RERUN with adjusted k_height                        │    │
│      │                                                         │    │
│      │ sam_boundary_straightness < 0.70?                       │    │
│      │   → RERUN with adjusted angle_deg                       │    │
│      └─────────────────────────────────────────────────────────┘    │
│                                                                     │
│   3. If all guardrails pass:                                        │
│      - Predict mIoU from intrinsic metrics                          │
│      - If predicted_mIoU < 0.35 → FLAG for review                   │
│      - If predicted_mIoU >= 0.35 → ACCEPT                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Next Steps

1. **Implement Phase 1 metrics** in `intrinsic_metrics.py`
2. **Collect correlation data** by running 50+ complex pattern configs
3. **Validate correlations** with GT mIoU
4. **Build predictor** using validated metrics
5. **Test reflection loop** on new tunnels

---

*Document created: 2026-02-02*
