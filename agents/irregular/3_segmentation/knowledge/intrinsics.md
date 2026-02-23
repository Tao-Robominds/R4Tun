# Segmentation Intrinsic Metrics — Irregular (Complex Staggered) Tunnels

## Overview

Six intrinsic metrics evaluate segmentation quality **without ground truth**, designed for geometric segmentation (GT-free approach).

| # | Metric | What it checks | Guardrail |
|---|--------|---------------|-----------|
| 1 | `seg_segment_type_completeness` | All 7 block types present? | Must be True |
| 2 | `seg_ring_completeness_avg` | Avg fraction of 7 types/ring | >= 0.80 |
| 3 | `seg_mask_coverage_pct` | Segmented / mappable points | [45%, 85%] |
| 4 | `seg_k_size_ratio` | K-block proportion of segmented area | [2%, 12%] |
| 5 | `seg_groove_score` | Boundary-groove alignment | >= 15.0 |
| 6 | `seg_block_size_variance_ratio` | max/min block area per ring | [3.0, 20.0] |

## Metric Details

### 1. `seg_segment_type_completeness`
- **Formula:** Check pred values 1–7 all present in `final.csv`
- **Threshold:** Must be True
- **Failure mode:** Missing block type → segmentation missed entire category

### 2. `seg_ring_completeness_avg`
- **Formula:** For each ring, count how many of the 7 types are present, average across rings
- **Threshold:** >= 0.80 (relaxed from regular 0.85 for wrap-around effects)
- **Failure mode:** Low value → some rings have missing segments

### 3. `seg_mask_coverage_pct`
- **Formula:** (points with pred in 1–7) / (points with pred != 8) × 100
- **Threshold:** [45%, 85%]
- **Failure mode:** Too low → many points unsegmented; too high → possible over-segmentation

### 4. `seg_k_size_ratio`
- **Formula:** (pred==1 count) / (pred in 1–7 count) × 100
- **Threshold:** [2%, 12%]
- **Failure mode:** K blocks physically small in 7-segment tunnels; deviation signals size miscalculation

### 5. `seg_groove_score`
- **Formula:** Mean gradient magnitude at label boundaries (Sobel on depth map)
- **Threshold:** >= 15.0
- **Note:** This is the key metric distinguishing geometric from SAM segmentation:
  - Geometric segmentation (GT-free) aligns to physical grooves: 5-1=53.53, 4-1=25.33
  - SAM segmentation (GT-assisted): 5-1=6.61, 4-1=4.55
- **Failure mode:** Low score → boundaries not at physical grooves → poor physical accuracy

### 6. `seg_block_size_variance_ratio`
- **Formula:** For each ring, max(block_point_count) / min(block_point_count), averaged
- **Threshold:** [3.0, 20.0]
- **Note:** Complex staggered tunnels have inherently high size variation (K is small, A blocks large)
- **Failure mode:** Extremely high ratio → one block type grossly oversized

## Known-Good Values

| Metric | 5-1 (SAM/GT) | 4-1 (SAM/GT) | 5-1 (Geo/GTfree) | 4-1 (Geo/GTfree) |
|--------|-------------|-------------|------------------|------------------|
| type_completeness | True | True | True | True |
| ring_completeness_avg | 1.00 | 0.86 | 0.735 | 0.778 |
| mask_coverage_pct | 85.0% | 83.2% | 22.4% | 15.0% |
| k_size_ratio | 9.1% | 2.4% | 7.0% | 9.1% |
| groove_score | 6.61 | 4.55 | 6.08 | 5.61 |
| block_size_variance_ratio | 5.7 | 15.7 | 7.4 | 6.3 |
| **mIoU** | **0.700** | **0.864** | **0.099** | **0.091** |

**Key Finding:** The GT-free template+geometric pipeline achieves only ~0.1 mIoU vs 0.7–0.86 with GT-assisted SAM. The massive gap is primarily due to template expansion placing blocks at regular geometric positions that don't match actual physical positions. The intrinsic metrics correctly flag this: `mask_coverage_pct` is very low (15–22%), and `groove_score` doesn't benefit from geometric segmentation because the underlying block positions are wrong.

**Implication:** For irregular tunnels, improving the GT-free pipeline requires better block position estimation (not just template expansion), or a fundamentally different approach to segmentation that doesn't rely on accurate per-block centroids.

## Implementation

- **Script:** `agents/irregular/3_segmentation/scripts/extract_intrinsics.py`
- **Output:** `data/wrap/{tunnel_id}/segmentation_intrinsics.json`
