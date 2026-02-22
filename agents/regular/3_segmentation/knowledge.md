# SAM Segmentation Stage - Parameter Tuning Guide

## Overview

SAM (Segment Anything Model) segmentation is the **highest impact stage** for mIoU improvement (+7.4% from BO experiments combined with detection). This stage uses detected K positions to generate segment masks for each block type.

## Parameter Classification

### Inherited from Preprocessing (NOT tuned here)

These values are automatically read from the preprocessing stage:

| Parameter | Source | Description |
|-----------|--------|-------------|
| `resolution` | `depth_map_resolution` | Pixel-to-mm conversion factor |
| `tunnel_diameter` | preprocessing params | Used to calculate K/AB heights |
| `K_height` | calculated | π × tunnel_diameter × 1000 / 16 |
| `AB_height` | calculated | 3 × K_height |

### HIGH Sensitivity Parameters

| Parameter | BO Range | Notes |
|-----------|----------|-------|
| `segment_width` | 1100-1250 | Width of segment crop window in mm. Affects how much context SAM sees |
| `angle_deg` | 6.0-8.5 | Oblique angle for K-block trapezoid masks. Must match actual joint angle |
| `k_mask_width` | 580-680 | K-block template mask width in mm |
| `k_mask_height_pos` | 550-680 | K-block template mask height (positive slope side) |
| `k_mask_height_neg` | 400-580 | K-block template mask height (negative slope side) |

### MEDIUM Sensitivity Parameters

| Parameter | BO Range | Notes |
|-----------|----------|-------|
| `ab_mask_width` | 580-700 | A/B block template mask width in mm |
| `ab_mask_height` | 1500-1700 | A/B block template mask height in mm |
| `padding` | 80-180 | Extra padding around crop windows in mm |
| `crop_margin` | 30-60 | Margin for cropping in mm |
| `min_quality_threshold` | 0.25-0.55 | Minimum detection quality to include ring. Lower = more rings, potentially noisier |

### FIXED Parameters (Not tuned)

These parameters are too numerous/interdependent for individual tuning - use fixed defaults:

| Category | Parameters | Notes |
|----------|------------|-------|
| K-block prompt points | outer_ring, middle_ring, inner_ring, center_ring, spacing | ~47 prompt points per K-block |
| AB-block prompt points | outer_ring, middle_ring, vertical_levels (8 levels) | ~107 prompt points per AB-block |
| Mask epsilon | 0.001 | Logit computation constant |

## Tuning Strategy

1. **Start with segment geometry** - `segment_width` and `angle_deg` are most critical
2. **Adjust template masks** - K-block masks affect K detection quality; AB masks affect A/B blocks
3. **Tune processing params** - `padding` and `crop_margin` affect context visible to SAM
4. **Quality threshold last** - `min_quality_threshold` is a trade-off between coverage and accuracy

## Why Runs Fail (mIoU < 0.5)

Analysis of failed BO runs shows:

1. **Wrong angle_deg**
   - Causes: Template masks don't align with actual oblique K-blocks
   - Symptoms: K-block IoU drops significantly

2. **Template masks too small**
   - Causes: SAM doesn't see enough of the block structure
   - Symptoms: Under-segmentation, blocks partially missing

3. **Template masks too large**
   - Causes: Overlap between adjacent blocks
   - Symptoms: Block boundaries bleed into neighbors

4. **min_quality_threshold too high**
   - Causes: Good rings excluded due to detection quality score
   - Symptoms: Entire rings missing from segmentation

## Cross-Stage Dependencies

```
Detection outputs used by SAM:
├── detected.csv → K positions (X, Y) for each ring
├── detected.csv quality → affects ring inclusion via min_quality_threshold
└── Ring count → determines segment processing order

Preprocessing outputs used by SAM:
├── depth_map.png → input image for SAM
├── enhanced.csv → point cloud for final projection
├── pixel_to_point.pkl → mapping for back-projection
├── depth_map_resolution → pixel scale
└── tunnel_diameter → K/AB height calculation
```

## Comparison with Other Stages

| Stage | mIoU Impact | Tuning Priority |
|-------|-------------|-----------------|
| Detection | +6.3% | 1st |
| **SAM** | +7.4% (combined) | **2nd** |
| Preprocessing | +0.1% | 3rd |

SAM and Detection together contribute the vast majority of mIoU gains. Focus tuning effort here.

## BO Results Summary

| Tunnel | Best mIoU | Key Param Differences |
|--------|-----------|----------------------|
| 1-4 | 74.8% | angle=6.5°, segment_width=1150, quality_threshold=0.5 |
| 2-2 | 76.0% | angle=7.05°, segment_width=1209, quality_threshold=0.39 |

Note: Different tunnels may need different angle_deg values due to K-block joint angle variations.
