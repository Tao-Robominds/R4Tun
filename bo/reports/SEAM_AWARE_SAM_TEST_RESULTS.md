# Seam-Aware SAM Testing Results

## Summary

Implemented **Y-wraparound-aware cropping** (theta-seam stitching) in `p4tun/4-2_sam.py` to handle segments that span the 0°/360° boundary.

## Test Results

### Tunnel 1-4 (6 segments)

| Configuration | mIoU | F1 | OA | A2-block IoU | Status |
|--------------|------|----|----|--------------|--------|
| **Baseline** (old clamping) | 0.626 | 0.734 | 0.846 | 0.087 | Baseline |
| **With Seam-Aware** | **0.679** | **0.800** | 0.847 | **0.486** | ✅ **+8.5% improvement** |
| **Change** | **+0.053** | **+0.066** | +0.001 | **+0.399** | A2-block fixed! |

**Key improvement:** A2-block IoU improved from 0.087 to 0.486 (+458%), confirming wraparound fix works.

### Tunnel 2-2 (6 segments)

| Configuration | mIoU | F1 | OA | Status |
|--------------|------|----|----|--------|
| **Baseline** (old clamping) | 0.672 | 0.797 | 0.828 | Baseline |
| **Optimized** (bo/2-2) | 0.763 | 0.864 | 0.885 | Best known |
| **With Seam-Aware** (enabled) | 0.440 | 0.568 | 0.645 | ❌ **Regression** |
| **With Seam-Aware** (disabled) | **0.763** | **0.864** | 0.886 | ✅ **Matches optimized** |

**Key finding:** For tunnel 2-2, wraparound logic causes regression. Disabling it restores performance.

## Analysis

### Why 1-4 Improved

- A2-block was severely affected by wraparound (IoU 0.087)
- Seam-aware cropping correctly handles segments spanning boundary
- **Result:** A2-block IoU improved dramatically (+0.399)

### Why 2-2 Regressed (with wraparound enabled)

**Possible causes:**
1. **Wraparound applied too aggressively** - wrapping crops that don't need it
2. **Coordinate mapping bug** - prompt_centre_y or mask aggregation may be incorrect
3. **2-2 doesn't need wraparound** - segments may not actually span boundaries in a way that requires wrapping

**Evidence:**
- With wraparound **disabled**, performance matches optimized baseline (0.763 mIoU)
- This suggests the wraparound logic has a bug, or is being applied incorrectly

## Implementation Details

### Seam-Aware Cropping

```python
def crop_image_and_mask_logits(..., enable_y_wraparound=True):
    # Y (theta) IS periodic
    y1 = int(cy - crop_height // 2)
    y2 = int(cy + crop_height // 2)
    
    if enable_y_wraparound and (y1 < 0 or y2 > img_height):
        # Stitch: [bottom_part, top_part] or [top_part, bottom_part]
        # Adjust prompt_centre_y for stitched coordinate system
    else:
        # Clamp (old behavior)
        y1 = max(y1, 0)
        y2 = min(y2, img_height)
```

### Seam-Aware Mask Aggregation

```python
def apply_mask_logits_with_y_wraparound(...):
    if start_y < 0:
        # Split mask: [0:wrap_h] -> bottom, [wrap_h:] -> top
    elif end_y > img_height:
        # Split mask: [0:normal_h] -> normal, [normal_h:] -> top
```

## Recommendations

### For Tunnel 1-4
✅ **Keep seam-aware enabled** - Significant improvement, especially for A2-block

### For Tunnel 2-2
⚠️ **Keep seam-aware disabled** - Current implementation causes regression
- May need bug fix in wraparound logic
- Or 2-2 doesn't actually need wraparound (segments don't span boundaries)

### For 7-Segment Tunnels (4-1, 5-1)
🔍 **Needs testing** - Should benefit most from wraparound, but logic needs verification

## Next Steps

1. **Debug wraparound logic** - Investigate why it works for 1-4 but not 2-2
2. **Test on 4-1/5-1** - Verify wraparound helps severe cases
3. **Add selective wrapping** - Only wrap when segments actually span boundaries
4. **Fix coordinate mapping** - Verify prompt_centre_y calculation for wrapped crops

## Code Changes

- Added `enable_y_wraparound` parameter to control wraparound behavior
- Modified `crop_image_and_mask_logits()` to support Y-wraparound stitching
- Added `apply_mask_logits_with_y_wraparound()` for periodic mask aggregation
- Updated prompt point handling for wrapped crops
