# Will Wraparound-Aware Segmentation Work for All Tunnels?

## Short Answer

**Partially, but with important limitations:**

✅ **Works for:** Tunnels with ground truth segment positions available  
⚠️ **Requires:** `all_segments.csv` with known segment positions  
❌ **Current limitation:** The implementation doesn't actually handle wraparound in the crop function!

---

## Current Implementation Analysis

### What `4-2_sam_wraparound.py` Actually Does

Looking at the code, the current implementation:

1. ✅ **Processes segments individually** - Each segment is handled separately
2. ✅ **Uses GT positions** - Requires `all_segments.csv` with X, Y coordinates
3. ❌ **Does NOT actually handle wraparound in cropping!**

**Critical Issue Found:**

```python
# From crop_image_and_mask_logits() - line 236
def crop_image_and_mask_logits(image, cx, cy, crop_width, crop_height, block, resolution):
    img_height, img_width, _ = image.shape
    x1 = max(cx - crop_width // 2, 0)      # Clamps to left edge
    y1 = max(cy - crop_height // 2, 0)
    x2 = min(cx + crop_width // 2, img_width)  # Clamps to right edge
    y2 = min(cy + crop_height // 2, img_height)
    
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    # ❌ This is just a normal crop! No wraparound handling!
```

**The problem:** If a segment spans the boundary:
- If `cx` is near the right edge, `x1` gets clamped to `cx - crop_width//2`
- The left part (which wraps around) is **still outside the crop**!
- This is the same problem as normal segmentation!

---

## What Would Actually Work

### True Wraparound-Aware Cropping

To actually handle wraparound, the crop function should:

```python
def crop_image_and_mask_logits_wraparound(image, cx, cy, crop_width, crop_height, block, resolution):
    img_height, img_width, _ = image.shape
    
    # Check if crop would span boundary
    x1 = cx - crop_width // 2
    x2 = cx + crop_width // 2
    
    if x1 < 0 or x2 >= img_width:
        # Wraparound case: crop spans boundary
        # Concatenate right edge + left edge
        right_part = image[:, max(0, x1):img_width]
        left_part = image[:, 0:min(img_width, x2)]
        cropped_image = np.concatenate([right_part, left_part], axis=1)
        
        # Adjust prompt_centre to account for wraparound
        if x1 < 0:
            prompt_centre_x = cx + (img_width - x1)
        else:
            prompt_centre_x = cx - img_width
    else:
        # Normal case: no wraparound
        cropped_image = image[int(cy - crop_height//2):int(cy + crop_height//2),
                              int(x1):int(x2)]
        prompt_centre_x = cx - x1
    
    # ... rest of function
```

**This is what's missing from the current implementation!**

---

## Requirements for Universal Solution

### 1. Ground Truth Segment Positions

**Current requirement:**
- `all_segments.csv` with columns: `Ring`, `Block`, `X`, `Y`
- Must have positions for ALL segments

**Availability by tunnel:**
- ✅ **4-1**: Has `all_segments.csv` (found in data/4-1/)
- ✅ **5-1**: Likely has it (7-segment tunnel)
- ❓ **1-4, 2-2, 3-1**: May or may not have it

**Alternative:** Pattern discovery can infer positions (see `4-2_sam_pattern.py`)

### 2. Wraparound Detection

**Need to detect:**
- Which segments span the 0°/360° boundary
- For each segment, whether crop would cross boundary

**Detection method:**
```python
def segment_spans_boundary(cx, crop_width, img_width):
    """Check if segment crop would span image boundary."""
    x1 = cx - crop_width // 2
    x2 = cx + crop_width // 2
    return x1 < 0 or x2 >= img_width
```

### 3. Actual Wraparound Handling

**Must implement:**
- Concatenate left + right edges when crop spans boundary
- Adjust template mask coordinates for wraparound
- Adjust prompt point coordinates for wraparound
- Handle coordinate mapping back to original image

---

## Will It Work for All Tunnels?

### By Tunnel Type

| Tunnel | Segments | Wraparound | GT Available? | Current Solution Works? | Needs Enhancement? |
|--------|----------|------------|---------------|------------------------|---------------------|
| **1-4** | 6 | Moderate (3 segments) | ❓ Maybe | ⚠️ Partial | ✅ Yes - add wraparound crop |
| **2-2** | 6 | Moderate (3 segments) | ❓ Maybe | ⚠️ Partial | ✅ Yes - add wraparound crop |
| **3-1** | 6 | Moderate (2 segments) | ❓ Maybe | ⚠️ Partial | ✅ Yes - add wraparound crop |
| **4-1** | 7 | Severe (ALL segments) | ✅ Yes | ⚠️ Partial | ✅ Yes - add wraparound crop |
| **5-1** | 7 | Severe (ALL segments) | ✅ Yes | ⚠️ Partial | ✅ Yes - add wraparound crop |

### Current Status

**What works:**
- ✅ Individual segment processing (better than row-based)
- ✅ Using GT positions (more accurate than detection-based)
- ✅ Processing all segments (not just K-blocks)

**What's missing:**
- ❌ Actual wraparound handling in crop function
- ❌ Boundary detection logic
- ❌ Coordinate adjustment for wrapped crops

---

## Universal Solution Design

### Enhanced Wraparound-Aware Segmentation

```python
def crop_image_wraparound_aware(image, cx, cy, crop_width, crop_height):
    """
    Crop image with wraparound handling.
    If crop spans 0°/360° boundary, concatenate edges.
    """
    img_height, img_width = image.shape[:2]
    
    x1 = cx - crop_width // 2
    x2 = cx + crop_width // 2
    y1 = max(0, cy - crop_height // 2)
    y2 = min(img_height, cy + crop_height // 2)
    
    # Check if wraparound needed
    if x1 < 0:
        # Crop extends past left edge - wrap to right
        right_part = image[y1:y2, max(0, x1 + img_width):img_width]
        left_part = image[y1:y2, 0:min(img_width, x2)]
        cropped = np.concatenate([right_part, left_part], axis=1)
        offset_x = img_width - max(0, x1 + img_width)  # Offset for coordinate mapping
        
    elif x2 >= img_width:
        # Crop extends past right edge - wrap to left
        right_part = image[y1:y2, max(0, x1):img_width]
        left_part = image[y1:y2, 0:min(img_width, x2 - img_width)]
        cropped = np.concatenate([right_part, left_part], axis=1)
        offset_x = 0  # No offset needed
        
    else:
        # Normal crop - no wraparound
        cropped = image[y1:y2, int(x1):int(x2)]
        offset_x = 0
    
    return cropped, offset_x

def adjust_coordinates_for_wraparound(points, offset_x, img_width):
    """Adjust prompt point coordinates for wraparound crop."""
    adjusted = points.copy()
    if offset_x > 0:
        # Points on right edge need to be shifted
        mask = adjusted[:, 0] < offset_x
        adjusted[mask, 0] += img_width
    return adjusted
```

---

## Recommendations

### For Immediate Use

1. **For tunnels with GT positions (4-1, 5-1):**
   - Use `4-2_sam_wraparound.py` as-is
   - It's better than standard processing (individual segments)
   - But still has the wraparound crop limitation

2. **For tunnels without GT positions (1-4, 2-2, 3-1):**
   - Use pattern discovery first (`agents/detecting/pattern_discovery.py`)
   - Generate `inferred_segments.csv`
   - Then use wraparound-aware processing

### For Complete Solution

**Enhance `4-2_sam_wraparound.py` to:**

1. ✅ **Add wraparound detection:**
   ```python
   def detect_wraparound(cx, crop_width, img_width):
       return (cx - crop_width//2 < 0) or (cx + crop_width//2 >= img_width)
   ```

2. ✅ **Implement wraparound crop:**
   - Concatenate left + right edges
   - Adjust coordinates accordingly

3. ✅ **Update template mask generation:**
   - Account for wrapped coordinates
   - Ensure mask covers both parts

4. ✅ **Update prompt point generation:**
   - Adjust coordinates for wrapped crop
   - Place points on both edges when needed

---

## Conclusion

**Current answer:** The wraparound-aware solution **partially works** but needs enhancement.

**What works:**
- ✅ Individual segment processing
- ✅ Using GT positions
- ✅ Better than standard row-based processing

**What's missing:**
- ❌ Actual wraparound handling in crop function
- ❌ The current implementation still has the same wraparound problem!

**To make it universal:**
1. Add wraparound detection
2. Implement true wraparound crop (concatenate edges)
3. Adjust all coordinates for wrapped crops
4. Support both GT and inferred segment positions

**Bottom line:** The concept is correct, but the implementation needs the actual wraparound crop logic to work properly for all tunnels, especially 7-segment ones where ALL segments span boundaries.
