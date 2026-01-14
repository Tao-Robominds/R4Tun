# Issue Summary: LLM Parameter Analysis

## Quick Summary

| Model | Status | Main Issue |
|-------|--------|------------|
| **Original** | ✅ Works (OA: 0.832) | Baseline |
| **Gemini 3** | ⚠️ Poor (OA: 0.449) | Too permissive thresholds |
| **Opus 4.5** | ⚠️ Poor (OA: 0.497) | Over-aggressive dilation |
| **GPT 5.2** | ❌ Crashed/Timeout | **Angle range too wide [5,12]** |
| **Group** | ❌ Crashed/Timeout | **Angle range too wide [5,12]** |

---

## 1. Gemini 3 - Why Performance Dropped

### Key Changes
- `hough_threshold_oblique`: 50 → **35** (-30%)
- `minLineLength_oblique`: 100 → **70** (-30%)
- `maxLineGap_oblique`: 40 → **60** (+50%)
- `angle_range`: ±[6,9] → **±[5,11]** (wider)
- `dilation_iterations`: 1 → **2**

### Problem
Too many false detections due to:
1. Lower threshold (35) accepts weak/noisy lines
2. Shorter min length (70) captures fragmented noise
3. Wider angles (±[5,11]) include near-horizontal clutter

### Result
- **OA: 0.449** (46% drop from 0.832)
- **K-block IoU: 0.048** (90% drop from 0.476)

---

## 2. Opus 4.5 - Why Performance Dropped

### Key Changes
- `morphological_kernel_size`: [3,3] → **[5,5]** (larger)
- `dilation_iterations`: 1 → **2**
- `hough_threshold_oblique`: 50 → **35** (-30%)
- `minLineLength_oblique`: 100 → **60** (-40%)
- `maxLineGap_oblique`: 40 → **70** (+75%)
- `angle_range`: ±[6,9] → **±[5,10]** (wider)

### Problem
Over-aggressive dilation + permissive detection:
1. 5x5 kernel + 2 iterations = very thick edges, connecting unrelated regions
2. Very short min length (60) captures noise
3. Large gap tolerance (70) bridges gaps incorrectly

### Result
- **OA: 0.497** (40% drop from 0.832)
- **A2-block IoU: 0.000** (completely missed this class)

---

## 3. GPT 5.2 - Why It Didn't Finish

### Key Changes (Same as Group)
- `morphological_kernel_size`: [3,3] → **[5,5]**
- `dilation_iterations`: 1 → **2**
- `hough_threshold_oblique`: 50 → **35**
- `minLineLength_oblique`: 100 → **70**
- `maxLineGap_oblique`: 40 → **80** (+100%)
- `angle_range_oblique_positive`: [6,9] → **[5, 12]** ⚠️ **TOO WIDE**
- `angle_range_oblique_negative`: [-9,-6] → **[-12, -5]** ⚠️ **TOO WIDE**

### Critical Issue: Angle Range [5, 12] and [-12, -5]

**Why this causes failure:**

1. **Includes angles too close to horizontal**: The range [5, 12] includes angles very close to 0° (horizontal), which should be filtered out
2. **Combined with low threshold**: `hough_threshold_oblique=35` + `minLineLength_oblique=70` + wide angles = **millions of line detections**
3. **Computational explosion**: 
   - `cv2.HoughLinesP()` returns enormous array
   - Filtering loop (lines 121-133) processes all lines
   - Intersection computation (lines 362-439) becomes intractable
4. **Memory/timeout**: Process likely runs out of memory or times out

### Code Flow
```
Line 96: cv2.HoughLinesP() → Returns HUGE array (millions of lines)
Line 121-133: Filter by angle → Still millions of lines pass
Line 362-439: Compute intersections → Nested loops become O(n²) or worse
→ CRASH/TIMEOUT
```

---

## 4. Group - Why It Didn't Finish

**Identical to GPT 5.2** - same parameters, same failure mode.

---

## Root Causes

### Performance Drop (Gemini 3, Opus 4.5)
1. **Over-relaxed thresholds**: Trying to improve left-side recall introduced too many false positives
2. **Wider angle ranges**: Including angles too close to horizontal captures noise
3. **Excessive dilation**: Connecting unrelated features

### Process Failure (GPT 5.2, Group)
1. **Angle range [5,12] is too wide**: Includes angles dangerously close to 0° (horizontal)
2. **Computational explosion**: Low threshold + short length + wide angles = millions of detections
3. **Processing bottleneck**: Intersection computation can't handle the volume

---

## Fixes Needed

### For GPT 5.2 and Group (Critical)
```json
{
  "angle_range_oblique_positive": [5, 10],  // NOT [5, 12]
  "angle_range_oblique_negative": [-10, -5], // NOT [-12, -5]
  "hough_threshold_oblique": 40,  // Increase from 35
  "morphological_kernel_size": [3, 3],  // Reduce from [5, 5]
  "minLineLength_oblique": 80  // Increase from 60-70
}
```

### For Gemini 3 and Opus 4.5
```json
{
  "angle_range_oblique_positive": [6, 9],  // Narrow back to original
  "angle_range_oblique_negative": [-9, -6],
  "hough_threshold_oblique": 45,  // Increase from 35
  "minLineLength_oblique": 85,  // Increase from 60-70
  "morphological_kernel_size": [3, 3]  // For Opus: reduce from [5, 5]
}
```

---

## Key Insight

**The LLM models correctly identified the problem** (left-side sparsity) but **overcorrected**:
- They relaxed parameters too much
- They widened angle ranges too far
- They didn't account for the computational cost

**The original parameters were actually well-tuned** - the issue might be in the data or preprocessing, not the detection parameters.











