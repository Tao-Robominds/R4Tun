# Parameter Analysis: Original vs LLM Models

## Performance Summary

| Model | OA | F1 | mIoU | Status |
|-------|----|----|------|--------|
| **Original (sample)** | **0.832** | **0.772** | **0.645** | ✅ Working |
| **gemini_3** | 0.449 | 0.327 | 0.230 | ⚠️ Completed but poor performance |
| **opus_4.5** | 0.497 | 0.359 | 0.256 | ⚠️ Completed but poor performance |
| **gpt_5.2** | - | - | - | ❌ Did not finish |
| **group** | - | - | - | ❌ Did not finish |

---

## Parameter Comparison

### Original (Working) Parameters
```json
{
  "binary_threshold": 127,
  "morphological_kernel_size": [3, 3],
  "dilation_iterations": 1,
  "hough_threshold_oblique": 50,
  "minLineLength_oblique": 100,
  "maxLineGap_oblique": 40,
  "hough_threshold_horizontal": 50,
  "minLineLength_horizontal": 100,
  "maxLineGap_horizontal": 10,
  "hough_threshold_vertical": 500,
  "angle_range_oblique_positive": [6, 9],
  "angle_range_oblique_negative": [-9, -6],
  "merge_distance": 3,
  "ring_spacing_constant": 1.2,
  "resolution": 0.005
}
```

---

## 1. Gemini 3 Analysis

### Parameter Changes
| Parameter | Original | Gemini 3 | Change | Impact |
|-----------|----------|----------|--------|--------|
| `dilation_iterations` | 1 | **2** | +100% | More connectivity, but more noise |
| `hough_threshold_oblique` | 50 | **35** | -30% | Lower threshold = more false positives |
| `minLineLength_oblique` | 100 | **70** | -30% | Shorter segments = more noise |
| `maxLineGap_oblique` | 40 | **60** | +50% | Larger gaps = more false connections |
| `angle_range_oblique_positive` | [6, 9] | **[5, 11]** | Wider | Captures more angles, but includes noise |
| `angle_range_oblique_negative` | [-9, -6] | **[-11, -5]** | Wider | Same issue |

### Why Performance Dropped
1. **Too permissive thresholds**: Lowering `hough_threshold_oblique` from 50 to 35 allows weak/noisy detections
2. **Shorter line segments**: Reducing `minLineLength_oblique` from 100 to 70 captures fragmented noise
3. **Wider angle ranges**: Expanding from ±[6,9] to ±[5,11] includes near-horizontal noise that shouldn't be detected
4. **More dilation**: 2 iterations instead of 1 creates thicker edges, connecting unrelated features

### Performance Impact
- **OA dropped 46%** (0.832 → 0.449): Too many false detections
- **mIoU dropped 64%** (0.645 → 0.230): Poor segmentation quality
- **K-block IoU**: 0.476 → 0.048 (90% drop) - severe misclassification

---

## 2. Opus 4.5 Analysis

### Parameter Changes
| Parameter | Original | Opus 4.5 | Change | Impact |
|-----------|----------|----------|--------|--------|
| `morphological_kernel_size` | [3, 3] | **[5, 5]** | Larger | Much more aggressive dilation |
| `dilation_iterations` | 1 | **2** | +100% | Combined with 5x5 = very thick edges |
| `hough_threshold_oblique` | 50 | **35** | -30% | More false positives |
| `minLineLength_oblique` | 100 | **60** | -40% | Very short segments = noise |
| `maxLineGap_oblique` | 40 | **70** | +75% | Large gaps = false connections |
| `angle_range_oblique_positive` | [6, 9] | **[5, 10]** | Wider | Includes more noise |
| `angle_range_oblique_negative` | [-9, -6] | **[-10, -5]** | Wider | Same issue |

### Why Performance Dropped
1. **Aggressive dilation**: 5x5 kernel + 2 iterations creates very thick edges, connecting unrelated regions
2. **Very permissive detection**: `minLineLength_oblique=60` and `hough_threshold_oblique=35` capture too much noise
3. **Large gap tolerance**: `maxLineGap_oblique=70` bridges gaps that shouldn't be connected
4. **Wider angles**: ±[5,10] includes angles too close to horizontal, causing false detections

### Performance Impact
- **OA dropped 40%** (0.832 → 0.497): Better than Gemini but still poor
- **mIoU dropped 60%** (0.645 → 0.256): Severe quality degradation
- **A2-block IoU**: 0.344 → 0.000 (100% failure) - completely missed this class

---

## 3. GPT 5.2 Analysis (Did Not Finish)

### Parameter Changes
| Parameter | Original | GPT 5.2 | Change | Impact |
|-----------|----------|---------|--------|--------|
| `morphological_kernel_size` | [3, 3] | **[5, 5]** | Larger | Very aggressive dilation |
| `dilation_iterations` | 1 | **2** | +100% | Combined = extremely thick edges |
| `hough_threshold_oblique` | 50 | **35** | -30% | More false positives |
| `minLineLength_oblique` | 100 | **70** | -30% | Shorter segments |
| `maxLineGap_oblique` | 40 | **80** | +100% | Very large gap tolerance |
| `hough_threshold_horizontal` | 50 | **60** | +20% | Stricter horizontal (good) |
| `minLineLength_horizontal` | 100 | **110** | +10% | Stricter horizontal (good) |
| `hough_threshold_vertical` | 500 | **550** | +10% | Slightly stricter vertical |
| `angle_range_oblique_positive` | [6, 9] | **[5, 12]** | **Much wider** | ⚠️ **PROBLEM** |
| `angle_range_oblique_negative` | [-9, -6] | **[-12, -5]** | **Much wider** | ⚠️ **PROBLEM** |

### Why It Didn't Finish

**CRITICAL ISSUE: Angle Range Too Wide**

The angle ranges `[5, 12]` and `[-12, -5]` are **dangerously close to horizontal**:
- Original: ±[6,9] degrees (3-degree window, safely away from horizontal)
- GPT: ±[5,12] degrees (7-degree window, includes angles very close to 0°)

**Potential Failure Modes:**
1. **Infinite loop or excessive computation**: With such wide angles and permissive thresholds, the Hough transform may detect an enormous number of lines
2. **Memory overflow**: Too many line detections could exhaust memory
3. **Processing timeout**: The intersection computation (lines 362-439) may take too long with excessive detections
4. **Division by zero or NaN errors**: With too many false detections, the midpoint calculations or distance computations may fail

**Code Risk Points:**
- Line 96: `cv2.HoughLinesP()` with low threshold (35) and wide angles could return millions of lines
- Lines 362-439: Intersection computation loops through all detected lines - could be extremely slow
- Line 406: `check_distance_pattern()` with tolerance=50 on many points could be computationally expensive

---

## 4. Group Analysis (Did Not Finish)

### Parameter Changes
| Parameter | Original | Group | Change | Impact |
|-----------|----------|-------|--------|--------|
| `morphological_kernel_size` | [3, 3] | **[5, 5]** | Larger | Very aggressive dilation |
| `dilation_iterations` | 1 | **2** | +100% | Combined = extremely thick edges |
| `hough_threshold_oblique` | 50 | **35** | -30% | More false positives |
| `minLineLength_oblique` | 100 | **60** | -40% | Very short segments |
| `maxLineGap_oblique` | 40 | **80** | +100% | Very large gap tolerance |
| `hough_threshold_horizontal` | 50 | **60** | +20% | Stricter horizontal |
| `minLineLength_horizontal` | 100 | **110** | +10% | Stricter horizontal |
| `angle_range_oblique_positive` | [6, 9] | **[5, 12]** | **Much wider** | ⚠️ **PROBLEM** |
| `angle_range_oblique_negative` | [-9, -6] | **[-12, -5]** | **Much wider** | ⚠️ **PROBLEM** |

### Why It Didn't Finish

**Same critical issue as GPT 5.2**: The angle ranges `[5, 12]` and `[-12, -5]` are too wide and include angles dangerously close to horizontal (0°).

The group parameters are essentially identical to GPT 5.2, so it suffers from the same computational explosion problem.

---

## Root Cause Summary

### Why Performance Dropped (Gemini 3, Opus 4.5)

1. **Over-aggressive relaxation**: All models lowered thresholds and minimum lengths too much
2. **Wider angle ranges**: Including angles too close to horizontal captures noise
3. **Excessive dilation**: Larger kernels and more iterations connect unrelated features
4. **Trade-off failure**: The models tried to improve left-side recall but introduced too many false positives on the right side

### Why GPT 5.2 and Group Didn't Finish

1. **Angle range too wide**: `[5, 12]` and `[-12, -5]` include angles very close to 0° (horizontal)
2. **Computational explosion**: Combined with low thresholds (35) and short min lengths (60-70), this creates millions of false detections
3. **Processing bottleneck**: The intersection computation loops become intractable with excessive line detections
4. **Memory/timeout**: Likely runs out of memory or takes too long to complete

---

## Recommendations

### For Working Models (Gemini 3, Opus 4.5)
1. **Narrow angle ranges**: Keep closer to original ±[6,9] or at most ±[5,10]
2. **Increase thresholds**: Raise `hough_threshold_oblique` to at least 40-45
3. **Increase min length**: Raise `minLineLength_oblique` to at least 80-90
4. **Reduce dilation**: Use 3x3 kernel with 1-2 iterations, not 5x5

### For Failed Models (GPT 5.2, Group)
1. **CRITICAL: Narrow angle ranges**: Must be ±[6,9] or ±[5,10] at most - never ±[5,12]
2. **Increase thresholds**: `hough_threshold_oblique` should be at least 40
3. **Reduce dilation**: Use 3x3 kernel, not 5x5
4. **Add validation**: Check if angle ranges are too wide before processing

### General Guidelines
- **Angle ranges**: Should stay within ±[5,10] degrees maximum
- **Hough thresholds**: Should be ≥ 40 for oblique lines
- **Min line lengths**: Should be ≥ 80 pixels
- **Dilation**: 3x3 kernel with 1-2 iterations is sufficient













