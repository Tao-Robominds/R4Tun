# Preprocessing Bayesian Optimization Report — Multi-Tunnel

**Date**: 2026-02-12  
**Branch**: `bayesian`  
**Pipelines**: `agents/simple_staggered`, `agents/continuous`, `agents/complex_staggered`  
**Tunnels**: 1-4, 2-2, 3-1, 4-1, 5-1

---

## 1. Session Overview

This report covers the entire workflow performed across multiple chat sessions, including
architectural refactoring, pipeline validation, and Bayesian Optimization of
preprocessing parameters for all 5 tunnels (1-4, 2-2, 3-1, 4-1, 5-1).

### Timeline

| Step | Action | Outcome |
|------|--------|---------|
| 1 | Created `bayesian` branch, committed as "init" | Branch ready |
| 2 | Ran full pipeline (preprocess + detect + SAM) on 1-4 with default params | OA 0.645, mIoU 0.333 |
| 3 | Architectural refactoring: moved enhancing from preprocessing to detection | Pipeline restructured |
| 4 | Created backup of `agents/simple_staggered/` | `agents/simple_staggered_backup/` (later removed) |
| 5 | Validated restructured pipeline end-to-end on 1-4 | OA 0.645, mIoU 0.333, F1 0.448 |
| 6 | Fixed bug in `save_best_params()` (stale `DEFAULT_TARGET_DISTANCES` reference) | NameError prevented |
| 7 | Cleaned stale logs (old 6D search space) | Kept trial 035 as reference |
| 8 | Ran preprocessing BO (20 calls, 4D search space) | Best F2 = 0.9811 (trial 015) |
| 9 | Set parameters to best BO result | `parameters_preprocessing.json` updated |

---

## 2. Architectural Refactoring

### Before (Old Pipeline)

```
Preprocessing (1_preprocessing.py)
├── Stage 1: Unfolding
├── Stage 2: Denoising
└── Stage 3: Enhancing (curvature, upsampling, depth map generation)

Detection (2_detection.py)
├── Line detection on depth_map_outlier.npy
├── Ring center computation
└── K-position calculation

SAM Segmentation (3_sam.py)
└── Template-based SAM segmentation
```

**Problem**: Enhancing parameters (`target_distances`, `curvature_neighbors`,
`depth_map_resolution`, `interpolation_window`) did not affect the preprocessing
ground truth metric (F2 score on denoised.csv), yet they were bundled in
preprocessing. This made BO inefficient — tuning parameters that had no effect
on the objective.

### After (New Pipeline)

```
Preprocessing (1_preprocessing.py)
├── Stage 1: Unfolding
└── Stage 2: Denoising
    → Outputs: denoised.csv, ring_count.txt
    → Ground truth: Retention F2 score

Detection (2_detection.py)
├── Step 0: Load denoised.csv
├── Step 1: Enhancing (moved here)
├── Step 2: Line detection
├── Step 3: Ring center computation
└── Step 4: K-position calculation
    → Outputs: detected.csv, detected_lines.png, depth_map.png, etc.
    → Ground truth: K-position accuracy

SAM Segmentation (3_sam.py)
└── Template-based SAM segmentation (unchanged)
    → Ground truth: mIoU
```

### Parameter Ownership After Refactoring

| Parameter | Old Owner | New Owner |
|-----------|-----------|-----------|
| `ring_spacing` | Preprocessing | Preprocessing |
| `tunnel_diameter` | Preprocessing | Preprocessing (physical constant) |
| `radius_min` | Preprocessing | Preprocessing |
| `radius_max` | Preprocessing | Preprocessing |
| `gradient_threshold` | Preprocessing | Preprocessing |
| `target_distances` | Preprocessing | **Detection** |
| `curvature_neighbors` | Preprocessing | **Detection** |
| `depth_map_resolution` | Preprocessing | **Detection** |
| `interpolation_window` | Preprocessing | **Detection** |
| `binary_threshold` | Detection | Detection |
| Hough parameters | Detection | Detection |
| SAM parameters | Segmentation | Segmentation |

---

## 3. Pipeline Validation (Pre-BO)

Ran the full restructured pipeline on tunnel 1-4 with default parameters to verify
end-to-end functionality.

### Preprocessing

- Input points: 1,361,648
- Detected rings: 10
- Denoised points: 973,342 (71.5% retention)

### Detection (with Enhancing)

- Upsampled at 3 resolutions (0.08, 0.04, 0.02)
- Generated depth map: 1664 x 1507 pixels
- Detected: 15 positive slope, 15 negative slope, 21 horizontal, 2 vertical lines
- K positions: 10 (5 midpoint, 5 assumed)

### SAM Segmentation

- Processed 10 rings, 6 segments each
- Final point cloud: 3,030,333 points

### Evaluation

| Metric | Value |
|--------|-------|
| Overall Accuracy (OA) | 0.645 |
| F1 Score (macro) | 0.448 |
| Mean IoU (mIoU) | 0.333 |

Per-class IoU: Background 0.794, K-block 0.241, B1-block 0.439, A1-block 0.222,
A2-block 0.000, A3-block 0.113, B2-block 0.520.

---

## 4. Bug Fix

**File**: `agents/simple_staggered/1_preprocessing/bo/run_preprocessing_bo.py`  
**Line**: 419 (in `save_best_params()`)

The function referenced `DEFAULT_TARGET_DISTANCES`, which was removed from
`1_preprocessing.py` during refactoring. It also saved `target_distances` into
the preprocessing JSON, which is no longer appropriate since that parameter
now belongs to detection.

**Fix**: Removed the `target_distances` line from `params_to_save`. The function
now only saves the 5 preprocessing parameters: `ring_spacing`, `tunnel_diameter`,
`radius_min`, `radius_max`, `gradient_threshold`.

---

## 5. Bayesian Optimization

### Configuration

| Setting | Value |
|---------|-------|
| Optimizer | `skopt.gp_minimize` (Gaussian Process) |
| Objective | Retention F2 score (β=2, maximize) |
| Total evaluations | 20 |
| Initial random points | 5 |
| GP-guided points | 15 |
| Warm-start | Trial 035 (F2=0.9826, from previous session) |
| Search space | 4 dimensions |

### Search Space

| Parameter | Range | Type | Description |
|-----------|-------|------|-------------|
| `ring_spacing` | [1.0, 1.4] | Real | Ring spacing in meters |
| `radius_min` | [2.60, 2.77] | Real | Inner radius filter |
| `radius_max` | [2.77, 2.95] | Real | Outer radius filter |
| `gradient_threshold` | [0.05, 0.5] | Real | Surface cutoff aggressiveness |

### F2 Score Formula

```
F₂ = (1 + β²) × P × R / (β² × P + R),  β = 2

Where:
  P = TP / (TP + FP)    Precision
  R = TP / (TP + FN)    Recall

  TP = lining points kept
  FP = non-lining points kept
  FN = lining points removed (irreversible loss)
  TN = non-lining points removed
```

F2 weights recall 4× more than precision. This is appropriate because false
negatives (removing true lining points) are irreversible — downstream
detection and SAM cannot recover them.

### Trial Results

| Trial | F2 | Precision | Recall | Kept% | ring_spacing | radius_min | radius_max | gradient_threshold | Runtime |
|-------|-----|-----------|--------|-------|-------------|-----------|-----------|-------------------|---------|
| 000 | 0.9735 | 0.9442 | 0.9811 | 73.0% | 1.3186 | 2.6312 | 2.9103 | 0.3186 | 27.5s |
| 001 | 0.8908 | 0.9308 | 0.8813 | 66.6% | 1.1783 | 2.6170 | 2.8527 | 0.2002 | 21.9s |
| 002 | 0.8706 | 0.9412 | 0.8546 | 63.8% | 1.0571 | 2.7107 | 2.7802 | 0.3749 | 21.9s |
| 003 | **0.9799** | 0.9398 | 0.9905 | 74.1% | 1.3754 | 2.6001 | 2.9486 | 0.3279 | 21.7s |
| 004 | 0.4338 | 0.8793 | 0.3850 | 30.8% | 1.2447 | 2.6012 | 2.7742 | 0.2861 | 21.5s |
| 005 | 0.8424 | 0.9255 | 0.8239 | 62.6% | 1.1852 | 2.7068 | 2.8500 | 0.4304 | 21.4s |
| 006 | 0.8898 | 0.9385 | 0.8784 | 65.8% | 1.0413 | 2.6000 | 2.9077 | 0.0686 | 21.2s |
| 007 | 0.0000 | — | — | — | 1.3505 | 2.7700 | 2.7700 | 0.2435 | 12.7s |
| 008 | 0.9776 | 0.9524 | 0.9840 | 72.6% | 1.4000 | 2.6733 | 2.7700 | 0.4687 | 20.6s |
| 009 | 0.9469 | 0.9419 | 0.9482 | 70.8% | 1.0280 | 2.6000 | 2.9213 | 0.2117 | 21.2s |
| 010 | 0.9643 | 0.9408 | 0.9703 | 72.5% | 1.0158 | 2.6003 | 2.9500 | 0.3613 | 21.1s |
| 011 | 0.1975 | 0.8969 | 0.1653 | 13.0% | 1.2253 | 2.7118 | 2.7700 | 0.4488 | 20.6s |
| 012 | 0.9631 | 0.9450 | 0.9677 | 72.0% | 1.3860 | 2.6357 | 2.9434 | 0.1369 | 20.8s |
| 013 | 0.9799 | 0.9397 | 0.9905 | 74.1% | 1.3053 | 2.6198 | 2.9227 | 0.3987 | 20.8s |
| 014 | 0.9810 | 0.9582 | 0.9868 | 72.4% | 1.1973 | 2.7071 | 2.7700 | 0.4092 | 20.8s |
| **015** | **0.9811** | **0.9360** | **0.9931** | **74.6%** | **1.3127** | **2.6122** | **2.9237** | **0.4620** | 21.0s |
| 016 | 0.9693 | 0.9424 | 0.9763 | 72.8% | 1.0085 | 2.7021 | 2.8234 | 0.4588 | 21.1s |
| 017 | 0.9550 | 0.9444 | 0.9577 | 71.3% | 1.1535 | 2.6125 | 2.7958 | 0.0796 | 21.0s |
| 018 | 0.9732 | 0.9392 | 0.9821 | 73.5% | 1.3782 | 2.6000 | 2.9500 | 0.3617 | 20.7s |
| 019 | 0.9726 | 0.9427 | 0.9803 | 73.1% | 1.4000 | 2.6294 | 2.8227 | 0.2455 | 20.7s |

**Failed trials**: Trial 007 (radius_min == radius_max = 2.77, zero-width band),
Trial 004 and 011 (extreme parameter combinations causing excessive point removal).

### Best Result: Trial 015

| Metric | Value |
|--------|-------|
| **F2 Score** | **0.9811** |
| Precision | 0.9360 |
| Recall | 0.9931 |
| Points kept | 74.6% |

**Parameters**:
```json
{
    "ring_spacing": 1.3127216281448195,
    "tunnel_diameter": 5.54,
    "radius_min": 2.61217778194112,
    "radius_max": 2.9237050682931147,
    "gradient_threshold": 0.46195284084885485
}
```

### Reference: Trial 035 (Pre-Refactoring Warm-Start)

| Metric | Value |
|--------|-------|
| F2 Score | 0.9826 |
| Precision | 0.9578 |
| Recall | 0.9890 |
| Points kept | 72.6% |

Note: Trial 035 was from a prior session with the old 6D search space
(included `curvature_neighbors` and `depth_map_resolution`). It did not
tune `ring_spacing` (fixed at ~1.19). Its F2 = 0.9826 is higher than
trial 015's 0.9811, but the new search space correctly includes
`ring_spacing` as tunable and excludes enhancing parameters that
have no effect on the denoised output.

---

## 6. Observations and Analysis

### Parameter Sensitivity

1. **`ring_spacing`**: Highly sensitive. Directly controls the number of detected
   rings (10 at ~1.19, 9 at ~1.31). Values near 1.0 or below tend to over-segment
   and yield lower F2. Best results cluster around 1.19-1.38.

2. **`radius_min` and `radius_max`**: The gap between these defines the radial band
   of kept points. Too narrow (e.g., trial 007 where min==max) causes total failure.
   Wider bands (2.60-2.95) retain more lining points at the cost of more noise.

3. **`gradient_threshold`**: Controls surface cutoff aggressiveness. Higher values
   (0.4-0.5) are more permissive, retaining more points. Very low values (0.05-0.08)
   aggressively remove points, hurting recall.

### Trade-off: Precision vs Recall

- **Trial 015** (best F2): Recall = 0.9931 (highest), Precision = 0.9360 (lower).
  Keeps 74.6% of points — more permissive, loses fewer lining points.
- **Trial 014** (runner-up): Recall = 0.9868, Precision = 0.9582 (higher).
  Keeps 72.4% — tighter filtering, better precision but slightly more lining loss.

F2 favors recall, so trial 015 wins despite lower precision. For downstream
detection where noise impacts depth map quality, trial 014's tighter filtering
may perform better end-to-end — this would be revealed by detection-stage BO.

### Ring Count Impact

| ring_spacing | Detected Rings | Example Trials |
|-------------|---------------|----------------|
| ~1.0-1.05 | 11+ | 002, 006, 009, 010 |
| ~1.19-1.25 | 10 | 004, 005, 011, 014 |
| ~1.30-1.40 | 9 | 000, 003, 008, 012, 013, 015, 018, 019 |

The correct ring count for tunnel 1-4 may be 9 or 10. This is determined by
`ring_spacing` and directly affects all downstream stages. Future detection-stage
BO will clarify which ring count produces better K-position accuracy.

---

## 7. Current State

### Active Parameters

**Preprocessing** (`parameters_preprocessing.json`):
```json
{
    "ring_spacing": 1.3127216281448195,
    "tunnel_diameter": 5.54,
    "radius_min": 2.61217778194112,
    "radius_max": 2.9237050682931147,
    "gradient_threshold": 0.46195284084885485
}
```

**Detection** (`parameters_detection.json`):
```json
{
    "target_distances": [0.08, 0.04, 0.02],
    "curvature_neighbors": 10,
    "depth_map_resolution": 0.008,
    "interpolation_window": 9,
    "binary_threshold": 149,
    "dilation_kernel_size": 2,
    "dilation_iterations": 1,
    "hough_oblique_threshold": 69,
    "hough_oblique_min_length": 99,
    "hough_oblique_max_gap": 60,
    "angle_positive_min": 5.509,
    "angle_positive_max": 8.652,
    "angle_negative_min": -8.652,
    "angle_negative_max": -5.509,
    "hough_vertical_threshold": 574
}
```

### Log Files

- `preproc_1-4_000.json` through `preproc_1-4_019.json`: 20 BO trials (this session)
- `preproc_1-4_035.json`: Warm-start reference from previous session (old 6D space)

---

## 8. Multi-Tunnel Extension

### 8.1 Radius Extraction Method Improvement

**Problem**: The original single-plane percentile method (80th percentile of radial distances from a global PCA-fitted center) overestimated radius for curved tunnels:
- Tunnel 3-1: Extracted 3.44m vs. known 2.75m (25% error)
- Tunnel 5-1: Extracted 4.08m vs. known 3.75m (9% error)

**Root Cause**: Projecting all points from a curved tunnel onto a single plane inflates the apparent radius because distant points along the curve project farther from the center.

**Solution**: Implemented **slice-and-fit method** in `extract_characteristics.py`:
1. PCA to find tunnel axis
2. Divide points into thin slices (~0.5m) along the axis
3. For each slice: project to local cross-section, fit circle (Kása method), record radius
4. Return **median** of per-slice radii (robust to outlier slices)

**Status**: Method implemented but still overestimates for highly curved tunnels. For now, using known `tunnel_diameter / 2` from `p4tun/parameters/` as fallback for 3-1, 4-1, 5-1. Future work: refine slice method or use spline-fitted centerline.

### 8.2 Auto-Adaptive Search Space

**Enhancement**: Modified `get_preprocessing_dimensions()` to accept `tunnel_id` and `agent_type`, loading `characteristics.json` to set radius bounds dynamically:

```python
radius = characteristics['cross_section_radius_m']
radius_min: [radius * 0.90, radius]  # 10% below
radius_max: [radius, radius * 1.10]  # 10% above
```

This ensures each tunnel's search space is appropriately scaled to its geometry, avoiding wasted evaluations in irrelevant parameter regions.

### 8.3 Filesystem Organization

**Initial Problem**: After copying preprocessing to `continuous` and `complex_staggered`, all parameter directories and logs were duplicated, creating confusion.

**Solution**: Cleaned up and organized by tunnel type:
- **simple_staggered**: Tunnels 1-4, 2-2 only
- **continuous**: Tunnel 3-1 only
- **complex_staggered**: Tunnels 4-1, 5-1 only

**Log Organization**:
- Fixed `PreprocessingObjective.__init__` to use `agent_type` for logs directory
- Moved misplaced logs to correct agent directories
- All future BO runs write logs to agent-specific locations

---

## 9. Multi-Tunnel BO Results

### 9.1 Final F2 Scores Summary

| Tunnel | Agent Type | Best F2 Score | BO Runs | Status |
|--------|------------|--------------|---------|--------|
| **1-4** | simple_staggered | **0.9811** | 20 | ✅ Optimized |
| **2-2** | simple_staggered | **0.9760** | 30 | ✅ Optimized |
| **3-1** | continuous | **0.9603** | 30 | ✅ Optimized |
| **4-1** | complex_staggered | **0.9379** | 50 | ✅ Optimized |
| **5-1** | complex_staggered | **0.9545** | 50 | ✅ Optimized |

**Summary Statistics**:
- **Best performing**: 1-4 (F2=0.9811)
- **Lowest performing**: 4-1 (F2=0.9379)
- **Average F2**: 0.9620
- **All scores ≥ 0.93**, indicating good preprocessing quality across all tunnel types

### 9.2 Tunnel 2-2 (simple_staggered)

**Configuration**: 30 runs, 5 initial random, warm-started from 1-4 optimal parameters

**Best Result**:
- **F2 Score**: 0.9760
- **Parameters**:
  - `ring_spacing`: 1.4000
  - `radius_min`: 2.6127
  - `radius_max`: 3.0563
  - `gradient_threshold`: 0.5000

**Observations**: Similar to 1-4, benefits from higher `ring_spacing` (1.4) and permissive `gradient_threshold` (0.5). Radius bounds slightly wider than 1-4, reflecting its slightly larger cross-section (2.78m vs 2.77m).

### 9.3 Tunnel 3-1 (continuous)

**Configuration**: 30 runs, 5 initial random, warm-started from 1-4 optimal parameters

**Best Result**:
- **F2 Score**: 0.9603
- **Parameters**:
  - `ring_spacing`: 1.1697
  - `radius_min`: 2.5689
  - `radius_max`: 3.0250
  - `gradient_threshold`: 0.4820

**Observations**: Lower F2 than simple_staggered tunnels. Continuous assembly type may have different point distribution characteristics. Optimal `ring_spacing` (1.17) is lower than 1-4/2-2, suggesting different ring spacing in the physical tunnel.

### 9.4 Tunnel 4-1 (complex_staggered)

**Configuration**: 50 runs, 8 initial random, warm-started from 1-4 optimal parameters

**Best Result**:
- **F2 Score**: 0.9379
- **Parameters**:
  - `ring_spacing`: 1.2560
  - `radius_min`: 3.5150
  - `radius_max`: 3.9073
  - `gradient_threshold`: 0.4565

**Observations**: Lowest F2 among all tunnels. Complex_staggered with 7 segments per ring may have more challenging geometry. Larger radius (3.75m) requires wider search space. More BO runs (50) needed to find good parameters.

### 9.5 Tunnel 5-1 (complex_staggered)

**Configuration**: 50 runs, 8 initial random, warm-started from 1-4 optimal parameters

**Best Result**:
- **F2 Score**: 0.9545
- **Parameters**:
  - `ring_spacing`: 1.4000
  - `radius_min`: 3.5257
  - `radius_max`: 4.0511
  - `gradient_threshold`: 0.4482

**Observations**: Better than 4-1 despite same assembly type. Optimal `ring_spacing` at upper bound (1.4), suggesting this tunnel has wider ring spacing. Similar `gradient_threshold` to 4-1 (~0.45).

---

## 10. Cross-Tunnel Insights

### 10.1 Parameter Patterns by Tunnel Type

**simple_staggered (1-4, 2-2)**:
- Higher F2 scores (0.976-0.981)
- Optimal `ring_spacing`: 1.31-1.40
- Optimal `gradient_threshold`: 0.46-0.50 (more permissive)
- Smaller radius (2.75-2.78m)

**continuous (3-1)**:
- Moderate F2 (0.960)
- Optimal `ring_spacing`: 1.17 (lower)
- Optimal `gradient_threshold`: 0.48
- Similar radius to simple_staggered (2.75m)

**complex_staggered (4-1, 5-1)**:
- Lower F2 scores (0.938-0.955)
- Optimal `ring_spacing`: 1.26-1.40 (variable)
- Optimal `gradient_threshold`: 0.45-0.46
- Larger radius (3.75m)

### 10.2 Radius Bounds Scaling

The 10% rule for radius bounds (`[radius × 0.90, radius]` and `[radius, radius × 1.10]`) works well across all tunnel sizes:
- Small tunnels (2.75m): bounds [2.48, 3.03]
- Large tunnels (3.75m): bounds [3.38, 4.13]

This ensures the search space is appropriately scaled without wasting evaluations.

### 10.3 Warm-Start Effectiveness

Warm-starting from 1-4 optimal parameters helped all tunnels:
- **2-2**: Quickly found good parameters (F2=0.9760 in 30 runs)
- **3-1**: Found good parameters despite different assembly type
- **4-1, 5-1**: Required more exploration (50 runs) but still benefited from warm-start

The 1-4 parameters serve as a good initialization point, especially for `ring_spacing` and `gradient_threshold`.

---

## 11. Current State

### 11.1 Active Parameters by Tunnel

All tunnels now have optimized `parameters_preprocessing.json` files in their respective agent directories:
- `agents/simple_staggered/1_preprocessing/parameters/{1-4,2-2}/`
- `agents/continuous/1_preprocessing/parameters/3-1/`
- `agents/complex_staggered/1_preprocessing/parameters/{4-1,5-1}/`

### 11.2 Log Files Organization

- **simple_staggered**: `preproc_1-4_*.json` (20 files), `preproc_2-2_*.json` (30 files)
- **continuous**: `preproc_3-1_*.json` (30 files)
- **complex_staggered**: `preproc_4-1_*.json` (50 files), `preproc_5-1_*.json` (50 files)

Total: 180 BO trial logs across all tunnels.

---

## 12. Next Steps

1. **Detection BO**: Optimize detection parameters (including the newly moved
   `target_distances`, `curvature_neighbors`, `depth_map_resolution`,
   `interpolation_window`) using K-position accuracy as the ground truth.
   Run for each tunnel type separately.

2. **SAM BO**: Optimize SAM segmentation parameters using mIoU as the ground truth.
   Run for each tunnel type separately.

3. **End-to-end validation**: After all three stages are individually optimized,
   run the full pipeline for all tunnels and evaluate final mIoU against baselines.

4. **Radius extraction refinement**: Improve slice-and-fit method or implement
   spline-fitted centerline approach to eliminate need for manual diameter fallback.
