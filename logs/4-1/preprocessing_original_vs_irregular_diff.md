# Preprocessing: Original vs Irregular (4-1)

## 1. Parameters JSON

| Aspect | Original `agents/1_preprocessing_original/parameters/4-1/` | Irregular `agents/irregular/1_preprocessing/parameters/4-1/` |
|--------|-----------------------------------------------------------|-------------------------------------------------------------|
| **Keys** | 19 keys only | 27 keys (same 19 + 8 outlier_* keys) |
| **Shared values** | Identical for shared keys: ring_spacing 1.816, num_slicing_planes 7, radius_min/radius_max, gradient_threshold 10.0, double_zero_cutoff false, smoothing_offset 0.0, target_distances, curvature_neighbors 6, depth_map_resolution 0.005, interpolation_window 5, tunnel_diameter 7.5 | Same |
| **Outlier params** | Not in JSON → script uses **hardcoded FIXED_*** (0.003, 0.008, rings 0–5, neighbors 20, max 5000, etc.) | In JSON → uses those values (same numbers as original’s FIXED_*) |

So for 4-1, **effective parameter values are the same**; only difference is where they come from (code vs JSON).

---

## 2. Code differences (script)

### 2.1 Imports
- **Original:** `from scipy.cluster.vq import kmeans2`
- **Irregular:** No `kmeans2` (used only in pattern classification, not in depth map)

### 2.2 Parameter loading
- **Original:** 4-level fallback:
  1. `parameters/<tunnel_id>/parameters_preprocessing.json`
  2. `data/<tunnel_id>/parameters_preprocessing.json`
  3. **`parameters/sample/parameters_preprocessing.json`** ← only in original
  4. Hardcoded defaults
- **Irregular:** 3-level (no `parameters/sample/` fallback)

### 2.3 Constants / defaults
- **Original:** All non-critical defaults as `FIXED_*` (e.g. `FIXED_SMOOTHING_OFFSET = -0.003`, `FIXED_SAMPLES_PER_RING`). Denoising default smoothing = `FIXED_SMOOTHING_OFFSET`.
- **Irregular:** BO-tunable as `DEFAULT_*` (e.g. `DEFAULT_SMOOTHING_OFFSET = -0.003`), rest `FIXED_*`. Denoising default smoothing = `DEFAULT_SMOOTHING_OFFSET`. Same numeric values.

### 2.4 Stage 2 (Denoising)
- Same signature and logic. Both call `denoise_point_cloud(df_unwrapped, n_planes, ...)`. Same `h_step = (h_max - h_min) / ring_count` (second arg is `n_planes`). No behavioral difference.

### 2.5 Stage 3 (Enhancing) – depth map
- **Original:** After saving `depth_map.png` and `depth_map_outlier.npy`, calls **`classify_tunnel_pattern(depth_map_outlier, tunnel_dir)`** and writes **`pattern_type.json`**.
- **Irregular:** No `classify_tunnel_pattern`; no `pattern_type.json`. No `kmeans2`, no HoughLinesP-based pattern classification.

Depth map construction (surface + boundary → grid → gap interpolation) is the same in both.

### 2.6 Other
- **Original:** `df['pred'] = 7` (literal) in denoise.
- **Irregular:** `df['pred'] = SURFACE_PRED` (7). Same.

---

## 3. What actually affects the depth map?

- **Parameters:** For 4-1, same effective values (including denoising and outlier defaults).
- **Pipeline steps that produce the depth map:** Unfolding → Denoising → Enhancing (curvature, upsampling, outlier boundaries, then `generate_depth_map`) are implemented the same; no intentional logic difference in extent, projection, or gap filling.
- **Extra in original:** Only `classify_tunnel_pattern` and `pattern_type.json`; these use the depth map but do not change it.

So under the same input and same 4-1 params, the two scripts should produce the same depth map unless:
- There is a subtle bug or different default path in one script, or
- The “good” run used a different input, different params, or a different code version.

---

## 4. Recommendations

1. **To reproduce the good result without touching `data/irregular/4-1/`:** Run the **original** script and write to a separate dir, e.g. `logs/4-1/preprocessing_run/` (or add an output-dir option and point it there).
2. **To align irregular with original:** Add the `parameters/sample/` fallback and, if desired, port `classify_tunnel_pattern` + `pattern_type.json` so outputs match; this does not change the depth map pixels.
3. **To debug remaining differences:** Run both scripts on the same `data/4-1.txt` with the same 19 parameters, compare `unwrapped.csv` / `denoised.csv` / `enhanced.csv` and depth map dimensions (height × width) and a few sample pixels; any divergence will point to the exact stage that differs.
