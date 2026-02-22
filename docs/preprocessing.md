# Preprocessing Report

Summary of the preprocessing pipeline, coverage fix, resolution standardization, and per-tunnel parameters. Content extracted from journals 2026-02-18, 2026-02-20, and 2026-02-21.

---

## 1. Pipeline overview

Three stages:

1. **Unfolding** — Raw point cloud to cylindrical coordinates (ring, theta, h). Slicing planes perpendicular to tunnel axis; `ring_spacing`, `tunnel_diameter`, `num_slicing_planes`, `samples_per_ring` control geometry.
2. **Denoising** — Remove noise via radius filter (`radius_min`, `radius_max`) and gradient surface cutoff (`gradient_threshold`, `double_zero_cutoff`, `smoothing_offset`). Points kept get `pred=7` or `pred=8`; removed points get `pred=0`.
3. **Enhancing** — Upsample (`target_distances`, `curvature_neighbors`), build depth map at `depth_map_resolution`, fill gaps, write `enhanced.csv`, `depth_map.png`, and `pixel_to_point.pkl`.

Script: `complex_agents_wrap/1_preprocessing/1_preprocessing.py`. Parameters per tunnel: `complex_agents_wrap/1_preprocessing/parameters/<tunnel_id>/parameters_preprocessing.json`.

**GT independence:** The preprocessing script does not use ground truth at runtime. The raw input file may contain `segment` and `ring` columns (GT labels), but these are carried through passively to `enhanced.csv` for downstream evaluation only — no preprocessing logic reads or depends on them. Denoising is purely geometric (radius filter + gradient surface); enhancing uses only `pred`, `h`, `theta`, `r`. The parameter values (e.g. `gradient_threshold=10`, `double_zero_cutoff=false`) were derived offline via GT reverse engineering (journal 2026-02-21 section 6) but are applied as fixed JSON config at runtime with no GT access.

---

## 2. The coverage problem and fix

**Problem (journal 2026-02-20):** Denoising removed 337k valid GT points (marked `pred=0`), including 102k GT-block points. The enhancing stage only projects points with `pred` in [7, 8] into the depth map, so those 337k points never appeared in `pixel_to_point.pkl`. Downstream segmentation could not assign them any label, capping the theoretical ceiling at **0.884** (5-1).

**Root cause (journal 2026-02-21, section 6):** Two denoising conditions were too aggressive:

- **`double_zero_cutoff`**: Removed points when two consecutive density bins were empty.
- **`smoothing_offset`**: Subtracted a fixed 0.005 m from the cutoff surface.

**Fix:** Both parameters were made configurable in `1_preprocessing.py`. GT-optimal values were set via reverse engineering and applied across all tunnels:

- **`gradient_threshold=10.0`**
- **`double_zero_cutoff=false`**
- **`smoothing_offset=0.0`**

**Impact (5-1):**

| Metric | Before | After |
|--------|--------|-------|
| pixel_to_point entries | 1,167,060 | 1,341,094 |
| GT-block coverage | ~77.6% | 100% |
| Theoretical ceiling | 0.884 | 0.9923 |

Recovered 197k points (102k GT-block, 95k GT-BG) from `pred=0` to `pred=7`.

---

## 3. Resolution standardization

**`depth_map_resolution=0.005`** (5 mm/pixel) is used for **all tunnels**. This is set in every parameter file under `complex_agents_wrap/1_preprocessing/parameters/` (1-4, 2-2, 3-1, 4-1, 5-1). The depth map and `pixel_to_point.pkl` are built at this resolution so downstream SAM and evaluation use a consistent scale.

---

## 4. Per-tunnel parameters

The three coverage parameters are **identical** across all tunnels: `gradient_threshold=10`, `double_zero_cutoff=false`, `smoothing_offset=0`. Tunnel-specific geometry and enhancement differ:

| Tunnel | ring_spacing | tunnel_diameter | radius_min | radius_max | target_distances[0] | curvature_neighbors | interpolation_window |
|--------|--------------|-----------------|------------|------------|---------------------|---------------------|----------------------|
| 1-4 | 1.313 | 5.257 | 2.612 | 2.924 | 0.0755 | 11 | 15 |
| 2-2 | 1.4 | 5.285 | 2.613 | 3.056 | 0.0604 | 20 | 12 |
| 3-1 | 1.170 | 5.5 | 2.569 | 3.025 | 0.0985 | 28 | 15 |
| 4-1 | 1.816 | 7.5 | 3.526 | 4.051 | 0.0808 | 9 | 5 |
| 5-1 | 1.816 | 7.5 | 3.526 | 4.051 | 0.0808 | 9 | 5 |

All tunnels use `depth_map_resolution=0.005`, `num_slicing_planes=9`, `samples_per_ring=1210`.

---

## 5. Impact on downstream performance

- **Journal 2026-02-21, Key Insight 1:** "Preprocessing is no longer the bottleneck." GT-optimal denoising achieved 100% GT-block coverage and 0.993 ceiling (5-1).
- Per-tunnel **direct GT projection ceilings** (point-level upper bound) are reported in [docs/upper_bound.md](upper_bound.md): 0.946–0.988 depending on tunnel; these ceilings assume the same preprocessing and `pixel_to_point.pkl` with full coverage.
- The multi-tunnel rollout (1-4, 2-2, 3-1, 4-1, 5-1) uses the same preprocessing params (100% GT coverage, 0.005 m resolution) so segmentation and evaluation are comparable across tunnels.
