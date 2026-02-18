Baseline for tunnel 5-1 (complex_staggered)
===========================================

This directory captures detection baselines for tunnel 5-1:

- **Baseline A (global steps)**: **mean_segment_distance_px ≈ 283.6**
- **Baseline B (per-ring steps)**: **mean_segment_distance_px ≈ 242.5**

- **Assembly**: `complex_staggered`
- **Tunnel**: `5-1`
- **Depth map**: `depth_map_outlier.npy` (copied from `data/5-1`)
- **GT**: `all_segments_gt.csv` (copied from `data/5-1`)
- **Detection parameters (Baseline A)**: `parameters_detection_283px.json`
- **Detection parameters (Baseline B)**: `parameters_detection_perring_242px.json`
- **Preprocessing parameters (HEAD)**: `parameters_preprocessing_head.json`
- **BO config**: `detect_5-1_config.json`

The 283.6px parameters were recovered from the BO run logged in
Cursor terminal file:

- `~/.cursor/projects/home-boringtao-Projects-Bayesian-R4Tun/terminals/391248.txt`

That run reported:

- Best composite score: **0.4998**
- Best mean distance: **283.6px**
- K F1: **0.656** (weighted F1 in log)

The exact parameter block from that log has been written into:

- `agents/complex_staggered/2_detection/parameters/5-1/parameters_detection.json`
- `baselines/5-1/parameters_detection_283px.json`

The 242.5px per-ring parameters were obtained from a later BO run
using a 14D per-ring expansion model (k_to_b_r0..r6, ab_step_r0..r6)
with K-detection frozen at the 283px settings.

To **reproduce** Baseline A (283.6px) from scratch:

1. Ensure `data/5-1/depth_map_outlier.npy` and `data/5-1/all_segments_gt.csv`
   match the copies in this directory.
2. Ensure `agents/complex_staggered/2_detection/parameters/5-1/parameters_detection.json`
   matches `parameters_detection_283px.json`.
3. Run detection, writing outputs to `data/bo` to avoid overwriting current results:

   ```bash
   venv/bin/python3 agents/complex_staggered/2_detection/2_detection.py 5-1 --data-dir data/bo
   ```

4. Optionally, re-evaluate using the BO objective helper (single call) to confirm:

   - `mean_segment_distance_px ≈ 283.55`
   - `composite_score ≈ 0.4998`
   - `K F1 ≈ 0.77`

To **reproduce** Baseline B (per-ring, 242.5px):

1. Ensure `data/5-1/depth_map_outlier.npy` and `data/5-1/all_segments_gt.csv`
   match the copies in this directory.
2. Ensure `agents/complex_staggered/2_detection/parameters/5-1/parameters_detection.json`
   matches `parameters_detection_perring_242px.json`.
3. Run detection, writing outputs to `data/bo`:

   ```bash
   venv/bin/python3 agents/complex_staggered/2_detection/2_detection.py 5-1 --data-dir data/bo
   ```


