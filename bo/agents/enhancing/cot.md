# Parameter reference — Enhancing (BO)

**Tunable:** upsampling_stage1_target_distance [0.055, 0.111], upsampling_stage2_target_distance [0.028, 0.056], upsampling_stage3_target_distance [0.014, 0.028], inter_radius [0.03, 0.08], n_segment_end [5, 21].

**Fixed:** curvature_threshold=0.005, depth_threshold_low=0.005, depth_threshold_high=0.015.

**Locked:** n_segment_start, duplicate_threshold, num_neighbors, num_interpolations, resolution, window_size — see `knowledge.md`.

**Constraints:** stage1 ≈ 2× stage2 ≈ 4× stage3; n_segment_end ≤ ring count when known.
