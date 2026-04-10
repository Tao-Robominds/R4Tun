# Parameter reference — Denoising (BO)

**Tunable:** `mask_r_low` [2.09, 3.75], `mask_r_high` [2.78, 4.38], `default_cutoff_z` [2.65, 6.27], `z_step` [0.003, 0.005].

**Fixed:** smoothing_window_size=5, smoothing_offset=-0.002, grad_threshold=0.15, y_step=0.4 — see `knowledge.md`.

**Constraints:** mask_r_low < mask_r_high; default_cutoff_z coherent with unfolding diameter.
