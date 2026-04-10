# BO search space — Denoising stage

## Tunable parameters

- **mask_r_low (m)** — Inner radial gate before depth histogramming. Range: **[2.09, 3.75]** (baseline 2.7).

- **mask_r_high (m)** — Outer radial gate. Range: **[2.78, 4.38]** (baseline 2.8).

- **default_cutoff_z (m)** — Fallback radial cutoff when a θ-bin lacks reliable counts. Range: **[2.65, 6.27]** (baseline 2.7).

- **z_step (m)** — Radial bin width per histogram column. Range: **[0.003, 0.005]** (baseline 0.001).

## Proven defaults (fixed, not searched)

| Parameter | Value |
|---|---|
| smoothing_window_size | 5 |
| smoothing_offset | -0.002 |
| grad_threshold | 0.15 |
| y_step | 0.4 |

## Locked parameters (baseline, not searched)

All other keys in the stage JSON remain at pipeline defaults unless explicitly listed elsewhere.

## Constraints

- Require **mask_r_low < mask_r_high**.
- **default_cutoff_z** should stay coherent with the unfolding stage **diameter** (order of magnitude ~ diameter/2 when diameter is in meters).
