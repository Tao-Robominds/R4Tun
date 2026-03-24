# Memory-only ablation — tunnel `1-4` — Opus 4.6 (regular-staggered)

**Layout family:** regular-staggered  
**Condition:** memory (`-m`) — raw characteristics only; no CoT, knowledge, or intermediate-stage JSON in the analyst prompt.


## Raw characteristics comparison (memory-only evidence)

| Property | Sample | 1-4 | Ratio (tunnel / sample) |
|---|---|---|---|
| total_points | 1109768 | 2005884 | 1.81× |
| length_x_axis (m) | 12.16 | 33.97 | 2.79× |
| width_y_axis (m) | 5.60 | 6.00 | 1.07× |
| height_z_axis (m) | 5.08 | 5.78 | 1.14× |
| actual_diameter (m) | 5.5 | 5.5 | same |
| median NN (m) | 0.006514481254712708 | 0.005415398201360711 | 0.83× |
| max NN (m) | 0.2442797068280462 | 0.5189269580942679 | 2.12× |


## Per-stage outcome (same methodology as `1-4`)

Each `parameters_*_m_opus4.6.json` is a **byte-for-byte copy** of this tunnel’s archived `parameters_*.json` at generation time. With memory-only context, raw global statistics do not justify changing cross-sectional pipeline parameters (unfolding/denoising/enhancing/detecting/SAM) without intermediate evidence.

| Stage | File | vs archive |
|---|---|---|
| Unfolding | `parameters_unfolding_m_opus4.6.json` | identical |
| Denoising | `parameters_denoising_m_opus4.6.json` | identical |
| Enhancing | `parameters_enhancing_m_opus4.6.json` | identical |
| Detecting | `parameters_detecting_m_opus4.6.json` | identical |
| SAM | `parameters_sam_m_opus4.6.json` | identical |

**Conclusion:** Memory-only baseline = **retain** archived parameters. Later ablation roots (`memory+state`, `memory+state+knowledge`, `reflection`) are where analyst context may diverge.

## Archive seeding

If a tunnel folder had no `parameters_*.json`, files were copied from `1-4/` as the regular-staggered default. Tunings that already existed (e.g. different `window_size`) were **not** overwritten.

## Regenerate Opus 4.6 copies + readmes

```bash
./venv/bin/python configurable/ablation/memory/sync_opus4.6_regular_staggered.py
```
