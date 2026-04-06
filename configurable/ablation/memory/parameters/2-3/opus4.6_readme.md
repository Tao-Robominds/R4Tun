# Memory-only ablation — tunnel `2-3` — Opus 4.6 (regular-staggered)

**Layout family:** regular-staggered  
**Condition:** memory (`-m`) — raw characteristics only; no CoT, knowledge, or intermediate-stage JSON in the analyst prompt.


## Raw characteristics comparison (memory-only evidence)

| Property | Sample | 2-3 | Ratio (tunnel / sample) |
|---|---|---|---|
| total_points | 1109768 | 2049003 | 1.85× |
| length_x_axis (m) | 12.16 | 27.74 | 2.28× |
| width_y_axis (m) | 5.60 | 5.64 | 1.01× |
| height_z_axis (m) | 5.08 | 6.04 | 1.19× |
| estimated_diameter (m) | 5.604292068996665 | 5.64122858327027 | 1.01× |
| median NN (m) | 0.006514481254712708 | 0.005415145612492822 | 0.83× |
| max NN (m) | 0.2442797068280462 | 1.2163606099645212 | 4.98× |


## Per-stage outcome (same methodology as `1-4`)

Each `parameters_*_m_opus4.6.json` is a **byte-for-byte copy** of this tunnel’s archived `parameters_*.json` at generation time. With memory-only context, raw global statistics do not justify changing cross-sectional pipeline parameters (unfolding/denoising/enhancing/detecting/SAM) without intermediate evidence.

| Stage | File | vs archive |
|---|---|---|
| Unfolding | `parameters_unfolding_m_opus4.6.json` | identical |
| Denoising | `parameters_denoising_m_opus4.6.json` | identical |
| Enhancing | `parameters_enhancing_m_opus4.6.json` | identical |
| Detecting | `parameters_detecting_m_opus4.6.json` | identical |
| SAM | `parameters_sam_m_opus4.6.json` | identical |

**Conclusion:** Memory-only baseline = **retain** archived parameters. Later ablation roots (`memory+state`, `memory+state+knowledge`) are where analyst context may diverge.

## Archive history

If this tunnel’s archive was initially seeded from tunnel `1-4`, tunings that already existed (e.g. different `window_size`) were **not** overwritten.

## Refresh for E2E

See [`process.md`](../../process.md). Regenerate prompts: `./venv/bin/python skills/scripts/export_llm_parameter_context.py 2-3` (repo root). After inference, sync executables:

```bash
./venv/bin/python skills/scripts/sync_inference_to_executable.py --tunnel-id 2-3
```
