# GT-detection ceiling report

First-principles ceiling: per-pixel dominant GT labelmap computed directly from the raw ring point cloud, back-projected to every raw point. Preprocessing and detection are bypassed entirely — the only loss source is the per-pixel mixing fraction (pixels where points from ≥2 GT segments share the same depth-map cell).

Resolution: 0.005 m  Tunnel diameter: 7.5 m  Source panel: `reference_panel.json`

## Headline

- Reference rings (n=6)
- Median mIoU: **0.9935**
- Mean mIoU:   0.9874
- Min / max:   0.9614 / 1.0000
- Acceptance gate: median mIoU ≥ 0.90 → **PASS**

## Per-ring summary

| tunnel/ring | regime | mIoU | OA | F1 (macro) | mixed-pixel % | n_points | gate |
|---|---|---:|---:|---:|---:|---:|---|
| `5-5/r258` | medium_full_normal_reversed_canonical | 0.9964 | 0.9981 | 0.9982 | 0.19% | 91,547 | ✓ |
| `4-9/r366` | dense_full_wide_canonical | 0.9773 | 0.9824 | 0.9884 | 1.82% | 396,950 | ✓ |
| `5-3/r190` | low_full_normal_canonical | 0.9989 | 0.9993 | 0.9994 | 0.07% | 20,512 | ✓ |
| `4-8/r337` | medium_full_narrow_reversed_canonical | 0.9905 | 0.9948 | 0.9952 | 0.54% | 157,262 | ✓ |
| `4-1/r116` | dense_partial_normal_reversed_canonical | 0.9614 | 0.9743 | 0.9802 | 2.83% | 579,860 | ✓ |
| `4-6/r283` | sparse_full_wide_canonical | 1.0000 | 1.0000 | 1.0000 | 0.00% | 7,938 | ✓ |

## Per-class IoU (per ring)

| ring | Background | K-block | B1-block | A1-block | A2-block | A3-block | A4-block | B2-block |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `5-5/r258` | 0.992 | 0.997 | 0.999 | 0.992 | 1.000 | 0.998 | 0.999 | 0.993 |
| `4-9/r366` | 0.942 | 0.995 | 0.952 | 0.959 | 0.997 | 0.997 | 0.999 | 0.979 |
| `5-3/r190` | 0.997 | 1.000 | 1.000 | 1.000 | 0.994 | 1.000 | 1.000 | 1.000 |
| `4-8/r337` | 0.978 | 0.989 | 0.991 | 0.995 | 0.996 | 0.982 | 0.993 | 1.000 |
| `4-1/r116` | 0.927 | 0.972 | 0.975 | 0.957 | 0.931 | 0.999 | 0.966 | 0.965 |
| `4-6/r283` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Notes

- The labelmap height is locked to the full circumference (`pi * tunnel_diameter / resolution`) so the same theta axis is used across rings of one family.
- A pixel is 'mixed' when raw points from ≥2 GT segments fall in the same depth-map cell. The first-principles ceiling loss is bounded above by the mixed-pixel fraction; tightening `--resolution` reduces it.
- This ceiling is an upper bound on what any segmentation/back-projection code-path can deliver on these inputs at this resolution.
