# Step 0 — Oracle ceiling (GT-derived layout diagnostic)

GT-derived optimal K position and A/B offsets → frozen agents detection + segmentation → GT-best `r_surface_min` sweep. Reported as a **design-time diagnostic ceiling**, not part of the deployable method.

**Pass criterion:** ceiling mIoU ≥ 0.85 per ring.

---

## BO calibration corpus (`data/bo/`)

**6 / 6 pass**

| Ring | Ceiling mIoU |
|------|-------------:|
| 4-9/r365 | 0.913 |
| 4-1/r116 | 0.905 |
| 5-5/r253 | 0.886 |
| 5-7/r323 | 0.884 |
| 4-7/r309 | 0.878 |
| 4-8/r336 | 0.878 |

| Stat | Value |
|------|------:|
| Mean | 0.891 |
| Min | 0.878 |
| Max | 0.913 |

**Evidence:** `data/bo/MANIFEST.json`

---

## Held-out evaluation corpus (`data/held-out/`)

**50 / 50 pass**

| Ring | Ceiling mIoU | Ring | Ceiling mIoU |
|------|-------------:|------|-------------:|
| 1-1/r18 | 0.882 | 3-1-2/r47 | 0.903 |
| 1-1/r19 | 0.894 | 3-1-2/r48 | 0.882 |
| 1-2/r58 | 0.913 | 3-1-3/r77 | 0.884 |
| 1-2/r59 | 0.896 | 3-1-3/r78 | 0.916 |
| 1-3/r125 | 0.907 | 3-1-3/r86 | 0.906 |
| 1-3/r131 | 0.889 | 4-1/r110 | 0.888 |
| 1-4/r197 | 0.941 | 4-10/r398 | 0.874 |
| 1-4/r205 † | 0.923 | 4-2/r142 | 0.868 |
| 1-5/r270 | 0.861 | 4-3/r177 | 0.860 |
| 1-5/r273 | 0.887 | 4-4/r212 | 0.855 |
| 2-1/r60 | 0.892 | 4-5/r249 | 0.878 |
| 2-1/r64 | 0.944 | 4-6/r275 † | 0.891 |
| 2-2/r141 | 0.903 | 4-7/r308 | 0.884 |
| 2-2/r143 | 0.937 | 4-8/r332 | 0.876 |
| 2-3/r220 | 0.892 | 4-7/r303 † | 0.881 |
| 2-3/r224 | 0.913 | 5-1/r118 | 0.875 |
| 2-4/r298 | 0.914 | 5-2/r140 | 0.861 |
| 2-4/r304 | 0.911 | 5-3/r192 | 0.864 |
| 2-5/r353 | 0.920 | 5-3/r195 | 0.872 |
| 2-5/r360 | 0.899 | 5-4/r227 | 0.868 |
| 3-1-1/r28 | 0.937 | 5-5/r254 | 0.858 |
| 3-1-1/r31 | 0.902 | 5-5/r259 | 0.884 |
| 3-1-1/r32 | 0.878 | 5-6/r286 † | 0.857 |
| 3-1-1/r29 † | 0.929 | 5-7/r317 | 0.864 |
| 3-1-2/r46 | 0.892 | 5-7/r322 | 0.857 |

† Replacement ring in the final panel.

| Stat | Value |
|------|------:|
| Mean | 0.891 |
| Min | 0.855 (4-4/r212) |
| Max | 0.944 (2-1/r64) |

**Evidence:** `logs/ablation_step0_heldout_v1/ceiling_summary.json`, panel `data/held-out/_manifests/data_v6_50ring_calibration_panel.csv`

---

## Summary

Both locked preprocessing corpora support high mIoU when the correct layout is applied with GT at design time. The bottleneck for deployment is layout recovery without GT, not depth-map quality.

| Corpus | Rings | Pass | Mean ceiling | Min | Max |
|--------|------:|-----:|-------------:|----:|----:|
| `data/bo/` | 6 | 6/6 | 0.891 | 0.878 | 0.913 |
| `data/held-out/` | 50 | 50/50 | 0.891 | 0.855 | 0.944 |
