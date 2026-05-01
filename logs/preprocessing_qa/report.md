# Ring-native preprocessing — intrinsic depth-map QA (299 rings)

## Batch reproduction

- Driver: `agents/1_preprocessing/scripts/run_all_rings.py` (resume mode; skips existing `depth_map.npy`).
- Initial batch: `ran=261`, `skipped=37`, `failed=1`.
- **Failure:** `5-4/r225` — `ValueError: max() iterable argument is empty` in `optimized_radius_filter` when midpoint candidates filtered to **0 rows** before radius filtering.
- **Fix:** early return for empty `h/theta` in `agents/1_preprocessing/_ring_enhancing.py::optimized_radius_filter`; reran `./venv/bin/python agents/1_preprocessing/scripts/run_all_rings.py --tunnels 5-4` → `5-4/r225` **OK** (~97s).
- **Artifacts:** **299/299** rings have `ring_count.txt`, `unwrapped.csv`, `denoised.csv`, `enhanced.csv`, `depth_map.png`, `depth_map.npy`, `depth_map_outlier.npy`.
- Per-run CSV log: `data/rings/preprocessing_log.csv` (includes the ERROR row for the first `r225` attempt and a later OK row after rerun).

## Triage (intrinsic FAIL rings)

All rings have outputs on disk. **23** rings are classified **FAIL** below (sparse coverage / large empty regions / banding). These are **documented** for this corpus pass; per-ring parameter tuning was **not** applied. Several coincide with very high NaN fraction in `depth_map.npy` (often **>0.9** `nan_ratio` in the batch log), i.e. acquisition/unwrapping sparsity rather than a missing-output bug.

**Note:** `5-4/r225` now completes successfully but remains **FAIL** under `many_empty_row_bands` — intrinsic QA reflects map structure, not the earlier crash.

## `data/rings` cleanup

- `agents/1_preprocessing/scripts/cleanup_rings_dir.py` **dry-run**: `0` proposed deletions (no legacy `data/rings/*.png`; canonical TXT set matches **299** rings). **No `--apply`** run.

---

- Total rings (canonical TXT list): **299**
- PASS: **276**
- WARN: **0**
- FAIL: **23**

Metrics: `valid_ratio`, `largest_empty_ratio`, empty row/column bands (occupancy < 1%).
Classification is intrinsic-only (no GT).

## FAIL rings

| tunnel | ring | reason | depth_map.png |
|---|---:|---|---|
| 1-1 | 19 | dominant_empty_component | `data/1-1/r19/depth_map.png` |
| 1-2 | 62 | many_empty_row_bands | `data/1-2/r62/depth_map.png` |
| 1-3 | 128 | dominant_empty_component | `data/1-3/r128/depth_map.png` |
| 1-3 | 131 | many_empty_row_bands | `data/1-3/r131/depth_map.png` |
| 1-4 | 198 | many_empty_row_bands | `data/1-4/r198/depth_map.png` |
| 1-4 | 202 | dominant_empty_component | `data/1-4/r202/depth_map.png` |
| 1-5 | 269 | dominant_empty_component | `data/1-5/r269/depth_map.png` |
| 1-5 | 272 | near_empty_valid_ratio | `data/1-5/r272/depth_map.png` |
| 2-3 | 218 | dominant_empty_component | `data/2-3/r218/depth_map.png` |
| 2-3 | 221 | many_empty_row_bands | `data/2-3/r221/depth_map.png` |
| 2-4 | 298 | near_empty_valid_ratio | `data/2-4/r298/depth_map.png` |
| 2-5 | 358 | near_empty_valid_ratio | `data/2-5/r358/depth_map.png` |
| 3-1-1 | 36 | dominant_empty_component | `data/3-1-1/r36/depth_map.png` |
| 4-10 | 395 | many_empty_row_bands | `data/4-10/r395/depth_map.png` |
| 4-2 | 145 | many_empty_row_bands | `data/4-2/r145/depth_map.png` |
| 4-3 | 175 | many_empty_row_bands | `data/4-3/r175/depth_map.png` |
| 4-4 | 214 | many_empty_row_bands | `data/4-4/r214/depth_map.png` |
| 4-5 | 245 | many_empty_row_bands | `data/4-5/r245/depth_map.png` |
| 4-6 | 276 | many_empty_row_bands | `data/4-6/r276/depth_map.png` |
| 4-7 | 305 | many_empty_row_bands | `data/4-7/r305/depth_map.png` |
| 4-8 | 335 | many_empty_row_bands | `data/4-8/r335/depth_map.png` |
| 5-3 | 194 | many_empty_row_bands | `data/5-3/r194/depth_map.png` |
| 5-4 | 225 | many_empty_row_bands | `data/5-4/r225/depth_map.png` |

## WARN rings (sample)

| tunnel | ring | reason | valid_ratio | largest_empty_ratio |
|---|---:|---|---:|---:|
