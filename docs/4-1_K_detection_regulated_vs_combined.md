# 4-1 K detection: regulated vs combined (research only, no reverts)

## What produces `data/irregular/4-1/depth_map_annotated.png`

From conversation history and code:

1. **Plot script**  
   `scripts/plot_depth_map_annotated.py` or `p4tun/plot_depth_map_k_annotated.py` builds the image: depth map + GT K (red) + optional detected K CSV (blue). So the **blue markers** on the annotated image come from whatever CSV is passed as `--detected-csv`.

2. **History (transcript)**  
   - User: add a **regulator** on to the detection result; reference p4tun/4-1_detection.  
   - User: show the **regulated** result on depth_map_annotated.png.  
   - Agent: the blue markers are the **regulated result**: one K per ring, X evenly spaced, **Y from the oblique-line regulator**.  
   - So the 4-1 annotated view that “must be from regulated methods” is: **base detection (e.g. DBSCAN) + regulator**, then that CSV is plotted in blue.

3. **Where “regulated” is implemented**  
   - **File:** `p4tun/4-1-1_geo_k_detection.py`  
   - **Function:** `apply_k_regulator()` (lines 104–224)  
   - **Used by:** `run_k_detection(..., use_regulator=True)` (default).  
   - **Behaviour:**  
     - Takes raw K from any base method (dbscan, groove_pair, banded, etc.).  
     - **Even X:** one K per ring at `vertical_x = (i + 0.5) * ring_width`.  
     - **Y from oblique lines:** at each ring’s X, finds pos/neg line crossings, picks pairs with gap ≈ `reg_target_gap` (K height), takes **midpoint**; blends with detected Y via `reg_blend_weight`; if line midpoint is > `reg_max_det_line_dist` from detected Y, keeps detected Y only.  
   - Output type is e.g. `dbscan_regulated` (base type + `_regulated`).

So **“regulated”** = any base method (e.g. DBSCAN) **then** `apply_k_regulator` (even X, Y from line geometry). No code changes or reverts were made; this is documentation only.

## Regulated vs “combined” (agents pipeline)

| Term | Where | Meaning |
|------|--------|--------|
| **Regulated** | p4tun `4-1-1_geo_k_detection.py` | Base method (e.g. dbscan or groove_pair) → then **apply_k_regulator**: even X, one K per ring, Y from oblique line pairs (with blend/fallback). Produces e.g. `detected_k_dbscan.csv` with Type `dbscan_regulated`. |
| **Combined** | agents `2_detection.py` | **calculate_k_positions_combined**: run both DBSCAN path and groove_pair path; per ring, collect candidates from both and pick best by groove alignment score. No regulator step; X/Y come from the chosen candidate. |

So:

- **Regulated** = base detection + **regulator** (geometry: even X, Y from lines).  
- **Combined** = **fusion** of DBSCAN and groove_pair per ring (no regulator).

The 4-1 annotated image that “must be from regulated methods” is from the **p4tun regulated pipeline** (base + `apply_k_regulator`), not from the agents “combined” method alone.

## Current 4-1 config (unchanged)

- `agents/irregular/2_detection/parameters/4-1/parameters_detection.json` has `"k_detection_method": "combined"`.  
- No reverts or edits were made to params in this research; only this doc was added.

## How to reproduce regulated 4-1 annotated image

1. Run K detection with regulator (p4tun):  
   `python p4tun/4-1-1_geo_k_detection.py 4-1 --data-dir data --method dbscan`  
   (default `use_regulator=True` → writes regulated K to e.g. `data/4-1/detected_k_dbscan.csv`.)

2. Plot with that CSV:  
   `python scripts/plot_depth_map_annotated.py --tunnel 4-1 --data-dir data --detected-csv data/4-1/detected_k_dbscan.csv`  
   (or use `p4tun/plot_depth_map_k_annotated.py` with the same `--detected-csv` if it lives under `data/irregular/4-1/`.)

Result: depth_map_annotated.png with red = GT K, blue = **regulated** K (even X, Y from oblique-line regulator).
