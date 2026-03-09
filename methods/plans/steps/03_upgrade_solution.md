# 03 Upgrade Solution

## Goal

Produce the final non-GT pipeline (`1_preprocessing.py`, `2_detection.py`,
`segmentation.py`) and document all changes from the original `sam4tun` in
`output/03_upgrade_solution_output.md`.

## Inputs

- `output/02_challenge_map_output.md` (challenge IDs and failure modes)
- `output/01_assumptions_output.md` (baseline assumptions)
- Existing agent code in `agents/irregular/`

## Actions

1. **Gap check**: Cross-reference every challenge ID against the candidate
   detection + segmentation methods.  Confirm which are addressed, which are
   stable, and which are structural gaps.

2. **Generate detection.py** (`agents/irregular/2_detection/2_detection.py`):
   - Use combined method only (DBSCAN + groove-pair fusion).
   - Remove standalone K-detection methods, method selector, dead helpers.
   - Wire dilation/canny params to `params.get()` with `DEFAULT_*` fallbacks.
   - Consolidate prints.

3. **Generate segmentation.py** (`agents/irregular/3_segmentation/segmentation.py`):
   - Base on `3_template_geometric.py` (trapezoid K/B, rectangle A).
   - Inline `project_back_to_point_cloud` and `compute_block_to_label_map`
     from `3_sam.py`.
   - Remove GT-dependent unmapped fallback.
   - Remove dynamic SAM import.

4. **Clean preprocessing** (`agents/irregular/1_preprocessing/1_preprocessing.py`):
   - Remove `classify_tunnel_pattern` + call site + `pattern_type.json` save.
   - Remove `scipy.cluster.vq.kmeans2` import.

5. **Delete experimental files** (6 files in `agents/irregular/3_segmentation/`):
   `3_sam.py`, `3_sam_wrap.py`, `3_sam_wrap_a.py`, `3_sam_wrap_b.py`,
   `3_geometric.py`, `3_template_geometric.py`.

6. **Fix evaluation.py** (`agents/irregular/evaluation.py`):
   - Fix `CLASS_NAMES_7` label ordering to match GT and segmentation
     (K=1, B1=2, A1=3, A2=4, A3=5, A4=6, B2=7).
   - Handle NaN GT labels from enhanced/upsampled points: filter before
     metric computation and print accurate exclusion count.

7. **Document**: Write `output/03_upgrade_solution_output.md` with:
   - 4 architecture shifts (walk-order, combined K, SAM→geometric, consolidated preproc).
   - Per-file change summary (lines removed, functions removed, params added).
   - Full challenge coverage table.
   - Remaining structural gaps (deferred).
   - Non-GT compliance statement.

8. **Verify on 5-1**: Run the full pipeline end-to-end on tunnel 5-1
   (preprocessing outputs already exist; run detection → segmentation →
   evaluation) to confirm:
   - The workflow completes without errors.
   - No GT columns are read during detection or segmentation.
   - `evaluation.py` produces a baseline report (mIoU if GT available,
     coverage stats otherwise).
   - The result does not need to be perfect — the goal is to confirm the
     workflow is functional and non-GT.

## Outputs

- `agents/irregular/1_preprocessing/1_preprocessing.py` (cleaned)
- `agents/irregular/2_detection/2_detection.py` (combined-only)
- `agents/irregular/3_segmentation/segmentation.py` (new, non-GT geometric)
- `agents/irregular/evaluation.py` (fixed class names + NaN handling)
- `output/03_upgrade_solution_output.md`
- `data/irregular/5-1/final.csv` (verification run output)
- `data/irregular/5-1/evaluation/performance.md` (baseline report)

## Verify Prompt

```
1. Does the pipeline have zero GT dependencies at inference?
   (No reads of `segment`/`ring` columns for assignment, no gt_angular_boundaries.json, no SAM.)
2. Does every challenge ID in 02_challenge_map_output.md appear in the coverage table?
3. Are structural gaps (E8 per-ring offsets, E6/E7 per-ring sizes, E11 unmapped)
   explicitly listed as deferred?
4. Can `2_detection.py` and `segmentation.py` each run standalone with
   `python <script>.py <tunnel_id>`?
5. Does `python evaluation.py 5-1 --data-dir data/irregular` complete and
   produce a report in data/irregular/5-1/evaluation/?
```

## Verify Script

```bash
cd agents/irregular
python 2_detection/2_detection.py 5-1 --data-dir data/irregular
python 3_segmentation/segmentation.py 5-1 --data-dir data/irregular
python evaluation.py 5-1 --data-dir data/irregular
cat data/irregular/5-1/evaluation/performance.md
```
