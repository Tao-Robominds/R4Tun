# Step 05 Output — GT Warm Start

**Depends on:** Step 04 output (full parameter inventory), `data/5-1.txt` (GT columns `segment`, `ring`), existing `data/irregular/5-1/` preprocessing outputs.

**Runtime path:** `data/irregular/5-1/gt_warm_start/`

---

## 1. Artifact List

| Artifact | Description |
|----------|-------------|
| `parameters_preprocessing.json` | GT-derived preprocessing parameters (schema: `agents/irregular/1_preprocessing` JSON). |
| `parameters_detection.json` | GT-derived detection parameters (schema: `agents/irregular/2_detection` JSON). |
| `parameters_segmentation.json` | GT-derived segmentation parameters (schema: `agents/irregular/3_segmentation` JSON). |
| `parameter_comparison.md` | Table: (parameter, old value, GT-derived value, delta, method) for every parameter. |
| `baseline_report.md` | mIoU, OA, F1, per-class IoU; comparison vs previous baseline; gap analysis. |
| `final.csv` | Pipeline output (point cloud with `pred` column) from full run with GT-derived params. |

Optional: copies of intermediate outputs (e.g. `enhanced.csv`, `depth_map.png`, `all_segments.csv`, `boundaries_per_ring.json`) may be kept under the same directory for traceability.

---

## 2. Parameter JSONs

- **Source of truth for schema:** `agents/irregular/{stage}/parameters/5-1/` existing JSONs.
- **Values:** All BO-critical and tunnel-physical parameters must be set from GT reverse-engineering (see step 05 actions 1–3). Safe-fixed parameters are not written to JSON.
- **Validation:** After writing, running the full pipeline with these JSONs (copied to `agents/irregular/*/parameters/5-1/`) must produce `final.csv` without missing-column or type errors.

---

## 3. parameter_comparison.md — Required Format

- **Section 1:** Preprocessing — table with columns: `Parameter`, `Old value`, `GT-derived value`, `Delta`, `Method`.
- **Section 2:** Detection — same table.
- **Section 3:** Segmentation — same table.
- Every parameter present in the step 04 inventory (BO-critical and tunnel-physical) must appear. Use "—" or "N/A" for delta when not applicable (e.g. boolean, or no old value).
- **Method** should be one line per parameter (e.g. "min(r) over GT surface − 0.01", "median K half-height across rings").

---

## 4. baseline_report.md — Required Format

- **Overall metrics:** mIoU, OA, F1 (one line or short table).
- **Per-class IoU:** Table with columns `Class` (or segment id), `IoU`.
- **Comparison vs previous baseline:** Previous mIoU (and source: e.g. "current 5-1 params" or "boundary-based 0.793") vs GT warm start mIoU; delta.
- **Gap analysis (optional but recommended):** Which parameter groups (preprocessing / detection / segmentation) contributed most to the gain; which had negligible effect; any safe-fixed validation outcome (e.g. "FIXED_* retained; no >0.5% GT loss").

---

## 5. Verification

### Verify prompt

1. Are all 3 parameter JSONs present with GT-derived values?
2. Does `parameter_comparison.md` show old vs GT-derived for every parameter?
3. Was the full pipeline run with GT-derived params (`final.csv` exists)?
4. Does `baseline_report.md` show mIoU, OA, per-class IoU?
5. Were safe-fixed params validated against GT retention?
6. Is the baseline mIoU higher than the previous baseline?

### Verify script

```bash
python methods/plans/scripts/verify_step.py --root data/irregular/5-1/gt_warm_start --step 05
```

---

## 6. Success Criteria

- All artifacts in §1 exist under `data/irregular/5-1/gt_warm_start/`.
- `parameter_comparison.md` and `baseline_report.md` follow the format in §3 and §4.
- GT warm start mIoU is strictly greater than the pre–warm start baseline (or documented if equal, with reason).
- Safe-fixed validation is recorded (in baseline_report or a one-line note in parameter_comparison); any FIXED_* that caused >0.5% GT loss is flagged for promotion to BO-critical.
