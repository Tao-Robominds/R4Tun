# Can the drift be fixed by BO or do we need banded/template?

## What the current drift looks like

- **Mean drift: ~2021 px**, max ~3953 px.
- **GT:** Ring 0 has **high X** (~2361–2396), Ring 6 has **low X** (~172). So in the depth map, ring index increases as X decreases.
- **Detected (DBSCAN + physical):** K positions are **sorted by X** (line 702 in `2_detection.py`). So the **leftmost** cluster becomes ring 0 (X≈404) and the **rightmost** becomes ring 6 (X≈2229). That is the **opposite** of GT: detected ring 0 ≈ GT ring 6, detected ring 6 ≈ GT ring 0.

So the drift has two parts:

1. **Ring order inverted** – DBSCAN assigns ring 0 to the left (low X), GT has ring 0 on the right (high X).
2. **K X-positions wrong** – Even after reversing ring indices, K X would be off (e.g. detected “ring 6” at 2229 vs GT ring 0 at 2396; detected “ring 0” at 404 vs GT ring 6 at 172). So there is still ~200–400 px X error per ring plus Y from physical expansion.

## Can BO fix it?

- **BO on the current pipeline (DBSCAN + physical expansion)** can:
  - Tune binary/Hough/angle/DBSCAN parameters so that clusters move a bit.
  - **Not** fix the systematic ring-order inversion by parameter tuning alone; that needs a **code change** (e.g. reverse ring indices after clustering, or sort by `-X`).
  - Re‑introduce **per‑ring expansion** params (`k_to_b_r0..r6`, `ab_step_r0..r6`) and tune them to correct residual X/Y. That could bring drift down to a few hundred px if K positions are at least in the right “column” per ring.
- So: **BO alone is unlikely to fix ~2000 px mean drift** unless we first fix the **ring order** (and possibly switch to a method that places K in the right X band). After a structural fix (e.g. reversal + banded or better K placement), BO can then tune expansion and thresholds to reduce residual drift.

## Banded K detection

- **`calculate_k_positions_banded`**:
  - Puts ring **i** at a **fixed X**: `(i + 0.5) * W / ring_count` (uniform bands).
  - Sets **Y** from oblique line intersections (or crossings) inside that band.
- So **X is correct by construction** (up to a convention for which band is ring 0). Right now band 0 = left (low X), so we still need a **ring-index convention** (e.g. reverse so ring 0 = right side to match GT).
- Once convention is aligned, banded removes the huge X-misplacement from DBSCAN clustering and should **cut mean drift a lot** (e.g. to the order of hundreds of px), with Y coming from line geometry instead of physical expansion from a wrong K.

## Template expansion

- **`expand_k_with_template`** uses a 7-step template (K→B1→…→B2→K) and can rotate it to fit oblique line intersections. It improves **angular/step layout** around each K.
- It helps **after** K is in the right place (correct ring and roughly correct X). So: **banded (or DBSCAN + ring reversal) first, then template expansion** is the right order.

## Recommendation

| Approach | Fix ring order? | Fix K X? | Fix expansion (Y/steps)? | Expected drift after |
|----------|------------------|----------|---------------------------|----------------------|
| BO only (current pipeline) | No (needs code change) | Partly (tuning) | Partly (per‑ring params) | Likely still large (~1000+ px) |
| **Ring reversal + BO** | Yes (one code change) | Partly | Partly (per‑ring params) | Could reach few hundred px |
| **Banded + (optional reversal) + physical/template** | Yes (via convention) | Yes (by construction) | Physical or template | Likely **much lower** (hundreds or less) |
| **Banded + template + BO** | Yes | Yes | Template + BO on step_template | Best chance for minimal drift |

**Practical order:**

1. **Quick check:** In the detection pipeline, **reverse ring indices** so that the current “ring 0” (leftmost K) becomes ring 6, etc. Re-run detection and `compare_drift.py`. If mean drift drops to the order of a few hundred px, then BO (and possibly per‑ring expansion) may be enough.
2. **If drift stays large:** Switch to **banded** K detection (and set ring-order convention to match GT). Keep physical expansion first; if drift is still high, switch to **template** expansion. Then run BO on the remaining parameters (e.g. template steps, band margin, line-detection thresholds).

So: **the current ~2000 px drift is not fixable by BO alone**; we need at least a **ring-order fix**. **Banded + (optional) template** is the right structural change to get K positions and expansion in the right place; **BO** is then for fine-tuning.
