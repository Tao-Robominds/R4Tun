# Detection layout BO (`bo/`)

Design-time GP-BO over **k_y + per-block A/B offsets + r_surface_min** with GT mIoU labels.
All writes go to `logs/<run_id>/`; corpora under `data/` are read-only.

## Layout

```
bo/
  run_layout_bo.py          # single entry point (all BO trial modes)
  lib/
    ceiling_gate.py         # GT layout, sandbox, ceiling sweep
    layout_bo.py            # encode/decode, GP-BO, ceiling-push loop
    manifest.py             # manifest loading, panel summaries
    verify.py               # GT round-trip smoke test
```

Post-BO analysis and corpus promotion live under `methods/plans/scripts/` (not BO trials).

## Search space

`x = [k_y_frac, off_frac[K], off_frac[B1], …, r_frac]` — **per-block offset fractions**, not cumulative arc widths.

## Commands

**Verify encoding before a run:**

```bash
./venv/bin/python bo/run_layout_bo.py verify \
  --manifest data/minimum/MANIFEST.json \
  --source-dir data/minimum
```

**Ceiling-push (manifest-driven, any N rings):**

```bash
./venv/bin/python bo/run_layout_bo.py ceiling-push \
  --manifest data/minimum/MANIFEST.json \
  --source-dir data/minimum \
  --run-root logs/minimum_detection_bo_v1 \
  --target-regret 0.1 \
  --eval-chunk 128 \
  --max-total-evals 1024
```

Single ring:

```bash
./venv/bin/python bo/run_layout_bo.py ceiling-push \
  --manifest data/minimum/MANIFEST.json \
  --source-dir data/minimum \
  --run-root logs/minimum_detection_bo_v1 \
  --only-ring 1-4/r206 \
  --target-regret 0.1
```

**6-ring experience collection (fixed 64 evals/ring):**

```bash
./venv/bin/python bo/run_layout_bo.py experience \
  --manifest data/bo/MANIFEST.json \
  --source-dir data/bo \
  --run-root logs/calib_detection_bo_v1 \
  --n-evals 64
```

**Single ring without manifest:**

```bash
./venv/bin/python bo/run_layout_bo.py experience \
  --source-dir data/bo \
  --run-root logs/calib_detection_bo_v1 \
  --tunnel-id 1-1 --ring-id 18 --n-evals 64
```

## Corpora

| Corpus | Path | Typical mode |
|---|---|---|
| Minimum (hard cases) | `data/minimum/MANIFEST.json` | `ceiling-push` |
| 6-ring BO calib | `data/bo/MANIFEST.json` | `experience` |

## Outputs per ring

`logs/<run_id>/<tunnel>/r<N>/`: `bo_trials.csv`, `ceiling.json`, `gt_layout.json`, `best_bo_trial.json`, `ceiling_push_report.json` (ceiling-push mode).

## Post-run analysis

```bash
./venv/bin/python methods/plans/scripts/analyze_detection_bo_experience_v1.py \
  --run-root logs/calib_detection_bo_v1
```
