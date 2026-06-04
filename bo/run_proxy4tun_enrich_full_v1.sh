#!/usr/bin/env bash
# Full Proxy4Tun trial enrichment for proxy_train_lk_v2 (read-only corpus).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./venv/bin/python
OUT=logs/proxy4tun/proxy_train_lk_v2
CORPUS=data/bo_calibration

mkdir -p "$OUT/replay"

echo "=== Enrich stream_l ==="
$PY bo/enrich_proxy4tun_trials_v1.py \
  --trials logs/proxy4tun/stream_l/bo_trials.csv \
  --out-csv "$OUT/records_L_enriched.csv" \
  --replay-root "$OUT/replay/stream_l" \
  --corpus "$CORPUS"

echo "=== Enrich stream_k ==="
$PY bo/enrich_proxy4tun_trials_v1.py \
  --trials logs/proxy4tun/stream_k/bo_trials.csv \
  --out-csv "$OUT/records_K_enriched.csv" \
  --replay-root "$OUT/replay/stream_k" \
  --corpus "$CORPUS"

echo "=== Enrich stream_full ==="
$PY bo/enrich_proxy4tun_trials_v1.py \
  --trials logs/proxy4tun/stream_full/bo_trials.csv \
  --out-csv "$OUT/records_LK_joint_enriched.csv" \
  --replay-root "$OUT/replay/stream_full" \
  --corpus "$CORPUS"

echo "=== Build LK_concat ==="
$PY - <<'PY'
import pandas as pd
from pathlib import Path
out = Path("logs/proxy4tun/proxy_train_lk_v2")
l = pd.read_csv(out / "records_L_enriched.csv")
k = pd.read_csv(out / "records_K_enriched.csv")
l["axis_source"] = "layout"
k["axis_source"] = "k"
pd.concat([l, k], ignore_index=True).to_csv(out / "records_LK_concat_enriched.csv", index=False)
print(f"Wrote concat {len(l)+len(k)} rows")
PY

echo "=== Done ==="
