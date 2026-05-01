# Baseline vs Warm-start

| metric | default | warm-start |
|---|---:|---:|
| median mIoU | 0.1116 | 0.1116 |
| min mIoU | 0.0588 | 0.0588 |
| max mIoU | 0.1299 | 0.1299 |

| provider | model |
|---|---|
| anthropic | claude-sonnet-4-6 |

## Per-ring deltas

| tunnel | ring | default mIoU | warm-start mIoU | delta |
|---|---:|---:|---:|---:|
| 4-1 | 116 | 0.1299 | 0.1299 | +0.0000 |
| 4-6 | 283 | 0.1126 | 0.1126 | +0.0000 |
| 4-8 | 337 | 0.0761 | 0.0760 | -0.0001 |
| 4-9 | 366 | 0.1162 | 0.1162 | +0.0000 |
| 5-3 | 190 | 0.0588 | 0.0588 | +0.0000 |
| 5-5 | 258 | 0.1107 | 0.1107 | +0.0000 |
