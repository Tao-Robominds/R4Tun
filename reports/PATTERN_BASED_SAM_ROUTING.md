# Pattern-Based SAM Routing

Stage 5 (SAM) selects the script from `pattern_type.json`:

| Pattern             | Tunnels (examples) | SAM script              |
|---------------------|--------------------|-------------------------|
| **simple_staggered**| T1, T2 (1-4, 2-2)  | `4-2_sam.py`            |
| **continuous**      | T3 (3-1)           | `4-2_sam.py`            |
| **complex_staggered** | T4, T5 (4-1, 5-1) | `4-2_sam_wrap_around.py` |

T1, T2, T3 use **standard SAM** (no wraparound). T4, T5 use **wrap_around** (always wraparound).

## A/B: standard SAM vs sam_continuous (T3 / 3-1)

Same `data/3-1`, same `detected.csv`, same params:

| Script                | OA   | F1   | mIoU  |
|-----------------------|------|------|-------|
| **4-2_sam** (standard)| 0.799| 0.727| **0.594** |
| 4-2_sam_continuous    | 0.671| 0.604| 0.457 |

**Standard SAM outperforms sam_continuous** on 3-1. Continuous (T3) therefore uses `4-2_sam`, not `4-2_sam_continuous`.

## Router

`p4tun/sam_router.py` reads `{tunnel_dir}/pattern_type.json` and prints the script name. The pipeline invokes that script for stage 5.
