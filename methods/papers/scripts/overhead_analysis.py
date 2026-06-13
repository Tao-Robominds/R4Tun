#!/usr/bin/env python3
"""Quantify LLM adaptation overhead vs SAM4Tun pipeline execution time.

Parses orchestrator.log files under logs/{tunnel}/rerun_*/{cond}/{model}/ to extract:
  - per-stage LLM API call latency (the adaptation overhead)
  - per-stage pipeline "took X seconds" prints (SAM4Tun execution cost)
Reports per-model means, and the adaptation:pipeline ratio.
"""
import re
import statistics as st
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
LOGS = ROOT / "logs"
STAGES = ["unfolding", "denoising", "enhancing", "detecting", "sam"]

api_by_model_stage = defaultdict(list)   # (model, stage) -> [seconds]
full_tunnel_llm = defaultdict(list)      # model -> [sum of 5 stage api times] (complete logs only)
pipeline_took = []                       # all "took X seconds" pipeline-internal times

for log in LOGS.glob("*/rerun_*/*/*/orchestrator.log"):
    parts = log.parts
    model = parts[-2]
    text = log.read_text()
    # per-stage API call latency
    stage_times = {}
    cur = None
    for line in text.splitlines():
        ms = re.match(r"--- Stage: (\w+) ---", line.strip())
        if ms:
            cur = ms.group(1)
            continue
        ma = re.search(r"API call:\s*([\d.]+)s", line)
        if ma and cur:
            sec = float(ma.group(1))
            stage_times[cur] = sec
            api_by_model_stage[(model, cur)].append(sec)
        mp = re.search(r"took ([\d.]+) seconds", line)
        if mp:
            pipeline_took.append(float(mp.group(1)))
    if all(s in stage_times for s in STAGES):
        full_tunnel_llm[model].append(sum(stage_times[s] for s in STAGES))

print("=" * 64)
print("LLM ADAPTATION LATENCY (per stage, seconds), by model")
print("=" * 64)
print(f"{'model':14s} {'stage':12s} {'n':>3s} {'mean':>7s} {'min':>7s} {'max':>7s}")
for (model, stage), vals in sorted(api_by_model_stage.items()):
    print(f"{model:14s} {stage:12s} {len(vals):3d} {st.mean(vals):7.1f} {min(vals):7.1f} {max(vals):7.1f}")

print("\n" + "=" * 64)
print("TOTAL LLM ADAPTATION TIME PER TUNNEL (5 stages summed)")
print("=" * 64)
print(f"{'model':14s} {'n_full_runs':>11s} {'mean_s':>8s} {'mean_min':>9s} {'range_s':>14s}")
for model, vals in sorted(full_tunnel_llm.items()):
    print(
        f"{model:14s} {len(vals):11d} {st.mean(vals):8.1f} {st.mean(vals)/60:9.2f} "
        f"{min(vals):6.0f}-{max(vals):<6.0f}"
    )

# Estimate per-stage mean across models (overall adaptation)
all_full = [v for vals in full_tunnel_llm.values() for v in vals]
if all_full:
    print(f"\nOverall mean LLM adaptation per tunnel: {st.mean(all_full):.0f}s "
          f"({st.mean(all_full)/60:.1f} min) over {len(all_full)} full runs")

print("\n" + "=" * 64)
print("SAM4Tun PIPELINE-INTERNAL TIMED OPS ('took X s' prints)")
print("=" * 64)
if pipeline_took:
    print(f"n={len(pipeline_took)}  mean={st.mean(pipeline_took):.1f}s  "
          f"min={min(pipeline_took):.1f}s  max={max(pipeline_took):.1f}s  "
          f"sum={sum(pipeline_took):.0f}s")
