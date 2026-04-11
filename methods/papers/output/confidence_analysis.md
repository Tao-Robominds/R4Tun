# Confidence analysis (bootstrap CI, Cohen's d, sign tests, spread)

Source: per-tunnel mIoU in `methods/journals/comparison_*.md`. Bootstrap: resample tunnels with replacement (10 000 iterations), mean of paired ΔmIoU each draw; report 2.5th and 97.5th percentiles.

## GPT-5.4

### Bootstrap 95% CI on mean paired ΔmIoU (n=30 tunnels, 10 000 resamples)

| Contrast | mean Δ | 95% CI | paired Cohen's d |
|----------|--------|--------|------------------|
| memory − baseline | +0.032 | [0.005, 0.058] | 0.424 |
| memory+state − baseline | +0.149 | [0.110, 0.190] | 1.323 |
| m+s+k − baseline | +0.171 | [0.140, 0.203] | 1.938 |
| m+s+k − memory+state (knowledge increment) | +0.021 | [-0.011, 0.049] | 0.249 |

### Knowledge increment: per-tunnel sign (m+s+k − m+s)

- Tunnels with strictly positive increment: **21/30** (one-sided binomial vs p=0.5, p=0.02139)
- Complex only (n=17): **15/17** positive (p=0.001175)

## Claude Opus 4.6

### Bootstrap 95% CI on mean paired ΔmIoU (n=30 tunnels, 10 000 resamples)

| Contrast | mean Δ | 95% CI | paired Cohen's d |
|----------|--------|--------|------------------|
| memory − baseline | -0.006 | [-0.027, 0.014] | -0.108 |
| memory+state − baseline | +0.162 | [0.126, 0.198] | 1.609 |
| m+s+k − baseline | +0.178 | [0.139, 0.216] | 1.609 |
| m+s+k − memory+state (knowledge increment) | +0.016 | [-0.003, 0.034] | 0.314 |

### Knowledge increment: per-tunnel sign (m+s+k − m+s)

- Tunnels with strictly positive increment: **21/30** (one-sided binomial vs p=0.5, p=0.02139)
- Complex only (n=17): **10/17** positive (p=0.3145)

## Gemini 3 Flash

### Bootstrap 95% CI on mean paired ΔmIoU (n=30 tunnels, 10 000 resamples)

| Contrast | mean Δ | 95% CI | paired Cohen's d |
|----------|--------|--------|------------------|
| memory − baseline | +0.049 | [0.006, 0.101] | 0.364 |
| memory+state − baseline | +0.152 | [0.117, 0.189] | 1.458 |
| m+s+k − baseline | +0.163 | [0.122, 0.203] | 1.435 |
| m+s+k − memory+state (knowledge increment) | +0.011 | [-0.028, 0.046] | 0.106 |

### Knowledge increment: per-tunnel sign (m+s+k − m+s)

- Tunnels with strictly positive increment: **19/30** (one-sided binomial vs p=0.5, p=0.1002)
- Complex only (n=17): **14/17** positive (p=0.006363)

## Aggregated across three LLMs (mean of per-LLM statistics)

### Table 4a inputs: overall (n=30 per LLM)

| Metric | sam4tun | memory | memory+state | m+s+k |
|--------|---------|--------|--------------|-------|
| Mean mIoU | 0.150 | 0.175 | 0.304 | 0.320 |
| Std of per-tunnel mIoU | 0.166 | 0.136 | 0.228 | 0.218 |
| Min tunnel mIoU | 0.032 | 0.042 | 0.082 | 0.072 |
| Max tunnel mIoU | 0.532 | 0.471 | 0.682 | 0.679 |
| Δ min mIoU vs baseline (floor lift, mean across LLMs) | — | +0.010 | +0.050 | +0.040 |

### Within-family std of per-tunnel mIoU (mean across 3 LLMs)

| Family | n | sam4tun | memory | memory+state | m+s+k |
|--------|---|---------|--------|--------------|-------|
| Regular ∪ continuous | 13 | 0.168 | 0.121 | 0.170 | 0.190 |
| Complex | 17 | 0.003 | 0.032 | 0.047 | 0.074 |

## Cross-LLM summary: knowledge increment bootstrap CI

- **GPT-5.4:** mean Δ = +0.021, 95% CI [-0.011, 0.049]
- **Claude Opus 4.6:** mean Δ = +0.016, 95% CI [-0.003, 0.034]
- **Gemini 3 Flash:** mean Δ = +0.011, 95% CI [-0.028, 0.046]
