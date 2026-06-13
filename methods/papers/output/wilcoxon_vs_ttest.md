# Paired t-test vs Wilcoxon signed-rank (mIoU)

Per-tunnel paired comparisons vs **sam4tun** baseline. t-test and Wilcoxon are both two-sided. Source tables: `methods/journals/comparison_*.md`.

## GPT-5.4


### Overall (n=30)

| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |
|-----------|---|------------|---------------|----------|
| memory vs baseline | 30 | +0.032 | p=0.02754 | p=0.02365 |
| memory+state vs baseline | 30 | +0.149 | p<0.0001 | p<0.0001 |
| m_s_k vs baseline | 30 | +0.171 | p<0.0001 | p<0.0001 |

### Regular ∪ continuous (n=13)

| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |
|-----------|---|------------|---------------|----------|
| memory vs baseline | 13 | -0.004 | p=0.8904 | p=0.8926 |
| memory+state vs baseline | 13 | +0.235 | p<0.0001 | p=0.0002441 |
| m_s_k vs baseline | 13 | +0.217 | p<0.0001 | p=0.0002441 |

### Alternated (n=10)

| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |
|-----------|---|------------|---------------|----------|
| memory vs baseline | 10 | -0.043 | p=0.05057 | p=0.1055 |
| memory+state vs baseline | 10 | +0.240 | p=0.0002118 | p=0.001953 |
| m_s_k vs baseline | 10 | +0.231 | p<0.0001 | p=0.001953 |

### Continuous (n=3)

| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |
|-----------|---|------------|---------------|----------|
| memory vs baseline | 3 | +0.127 | p=0.04385 | p=0.25 |
| memory+state vs baseline | 3 | +0.218 | p=0.1184 | p=0.25 |
| m_s_k vs baseline | 3 | +0.168 | p=0.08822 | p=0.25 |

### Complex (n=17)

| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |
|-----------|---|------------|---------------|----------|
| memory vs baseline | 17 | +0.059 | p<0.0001 | p=0.0002919 |
| memory+state vs baseline | 17 | +0.084 | p<0.0001 | p=0.0002925 |
| m_s_k vs baseline | 17 | +0.135 | p<0.0001 | p=0.0002913 |

## Claude Opus 4.6


### Overall (n=30)

| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |
|-----------|---|------------|---------------|----------|
| memory vs baseline | 30 | -0.006 | p=0.5583 | p=0.9095 |
| memory+state vs baseline | 30 | +0.162 | p<0.0001 | p<0.0001 |
| m_s_k vs baseline | 30 | +0.178 | p<0.0001 | p<0.0001 |

### Regular ∪ continuous (n=13)

| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |
|-----------|---|------------|---------------|----------|
| memory vs baseline | 13 | -0.031 | p=0.1843 | p=0.1909 |
| memory+state vs baseline | 13 | +0.225 | p<0.0001 | p=0.0002441 |
| m_s_k vs baseline | 13 | +0.244 | p<0.0001 | p=0.0002441 |

### Alternated (n=10)

| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |
|-----------|---|------------|---------------|----------|
| memory vs baseline | 10 | -0.055 | p=0.02383 | p=0.02734 |
| memory+state vs baseline | 10 | +0.232 | p=0.0001729 | p=0.001953 |
| m_s_k vs baseline | 10 | +0.253 | p<0.0001 | p=0.001953 |

### Continuous (n=3)

| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |
|-----------|---|------------|---------------|----------|
| memory vs baseline | 3 | +0.049 | p=0.415 | p=0.5 |
| memory+state vs baseline | 3 | +0.201 | p=0.04254 | p=0.25 |
| m_s_k vs baseline | 3 | +0.213 | p=0.003275 | p=0.25 |

### Complex (n=17)

| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |
|-----------|---|------------|---------------|----------|
| memory vs baseline | 17 | +0.013 | p=0.004683 | p=0.007632 |
| memory+state vs baseline | 17 | +0.113 | p<0.0001 | p<0.0001 |
| m_s_k vs baseline | 17 | +0.127 | p<0.0001 | p<0.0001 |

## Gemini 3 Flash


### Overall (n=30)

| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |
|-----------|---|------------|---------------|----------|
| memory vs baseline | 30 | +0.049 | p=0.05591 | p=0.03143 |
| memory+state vs baseline | 30 | +0.152 | p<0.0001 | p<0.0001 |
| m_s_k vs baseline | 30 | +0.163 | p<0.0001 | p<0.0001 |

### Regular ∪ continuous (n=13)

| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |
|-----------|---|------------|---------------|----------|
| memory vs baseline | 13 | +0.053 | p=0.374 | p=0.7334 |
| memory+state vs baseline | 13 | +0.240 | p<0.0001 | p=0.0002441 |
| m_s_k vs baseline | 13 | +0.204 | p=0.0002489 | p=0.001709 |

### Alternated (n=10)

| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |
|-----------|---|------------|---------------|----------|
| memory vs baseline | 10 | -0.035 | p=0.3251 | p=0.25 |
| memory+state vs baseline | 10 | +0.223 | p<0.0001 | p=0.001953 |
| m_s_k vs baseline | 10 | +0.219 | p=0.00207 | p=0.009766 |

### Continuous (n=3)

| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |
|-----------|---|------------|---------------|----------|
| memory vs baseline | 3 | +0.346 | p=0.09975 | p=0.25 |
| memory+state vs baseline | 3 | +0.295 | p=0.01069 | p=0.25 |
| m_s_k vs baseline | 3 | +0.154 | p=0.006161 | p=0.25 |

### Complex (n=17)

| Condition | n | mean ΔmIoU | paired t-test | Wilcoxon |
|-----------|---|------------|---------------|----------|
| memory vs baseline | 17 | +0.046 | p<0.0001 | p=0.0002925 |
| memory+state vs baseline | 17 | +0.084 | p<0.0001 | p=0.0002919 |
| m_s_k vs baseline | 17 | +0.131 | p<0.0001 | p<0.0001 |
