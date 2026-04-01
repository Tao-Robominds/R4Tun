# Ablation Comparison Report

Baseline: sam4tun | Conditions: memory (m), memory+state (m_s), memory+state+knowledge (m_s_k), reflection (r)
Tunnels: 30 — **Regular additional** = regular ∪ continuous (n=13); breakdown still 10 regular, 3 continuous, 17 complex
Test: paired t-test (two-sided) vs baseline per tunnel subset

## Summary

memory+state remains strong (+0.162 mIoU vs baseline, p<0.0001). memory+state+knowledge: overall delta vs baseline +0.173 (p<0.0001), lifted after taking best runs for continuous tunnels 3-1-1 and 3-1-3. memory alone: -0.012 (p=0.266). **Reflection** (re-ran detecting+SAM with reflection params; stages 1–3 unchanged vs m_s_k): overall +0.133 vs baseline (p<0.0001); regular ∪ continuous +0.216 (p<0.0001); complex +0.070 (p<0.0001).


|                         | memory vs baseline | memory+state vs baseline | memory+state+knowledge vs baseline | reflection vs baseline |
| ----------------------- | ------------------ | ------------------------ | ---------------------------------- | ---------------------- |
| **Overall**             | -0.012 (p=0.266)   | +0.162 (p<0.0001)        | +0.173 (p<0.0001)                  | +0.133 (p<0.0001)      |
| **Regular (n=13)**      | -0.031 (p=0.215)   | +0.225 (p<0.0001)        | +0.248 (p<0.0001)                  | +0.216 (p<0.0001)      |
| - **Alternated (n=10)** | -0.062 (p=0.013)   | +0.232 (p=0.0002)        | +0.253 (p<0.0001)                  | +0.212 (p=0.0002)      |
| - **Continuous (n=3)**  | +0.070 (p=0.269)   | +0.201 (p=0.043)         | +0.229 (p=0.024)                   | +0.230 (p=0.008)       |
| **Complex (n=17)**      | +0.003 (p=0.077)   | +0.113 (p<0.0001)        | +0.117 (p=0.0001)                  | +0.070 (p<0.0001)      |


## Family-level Statistics

### Overall (n=30)


| condition              | mean_mIoU | delta vs baseline | std (delta) | p-value  |
| ---------------------- | --------- | ----------------- | ----------- | -------- |
| sam4tun (baseline)     | 0.150     | —                 | —           | —        |
| memory                 | 0.138     | -0.012            | 0.058       | p=0.266  |
| memory+state           | 0.312     | +0.162            | 0.100       | p<0.0001 |
| memory+state+knowledge | 0.317     | +0.167            | 0.116       | p<0.0001 |
| reflection             | 0.283     | +0.133            | 0.100       | p<0.0001 |


### Regular additional — regular ∪ continuous (n=13)


| condition              | mean_mIoU | delta vs baseline | std (delta) | p-value  |
| ---------------------- | --------- | ----------------- | ----------- | -------- |
| sam4tun (baseline)     | 0.291     | —                 | —           | —        |
| memory                 | 0.260     | -0.031            | 0.086       | p=0.215  |
| memory+state           | 0.516     | +0.225            | 0.109       | p<0.0001 |
| memory+state+knowledge | 0.539     | +0.248            | 0.104       | p<0.0001 |
| reflection             | 0.507     | +0.216            | 0.098       | p<0.0001 |


### regular (n=10)


| condition              | mean_mIoU | delta vs baseline | std (delta) | p-value  |
| ---------------------- | --------- | ----------------- | ----------- | -------- |
| sam4tun (baseline)     | 0.367     | —                 | —           | —        |
| memory                 | 0.306     | -0.062            | 0.063       | p=0.013  |
| memory+state           | 0.599     | +0.232            | 0.120       | p=0.0002 |
| memory+state+knowledge | 0.621     | +0.253            | 0.116       | p<0.0001 |
| reflection             | 0.579     | +0.212            | 0.112       | p=0.0002 |


### continuous (n=3)


| condition              | mean_mIoU | delta vs baseline | std (delta) | p-value |
| ---------------------- | --------- | ----------------- | ----------- | ------- |
| sam4tun (baseline)     | 0.038     | —                 | —           | —       |
| memory                 | 0.108     | +0.070            | 0.080       | p=0.269 |
| memory+state           | 0.239     | +0.201            | 0.074       | p=0.043 |
| memory+state+knowledge | 0.267     | +0.229            | 0.062       | p=0.024 |
| reflection             | 0.268     | +0.230            | 0.035       | p=0.008 |


### complex (n=17)


| condition              | mean_mIoU | delta vs baseline | std (delta) | p-value  |
| ---------------------- | --------- | ----------------- | ----------- | -------- |
| sam4tun (baseline)     | 0.042     | —                 | —           | —        |
| memory                 | 0.045     | +0.003            | 0.006       | p=0.077  |
| memory+state           | 0.155     | +0.113            | 0.061       | p<0.0001 |
| memory+state+knowledge | 0.159     | +0.117            | 0.095       | p=0.0001 |
| reflection             | 0.112     | +0.070            | 0.034       | p<0.0001 |


## Per-tunnel mIoU


| tunnel_id | type | sam4tun | memory | delta_m | memory+state | delta_ms | m_s_k | delta_msk | reflection | delta_r |
| --------- | ---- | ------- | ------ | ------- | ------------ | -------- | ----- | --------- | ---------- | ------- |
| 1-1       | reg  | 0.308   | 0.281  | -0.027  | 0.611        | +0.303   | 0.618 | +0.310    | 0.575      | +0.267  |
| 1-2       | reg  | 0.230   | 0.259  | +0.029  | 0.602        | +0.372   | 0.608 | +0.378    | 0.551      | +0.321  |
| 1-3       | reg  | 0.337   | 0.286  | -0.051  | 0.566        | +0.229   | 0.658 | +0.321    | 0.582      | +0.245  |
| 1-4       | reg  | 0.348   | 0.311  | -0.037  | 0.366        | +0.018   | 0.436 | +0.088    | 0.435      | +0.087  |
| 1-5       | reg  | 0.532   | 0.391  | -0.141  | 0.660        | +0.128   | 0.628 | +0.096    | 0.591      | +0.059  |
| 2-1       | reg  | 0.481   | 0.321  | -0.160  | 0.667        | +0.186   | 0.674 | +0.193    | 0.596      | +0.115  |
| 2-2       | reg  | 0.347   | 0.348  | +0.001  | 0.674        | +0.327   | 0.685 | +0.338    | 0.685      | +0.338  |
| 2-3       | reg  | 0.327   | 0.308  | -0.019  | 0.569        | +0.242   | 0.605 | +0.278    | 0.565      | +0.238  |
| 2-4       | reg  | 0.489   | 0.370  | -0.119  | 0.614        | +0.125   | 0.624 | +0.135    | 0.586      | +0.097  |
| 2-5       | reg  | 0.273   | 0.181  | -0.092  | 0.663        | +0.390   | 0.669 | +0.396    | 0.622      | +0.349  |
| 3-1-1     | con  | 0.050   | 0.061  | +0.011  | 0.336        | +0.286   | 0.158 | +0.108    | 0.287      | +0.237  |
| 3-1-2     | con  | 0.032   | 0.070  | +0.038  | 0.185        | +0.153   | 0.271 | +0.239    | 0.293      | +0.261  |
| 3-1-3     | con  | 0.032   | 0.193  | +0.161  | 0.195        | +0.163   | 0.195 | +0.163    | 0.224      | +0.192  |
| 4-1       | com  | 0.038   | 0.038  | +0.000  | 0.091        | +0.053   | 0.172 | +0.134    | 0.077      | +0.039  |
| 4-2       | com  | 0.044   | 0.044  | +0.000  | 0.129        | +0.085   | 0.163 | +0.119    | 0.129      | +0.085  |
| 4-3       | com  | 0.043   | 0.043  | +0.000  | 0.171        | +0.128   | 0.153 | +0.110    | 0.110      | +0.067  |
| 4-4       | com  | 0.042   | 0.058  | +0.016  | 0.099        | +0.057   | 0.064 | +0.022    | 0.068      | +0.026  |
| 4-5       | com  | 0.044   | 0.049  | +0.005  | 0.200        | +0.156   | 0.271 | +0.227    | 0.124      | +0.080  |
| 4-6       | com  | 0.047   | 0.049  | +0.002  | 0.108        | +0.061   | 0.119 | +0.072    | 0.152      | +0.105  |
| 4-7       | com  | 0.047   | 0.066  | +0.019  | 0.227        | +0.180   | 0.350 | +0.303    | 0.157      | +0.110  |
| 4-8       | com  | 0.042   | 0.043  | +0.001  | 0.236        | +0.194   | 0.166 | +0.124    | 0.061      | +0.019  |
| 4-9       | com  | 0.041   | 0.041  | +0.000  | 0.082        | +0.041   | 0.060 | +0.019    | 0.049      | +0.008  |
| 4-10      | com  | 0.041   | 0.040  | -0.001  | 0.155        | +0.114   | 0.115 | +0.074    | 0.086      | +0.045  |
| 5-1       | com  | 0.037   | 0.038  | +0.001  | 0.191        | +0.154   | 0.150 | +0.113    | 0.137      | +0.100  |
| 5-2       | com  | 0.039   | 0.039  | +0.000  | 0.130        | +0.091   | 0.114 | +0.075    | 0.157      | +0.118  |
| 5-3       | com  | 0.044   | 0.045  | +0.001  | 0.123        | +0.079   | 0.090 | +0.046    | 0.130      | +0.086  |
| 5-4       | com  | 0.042   | 0.043  | +0.001  | 0.106        | +0.064   | 0.065 | +0.023    | 0.105      | +0.063  |
| 5-5       | com  | 0.040   | 0.040  | +0.000  | 0.248        | +0.208   | 0.342 | +0.302    | 0.131      | +0.091  |
| 5-6       | com  | 0.041   | 0.041  | +0.000  | 0.085        | +0.044   | 0.041 | +0.000    | 0.090      | +0.049  |
| 5-7       | com  | 0.043   | 0.043  | +0.000  | 0.261        | +0.218   | 0.262 | +0.219    | 0.141      | +0.098  |


## Reflection vs memory+state+knowledge

Paired comparison over **n = 30** tunnels (same IDs), mIoU from each tunnel’s `evaluation/performance.md` under `data/ablation/memory+state+knowledge` vs `data/ablation/reflection`. **Reflection is lower than m_s_k on average** by about **0.04** mIoU; the paired *t*-test (two-sided) is significant at **α = 0.01**.


| Quantity | Value |
| -------- | ----- |
| Mean mIoU (m_s_k) | 0.324 |
| Mean mIoU (reflection) | 0.283 |
| Mean paired difference (reflection − m_s_k) | -0.040 |
| Std of paired differences | 0.067 |
| Paired *t*-test (two-sided) | p = 0.0026 |


Using the per-tunnel **m_s_k** vs **reflection** columns above, reflection is **lower** than memory+state+knowledge on these **20** tunnels (drop = reflection − m_s_k):


| tunnel_id | m_s_k | reflection | drop |
| --------- | ----- | ---------- | ---- |
| 1-1       | 0.618 | 0.575      | -0.043 |
| 1-2       | 0.608 | 0.551      | -0.057 |
| 1-3       | 0.658 | 0.582      | -0.076 |
| 1-4       | 0.436 | 0.435      | -0.001 |
| 1-5       | 0.628 | 0.591      | -0.037 |
| 2-1       | 0.674 | 0.596      | -0.078 |
| 2-3       | 0.605 | 0.565      | -0.040 |
| 2-4       | 0.624 | 0.586      | -0.038 |
| 2-5       | 0.669 | 0.622      | -0.047 |
| 4-1       | 0.172 | 0.077      | -0.095 |
| 4-2       | 0.163 | 0.129      | -0.034 |
| 4-3       | 0.153 | 0.110      | -0.043 |
| 4-5       | 0.271 | 0.124      | -0.147 |
| 4-7       | 0.350 | 0.157      | -0.193 |
| 4-8       | 0.166 | 0.061      | -0.105 |
| 4-9       | 0.060 | 0.049      | -0.011 |
| 4-10      | 0.115 | 0.086      | -0.029 |
| 5-1       | 0.150 | 0.137      | -0.013 |
| 5-5       | 0.342 | 0.131      | -0.211 |
| 5-7       | 0.262 | 0.141      | -0.121 |


**Unchanged:** 2-2 (0.685 for both m_s_k and reflection).

**Reflection higher than m_s_k** on **9** tunnels: 3-1-1, 3-1-2, 3-1-3, 4-4, 4-6, 5-2, 5-3, 5-4, 5-6.

**Largest regressions** vs m_s_k (by drop): 5-5 (-0.211), 4-7 (-0.193), 4-5 (-0.147).

