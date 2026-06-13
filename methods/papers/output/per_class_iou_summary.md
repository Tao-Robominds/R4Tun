# Per-class IoU summary (from existing performance.md)

Mean IoU per class, aggregated over tunnels in each family.

- **regular_all**: alternated (1-*, 2-*) ∪ continuous (3-*), n=13
- **complex**: 4-*, 5-*, n=17

Note: `sam4tun` rows for `gemini3flash` use the shared GPT snapshot baseline (same as comparison journals).

## LLM: gpt5.4

### Family: regular_all

| class | sam4tun | memory | memory+state | memory+state+knowledge |
|---|---|---|---|---|
| Background | 0.640 | 0.584 | 0.724 | 0.743 |
| A1-block | 0.253 | 0.273 | 0.568 | 0.543 |
| A2-block | 0.150 | 0.147 | 0.347 | 0.351 |
| A3-block | 0.286 | 0.244 | 0.545 | 0.534 |
| B1-block | 0.261 | 0.251 | 0.562 | 0.516 |
| B2-block | 0.256 | 0.303 | 0.552 | 0.540 |
| K-block | 0.192 | 0.212 | 0.387 | 0.329 |

### Family: complex

| class | sam4tun | memory | memory+state | memory+state+knowledge |
|---|---|---|---|---|
| Background | 0.337 | 0.408 | 0.522 | 0.582 |
| A1-block | 0.000 | 0.049 | 0.035 | 0.124 |
| A2-block | 0.000 | 0.057 | 0.059 | 0.115 |
| A3-block | 0.000 | 0.050 | 0.119 | 0.138 |
| A4-block | 0.000 | 0.057 | 0.096 | 0.128 |
| B1-block | 0.000 | 0.073 | 0.053 | 0.099 |
| B2-block | 0.000 | 0.000 | 0.000 | 0.095 |
| K-block | 0.000 | 0.118 | 0.124 | 0.138 |

## LLM: opus4.6

### Family: regular_all

| class | sam4tun | memory | memory+state | memory+state+knowledge |
|---|---|---|---|---|
| Background | 0.640 | 0.559 | 0.741 | 0.751 |
| A1-block | 0.253 | 0.276 | 0.537 | 0.542 |
| A2-block | 0.150 | 0.159 | 0.380 | 0.420 |
| A3-block | 0.286 | 0.259 | 0.519 | 0.550 |
| B1-block | 0.261 | 0.206 | 0.527 | 0.540 |
| B2-block | 0.256 | 0.232 | 0.534 | 0.558 |
| K-block | 0.192 | 0.129 | 0.373 | 0.386 |

### Family: complex

| class | sam4tun | memory | memory+state | memory+state+knowledge |
|---|---|---|---|---|
| Background | 0.337 | 0.358 | 0.513 | 0.520 |
| A1-block | 0.000 | 0.005 | 0.128 | 0.155 |
| A2-block | 0.000 | 0.006 | 0.143 | 0.116 |
| A3-block | 0.000 | 0.017 | 0.092 | 0.119 |
| A4-block | 0.000 | 0.019 | 0.100 | 0.104 |
| B1-block | 0.000 | 0.001 | 0.084 | 0.135 |
| B2-block | 0.000 | 0.000 | 0.000 | 0.043 |
| K-block | 0.000 | 0.034 | 0.183 | 0.159 |

## LLM: gemini3flash

### Family: regular_all

| class | sam4tun | memory | memory+state | memory+state+knowledge |
|---|---|---|---|---|
| Background | 0.640 | 0.623 | 0.741 | 0.729 |
| A1-block | 0.253 | 0.367 | 0.589 | 0.500 |
| A2-block | 0.150 | 0.252 | 0.411 | 0.364 |
| A3-block | 0.286 | 0.368 | 0.554 | 0.484 |
| B1-block | 0.261 | 0.267 | 0.534 | 0.510 |
| B2-block | 0.256 | 0.338 | 0.539 | 0.516 |
| K-block | 0.192 | 0.195 | 0.347 | 0.362 |

### Family: complex

| class | sam4tun | memory | memory+state | memory+state+knowledge |
|---|---|---|---|---|
| Background | 0.337 | 0.396 | 0.484 | 0.575 |
| A1-block | 0.000 | 0.039 | 0.087 | 0.158 |
| A2-block | 0.000 | 0.054 | 0.083 | 0.135 |
| A3-block | 0.000 | 0.052 | 0.067 | 0.109 |
| A4-block | 0.000 | 0.056 | 0.074 | 0.071 |
| B1-block | 0.000 | 0.023 | 0.081 | 0.130 |
| B2-block | 0.000 | 0.000 | 0.000 | 0.028 |
| K-block | 0.000 | 0.087 | 0.133 | 0.178 |
