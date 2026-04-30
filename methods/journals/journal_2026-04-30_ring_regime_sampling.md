# Daily Journal — 30 April 2026

## Objective

Define the sampling philosophy for step 01 ring-regime discovery: choose ring proportions that support the paper claim while preserving a small sanity check on simpler regular tunnels.

---

## Decision

Use `data/subsets/` as the source pool for ring-regime discovery.

Sampling target:

| Pool | Role | Recommended share |
|------|------|-------------------|
| Irregular families `4-*`, `5-*` | Main paper target; BO/proxy/guardrail calibration | 80% |
| Regular families `1-*`, `2-*`, `3-*` | Sanity/generalisation check only | 20% |

Concrete panels:

| Panel | `4-*` | `5-*` | `1/2/3` regular | Total |
|-------|------:|------:|----------------:|------:|
| `panel_20` | 9 | 7 | 4 | 20 |
| `panel_30` | 14 | 10 | 6 | 30 |

---

## Rationale

The paper studies ring-wise BO and fixed-rule reflection for structurally difficult irregular tunnels. Therefore, the sampling distribution should be dominated by the target domain: irregular tunnel families `4-*` and `5-*`.

Regular tunnels are expected to be easier, but this should not be stated as a solved fact without evidence. A smaller regular subset is useful as a sanity/generalisation check: it tests whether the irregular-tuned framework degrades on simpler geometry without letting regular data drive the method.

This avoids two weak claims:

1. Overclaiming that success on irregular rings automatically proves success on regular rings.
2. Diluting the BO calibration set with simpler rings that do not express the failure modes studied in the paper.

---

## Scientific Framing

Use this wording:

> We sample primarily from irregular tunnel families because they define the target distribution and contain the structural failure modes studied in this paper. A smaller regular subset is included only as a sanity/generalisation check to test whether the framework degrades on simpler tunnel geometries.

Avoid this wording:

> If it works on irregular, it automatically works on regular.

---

## Implementation Implication

In `regime_sampling_panel.json`, regular rings should be explicitly flagged as sanity rows, for example:

```json
{
  "domain_role": "sanity_regular"
}
```

They should not be used for:

- BO optimisation
- proxy fitting or calibration
- guardrail threshold selection
- tuning memory construction for the irregular-target method

They can be used for:

- sanity evaluation
- out-of-distribution reporting
- qualitative evidence that the method does not break on simpler geometry
