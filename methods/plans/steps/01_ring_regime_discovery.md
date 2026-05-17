# 01 Ring-Condition Coverage and Dataset Roles

## Goal

Define the condition space used to justify ring-level optimisation and proxy confidence. Diversity is measured by tunnel/ring conditions, not just by the count of tunnels.

## Runtime Path

Sandbox path: `logs/{run_id}/01_ring_condition_coverage/`

## Inputs

- Canonical per-ring point clouds and baseline artifacts, read only.
- Existing six-tunnel BO/proxy experience.
- Intrinsic QA summaries and baseline failure notes, read only.
- GT labels only for split auditing and later mIoU evaluation.

## Condition Descriptors

Compute one descriptor row per ring before selecting any split:

- point density and density percentile;
- valid coverage, empty-band ratio, and largest missing region;
- diameter/segment-count proxy;
- assembly pattern, e.g. staggered or continuous;
- K-position span, segment order pattern, and boundary clarity;
- noise/clutter proxy and non-structural interference;
- baseline quality indicators and failure reason;
- candidate score spread when sampled candidates already exist.

Normalize numeric descriptors and keep categorical descriptors as strata.

## Dataset Roles

| Role | Purpose | Typical Size |
|------|---------|-------------:|
| One-shot seed | First labelled sparse/irregular ring used to start proxy learning | 1 ring |
| Few-shot BO set | Additional labelled rings chosen to expand condition coverage | 3-12 rings |
| Proxy validation set | Choose confidence definitions and test calibration | 30-50 rings |
| Held-out confidence test | Final unseen evaluation of confidence and runtime selection | 50+ rings preferred |
| Hard-negative bank | Cases where proxy confidence or ranking fails | grows during step 06 |

The same physical tunnel may contain multiple condition regimes. Conversely, two different tunnels may occupy a similar condition region. The paper should therefore claim coverage in descriptor space, not only tunnel count.

## One-Shot Seed

Start with a single labelled sparse/irregular ring before adding more shots.

Recommended seed: `4-6/r276`.

Rationale:

- irregular family-4 ring;
- sparse/depth-risk condition;
- weak deterministic-baseline result but not hopeless;
- existing v5 evidence shows candidate variation and recoverable improvement, so BO can learn both parameter sensitivity and proxy-feature movement.

Do not choose a completely collapsed ring as the first shot. The one-shot seed should expose a learnable relationship between parameter changes, observable features, and GT mIoU.

## Split Rules

1. Select the first one-shot ring from a sparse and irregular condition region.
2. Exclude BO rings from validation and held-out confidence testing.
3. Use max-min sampling in descriptor space for few-shot expansion.
4. Reserve some descriptor clusters as deliberately uncovered regions for out-of-distribution evaluation.
5. Freeze validation and held-out splits before proxy model comparison.

## Required Evidence

- descriptor coverage plot/table for all rings;
- condition clusters or bins;
- seed/few-shot/validation/held-out membership;
- explicitly marked covered and uncovered regions;
- reason why ring-level optimisation is needed: tunnel-level parameters do not generalise across within-tunnel ring variation.

## Outputs

- `ring_condition_descriptors.csv`
- `condition_clusters.csv`
- `dataset_roles.json`
- `coverage_summary.md`
- `ood_regions.json`

## Verify Prompt

Are ring conditions measured before split selection, are BO/validation/held-out roles isolated, and does the split include both covered and intentionally uncovered condition regions?
