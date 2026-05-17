# 06 Diversity Expansion and Hard-Negative Loop

## Goal

Test whether adding more diverse BO experience expands the proxy's reliable region. This is the main answer to the question: if the proxy sees more tunnel conditions, does it become confident on a wider range of cases?

## Runtime Path

Sandbox path: `logs/{run_id}/{tunnel_id}/r{ring_id}/06_diversity_expansion/`

## Inputs

- Confidence results from step 05.
- Hard-negative candidates: high proxy score or high confidence but low GT mIoU.
- Ring-condition descriptors from step 01.
- BO experience from step 02.

## Covered vs Uncovered Failure

For each hard negative, decide whether it lies in:

- **covered region**: close to BO experience in condition and feature space;
- **uncovered region**: far from BO experience, rare condition cluster, or new failure mode.

Covered failures suggest the proxy formulation is weak. Uncovered failures suggest the BO dataset lacks diversity.

## Actions

1. Cluster hard negatives by condition descriptors and failure type.
2. Label each cluster as covered or uncovered using condition distance and nearest BO examples.
3. If covered, refine the proxy using existing BO experience:
   - feature normalization;
   - feature subset;
   - robust loss;
   - confidence threshold.
4. If uncovered, add one or more new BO shots from that condition region.
5. Retrain the proxy and re-evaluate confidence on unchanged validation data.
6. Build a learning curve:
   - one-shot;
   - few-shot same-tunnel;
   - few-shot multi-condition;
   - expanded hard-negative coverage.
7. Stop expansion before held-out testing and freeze the proxy/confidence definition.

## Evidence to Report

- coverage increase in descriptor space;
- change in high-confidence success rate;
- change in low-confidence success rate;
- change in confidently wrong rate;
- success rate on formerly uncovered condition clusters;
- whether improvement came from better accuracy or better abstention.

## Outputs

- `hard_negative_taxonomy.csv`
- `coverage_before_after.csv`
- `diversity_learning_curve.csv`
- `expanded_bo_manifest.json`
- `proxy_after_expansion.json`
- `diversity_expansion_report.md`

## Verify Prompt

`Does adding diverse BO experience reduce proxy collapse in previously uncovered regions without leaking held-out confidence-test cases into training?`
