# 05 Data-Flow Graph

## Goal
Produce `graph.md`.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/05_data_flow_graph/graph.md`

## Inputs
- parameter inventory
- code flow

## Actions
1. List graph nodes.
2. List directed edges.
3. Record stage transitions and critical path.

## Outputs
- `graph.md`

## Verify Prompt
`Does the graph artifact list nodes, edges, and the critical path?`

## Verify Script
```bash
python plans/scripts/verify_step.py --root data/{tunnel_id}/workflow/{run_id} --step 05
```
