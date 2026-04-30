# Coding Plans (Implementation Guides)

## Purpose
This folder is the implementation companion to `methods/plans/`.

- `methods/plans/steps/` defines methodology and study logic.
- `methods/ablation/steps/` defines concrete coding guidance to build that methodology.

## What each coding guide contains

Each step file documents:
- plan reference (`methods/plans/steps/NN_*.md`)
- target artifacts (exact output files)
- files to create/modify
- suggested public function signatures
- data-flow sketch
- reuse points in current codebase
- run commands and verification checks

## Scope boundaries

Implementation guidance follows paper constraints:
- use fixed-rule reflection only
- no LLM routing
- no RL routing/policy learning
- no adaptive correction sequencing

## Step mapping

`methods/ablation/steps/01..07` maps one-to-one to `methods/plans/steps/01..07`.
