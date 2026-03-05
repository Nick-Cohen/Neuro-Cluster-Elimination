---
phase: quick
plan: 6
subsystem: docs
tags: [nbe, evaluation, planning]
key_files:
  created:
    - docs/nbe_evaluation_plan.md
decisions:
  - "5-phase evaluation structure: smoke test → deep dive → full benchmark → early stopping → paper comparison"
  - "Found loss_fn name mismatch bug: config uses 'weighted_mse' but function registered as 'weighted_logspace_mse'"
  - "Found backward_iB hardcoded to 10, should match per-model iB"
metrics:
  duration: "~3 min"
  completed: "2026-03-05"
  tasks_completed: 1
  tasks_total: 1
---

## One-liner
Created comprehensive 5-phase NBE evaluation plan with pre-requisite bug fixes, ablation study design, and paper comparison methodology.

## What Changed
- Created `docs/nbe_evaluation_plan.md` with full evaluation plan

## Key Findings
- **Bug:** `loss_fn: 'weighted_mse'` in nbe_sanity_check configs won't match registered `'weighted_logspace_mse'`
- **Bug:** `backward_iB` hardcoded to 10 in all configs, but iB varies per model (10 or 20)
- **Plan:** 5 phases from smoke test through paper comparison, with ablation variants for rbm_20

## Verification
- [x] Plan document written and saved
- [x] Pre-requisites checklist included
- [x] All phases have clear setup, metrics, and success criteria
