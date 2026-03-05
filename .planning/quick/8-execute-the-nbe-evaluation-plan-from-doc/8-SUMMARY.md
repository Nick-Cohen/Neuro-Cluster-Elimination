---
phase: quick
plan: 8
subsystem: nbe-evaluation
tags: [nbe, evaluation, benchmarks, graphical-models, ablation]
dependency_graph:
  requires: [quick-7 (NBE configs + pre-smoke scripts)]
  provides: [nbe-evaluation-results, phase0a-4-scripts]
  affects: [nbe_sanity_check configs, train.py set_size logic]
tech_stack:
  added: []
  patterns: [try/except per problem, graceful failure on network errors]
key_files:
  created:
    - notebooks/March-2026/claude_experiments/nbe_eval_phase1_smoke.py
    - notebooks/March-2026/claude_experiments/nbe_eval_phase2_rbm20.py
    - notebooks/March-2026/claude_experiments/nbe_eval_phase3_full.py
    - notebooks/March-2026/claude_experiments/nbe_eval_results/phase0a/output.txt
    - notebooks/March-2026/claude_experiments/nbe_eval_results/phase0b/output.txt
    - notebooks/March-2026/claude_experiments/nbe_eval_results/phase1/practice_output.txt
    - notebooks/March-2026/claude_experiments/nbe_eval_results/phase1/full_output.txt
    - notebooks/March-2026/claude_experiments/nbe_eval_results/phase2/output.txt
    - notebooks/March-2026/claude_experiments/nbe_eval_results/phase3/output.txt
  modified:
    - nce/neural_networks/train.py (set_size clamp fix)
    - notebooks/March-2026/claude_experiments/nbe_eval_phase0a_exact.py (model.X -> model.num_vars)
    - notebooks/March-2026/claude_experiments/nbe_eval_phase0b_single_bucket.py (model.X + FactorNN.tensor fix)
    - notebooks/March-2026/claude_experiments/nbe_eval_practice_1epoch.py (model.X -> model.num_vars)
decisions:
  - "ecl=2^22 benchmark config causes num_trained=0 for ALL 5 sanity check models"
  - "grid40x40.f10 cannot be loaded without network access to sli.ics.uci.edu (ord file missing)"
  - "FactorNN.tensor=None (lazy representation); scripts must not access .tensor.shape"
  - "NBE adaptive sampling (nbe,0.1 -> 29602 samples) < set_size=50000 was producing 0 training iterations; fixed with set_size clamp"
metrics:
  duration_min: 37
  completed_date: "2026-03-05"
  tasks_completed: 2
  tasks_total: 3
  files_created: 9
  files_modified: 4
---

# Phase Quick Plan 8: Execute NBE Evaluation Plan Summary

**One-liner:** Executed NBE evaluation Phases 0a-3 producing exact WMB-BE baselines across 5 benchmark problems, uncovering that ecl=2^22 config prevents any NN training (num_trained=0 for all models).

## What Was Built

Executed the NBE algorithm evaluation plan through Phases 0a, 0b, 1 (practice + full), 2 (rbm_20 ablation), and 3 (full benchmark). Three existing pre-smoke scripts were validated and fixed. Three new scripts were created and executed.

### Phase 0a: Exact Baseline
- Grid10x10.f5.wrap with ecl=2^30, iB=30
- **Result:** log Z = 169.40834045410156, num_trained=0
- Time: 6.23s

### Phase 0b: Single Bucket NN Test
- Pedigree13, ecl=2^30, iB=30, num_epochs=1
- Target bucket 302: 16 vars, message_size=98304
- NBE formula (eps=0.1) -> 29602 samples
- **Result:** Training succeeded after fixing set_size clamp bug
- Time: 29.76s for 1 epoch

### Phase 1 Practice (1-epoch)
- Grid10x10.f5.wrap with benchmark defaults (ecl=2^22, iB=10)
- **Result:** log Z = 169.40834, num_trained=0 (max message size 2^21 < ecl=2^22)
- Grid10x10 runs entirely exact with benchmark config

### Phase 1 Full (500-epoch)
- Same as practice, confirming exact execution
- **Result:** log Z = 169.4082, num_trained=0, absolute error vs Phase 0a = 0.000137
- Fast: 0.21s (no NN training, just exact bucket elimination)

### Phase 2: rbm_20 Ablation (5 variants)

| Variant | Log Z | Num Trained | Time |
|---------|-------|-------------|------|
| NBE-full | 58.5306 | 0 | 0.1s |
| NBE-fixed-samples | 58.5306 | 0 | 0.1s |
| NBE-fixed-arch | 58.5306 | 0 | 0.1s |
| NBE-fixed-loss | 58.5306 | 0 | 0.1s |
| Baseline | 58.5306 | 0 | 0.1s |

All identical because rbm_20 max message size (2^20 = 1M) < ecl (2^22 = 4M).

### Phase 3: Full Benchmark (all 5 problems)

| Problem | num_vars | num_trained | log_Z_estimate | time | status |
|---------|----------|-------------|----------------|------|--------|
| pedigree13 | 1077 | 0 | -23.4234 | 4.7s | OK |
| grid40x40.f10 | 1600 | N/A | ERROR | N/A | FAIL |
| grid20x20.f10 | 400 | 0 | 1332.5142 | 1.3s | OK |
| rbm_20 | 40 | 0 | 58.5306 | 0.1s | OK |
| grid10x10.f5.wrap | 100 | 0 | 169.4082 | 0.1s | OK |

grid40x40.f10 failed because the .uai.ord (elimination order) file is not cached and sli.ics.uci.edu was unreachable during execution.

## Key Findings

### Critical: ecl=2^22 Prevents All NN Training
The benchmark NBE config uses ecl=2^22 (4,194,304). With iB bounding message size to 2^iB:
- pedigree13 (iB=20): max message 2^20 = 1M < 4M -> exact
- grid40x40.f10 (iB=20): max message 2^20 = 1M < 4M -> exact
- grid20x20.f10 (iB=10): max message 2^10 = 1K < 4M -> exact
- rbm_20 (iB=20): max message 2^20 = 1M < 4M -> exact
- grid10x10.f5.wrap (iB=10): max message 2^10 = 1K < 4M -> exact

**The benchmark config tests WMB exact bucket elimination (WMB-BE), not NBE approximation.**
To test actual NN training, ecl must be lowered (e.g., ecl=2^15 would force NN training on pedigree13 and grid40x40 where message sizes reach 2^16+).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] model.X does not exist on catalog Model objects**
- **Found during:** Task 1 (Phase 0a first run)
- **Issue:** Scripts used `len(model.X)` but catalog Model uses `model.num_vars`
- **Fix:** Changed all 3 pre-smoke scripts to use `model.num_vars`
- **Files modified:** nbe_eval_phase0a_exact.py, nbe_eval_phase0b_single_bucket.py, nbe_eval_practice_1epoch.py
- **Commit:** b769b06

**2. [Rule 1 - Bug] set_size > num_samples caused 0 training iterations**
- **Found during:** Task 1 (Phase 0b first run)
- **Issue:** NBE adaptive sampling for pedigree13 bucket 302 computed 29,602 samples. Default `set_size=50000` caused `num_sets = 29602 // 50000 = 0`, producing 0 training iterations.
- **Fix:** Added `if num_samples < set_size: set_size = num_samples` clamp before computing num_sets
- **Files modified:** nce/neural_networks/train.py
- **Commit:** b769b06

**3. [Rule 1 - Bug] FactorNN.tensor is None (lazy representation)**
- **Found during:** Task 1 (Phase 0b second run)
- **Issue:** Script accessed `message.tensor.shape` but FactorNN uses lazy evaluation (tensor=None)
- **Fix:** Changed to print `type(message).__name__`, `message.labels`, `message.is_nn`
- **Files modified:** nbe_eval_phase0b_single_bucket.py
- **Commit:** b769b06

### Deferred Items

**grid40x40.f10 elimination order unavailable (network issue)**
- `.uai.ord` file not cached; `sli.ics.uci.edu` unreachable during Phase 3 execution
- Phase 3 script catches the error gracefully and continues with other models
- Deferred: Pre-download the .ord file when network is available

**ecl threshold: No NN training triggered for any benchmark model**
- The benchmark ecl=2^22 is too large for all 5 sanity check models to require NN training
- Deferred: Create separate "NBE with NN" phase using ecl=2^15 or lower to actually test NN path
- See Phase 4-5 in the eval plan for next steps

## Task 3: Checkpoint - Awaiting Human Review

Execution paused at Task 3 (checkpoint:human-verify). User should:
1. Review results in `notebooks/March-2026/claude_experiments/nbe_eval_results/`
2. Note that num_trained=0 for all models with ecl=2^22 - this is expected behavior
3. Decide whether to proceed with Phase 4 (early stopping) or lower ecl to test NN training
4. Pre-download grid40x40.f10 order file when network is available

## Self-Check

### Files Created/Modified
- [x] nce/neural_networks/train.py - modified with set_size clamp
- [x] notebooks/March-2026/claude_experiments/nbe_eval_phase1_smoke.py
- [x] notebooks/March-2026/claude_experiments/nbe_eval_phase2_rbm20.py
- [x] notebooks/March-2026/claude_experiments/nbe_eval_phase3_full.py
- [x] nbe_eval_results/phase0a/output.txt
- [x] nbe_eval_results/phase0b/output.txt
- [x] nbe_eval_results/phase1/practice_output.txt
- [x] nbe_eval_results/phase1/full_output.txt
- [x] nbe_eval_results/phase2/output.txt
- [x] nbe_eval_results/phase3/output.txt

### Commits
- b769b06: fix + script fixes (Task 1)
- fb93b9f: new Phase 1-3 scripts (Task 2)

## Self-Check: PASSED
