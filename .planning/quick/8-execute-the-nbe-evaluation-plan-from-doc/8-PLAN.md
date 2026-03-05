---
phase: quick
plan: 8
type: execute
wave: 1
depends_on: []
files_modified:
  - notebooks/March-2026/claude_experiments/nbe_eval_phase1_smoke.py
  - notebooks/March-2026/claude_experiments/nbe_eval_phase2_rbm20.py
  - notebooks/March-2026/claude_experiments/nbe_eval_phase3_full.py
  - notebooks/March-2026/claude_experiments/nbe_eval_results/
autonomous: false
requirements: []

must_haves:
  truths:
    - "Phase 0a exact baseline produces a log Z value for grid10x10.f5.wrap"
    - "Phase 0b single bucket NN trains without error on pedigree13"
    - "Phase 1 practice (1-epoch) runs end-to-end producing a log Z estimate"
    - "Phase 1 full (500-epoch) runs end-to-end on grid10x10 with reasonable accuracy"
    - "Phase 2 ablation produces results table for rbm_20 comparing 5 variants"
    - "Phase 3 full benchmark produces results for all 5 problems"
  artifacts:
    - path: "notebooks/March-2026/claude_experiments/nbe_eval_phase1_smoke.py"
      provides: "Full 500-epoch smoke test script for grid10x10"
    - path: "notebooks/March-2026/claude_experiments/nbe_eval_phase2_rbm20.py"
      provides: "rbm_20 deep dive with 5 ablation variants"
    - path: "notebooks/March-2026/claude_experiments/nbe_eval_phase3_full.py"
      provides: "Full benchmark evaluation on all 5 problems"
    - path: "notebooks/March-2026/claude_experiments/nbe_eval_results/"
      provides: "Output directory with phase results"
  key_links:
    - from: "nbe_sanity_check benchmark configs"
      to: "FastGM constructor"
      via: "config dict passed to FastGM(model=model, nn_config=config)"
    - from: "scripts"
      to: "nbe_eval_results/"
      via: "results written to phase subdirectories"
---

<objective>
Execute the NBE evaluation plan from docs/nbe_evaluation_plan.md through Phases 0a-3.

Purpose: Validate the NBE algorithm implementation end-to-end and collect benchmark results comparing NBE to baseline approaches across 5 graphical models.

Output: Execution results from Phases 0a, 0b, 1 (practice + full), 2 (rbm_20 ablation), and 3 (full benchmark). New scripts for phases 1-full, 2, and 3. Results saved to nbe_eval_results/.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@docs/nbe_evaluation_plan.md
@nce/benchmark_problems/nbe_sanity_check.py
@notebooks/March-2026/claude_experiments/nbe_eval_phase0a_exact.py
@notebooks/March-2026/claude_experiments/nbe_eval_phase0b_single_bucket.py
@notebooks/March-2026/claude_experiments/nbe_eval_practice_1epoch.py

<interfaces>
<!-- Key APIs used by all scripts -->
From nce/benchmark_problems/__init__.py:
```python
from nce.benchmark_problems import nbe_sanity_check
# nbe_sanity_check.problems  -> list of 5 Model objects (indices 0-4)
# nbe_sanity_check.configs['nbe']  -> list of 5 config dicts (same order)
# Index mapping: 0=pedigree13, 1=grid40x40.f10, 2=grid20x20.f10, 3=rbm_20, 4=grid10x10.f5.wrap
```

From nce/inference/graphical_model.py:
```python
from nce.inference.graphical_model import FastGM
fastgm = FastGM(model=model, nn_config=config, device='cpu')  # or 'cuda'
# Constructor auto-calls dope_factors() when config['dope_factors']=True
log_z = fastgm.get_log_partition_function()
# fastgm.num_trained -> int: number of buckets that used NN training
```

GPU environment: 4x CUDA GPUs available. Use device='cuda' for training-heavy scripts.

Benchmark config key parameters per model:
| Index | Model | iB | ecl | hidden_sizes | num_samples |
|-------|-------|----|-----|-------------|-------------|
| 0 | pedigree13 | 20 | 2^22 | nbe,3 | nbe,0.1 |
| 1 | grid40x40.f10 | 20 | 2^22 | nbe,1 | nbe,0.35 |
| 2 | grid20x20.f10 | 10 | 2^22 | nbe,1 | nbe,0.35 |
| 3 | rbm_20 | 20 | 2^22 | nbe,3 | nbe,0.1 |
| 4 | grid10x10.f5.wrap | 10 | 2^22 | nbe,1 | nbe,0.35 |
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Run existing pre-smoke scripts (Phase 0a, 0b, Phase 1 practice) and validate pipeline</name>
  <files>notebooks/March-2026/claude_experiments/nbe_eval_results/phase0a/output.txt, notebooks/March-2026/claude_experiments/nbe_eval_results/phase0b/output.txt, notebooks/March-2026/claude_experiments/nbe_eval_results/phase1/practice_output.txt</files>
  <action>
Run the three existing scripts sequentially, capturing output to results files. These validate that the core pipeline works before creating new scripts.

1. Create results directories:
   ```
   mkdir -p notebooks/March-2026/claude_experiments/nbe_eval_results/{phase0a,phase0b,phase1,phase2,phase3}
   ```

2. Run Phase 0a (exact baseline on grid10x10.f5.wrap):
   ```
   /home/cohenn1/NCE/venv/bin/python notebooks/March-2026/claude_experiments/nbe_eval_phase0a_exact.py 2>&1 | tee notebooks/March-2026/claude_experiments/nbe_eval_results/phase0a/output.txt
   ```
   Expected: Completes in seconds, prints log Z value, num_trained=0.

3. Run Phase 0b (single bucket NN on pedigree13):
   ```
   /home/cohenn1/NCE/venv/bin/python notebooks/March-2026/claude_experiments/nbe_eval_phase0b_single_bucket.py 2>&1 | tee notebooks/March-2026/claude_experiments/nbe_eval_results/phase0b/output.txt
   ```
   Expected: Finds large buckets, trains 1 epoch on one, prints message shape. May take a few minutes for the exact elimination steps preceding the target bucket.

4. Run Phase 1 practice (1-epoch full run on grid10x10.f5.wrap):
   ```
   /home/cohenn1/NCE/venv/bin/python notebooks/March-2026/claude_experiments/nbe_eval_practice_1epoch.py 2>&1 | tee notebooks/March-2026/claude_experiments/nbe_eval_results/phase1/practice_output.txt
   ```
   Expected: Completes end-to-end, shows how many buckets trained, prints log Z (accuracy will be poor with 1 epoch).

If ANY script fails, diagnose the error and fix the underlying issue in the codebase (nce/ package) before proceeding. Common issues:
- Missing loss function name in losses.py registry
- 'nbe,X' string not being resolved to integer in compute_message_nn
- Device mismatch (factor on CPU but model expects CUDA)

Record the exact log Z from Phase 0a -- this is the baseline for Phase 1 comparison.
  </action>
  <verify>
All three scripts complete without errors. Output files exist in nbe_eval_results/phase0a/, phase0b/, phase1/. Phase 0a produces a log Z value with num_trained=0. Phase 0b trains 1 epoch on a single bucket. Phase 1 practice produces an end-to-end log Z estimate.

```
ls notebooks/March-2026/claude_experiments/nbe_eval_results/phase0a/output.txt notebooks/March-2026/claude_experiments/nbe_eval_results/phase0b/output.txt notebooks/March-2026/claude_experiments/nbe_eval_results/phase1/practice_output.txt
```
  </verify>
  <done>Phase 0a, 0b, and Phase 1 practice scripts all execute successfully. Output captured. Exact log Z baseline recorded from Phase 0a. Pipeline validated -- ready for full training runs.</done>
</task>

<task type="auto">
  <name>Task 2: Create Phase 1 full, Phase 2, Phase 3 scripts and execute all phases</name>
  <files>notebooks/March-2026/claude_experiments/nbe_eval_phase1_smoke.py, notebooks/March-2026/claude_experiments/nbe_eval_phase2_rbm20.py, notebooks/March-2026/claude_experiments/nbe_eval_phase3_full.py</files>
  <action>
Create the remaining evaluation scripts and execute them. Use CUDA for all training runs.

**Script 1: nbe_eval_phase1_smoke.py (Phase 1 full, 500-epoch on grid10x10)**

Create script that:
- Loads grid10x10.f5.wrap (index 4) with its NBE config (no overrides except device='cuda')
- num_epochs=500 (already the benchmark default), ecl=2**22, iB=10
- Runs `fastgm.get_log_partition_function()` with timing
- Prints: log Z estimate, num_trained, total time, comparison to Phase 0a exact value
- Saves results to nbe_eval_results/phase1/full_output.txt

Run this script:
```
/home/cohenn1/NCE/venv/bin/python notebooks/March-2026/claude_experiments/nbe_eval_phase1_smoke.py 2>&1 | tee notebooks/March-2026/claude_experiments/nbe_eval_results/phase1/full_output.txt
```
This should be relatively fast (grid10x10 is the smallest problem, 100 vars).

**Script 2: nbe_eval_phase2_rbm20.py (Phase 2, rbm_20 ablation)**

Create script that runs 5 ablation variants on rbm_20 (index 3, 40 vars, width 20) sequentially:

| Variant | num_samples | hidden_sizes | loss_fn |
|---------|-------------|--------------|---------|
| NBE-full | nbe,0.1 | nbe,3 | weighted_logspace_mse |
| NBE-fixed-samples | 50000 | nbe,3 | weighted_logspace_mse |
| NBE-fixed-arch | nbe,0.1 | [3,3] | weighted_logspace_mse |
| NBE-fixed-loss | nbe,0.1 | nbe,3 | unnormalized_kl |
| Baseline | 50000 | [3,3] | unnormalized_kl |

For each variant:
- Create a fresh config from nbe_sanity_check.configs['nbe'][3]
- Override the relevant parameters per the table above
- Set device='cuda', track_errors=True
- Run FastGM and get_log_partition_function() with timing
- Record: variant name, log Z, num_trained, total time

Print a summary table at the end. Save to nbe_eval_results/phase2/output.txt.

Run this script:
```
/home/cohenn1/NCE/venv/bin/python notebooks/March-2026/claude_experiments/nbe_eval_phase2_rbm20.py 2>&1 | tee notebooks/March-2026/claude_experiments/nbe_eval_results/phase2/output.txt
```

**Script 3: nbe_eval_phase3_full.py (Phase 3, all 5 problems)**

Create script that runs NBE on all 5 benchmark problems sequentially:
- For each problem (indices 0-4), use its benchmark NBE config with device='cuda'
- Run get_log_partition_function() with timing
- Record: problem name, num_vars, num_trained, log_z, time
- Print summary table at the end matching the format in the eval plan
- Save to nbe_eval_results/phase3/output.txt

NOTE on Phase 3 execution: pedigree13 (1077 vars, width 32) and grid40x40.f10 (1600 vars, width 54) may take a very long time. Run Phase 3 in background if Phase 1 and 2 succeed. If any single problem takes more than 30 minutes, it is acceptable to note that and move on.

Run Phase 3:
```
/home/cohenn1/NCE/venv/bin/python notebooks/March-2026/claude_experiments/nbe_eval_phase3_full.py 2>&1 | tee notebooks/March-2026/claude_experiments/nbe_eval_results/phase3/output.txt
```

Important implementation notes for all scripts:
- Always use `dict(nbe_sanity_check.configs['nbe'][idx])` to copy the config (never mutate the original)
- Constructor auto-calls dope_factors() when config['dope_factors']=True -- do NOT call it manually
- get_log_partition_function() is the correct API (not run())
- Use try/except around each run to capture and report errors without crashing the whole script
- Print clear section headers and separator lines for readability
  </action>
  <verify>
Phase 1 full script produces a log Z estimate closer to the Phase 0a exact value than the 1-epoch practice run. Phase 2 ablation script produces results for all 5 variants. Phase 3 script runs (may still be in progress for larger models).

```
ls notebooks/March-2026/claude_experiments/nbe_eval_phase1_smoke.py notebooks/March-2026/claude_experiments/nbe_eval_phase2_rbm20.py notebooks/March-2026/claude_experiments/nbe_eval_phase3_full.py
```
  </verify>
  <done>Three new scripts created. Phase 1 full (500 epochs on grid10x10) produces a log Z value with measurable improvement over 1-epoch practice. Phase 2 ablation table populated for rbm_20. Phase 3 script executes on all 5 problems (large models may take significant time). All results saved to nbe_eval_results/.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 3: Review NBE evaluation results and decide on Phases 4-5</name>
  <files>notebooks/March-2026/claude_experiments/nbe_eval_results/</files>
  <action>
Present all evaluation results to user for review. Executed NBE evaluation Phases 0a through 3 per the evaluation plan in docs/nbe_evaluation_plan.md. Three existing scripts validated the pipeline (Phase 0a exact baseline, Phase 0b single bucket NN, Phase 1 practice). Three new scripts created and run: Phase 1 full (500 epochs on grid10x10), Phase 2 (rbm_20 ablation with 5 variants), Phase 3 (full benchmark on all 5 problems).
  </action>
  <verify>
User reviews results:
1. Review results in notebooks/March-2026/claude_experiments/nbe_eval_results/
2. Check Phase 1: Does the 500-epoch log Z improve over the 1-epoch practice run?
3. Check Phase 2: Do the 5 ablation variants show meaningful differences? Does NBE-full outperform Baseline?
4. Check Phase 3: Did all 5 problems produce results? Were any too slow?
5. Decide whether to proceed with Phase 4 (early stopping) and Phase 5 (paper comparison) as separate tasks
  </verify>
  <done>User has reviewed results and approved, or provided feedback for follow-up. Decision made on Phase 4-5 execution.</done>
</task>

</tasks>

<verification>
- Phase 0a: exact log Z produced (num_trained=0)
- Phase 0b: single bucket trained successfully
- Phase 1 practice: end-to-end pipeline validated
- Phase 1 full: 500-epoch log Z closer to exact baseline than 1-epoch
- Phase 2: 5 ablation variants produce results table for rbm_20
- Phase 3: results for all 5 benchmark problems (or clear explanation of timeouts)
- All output captured in nbe_eval_results/ subdirectories
</verification>

<success_criteria>
Phases 0a-3 of the NBE evaluation plan executed. Pipeline validated through pre-smoke tests. Full training results collected for grid10x10 (Phase 1), rbm_20 ablation (Phase 2), and all 5 problems (Phase 3). Results saved and ready for analysis. Phases 4-5 deferred for user decision.
</success_criteria>

<output>
After completion, create `.planning/quick/8-execute-the-nbe-evaluation-plan-from-doc/8-SUMMARY.md`
</output>
