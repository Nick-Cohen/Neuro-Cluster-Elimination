---
phase: 13-design-benchmark-experiment
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/EXPERIMENT_DESIGN.md
  - notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/OPEN_QUESTIONS.md
  - notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/verify_full_data_training.py
  - notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/verify_config_correctness.py
  - notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/wmse_config_output.txt
autonomous: true
must_haves:
  truths:
    - "Experiment design document specifies all 5 experiment configurations completely"
    - "Open questions are documented with clear impact analysis"
    - "Verification script confirms entire message is trained on with sampling_scheme='all'"
    - "WMSE config is written to file showing all 42 fields"
    - "A future Claude instance can execute the experiment from the design document alone"
  artifacts:
    - path: "notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/EXPERIMENT_DESIGN.md"
      provides: "Complete experiment specification"
    - path: "notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/OPEN_QUESTIONS.md"
      provides: "Documented open questions and assumptions"
    - path: "notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/verify_full_data_training.py"
      provides: "Verification that full message is used in training"
    - path: "notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/verify_config_correctness.py"
      provides: "Verification that configs produce expected nn_config dicts"
    - path: "notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/wmse_config_output.txt"
      provides: "Full WMSE config dump for all problems"
  key_links:
    - from: "EXPERIMENT_DESIGN.md"
      to: "notebooks/_1-2026/worker.py"
      via: "YAML config format must match worker.py's build_nn_config()"
    - from: "verify_full_data_training.py"
      to: "nce/neural_networks/train.py"
      via: "Confirms sampling_scheme='all' forces batch_size=message_size"
---

<objective>
Design a comprehensive benchmark experiment comparing WMSE (weighted_logspace_mse) and UKL (unnormalized_kl) loss functions across the benchmarks_12_4_2025 problem set with varying backward information levels.

Purpose: Create a self-contained experiment specification that a future Claude instance can execute without ambiguity. The experiment compares two loss functions -- WMSE (no backward info) vs UKL (with 4 levels of backward info) -- across all 24 unique problems in the small_problems benchmark set.

Output: Experiment design document, open questions file, verification scripts, and WMSE config dump.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@/home/cohenn1/NCE/CLAUDE.md
@/home/cohenn1/NCE/nce/benchmark_problems/small_problems.py
@/home/cohenn1/NCE/nce/neural_networks/losses.py (lines 245-250 for logspace_mse_fdb, lines 751-762 for weighted_logspace_mse, lines 35-105 for unnormalized_kl)
@/home/cohenn1/NCE/nce/neural_networks/train.py (lines 190-275 for batch/sampling logic)
@/home/cohenn1/NCE/notebooks/_1-2026/worker.py (lines 115-185 for build_nn_config, lines 55-92 for generate_experiments)
@/home/cohenn1/NCE/notebooks/_1-2026/experiment_config.py
@/home/cohenn1/NCE/notebooks/_1-2026/experiment_runner.py
@/home/cohenn1/NCE/notebooks/_1-2026/examples/probs12_4/bn_BN_1_iB10.yaml (example YAML config)
@/home/cohenn1/NCE/notebooks/_1-2026/examples/grid10x10_bw_sweep.yaml (example bw_ecl sweep config)

<interfaces>
<!-- Key types and contracts from codebase -->

From nce/benchmark_problems/small_problems.py:
- `small_problems`: BenchmarkSet with 24 problems, configs['default'] has auto_ecl per problem
- `set_bw_ecl(benchmark_set, config_name, value)`: sets bw_ecl, backward_ecl, populate_bw_factors
- `_MODELS`: 24 models (9 iB10-only, 6 shared, 9 iB15-only)
- `_AUTO_ECL`: per-model ecl values (e.g. BN_3: 16383, grid10x10.f5.wrap: 1048575)
- Default config: loss_fn='unnormalized_kl', sampling_scheme='all', batch_size=100000, num_epochs=10000

From nce/neural_networks/train.py:
- When sampling_scheme='all': set_size = message_size, num_samples = message_size
- batch_size='all' -> batch_size = message_size (full single batch)
- batch_size=integer -> uses that value, computes num_batches_per_set = ceil(set_size/batch_size)
- skip_early_stopping=True disables early stopping

From nce/neural_networks/losses.py:
- `weighted_logspace_mse(outputs, targets, bw_hat=None)`: NeuroBE WMSE loss
  - Normalizes targets to [0,1], creates weights proportional to normalized targets
  - Formula: weights = N * normalized_targets / sum(normalized_targets)
  - Loss = mean(weights * (outputs - targets)^2)
  - Does NOT use bw_hat (ignores backward info)
- `unnormalized_kl(outputs, targets, bw_hat=None, ...)`: UKL loss
  - Supports bw_hat for backward message weighting
  - Formula: sum(p_tilde * (log_p_tilde - log_q_tilde) - p_tilde + q_tilde)

From notebooks/_1-2026/worker.py build_nn_config():
- loss_fn = config.get('loss', 'unnormalized_kl')
- bw_ecl=0 -> use_bw_approx=False, populate_bw_factors=False, bw_ecl=None
- bw_ecl>0 -> use_bw_approx=True, populate_bw_factors=True, bw_ecl=value
- ecl = config.get('ecl', 2**10)
- sampling_scheme = config.get('sampling_scheme', 'all')
- batch_size = config.get('batch_size', 100000)

YAML config format (from experiment_config.py):
- Required: problem, loss, architectures, bw_ecl
- Optional with defaults: epochs (30000), gpus ([0,1,2,3]), num_runs (1),
  sampling_scheme ('all'), val_set ('all'), batch_size (100000), set_size (100000),
  num_samples (100000), num_batches_per_set (1)
- Additional optional: ecl, seed
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Design the experiment and document open questions</name>
  <files>
    notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/EXPERIMENT_DESIGN.md
    notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/OPEN_QUESTIONS.md
    notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/wmse_config_output.txt
  </files>
  <action>
Create the experiment design document (EXPERIMENT_DESIGN.md) that fully specifies the benchmark experiment.
The document must be self-contained enough for a future Claude instance to execute without interpretation.

## Experiment Structure

The experiment has 5 configurations (not 5 separate experiments -- they share the same problem set):

**Configuration 1: WMSE (no backward info)**
- loss_fn: 'weighted_logspace_mse'  (this is the loss function name in _get_loss_fn -- NOT 'weighted_mse' or 'wmse')
- bw_ecl: 0 (no backward info)
- use_bw_approx: False
- populate_bw_factors: False
- NOTE: weighted_logspace_mse ignores bw_hat parameter, so no backward info is inherent

**Configuration 2: UKL (no backward info)**
- loss_fn: 'unnormalized_kl'
- bw_ecl: 0
- use_bw_approx: False
- populate_bw_factors: False

**Configuration 3: UKL + bw_ecl = 2^3 = 8**
- loss_fn: 'unnormalized_kl'
- bw_ecl: 8
- use_bw_approx: True
- populate_bw_factors: True

**Configuration 4: UKL + bw_ecl = ecl (same as forward)**
- loss_fn: 'unnormalized_kl'
- bw_ecl: <per-problem auto_ecl from _AUTO_ECL dict>
- use_bw_approx: True
- populate_bw_factors: True
- NOTE: This is the tricky one -- bw_ecl varies per problem. Document this clearly.
  The YAML config format does NOT directly support per-problem bw_ecl.
  This means either (a) we need per-problem YAML configs for this configuration,
  or (b) we modify the runner to support 'ecl' as a bw_ecl value.
  Document this as an open question.

**Configuration 5: UKL + bw_ecl = 2^30 (effectively exact backward)**
- loss_fn: 'unnormalized_kl'
- bw_ecl: 1073741824
- use_bw_approx: True
- populate_bw_factors: True

## Common Settings (all configurations)
- num_epochs: 5000 (user specified)
- skip_early_stopping: True (no early stopping)
- nbe_early_stopping: False
- sampling_scheme: 'all' (enumerate full message space)
- batch_size: 'all' (user said batch_size=message_size -> use 'all' string which train.py translates to message_size)
  ACTUALLY: The user said batch_size=message_size. With sampling_scheme='all', setting batch_size='all' achieves this.
  But the current YAML config system and worker.py's build_nn_config do NOT support batch_size='all'.
  The worker.py defaults batch_size to config.get('batch_size', 100000).
  Document this as an OPEN QUESTION: either (a) set batch_size very large (e.g. 10000000) so it always exceeds message_size,
  or (b) modify worker.py to support batch_size='all'.
  The train.py code (line 258-260) already supports batch_size='all' as a config value.
- hidden_sizes: [3, 3] (the default from small_problems config template)
- ecl: per-problem auto_ecl values from _AUTO_ECL
- iB: 100 (effectively unlimited)
- seed: 42

## Problem Set
All 24 problems from small_problems (nce/benchmark_problems/small_problems.py).
List all 24 problems with their catalog key, modelfile, and auto_ecl value.

## YAML Config Strategy

For configs 1, 2, 3, 5: can use a single YAML per problem with different loss/bw_ecl combos.
But the existing YAML format assumes ONE loss per config. So we need:
- One set of YAML configs with loss='weighted_logspace_mse', bw_ecl=[0] (config 1)
- One set of YAML configs with loss='unnormalized_kl', bw_ecl=[0, 3, 30] (configs 2, 3, 5)
- Per-problem YAML configs for config 4 (bw_ecl=ecl varies per problem)

OR: A Python script that builds and runs all configs programmatically using small_problems.

Document both approaches and their tradeoffs.

## WMSE Config Output
Generate wmse_config_output.txt by dumping the full expected nn_config dict for the WMSE configuration.
This should show all 42 fields that will be passed to FastGM for one representative problem (e.g. BN_1).
Format it as a Python dict with field comments.

## Open Questions (OPEN_QUESTIONS.md)

Document the following open questions with impact analysis:

1. **batch_size='all' support in worker.py**: worker.py's build_nn_config() reads batch_size as integer from YAML. train.py supports batch_size='all' string. How to bridge this gap?
   - Option A: Set batch_size to a very large number (e.g. 10_000_000) -- guaranteed to exceed any message_size
   - Option B: Modify build_nn_config() to pass through 'all' string
   - Option C: Run experiment via Python script (bypassing YAML/worker.py)
   - Recommendation: Option A is simplest and requires no code changes

2. **bw_ecl=ecl (config 4) per-problem variation**: The auto_ecl varies per problem (16383 to 19487170). Current YAML format has one bw_ecl list for all experiments.
   - Option A: Generate 24 individual YAML configs for config 4
   - Option B: Add 'ecl' as a special bw_ecl value in worker.py
   - Option C: Run config 4 via Python script
   - Recommendation: Option A (24 YAML files) or Option C (Python script)

3. **Architecture**: User specified the experiment but didn't specify architecture. small_problems defaults to [3,3]. Should we also test [] (linear) and [2] (simple_nn)?
   - Impact: Each additional architecture multiplies total experiments by 1
   - Recommendation: Start with [3,3] only to keep experiment focused. Add architectures in follow-up.

4. **num_runs**: User didn't specify number of runs. Default is 1. Should we do multiple runs for statistical significance?
   - Impact: Each additional run multiplies wall-clock time by ~1
   - Recommendation: 1 run first, add runs if results look noisy

5. **ecl values**: small_problems uses auto_ecl (varies per problem). The existing probs12_4 YAML configs use ecl: 1024. Which to use?
   - The small_problems module was built with auto_ecl values specifically for these 24 problems
   - The probs12_4 YAML examples use ecl: 1024 (2^10)
   - Impact: Higher ecl means fewer NN-trained buckets (more exact computation)
   - Recommendation: Use auto_ecl from small_problems (these were computed for each problem's specific width)

6. **iB20 problems**: The benchmarks_12_4_2025 has iB10, iB15, AND iB20 problem sets. The small_problems module only includes iB10 and iB15 (24 problems). Should iB20 be included?
   - small_problems was specifically designed to exclude iB20 (those problems are much larger, some have no known Z)
   - Recommendation: Use only small_problems (24 problems, iB10+iB15)

7. **weighted_logspace_mse loss and backward info**: The WMSE loss function signature accepts bw_hat but ignores it. When bw_ecl=0, no backward factors are populated, so bw_hat will be None in training. This is correct behavior. But should we also test WMSE WITH backward info to verify it truly ignores bw_hat?
   - Recommendation: Not needed -- code inspection confirms bw_hat is unused in the loss computation

8. **5000 epochs**: The user specified 5000 epochs. Previous experiments used 10000 (small_problems default) or 30000 (worker.py default). Is 5000 sufficient for convergence?
   - Impact: Shorter training may not converge on harder problems
   - Recommendation: Honor user's specification of 5000 epochs

9. **Output format**: Should results be saved as JSON (current runner format) or also as aggregated CSV/pickle?
   - Recommendation: Use existing JSON format from experiment_runner.py

10. **dope_factors**: small_problems configs have dope_factors=False. NeuroBE config uses dope_factors=True. Which is correct for this experiment?
    - WMSE (NeuroBE loss) was designed for doped factors
    - UKL experiments in small_problems use dope_factors=False
    - Impact: doping replaces -inf values with finite values, affects training stability
    - Recommendation: Document both options. Use dope_factors=False for consistency with small_problems (both losses use same data).
  </action>
  <verify>
    All three files exist:
    - test -f notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/EXPERIMENT_DESIGN.md
    - test -f notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/OPEN_QUESTIONS.md
    - test -f notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/wmse_config_output.txt
    - EXPERIMENT_DESIGN.md contains all 5 configurations
    - wmse_config_output.txt contains all 42 config fields
  </verify>
  <done>
    - EXPERIMENT_DESIGN.md fully specifies all 5 experiment configurations with common settings
    - OPEN_QUESTIONS.md documents 10+ open questions with options and recommendations
    - wmse_config_output.txt shows complete nn_config dict for WMSE with all 42 fields
    - All three files are self-contained enough for a future Claude to execute from
  </done>
</task>

<task type="auto">
  <name>Task 2: Create verification scripts</name>
  <files>
    notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/verify_full_data_training.py
    notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/verify_config_correctness.py
  </files>
  <action>
Create two verification scripts that confirm the experiment will work correctly.

**verify_full_data_training.py**: Verify that sampling_scheme='all' and batch_size='all' causes the trainer to use the ENTIRE message for training (no data is left out).

The script should:
1. Import the necessary modules (FastGM, SampleGenerator, small_problems)
2. Pick one small problem (e.g., the one with smallest message_size) from small_problems
3. Create a config with sampling_scheme='all', batch_size='all', num_epochs=1, loss_fn='unnormalized_kl'
4. Create a FastGM with this config
5. For the FIRST NN-trained bucket (first bucket where message_size > ecl):
   a. Get the SampleGenerator's message_size
   b. Create the DataLoader and call load_all()
   c. Verify that the total number of training samples equals message_size
   d. If using batch_size='all', verify batch contains all samples
   e. Print: "PASS: Full message trained on. message_size={X}, training_samples={Y}"
6. Also verify batch_size=100000000 (very large int) results in the same behavior as batch_size='all' when sampling_scheme='all' (since train.py computes num_batches_per_set = ceil(message_size / batch_size) = 1 when batch_size > message_size)

Key code path to verify (from train.py lines 254-263):
```python
if self.dataloader.sample_generator.sampling_scheme == 'all':
    set_size = int(self.message_size)
    num_samples = int(self.message_size)
    if self.config['batch_size'] == 'all':
        batch_size = int(self.message_size)
    else:
        batch_size = self.config['batch_size']
    num_batches_per_set = (set_size + batch_size - 1) // batch_size
```

The test should prove that:
- With sampling_scheme='all', the full enumeration of the message space is used
- With batch_size='all' OR batch_size >= message_size, all data is in one batch
- No data is dropped or subsampled

Run with: /home/cohenn1/NCE/venv/bin/python verify_full_data_training.py
Device: cpu (no GPU needed for this verification)

**verify_config_correctness.py**: Verify that the experiment configs produce the expected nn_config dicts.

The script should:
1. Import small_problems from nce.benchmark_problems
2. For each of the 5 configurations, build what the nn_config dict WOULD look like
3. Verify key fields:
   - Config 1 (WMSE): loss_fn='weighted_logspace_mse', bw_ecl=None or 0, use_bw_approx=False, populate_bw_factors=False
   - Config 2 (UKL no bw): loss_fn='unnormalized_kl', bw_ecl=None or 0, use_bw_approx=False
   - Config 3 (UKL bw=8): loss_fn='unnormalized_kl', bw_ecl=8, use_bw_approx=True, populate_bw_factors=True
   - Config 4 (UKL bw=ecl): loss_fn='unnormalized_kl', bw_ecl=auto_ecl, use_bw_approx=True
   - Config 5 (UKL bw=2^30): loss_fn='unnormalized_kl', bw_ecl=2**30, use_bw_approx=True
4. Verify num_epochs=5000, skip_early_stopping=True, sampling_scheme='all'
5. Print PASS/FAIL for each configuration with details

Run with: /home/cohenn1/NCE/venv/bin/python verify_config_correctness.py
  </action>
  <verify>
    /home/cohenn1/NCE/venv/bin/python notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/verify_config_correctness.py
  </verify>
  <done>
    - verify_full_data_training.py runs and prints PASS confirming full message is used
    - verify_config_correctness.py runs and prints PASS for all 5 configurations
    - Both scripts run without errors from the NCE project root
  </done>
</task>

<task type="auto">
  <name>Task 3: Ping Nick on Discord when plan is complete</name>
  <files></files>
  <action>
After Tasks 1 and 2 are complete, run the Discord ping script to notify Nick:
```bash
~/.claude/ai-ops/scripts/ping_nick.sh "Benchmark experiment design complete: WMSE vs UKL across benchmarks_12_4_2025 (24 problems, 5 configs). Design doc, open questions, and verification scripts ready at notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/"
```
  </action>
  <verify>
    The ping script exits with code 0.
  </verify>
  <done>
    Nick has been notified via Discord that the experiment design is complete.
  </done>
</task>

</tasks>

<verification>
1. EXPERIMENT_DESIGN.md exists and specifies all 5 configurations with common settings
2. OPEN_QUESTIONS.md exists and documents open questions with options/recommendations
3. wmse_config_output.txt exists and shows complete 42-field config dict
4. verify_full_data_training.py runs successfully and confirms full message training
5. verify_config_correctness.py runs successfully and confirms all 5 configs are correct
6. Nick has been pinged on Discord
</verification>

<success_criteria>
- A future Claude instance can read EXPERIMENT_DESIGN.md and execute the full benchmark without ambiguity
- All open questions are documented with clear recommendations
- Verification scripts prove the experiment setup is correct (full data training, correct configs)
- WMSE config is dumped to file showing all expected fields
</success_criteria>

<output>
After completion, create `.planning/quick/13-design-benchmark-experiment-with-wmse-an/13-SUMMARY.md`
</output>
