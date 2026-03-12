---
phase: quick-7
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - nce/benchmark_problems/nbe_sanity_check.py
  - docs/nbe_evaluation_plan.md
  - notebooks/March-2026/claude_experiments/nbe_eval_phase0a_exact.py
  - notebooks/March-2026/claude_experiments/nbe_eval_phase0b_single_bucket.py
  - notebooks/March-2026/claude_experiments/nbe_eval_practice_1epoch.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "nbe_sanity_check configs match NeuroBE paper parameters (lr=0.001, num_epochs=500, dope_factors=True, loss_fn='weighted_logspace_mse')"
    - "backward_iB matches per-model iB for every config entry"
    - "Evaluation plan documents pre-smoke test phases (0a, 0b) and corrects all API usage"
    - "Phase 0a script runs grid10x10 in exact mode and prints log Z"
    - "Phase 0b script finds a large bucket, eliminates up to it, and trains 1 epoch on it"
    - "Practice script runs full inference with num_epochs=1 on grid10x10"
  artifacts:
    - path: "nce/benchmark_problems/nbe_sanity_check.py"
      provides: "Corrected NBE configs"
      contains: "weighted_logspace_mse"
    - path: "docs/nbe_evaluation_plan.md"
      provides: "Updated evaluation plan with pre-smoke phases"
      contains: "Phase 0a"
    - path: "notebooks/March-2026/claude_experiments/nbe_eval_phase0a_exact.py"
      provides: "Exact computation test script"
      contains: "get_log_partition_function"
    - path: "notebooks/March-2026/claude_experiments/nbe_eval_phase0b_single_bucket.py"
      provides: "Single bucket NN test script"
      contains: "compute_message_nn"
    - path: "notebooks/March-2026/claude_experiments/nbe_eval_practice_1epoch.py"
      provides: "Practice 1-epoch full run script"
      contains: "num_epochs"
  key_links:
    - from: "nbe_sanity_check.py configs"
      to: "losses.py::weighted_logspace_mse"
      via: "loss_fn config key"
      pattern: "weighted_logspace_mse"
    - from: "phase0b script"
      to: "graphical_model.py::eliminate_variables + get_bucket"
      via: "matching_var() converts int label to Var for up_to"
      pattern: "matching_var.*eliminate_variables"
---

<objective>
Fix all NBE config bugs in nbe_sanity_check.py, update the evaluation plan document with user feedback (new pre-smoke phases, API corrections, simplified goals), and create three pre-smoke test scripts.

Purpose: The current configs have 6 bugs that would cause failures or wrong results. The eval plan needs restructuring with pre-smoke phases. Scripts provide runnable verification before the main evaluation.
Output: Corrected configs, updated plan document, 3 runnable Python scripts.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@nce/benchmark_problems/nbe_sanity_check.py
@docs/nbe_evaluation_plan.md
@nce/inference/graphical_model.py (get_log_partition_function, get_bucket, get_large_message_buckets, dope_factors, show_elimination, eliminate_variables, matching_var)
@nce/inference/bucket.py (compute_message_nn, compute_nbe_num_samples, get_message_scope, get_message_dimension, get_message_size)
@nce/neural_networks/losses.py (weighted_logspace_mse is the registered name)

<interfaces>
<!-- Key APIs the scripts need to call correctly -->

From nce/inference/graphical_model.py:
```python
class FastGM:
    def get_log_partition_function(self):
        """Returns the log partition function. Computes if not already done."""

    def get_bucket(self, bucket):
        """Get bucket by int label or Var object."""

    def get_large_message_buckets(self, iB=None, ecl=None, debug=False):
        """Returns list of INTEGER labels (var.label) for buckets with large messages."""

    def dope_factors(self, new_min=-5):
        """Replace -inf values in factor tensors with new_min."""

    def show_elimination(self, elim_vars=None, up_to=None, through=None, all=False):
        """Print elimination scheme."""

    def eliminate_variables(self, elim_vars=None, up_to=None, through=None, all=False, all_but=None, exact=False):
        """Eliminate variables. up_to expects a Var object (from self.elim_order)."""

    def matching_var(self, var_index):
        """Convert integer var label to Var object."""
```

IMPORTANT: get_large_message_buckets returns INTEGER labels, but eliminate_variables(up_to=...) expects a Var object.
Must use fastgm.matching_var(int_label) to convert before passing to eliminate_variables.
get_bucket() accepts either int or Var.

From nce/inference/bucket.py:
```python
class FastBucket:
    def compute_message_nn(self, loss_fn='None', loss_fn2=None):
        """Train NN and compute message for this bucket."""

    def get_message_scope(self):
        """Returns sorted list of variable labels in scope (excluding bucket's own var)."""

    def get_message_size(self):
        """Returns product of domain sizes for all variables in message scope."""
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Fix all NBE config bugs in nbe_sanity_check.py</name>
  <files>nce/benchmark_problems/nbe_sanity_check.py</files>
  <action>
  In _build_nbe_configs(), apply these 6 corrections to the config dict:

  1. Change `'loss_fn': 'weighted_mse'` to `'loss_fn': 'weighted_logspace_mse'`
     - Reason: 'weighted_mse' is not a registered loss function name; the actual function in losses.py is 'weighted_logspace_mse'

  2. Change `'backward_iB': 10` to `'backward_iB': _IB_MAP[key]`
     - Reason: backward_iB should match the per-model iB, not be hardcoded to 10

  3. Change `'dope_factors': False` to `'dope_factors': True`
     - Reason: Models with deterministic factors have -inf values that cause NaN during training; doping replaces these with a small finite value

  4. Change `'num_epochs': 10000` to `'num_epochs': 500`
     - Reason: NeuroBE paper uses 500 epochs (from Config.h)

  5. Change `'lr': 0.01` to `'lr': 0.001`
     - Reason: NeuroBE paper uses lr=0.001 (from Config.h)

  6. Change `'backward_ecl': 2**22` to `'backward_ecl': None`
     - Reason: backward_ecl is irrelevant when use_bw_approx=False (which is the NBE default). Setting to None makes this explicit.

  Also update the module docstring to note the corrected NeuroBE hyperparameters (num_epochs=500, lr=0.001).
  </action>
  <verify>
  Run: /home/cohenn1/NCE/venv/bin/python -c "
from nce.benchmark_problems import nbe_sanity_check
configs = nbe_sanity_check.configs['nbe']
for i, c in enumerate(configs):
    assert c['loss_fn'] == 'weighted_logspace_mse', f'Config {i}: wrong loss_fn: {c[\"loss_fn\"]}'
    assert c['num_epochs'] == 500, f'Config {i}: wrong num_epochs: {c[\"num_epochs\"]}'
    assert c['lr'] == 0.001, f'Config {i}: wrong lr: {c[\"lr\"]}'
    assert c['dope_factors'] == True, f'Config {i}: dope_factors not True'
    assert c['backward_ecl'] is None, f'Config {i}: backward_ecl not None'
# Check backward_iB matches iB for each config
for c in configs:
    assert c['backward_iB'] == c['iB'], f'backward_iB {c[\"backward_iB\"]} != iB {c[\"iB\"]}'
print('All 6 config fixes verified.')
"
  </verify>
  <done>All 6 config bugs fixed: loss_fn corrected, backward_iB matches per-model iB, dope_factors=True, num_epochs=500, lr=0.001, backward_ecl=None.</done>
</task>

<task type="auto">
  <name>Task 2: Update docs/nbe_evaluation_plan.md with user feedback</name>
  <files>docs/nbe_evaluation_plan.md</files>
  <action>
  Rewrite docs/nbe_evaluation_plan.md incorporating all user feedback. Major changes:

  **Pre-Requisites section:**
  - Mark loss_fn, backward_iB, dope_factors, num_epochs, lr fixes as DONE (completed in Task 1)
  - Note that bw_ib/bw_ecl/backward_ecl are irrelevant for NBE (use_bw_approx=False)

  **Add Phase 0a (before existing Phase 1):**
  - Title: "Phase 0a: Exact Computation Baseline"
  - Purpose: Verify exact bucket elimination works on grid10x10.f5.wrap by setting ecl=2**30, iB=30
  - Use get_log_partition_function() (NOT run())
  - Script: nbe_eval_phase0a_exact.py

  **Add Phase 0b (before existing Phase 1):**
  - Title: "Phase 0b: Single Bucket NN Test"
  - Purpose: Test NN training on ONE large bucket in isolation
  - Load a problem (pedigree13 for largest buckets), set ecl=2**30/iB=30 for exact mode, num_epochs=1
  - Use get_large_message_buckets(iB=15) to find large buckets (returns INTEGER labels)
  - Use matching_var() to convert int label to Var before calling eliminate_variables(up_to=var)
  - Then get_bucket(target_label) and bucket.compute_message_nn()
  - Script: nbe_eval_phase0b_single_bucket.py

  **Fix Phase 1 (existing smoke test, renumber to Phase 1):**
  - Remove ecl/iB override -- use the benchmark config values as-is (they already have correct per-model iB and ecl)
  - Use get_log_partition_function() instead of run()
  - Add a practice step: run with num_epochs=1 first, then with correct num_epochs=500
  - Script: nbe_eval_practice_1epoch.py for the practice run

  **Fix hidden_sizes explanation throughout:**
  - 'nbe,1' means scaling factor b=1, so h = 1 * ceil(log2(message_size)), giving ONE hidden layer of that size (not [1,1])
  - 'nbe,3' means h = 3 * ceil(log2(message_size))

  **Fix rbm_20 note:**
  - rbm_20 does NOT have the highest width overall; it just has the highest width (20) among the sanity check set
  - pedigree13 has width 32, grid40x40.f10 has width 54

  **Simplify Phase 3 final output:**
  - Replace the dual NBE-vs-baseline table with a single table:
    Columns: problem name, num_trained, num_vars, total time, log Z true, log Z estimate, abs err

  **Add note about NeuroBE parameters:**
  - num_epochs=500 from NeuroBE Config.h
  - lr=0.001 from NeuroBE Config.h
  - Log Z comparison values can come from NeuroBE paper (prompts/NeuroBE.pdf) but exact match not expected due to different num_trained counts

  **Update execution order:**
  - Phase 0a -> Phase 0b -> Phase 1 (practice 1-epoch) -> Phase 1 (full run) -> Phase 2 -> ...

  Keep the overall structure (phases, scripts, file structure) but apply ALL corrections above.
  </action>
  <verify>
  Run: grep -c "Phase 0a" /home/cohenn1/NCE/docs/nbe_evaluation_plan.md && grep -c "Phase 0b" /home/cohenn1/NCE/docs/nbe_evaluation_plan.md && grep -c "get_log_partition_function" /home/cohenn1/NCE/docs/nbe_evaluation_plan.md && grep -c "matching_var" /home/cohenn1/NCE/docs/nbe_evaluation_plan.md && echo "All key content present"
  Expected: Each grep returns >= 1 match.
  </verify>
  <done>Evaluation plan updated with Phase 0a/0b, all API corrections (get_log_partition_function not run(), matching_var for Var conversion), corrected hidden_sizes explanation, simplified output table, NeuroBE parameter notes.</done>
</task>

<task type="auto">
  <name>Task 3: Create three pre-smoke test scripts</name>
  <files>
    notebooks/March-2026/claude_experiments/nbe_eval_phase0a_exact.py
    notebooks/March-2026/claude_experiments/nbe_eval_phase0b_single_bucket.py
    notebooks/March-2026/claude_experiments/nbe_eval_practice_1epoch.py
  </files>
  <action>
  Create directory notebooks/March-2026/claude_experiments/ if it doesn't exist.

  **Script 1: nbe_eval_phase0a_exact.py**
  - Load grid10x10.f5.wrap (index 4) from nbe_sanity_check
  - Copy its config with dict(), override ecl=2**30, iB=30, device='cpu' to force exact mode
  - Create FastGM, call dope_factors()
  - Call get_log_partition_function() and print result
  - Print model info (modelfile, num_vars)
  - If model has ln_z attribute, print exact value and absolute error

  **Script 2: nbe_eval_phase0b_single_bucket.py**
  - Load pedigree13 (index 0) -- 1077 vars, width 32, should have large buckets
  - Copy config, override ecl=2**30, iB=30, device='cpu', num_epochs=1
  - Create FastGM, call dope_factors()
  - Call get_large_message_buckets(iB=15, debug=True) to find buckets with >15 vars in scope
  - CRITICAL: get_large_message_buckets returns INTEGER labels, not Var objects
  - If no results, try iB=10
  - Pick first large bucket label (an integer)
  - Use fastgm.matching_var(target_label) to get the Var object
  - Call show_elimination(all=True) to display structure
  - Call eliminate_variables(up_to=target_var) where target_var is the Var object
  - Call get_bucket(target_label) -- integer works here
  - Print bucket info: scope size, dimensions, message size
  - Call bucket.compute_message_nn() and print message shape
  - Add note: pedigree13 on CPU may be slow; suggest cuda if available
  - Add try/except around the training call with helpful error message

  **Script 3: nbe_eval_practice_1epoch.py**
  - Load grid10x10.f5.wrap (index 4)
  - Copy config, override num_epochs=1, device='cpu'
  - DO NOT override ecl or iB -- use the benchmark defaults (ecl=2**22, iB=10)
  - Create FastGM, call dope_factors()
  - Call show_elimination(all=True)
  - Time the call to get_log_partition_function()
  - Print results: problem name, num_vars, log Z estimate, time elapsed
  - Print fastgm.num_trained (how many buckets used NN)
  - If model has ln_z, print true value and absolute error

  All scripts should use # %% cell markers for Jupyter-style execution in VS Code.
  </action>
  <verify>
  Run: /home/cohenn1/NCE/venv/bin/python -c "
import ast, sys
scripts = [
    'notebooks/March-2026/claude_experiments/nbe_eval_phase0a_exact.py',
    'notebooks/March-2026/claude_experiments/nbe_eval_phase0b_single_bucket.py',
    'notebooks/March-2026/claude_experiments/nbe_eval_practice_1epoch.py',
]
for s in scripts:
    try:
        with open(s) as f:
            ast.parse(f.read())
        print(f'PASS: {s} parses OK')
    except SyntaxError as e:
        print(f'FAIL: {s} syntax error: {e}')
        sys.exit(1)
print('All 3 scripts have valid Python syntax.')
"
  </verify>
  <done>Three scripts created: Phase 0a (exact computation on grid10x10), Phase 0b (single bucket NN on pedigree13 with correct int-to-Var conversion), practice 1-epoch run (grid10x10 with benchmark config defaults).</done>
</task>

</tasks>

<verification>
1. All 6 config fixes verified by import + assertions in Task 1 verify command
2. Evaluation plan contains Phase 0a, Phase 0b, correct API usage (grep checks in Task 2)
3. All 3 scripts parse without syntax errors (Task 3)
4. Scripts use correct API: get_large_message_buckets returns int labels, matching_var() converts to Var for eliminate_variables
</verification>

<success_criteria>
- nbe_sanity_check configs load with all 6 corrections (loss_fn, backward_iB, dope_factors, num_epochs, lr, backward_ecl)
- docs/nbe_evaluation_plan.md has Phase 0a, Phase 0b, corrected API usage, simplified output table
- Three scripts exist, parse cleanly, and use correct API patterns (matching_var for int-to-Var conversion)
</success_criteria>

<output>
After completion, create `.planning/quick/7-fix-nbe-configs-update-eval-plan-create-/7-SUMMARY.md`
</output>
