---
phase: quick-5
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - nce/inference/bucket.py
  - nce/benchmark_problems/nbe_sanity_check.py
  - notebooks/March-2026/claude_experiments/test_nbe_num_samples.py
autonomous: true
requirements: [NBE-SAMPLES-01, NBE-CONFIG-02, NBE-TEST-03]

must_haves:
  truths:
    - "nbe num_samples function computes correct sample count from bucket width and domain sizes"
    - "nbe_sanity_check configs have correct epsilon values and updated fields"
    - "grid10x10.f5.wrap is included as 5th model in nbe_sanity_check"
    - "Test file loads model, runs elimination, and calls nbe num_samples on a width-10 bucket"
  artifacts:
    - path: "nce/inference/bucket.py"
      provides: "compute_nbe_num_samples() static/class method on FastBucket"
      contains: "compute_nbe_num_samples"
    - path: "nce/benchmark_problems/nbe_sanity_check.py"
      provides: "Updated configs with nbe,<epsilon> and 5th model"
      contains: "grid10x10.f5.wrap"
    - path: "notebooks/March-2026/claude_experiments/test_nbe_num_samples.py"
      provides: "Test script exercising the nbe num_samples function"
      contains: "compute_nbe_num_samples"
  key_links:
    - from: "nce/inference/bucket.py"
      to: "nce/neural_networks/train.py"
      via: "config['num_samples'] resolved before Trainer.train()"
      pattern: "compute_nbe_num_samples"
    - from: "nce/benchmark_problems/nbe_sanity_check.py"
      to: "nce/inference/bucket.py"
      via: "config num_samples='nbe,<epsilon>' parsed by bucket"
      pattern: "nbe,"
---

<objective>
Implement the NeuroBE num_samples function, update nbe_sanity_check configs with correct epsilon values and additional changes, add grid10x10.f5.wrap as 5th benchmark model, and create a test file that exercises the function on a width-10 bucket.

Purpose: Enable NeuroBE-style sample count computation per bucket (based on bucket width and max domain size), replacing the placeholder 'nbe' string with the actual formula.
Output: Working nbe num_samples function callable from bucket, updated benchmark configs, test script.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@nce/inference/bucket.py
@nce/benchmark_problems/nbe_sanity_check.py
@nce/neural_networks/train.py
@nce/sampling/sample_generator.py
@nce/inference/graphical_model.py
@docs/neurobe_epsilon_values.md

<interfaces>
<!-- Key methods and patterns the executor needs -->

From nce/inference/bucket.py (FastBucket):
```python
def get_message_size(self):
    scopes = self.get_message_dimension()
    return np.prod([float(d) for d in scopes])

def get_message_scope(self):
    # returns list of variable labels in the message scope

def get_message_dimension(self):
    return [self.gm.matching_var(idx).states for idx in self.get_message_scope()]

# Hidden sizes nbe pattern (lines 197-208) - follow this pattern for num_samples:
if isinstance(hidden_sizes, str) and hidden_sizes.startswith('nbe'):
    if ',' in hidden_sizes:
        b = int(hidden_sizes.split(',')[1])
    else:
        b = 1
    message_size = self.get_message_size()
    h = b * math.ceil(math.log2(message_size)) if message_size > 1 else b
    hidden_sizes = [h, h]
```

From nce/inference/graphical_model.py (FastGM):
```python
def show_elimination(self, elim_vars=None, up_to=None, through=None, all=False)
def eliminate_variables(self, elim_vars=None, up_to=None, through=None, all=False, all_but=None, exact=False)
# up_to takes a Var object, eliminates all vars BEFORE that index (not including it)
```

From nce/benchmark_problems/nbe_sanity_check.py:
```python
_MODEL_KEYS = ['pedigree/pedigree13', 'grids/grid40x40.f10', 'grids/grid20x20.f10', 'dbn/rbm_20']
_HIDDEN_SIZES_MAP = { ... per-model 'nbe,{b}' values ... }
```

From nce/neural_networks/train.py (line 225, 244):
```python
nbe_val_size = max(1, self.config['num_samples'] // 9)  # needs num_samples to be int
num_samples = self.config['num_samples']  # used for num_sets = num_samples // set_size
```

From nce/inference/bucket.py (lines 105, 529):
```python
nbe_val_size = max(1, self.config['num_samples'] // 9)  # also needs int
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement compute_nbe_num_samples and resolve num_samples in bucket + update configs</name>
  <files>nce/inference/bucket.py, nce/benchmark_problems/nbe_sanity_check.py</files>
  <action>
**Part A: Add `compute_nbe_num_samples` to FastBucket in `nce/inference/bucket.py`**

Add a static method `compute_nbe_num_samples(w, l, epsilon)` to the FastBucket class that implements the NeuroBE formula:
```python
@staticmethod
def compute_nbe_num_samples(w, l, epsilon):
    """Compute NeuroBE number of samples for a bucket.

    Formula: nSamples = floor((pd + ln(1000)) / epsilon)
    where pd = temp * ln(temp/l), temp = (l-1)*w^2 + l*w + 4

    NeuroBE uses 80:20 split: 80% training, 20% validation.

    Args:
        w: bucket width (number of variables in message scope)
        l: max domain size of variables in scope
        epsilon: error tolerance parameter

    Returns:
        dict with keys: 'total', 'n_train', 'n_val'
    """
    import math
    temp = (l - 1) * w**2 + l * w + 4
    pd = temp * math.log(temp / l)
    n_total = int(math.floor((pd + math.log(1000)) / epsilon))
    n_train = int(math.floor(n_total * 0.8))
    n_val = n_total - n_train
    return {'total': n_total, 'n_train': n_train, 'n_val': n_val}
```

Also add a convenience instance method:
```python
def get_nbe_num_samples(self, epsilon):
    """Compute NeuroBE num_samples for this bucket using its actual width and domain sizes."""
    w = len(self.get_message_scope())
    dims = self.get_message_dimension()
    l = max(dims) if dims else 2
    return FastBucket.compute_nbe_num_samples(w, l, epsilon)
```

**Part B: Resolve 'nbe,<epsilon>' string in compute_message_nn**

In `compute_message_nn`, right after the hidden_sizes resolution block (around line 210, after `net = Net(...)` and `t = Trainer(...)` are created but BEFORE `t.train()` is called at line 299), add num_samples resolution. Actually, the resolution needs to happen BEFORE the Trainer is created because Trainer.__init__ reads config['num_samples']. Better approach: resolve it right after the hidden_sizes block, before creating Net/Trainer. Insert right before `net = Net(self, hidden_sizes=hidden_sizes)` at line 210:

```python
# Handle "nbe,<epsilon>" string format for num_samples
num_samples_cfg = self.config.get('num_samples')
if isinstance(num_samples_cfg, str) and num_samples_cfg.startswith('nbe'):
    if ',' in num_samples_cfg:
        epsilon = float(num_samples_cfg.split(',')[1])
    else:
        epsilon = 0.25  # default from NeuroBE Config.h
    nbe_result = self.get_nbe_num_samples(epsilon)
    self.config['num_samples'] = nbe_result['total']
    print(f"Bucket {self.label}: NBE num_samples (eps={epsilon}): total={nbe_result['total']}, train={nbe_result['n_train']}, val={nbe_result['n_val']}")
```

IMPORTANT: Also add the same resolution in the memorizer path (around line 96-106 area) before `nbe_val_size = max(1, self.config['num_samples'] // 9)` at line 105.

The Trainer/train.py code at lines 225 and 244 will then see an integer and work correctly. The 80:20 split is informational -- the actual validation split in train.py uses `num_samples // 9` which is roughly 10% for validation. The NeuroBE paper's 80:20 is different from this codebase's split; just resolve to the total and let the existing code handle the split as-is.

**Part C: Update nbe_sanity_check.py**

1. Add 'grids/grid10x10.f5.wrap' as 5th entry to `_MODEL_KEYS` list.

2. Add its entry to `_HIDDEN_SIZES_MAP`:
   ```python
   'grids/grid10x10.f5.wrap': 'nbe,1',  # grid type
   ```

3. Add a `_NUM_SAMPLES_MAP` dict (similar to `_HIDDEN_SIZES_MAP`):
   ```python
   _NUM_SAMPLES_MAP = {
       'pedigree/pedigree13': 'nbe,0.1',
       'grids/grid40x40.f10': 'nbe,0.35',
       'grids/grid20x20.f10': 'nbe,0.35',
       'dbn/rbm_20': 'nbe,0.1',
       'grids/grid10x10.f5.wrap': 'nbe,0.35',
   }
   ```

4. Add a `_IB_MAP` dict for per-model iB values:
   ```python
   _IB_MAP = {
       'pedigree/pedigree13': 20,
       'grids/grid40x40.f10': 20,
       'grids/grid20x20.f10': 10,
       'dbn/rbm_20': 20,
       'grids/grid10x10.f5.wrap': 10,  # small grid, iB=10 reasonable
   }
   ```

5. In `_build_nbe_configs()`, update the config dict to use per-model values:
   - `'num_samples': _NUM_SAMPLES_MAP[key]` (was `'nbe'`)
   - `'iB': _IB_MAP[key]` (was `10`)
   - `'skip_early_stopping': False` (was `True`)
   - `'loss_fn': 'weighted_mse'` (was `'unnormalized_kl'`)
   - `'use_bw_approx': False` (was `True`)

6. Update the module docstring to mention grid10x10.f5.wrap as the 5th model and update "4 models" to "5 models".
  </action>
  <verify>
Run: `/home/cohenn1/NCE/venv/bin/python -c "from nce.inference.bucket import FastBucket; r = FastBucket.compute_nbe_num_samples(20, 3, 0.1); print(r); assert r['total'] == 48999, f'Expected 48999, got {r[\"total\"]}'"` (matches the doc's example for w=20, l=3, eps=0.1).

Run: `/home/cohenn1/NCE/venv/bin/python -c "from nce.benchmark_problems import nbe_sanity_check; print(len(nbe_sanity_check.problems)); assert len(nbe_sanity_check.problems) == 5"` to verify 5 models load.

Run: `/home/cohenn1/NCE/venv/bin/python -c "from nce.benchmark_problems import nbe_sanity_check; c = nbe_sanity_check.configs['nbe'][0]; print(c['num_samples'], c['loss_fn'], c['skip_early_stopping'], c['use_bw_approx']); assert c['num_samples'] == 'nbe,0.1'; assert c['loss_fn'] == 'weighted_mse'; assert c['skip_early_stopping'] == False; assert c['use_bw_approx'] == False"` to verify config changes.
  </verify>
  <done>
- `FastBucket.compute_nbe_num_samples(w, l, epsilon)` returns correct sample counts matching doc examples
- `FastBucket.get_nbe_num_samples(epsilon)` works as instance method using bucket's actual scope
- 'nbe,<epsilon>' string in config is resolved to integer before Trainer sees it
- nbe_sanity_check has 5 models including grid10x10.f5.wrap
- All configs use correct per-model epsilon, iB, loss_fn='weighted_mse', skip_early_stopping=False, use_bw_approx=False
  </done>
</task>

<task type="auto">
  <name>Task 2: Find width-10 bucket and create test file</name>
  <files>notebooks/March-2026/claude_experiments/test_nbe_num_samples.py</files>
  <action>
**Part A: Find a width-10 bucket in grid10x10.f5.wrap**

Run the following to identify a bucket of width 10:
```python
from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM

# grid10x10.f5.wrap is the 5th model (index 4)
model = nbe_sanity_check.problems[4]
config = dict(nbe_sanity_check.configs['nbe'][4])
config['ecl'] = 2**30  # force exact
config['iB'] = 30      # force exact

fastgm = FastGM(model=model, nn_config=config, device='cpu')
fastgm.show_elimination(all=True)
```

From the output, identify a bucket with width exactly 10 (or closest to 10). Record the Var object (its label). Report this bucket index to the user in the test file output.

**Part B: Create test file**

Create `notebooks/March-2026/claude_experiments/test_nbe_num_samples.py` with:

```python
"""Test NeuroBE num_samples function on grid10x10.f5.wrap.

Loads the benchmark set, finds a bucket of width 10, eliminates up to it,
then calls the nbe num_samples function on that bucket.
"""
import sys
sys.path.insert(0, '/home/cohenn1/NCE')

from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM
from nce.inference.bucket import FastBucket

# --- Static function test ---
print("=== Static function test ===")
# Test case from docs: w=20, l=3, eps=0.1 -> 48999
result = FastBucket.compute_nbe_num_samples(20, 3, 0.1)
print(f"w=20, l=3, eps=0.1: {result}")
assert result['total'] == 48999, f"Expected 48999, got {result['total']}"

# Additional test cases from the doc
result2 = FastBucket.compute_nbe_num_samples(20, 3, 0.35)
print(f"w=20, l=3, eps=0.35: {result2}")
assert result2['total'] == 14000, f"Expected 14000, got {result2['total']}"
print("Static tests passed!\n")

# --- Load grid10x10.f5.wrap ---
print("=== Loading grid10x10.f5.wrap ===")
model = nbe_sanity_check.problems[4]  # 5th model
config = dict(nbe_sanity_check.configs['nbe'][4])
config['ecl'] = 2**30  # exact
config['iB'] = 30      # exact

fastgm = FastGM(model=model, nn_config=config, device='cpu')

# Show elimination to find width-10 bucket
print("\n=== Elimination scheme ===")
fastgm.show_elimination(all=True)

# Find the bucket with width 10
# (The actual variable label will be determined by running show_elimination above)
# Iterate through elim_order to find a width-10 bucket
target_var = None
for var in fastgm.elim_order:
    bucket = fastgm.buckets[var]
    w = len(bucket.get_message_scope())
    if w == 10:
        target_var = var
        print(f"\nFound width-10 bucket: var {var.label}")
        break

if target_var is None:
    # Try width closest to 10
    best_var, best_w = None, 0
    for var in fastgm.elim_order:
        bucket = fastgm.buckets[var]
        w = len(bucket.get_message_scope())
        if abs(w - 10) < abs(best_w - 10) or best_var is None:
            best_var, best_w = var, w
    target_var = best_var
    print(f"\nNo width-10 bucket found. Using closest: var {target_var.label} (width {best_w})")

# Eliminate up to (but not including) the target bucket
print(f"\n=== Eliminating up to bucket {target_var.label} ===")
fastgm.eliminate_variables(up_to=target_var)

# Get the bucket and compute nbe num_samples
bucket = fastgm.buckets[target_var]
w = len(bucket.get_message_scope())
dims = bucket.get_message_dimension()
l = max(dims) if dims else 2
epsilon = 0.35  # grid type

print(f"\n=== NBE num_samples for bucket {target_var.label} ===")
print(f"  Width (w): {w}")
print(f"  Max domain size (l): {l}")
print(f"  Epsilon: {epsilon}")

# Call instance method
result = bucket.get_nbe_num_samples(epsilon)
print(f"  Result: {result}")
print(f"  Total samples: {result['total']}")
print(f"  Training (80%): {result['n_train']}")
print(f"  Validation (20%): {result['n_val']}")

# Also verify static call matches
result_static = FastBucket.compute_nbe_num_samples(w, l, epsilon)
assert result == result_static, "Instance and static methods should match"
print("\nAll tests passed!")
```

Make sure the directory `notebooks/March-2026/claude_experiments/` exists (create if needed).

Run the test file and report:
1. The bucket variable label (index) that has width 10
2. The computed nbe num_samples for that bucket
  </action>
  <verify>
Run: `/home/cohenn1/NCE/venv/bin/python notebooks/March-2026/claude_experiments/test_nbe_num_samples.py`

Expected output shows:
- Static tests pass (48999 for w=20,l=3,eps=0.1)
- grid10x10.f5.wrap loads successfully
- A bucket with width 10 (or near 10) is found and reported
- Elimination runs to that point
- nbe num_samples returns a valid result dict
- "All tests passed!" printed at end
  </verify>
  <done>
- Test file runs end-to-end without errors
- Width-10 bucket identified and its variable label reported
- nbe num_samples function correctly applied to that bucket
- Both static and instance methods produce consistent results
  </done>
</task>

</tasks>

<verification>
1. `FastBucket.compute_nbe_num_samples(20, 3, 0.1)` returns `{'total': 48999, 'n_train': 39199, 'n_val': 9800}` matching the docs
2. `FastBucket.compute_nbe_num_samples(20, 3, 0.35)` returns `{'total': 14000, ...}` matching the docs
3. `nbe_sanity_check.problems` has 5 elements (grid10x10.f5.wrap added)
4. All configs have `loss_fn='weighted_mse'`, `skip_early_stopping=False`, `use_bw_approx=False`
5. Test file runs and identifies a width-10 bucket in grid10x10.f5.wrap
</verification>

<success_criteria>
- NeuroBE num_samples formula implemented and verified against known values from the documentation
- 'nbe,<epsilon>' config string resolved to integer in bucket before training
- nbe_sanity_check updated with 5 models and correct per-model configs
- Test file demonstrates end-to-end usage on a real model
- Width-10 bucket index reported to user
</success_criteria>

<output>
After completion, create `.planning/quick/5-implement-nbe-num-samples-function-updat/5-SUMMARY.md`
</output>
