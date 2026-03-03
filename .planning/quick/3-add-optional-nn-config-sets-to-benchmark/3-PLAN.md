---
phase: quick-3
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - nce/benchmark_problems/neuro_be_sanity_check.py
  - nce/benchmark_problems/__init__.py
  - notebooks/March-2026/test_benchmark_configs.py
autonomous: true
requirements: [QUICK-3]

must_haves:
  truths:
    - "Benchmark sets can have optional nn_config sets paired with each model"
    - "neuro_be_sanity_check exposes a configs dict mapping model filenames to neuroBE configs"
    - "Test script loads benchmark set and paired configs, prints each config clearly"
  artifacts:
    - path: "nce/benchmark_problems/neuro_be_sanity_check.py"
      provides: "neuroBE config dicts paired with each benchmark model"
      contains: "neuro_be_sanity_check_configs"
    - path: "nce/benchmark_problems/__init__.py"
      provides: "Export of config sets alongside model sets"
      contains: "neuro_be_sanity_check_configs"
    - path: "notebooks/March-2026/test_benchmark_configs.py"
      provides: "Interactive test script with #%% cells"
      contains: "# %%"
  key_links:
    - from: "nce/benchmark_problems/neuro_be_sanity_check.py"
      to: "nce/benchmark_problems/__init__.py"
      via: "import export"
      pattern: "from .neuro_be_sanity_check import neuro_be_sanity_check_configs"
---

<objective>
Add optional nn_config sets to the benchmark_problems module, starting with neuroBE configs for
the neuro_be_sanity_check benchmark set. Produce an interactive test script for user verification.

Purpose: Allow benchmark sets to bundle recommended nn_config dicts alongside model lists, so
experiments can load both together without manually defining configs each time.

Output: Updated neuro_be_sanity_check.py with config dicts, updated __init__.py exports, and a
test script with #%% cells the user can run to verify configs match expectations.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@nce/benchmark_problems/neuro_be_sanity_check.py
@nce/benchmark_problems/__init__.py
@nce/benchmark_problems/catalog_utils.py
@configs/example_nn_config.py

<interfaces>
<!-- Key types and contracts the executor needs. -->

From nce/benchmark_problems/neuro_be_sanity_check.py (current):
```python
def _load_benchmark_set():
    catalog = get_catalog()
    models = [
        catalog['pedigree/pedigree13'],
        catalog['grids/grid40x40.f10'],
        catalog['grids/grid20x20.f10'],
        catalog['dbn/rbm_20'],
    ]
    return models

neuro_be_sanity_check = _load_benchmark_set()
```

From nce/inference/bucket.py (hidden_sizes 'nbe' parsing, lines 198-208):
```python
# Handle "nbe" or "nbe,{b}" string format for hidden sizes
if isinstance(hidden_sizes, str) and hidden_sizes.startswith('nbe'):
    import math
    if ',' in hidden_sizes:
        b = int(hidden_sizes.split(',')[1])
    else:
        b = 1
    message_size = self.get_message_size()
    h = b * math.ceil(math.log2(message_size)) if message_size > 1 else b
    hidden_sizes = [h, h]
```

From nce/inference/graphical_model.py (config keys used, lines 32-67):
```python
self.iB = self.config.get('iB', 0)
self.ecl = self.config.get('ecl', 0)
self.hidden_sizes = self.config.get('hidden_sizes', [])
self.loss_fn = self.config.get('loss_fn')
self.optimizer = self.config.get('optimizer')
self.lr = self.config.get('lr')
self.num_samples = self.config.get('num_samples')
self.num_epochs = self.config.get('num_epochs')
self.batch_size = self.config.get('batch_size')
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add neuroBE config dicts to neuro_be_sanity_check and export from __init__.py</name>
  <files>
    nce/benchmark_problems/neuro_be_sanity_check.py
    nce/benchmark_problems/__init__.py
  </files>
  <action>
In `nce/benchmark_problems/neuro_be_sanity_check.py`:

1. Add a new function `_build_configs()` that returns a dict mapping model identifiers to nn_config dicts.
   The keys should match the catalogue keys used in `_load_benchmark_set()` (e.g., `'pedigree/pedigree13'`).
   This makes it easy to pair models with configs.

2. Each config dict contains the neuroBE parameters. The per-model configs are:

   - `'pedigree/pedigree13'`: `{ 'batch_size': 256, 'hidden_sizes': 'nbe,3', 'num_samples': 'nbe' }`
   - `'grids/grid40x40.f10'`: `{ 'batch_size': 256, 'hidden_sizes': 'nbe,1', 'num_samples': 'nbe' }`
   - `'grids/grid20x20.f10'`: `{ 'batch_size': 256, 'hidden_sizes': 'nbe,1', 'num_samples': 'nbe' }`
   - `'dbn/rbm_20'`: `{ 'batch_size': 256, 'hidden_sizes': 'nbe,3', 'num_samples': 'nbe' }`

   Note: `'nbe,N'` is already a recognized string format in `bucket.py` for hidden_sizes.
   `'nbe'` for num_samples is a placeholder string for a function not yet implemented -- leave it as the string literal `'nbe'`.

3. Export the configs as a module-level variable: `neuro_be_sanity_check_configs = _build_configs()`

4. Also refactor `_load_benchmark_set()` to store the catalogue keys as a module-level constant
   `_MODEL_KEYS` so both functions reference the same list, avoiding duplication:
   ```python
   _MODEL_KEYS = [
       'pedigree/pedigree13',
       'grids/grid40x40.f10',
       'grids/grid20x20.f10',
       'dbn/rbm_20',
   ]
   ```

5. Update the module docstring to mention the config set.

In `nce/benchmark_problems/__init__.py`:

6. Add import: `from .neuro_be_sanity_check import neuro_be_sanity_check_configs`

7. Update the module docstring to show how to use configs:
   ```python
   """Benchmark problem sets for NCE experiments.

   Usage:
       from nce.benchmark_problems import neuro_be_sanity_check, neuro_be_sanity_check_configs

       for model in neuro_be_sanity_check:
           config = neuro_be_sanity_check_configs.get(model.modelfile)
           print(model.modelfile, config)
   """
   ```

   IMPORTANT: The config dict keys use catalogue keys (e.g. `'pedigree/pedigree13'`), but
   model objects have `.modelfile` (e.g. `'pedigree13.uai'`). So also add a helper function
   `get_configs_for_benchmark(models, configs)` that returns a list of configs in the same
   order as the models list. This function should match models to configs by extracting the
   model name from `model.modelfile` (strip `.uai` suffix) and searching the config keys for
   a match. Export this helper from `__init__.py`.

   Actually, simpler approach: make `_build_configs()` return BOTH a dict keyed by catalogue key
   AND provide a convenience function in neuro_be_sanity_check.py that takes a model and returns
   its config. Or even simpler: return a list of (model, config) tuples alongside the separate
   exports.

   SIMPLEST approach (do this one): In addition to the dict keyed by catalogue keys, also
   export `neuro_be_sanity_check_configs_list` which is a list of config dicts in the same
   order as `neuro_be_sanity_check` (the models list). This way users can just `zip()`:
   ```python
   for model, config in zip(neuro_be_sanity_check, neuro_be_sanity_check_configs_list):
       ...
   ```
   Both the dict (keyed by catalogue key) and the list should be exported.
  </action>
  <verify>
    <automated>/home/cohenn1/NCE/venv/bin/python -c "from nce.benchmark_problems import neuro_be_sanity_check, neuro_be_sanity_check_configs; print(len(neuro_be_sanity_check), len(neuro_be_sanity_check_configs)); assert len(neuro_be_sanity_check_configs) == 4; print('OK')"</automated>
  </verify>
  <done>
    - neuro_be_sanity_check_configs is a dict with 4 entries keyed by catalogue keys
    - neuro_be_sanity_check_configs_list is a list of 4 config dicts matching model order
    - Each config has batch_size=256, hidden_sizes='nbe,N' with correct N per model, num_samples='nbe'
    - __init__.py exports both neuro_be_sanity_check_configs and neuro_be_sanity_check_configs_list
  </done>
</task>

<task type="auto">
  <name>Task 2: Create interactive test script with #%% cells</name>
  <files>
    notebooks/March-2026/test_benchmark_configs.py
  </files>
  <action>
Create `notebooks/March-2026/test_benchmark_configs.py` (create the March-2026 directory if it
does not exist). This is a Python script with `# %%` cell markers for use in VS Code / Spyder
interactive mode.

The script should have these cells:

**Cell 1: Imports**
```python
# %%
from nce.benchmark_problems import (
    neuro_be_sanity_check,
    neuro_be_sanity_check_configs,
    neuro_be_sanity_check_configs_list,
)
```

**Cell 2: Print all models and their paired configs**
```python
# %%
print("=== neuro_be_sanity_check: Models + Configs ===\n")
for model, config in zip(neuro_be_sanity_check, neuro_be_sanity_check_configs_list):
    print(f"Model: {model.modelfile}")
    print(f"  num_vars: {model.num_vars}, width: {model.width}")
    print(f"  Config: {config}")
    print()
```

**Cell 3: Print the full configs dict (keyed by catalogue key)**
```python
# %%
print("=== Configs dict (keyed by catalogue key) ===\n")
for key, config in neuro_be_sanity_check_configs.items():
    print(f"  {key}: {config}")
```

**Cell 4: Verify expected values**
```python
# %%
print("=== Verification ===\n")
# Check expected hidden_sizes per model
expected = {
    'pedigree/pedigree13': 'nbe,3',
    'grids/grid40x40.f10': 'nbe,1',
    'grids/grid20x20.f10': 'nbe,1',
    'dbn/rbm_20': 'nbe,3',
}
for key, expected_hs in expected.items():
    actual_hs = neuro_be_sanity_check_configs[key]['hidden_sizes']
    actual_bs = neuro_be_sanity_check_configs[key]['batch_size']
    status = "PASS" if actual_hs == expected_hs and actual_bs == 256 else "FAIL"
    print(f"  [{status}] {key}: hidden_sizes={actual_hs}, batch_size={actual_bs}")

print("\nAll checks passed!" if all(
    neuro_be_sanity_check_configs[k]['hidden_sizes'] == v and
    neuro_be_sanity_check_configs[k]['batch_size'] == 256
    for k, v in expected.items()
) else "\nSome checks FAILED!")
```

Keep it clean and readable. The user will run this interactively cell-by-cell.
  </action>
  <verify>
    <automated>/home/cohenn1/NCE/venv/bin/python /home/cohenn1/NCE/notebooks/March-2026/test_benchmark_configs.py 2>&1 | grep -c "PASS"</automated>
  </verify>
  <done>
    - Test script exists at notebooks/March-2026/test_benchmark_configs.py
    - Script has 4 cells with # %% markers
    - Running the script prints model info, configs, and passes all verification checks
    - All 4 models show PASS for expected hidden_sizes and batch_size values
  </done>
</task>

</tasks>

<verification>
Run the test script end-to-end:
```bash
/home/cohenn1/NCE/venv/bin/python /home/cohenn1/NCE/notebooks/March-2026/test_benchmark_configs.py
```
Expected: All 4 models print PASS, final line says "All checks passed!"
</verification>

<success_criteria>
- `from nce.benchmark_problems import neuro_be_sanity_check_configs` works
- Config dict has 4 entries with correct per-model neuroBE parameters
- `zip(neuro_be_sanity_check, neuro_be_sanity_check_configs_list)` pairs models to configs correctly
- Test script runs cleanly and all checks pass
- User can run the test script interactively with #%% cells to inspect configs
</success_criteria>

<output>
After completion, create `.planning/quick/3-add-optional-nn-config-sets-to-benchmark/3-SUMMARY.md`
</output>
