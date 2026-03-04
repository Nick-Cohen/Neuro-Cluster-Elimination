---
phase: quick-4
plan: 4
type: execute
wave: 1
depends_on: []
files_modified:
  - nce/benchmark_problems/nbe_sanity_check.py
  - nce/benchmark_problems/__init__.py
  - notebooks/March-2026/test_benchmark_configs.py
  - CLAUDE.md
  - docs/creating_benchmark_sets.md
autonomous: true
requirements: [QUICK-4]

must_haves:
  truths:
    - "from nce.benchmark_problems import nbe_sanity_check works"
    - "nbe_sanity_check.problems is a list of 4 Model objects"
    - "nbe_sanity_check.configs['nbe'] is a list of 4 fully-populated config dicts"
    - "Each config dict contains all fields from the reference get_config() function"
    - "test_benchmark_configs.py runs and prints full configs"
  artifacts:
    - path: "nce/benchmark_problems/nbe_sanity_check.py"
      provides: "BenchmarkSet class with .problems and .configs attributes"
      contains: "class BenchmarkSet"
    - path: "nce/benchmark_problems/__init__.py"
      provides: "Exports nbe_sanity_check as single import"
      contains: "nbe_sanity_check"
    - path: "notebooks/March-2026/test_benchmark_configs.py"
      provides: "Updated test script using new import pattern"
      contains: "nbe_sanity_check.configs"
  key_links:
    - from: "nce/benchmark_problems/__init__.py"
      to: "nce/benchmark_problems/nbe_sanity_check.py"
      via: "from .nbe_sanity_check import nbe_sanity_check"
      pattern: "from \\.nbe_sanity_check import nbe_sanity_check"
---

<objective>
Restructure the benchmark_problems module: rename neuro_be_sanity_check to nbe_sanity_check,
consolidate the triple-import pattern into a single object with .problems and .configs['nbe']
attributes, and fully populate the config dicts based on the reference experiment file
(full_algorithm_12_4.py get_config()).

Purpose: Cleaner API, shorter name, and configs that are complete enough to run experiments
without needing to manually add missing fields.

Output: Renamed module file, updated __init__.py, updated test script, updated docs.
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
@notebooks/_1-2026/claude_files/full_algorithm_12_4.py
@notebooks/March-2026/test_benchmark_configs.py
@CLAUDE.md
@docs/creating_benchmark_sets.md

<interfaces>
<!-- Current module structure being replaced -->
From nce/benchmark_problems/neuro_be_sanity_check.py:
```python
_MODEL_KEYS = [
    'pedigree/pedigree13',
    'grids/grid40x40.f10',
    'grids/grid20x20.f10',
    'dbn/rbm_20',
]
_HIDDEN_SIZES_MAP = {
    'pedigree/pedigree13': 'nbe,3',
    'grids/grid40x40.f10': 'nbe,1',
    'grids/grid20x20.f10': 'nbe,1',
    'dbn/rbm_20': 'nbe,3',
}
neuro_be_sanity_check = _load_benchmark_set()  # list of Model objects
neuro_be_sanity_check_configs = _build_configs()  # dict keyed by catalogue key
neuro_be_sanity_check_configs_list = [...]  # list, same order as models
```

From notebooks/_1-2026/claude_files/full_algorithm_12_4.py (reference config template):
```python
def get_config(model_type, iB):
    return {
        'device': 'cuda',
        'hidden_sizes': hidden_sizes,
        'optimizer': 'adam',
        'lr': 0.01,
        'lr_decay': 1.0,
        'momentum': 0.9,
        'inverse_time_decay_constant': 100,
        'patience': 20,
        'min_lr': 1e-8,
        'num_epochs': 10000,
        'num_epochs2': 0,
        'nbe_early_stopping': False,
        'nbe_warmup_epochs': 0,
        'skip_early_stopping': True,
        'sampling_scheme': 'uniform',
        'batch_size': 50000,
        'set_size': 50000,
        'num_samples': 50000,
        'num_batches_per_set': 1,
        'loss_fn': 'unnormalized_kl',
        'traced_losses': [],
        'val_set': True,
        'fdb': False,
        'use_bw_approx': True,
        'populate_bw_factors': False,
        'ecl': 2**22,
        'iB': iB,
        'approximation_method': 'wmb',
        'bw_ecl': None,
        'backward_ecl': 2**22,
        'backward_iB': iB,
        'use_linspace_bias': False,
        'use_memorizer': False,
        'display_intermediate': False,
        'track_errors': False,
        'plot_messages': False,
        'debug': False,
        'lower_dim': False,
        'dope_factors': False,
        'gather_message_stats': False,
        'stratify_samples': False,
        'seed': 42,
    }
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create nbe_sanity_check.py with BenchmarkSet class and fully-populated configs</name>
  <files>
    nce/benchmark_problems/nbe_sanity_check.py
    nce/benchmark_problems/__init__.py
  </files>
  <action>
1. Create `nce/benchmark_problems/nbe_sanity_check.py` (new file, replacing neuro_be_sanity_check.py):

Define a simple `BenchmarkSet` class at the top of the file:
```python
class BenchmarkSet:
    """A named collection of benchmark problems with associated config sets."""
    def __init__(self, problems, configs):
        self.problems = problems   # list of Model objects
        self.configs = configs     # dict: config_name -> list of config dicts (same order as problems)
```

Keep the existing `_MODEL_KEYS` list and `_HIDDEN_SIZES_MAP` dict unchanged.

Keep `_load_benchmark_set()` returning the list of Model objects.

Rewrite `_build_configs()` to return a **list** of fully-populated config dicts (one per model, same order as `_MODEL_KEYS`). Each config dict must contain ALL fields from the reference `get_config()` in `full_algorithm_12_4.py`, with these neuroBE-specific values:
- `device`: `'cuda'`
- `hidden_sizes`: per-model from `_HIDDEN_SIZES_MAP` (e.g. `'nbe,3'`)
- `optimizer`: `'adam'`
- `lr`: `0.01`
- `lr_decay`: `1.0`
- `momentum`: `0.9`
- `inverse_time_decay_constant`: `100`
- `patience`: `20`
- `min_lr`: `1e-8`
- `num_epochs`: `10000`
- `num_epochs2`: `0`
- `nbe_early_stopping`: `False`
- `nbe_warmup_epochs`: `0`
- `skip_early_stopping`: `True`
- `sampling_scheme`: `'uniform'`
- `batch_size`: `256` (neuroBE uses 256, NOT the 50000 from 12_4 reference)
- `set_size`: `50000`
- `num_samples`: `'nbe'` (placeholder -- neuroBE uses special sampling)
- `num_batches_per_set`: `1`
- `loss_fn`: `'unnormalized_kl'`
- `traced_losses`: `[]`
- `val_set`: `True`
- `fdb`: `False`
- `use_bw_approx`: `True`
- `populate_bw_factors`: `False`
- `ecl`: `2**22`
- `iB`: `10` (best guess for sanity check models)
- `approximation_method`: `'wmb'`
- `bw_ecl`: `None`
- `backward_ecl`: `2**22`
- `backward_iB`: `10` (same as iB)
- `use_linspace_bias`: `False`
- `use_memorizer`: `False`
- `display_intermediate`: `False`
- `track_errors`: `False`
- `plot_messages`: `False`
- `debug`: `False`
- `lower_dim`: `False`
- `dope_factors`: `False`
- `gather_message_stats`: `False`
- `stratify_samples`: `False`
- `seed`: `42`

At module level, instantiate:
```python
nbe_sanity_check = BenchmarkSet(
    problems=_load_benchmark_set(),
    configs={'nbe': _build_nbe_configs()},
)
```

2. Delete the old `nce/benchmark_problems/neuro_be_sanity_check.py` file using `os.remove` or `rm`.

3. Update `nce/benchmark_problems/__init__.py`:
- Remove the three old imports (`from .neuro_be_sanity_check import ...`)
- Add single import: `from .nbe_sanity_check import nbe_sanity_check`
- Also export `BenchmarkSet` for type hinting: `from .nbe_sanity_check import BenchmarkSet`
- Update docstring to show new usage pattern:
  ```
  from nce.benchmark_problems import nbe_sanity_check
  for model, config in zip(nbe_sanity_check.problems, nbe_sanity_check.configs['nbe']):
      print(model.modelfile, config)
  ```
- Keep the `from .catalog_utils import get_catalog` import unchanged.
  </action>
  <verify>
    <automated>/home/cohenn1/NCE/venv/bin/python -c "
from nce.benchmark_problems import nbe_sanity_check
assert hasattr(nbe_sanity_check, 'problems'), 'Missing .problems'
assert hasattr(nbe_sanity_check, 'configs'), 'Missing .configs'
assert len(nbe_sanity_check.problems) == 4, f'Expected 4 problems, got {len(nbe_sanity_check.problems)}'
assert 'nbe' in nbe_sanity_check.configs, 'Missing nbe config set'
assert len(nbe_sanity_check.configs['nbe']) == 4, f'Expected 4 configs, got {len(nbe_sanity_check.configs[\"nbe\"])}'
c = nbe_sanity_check.configs['nbe'][0]
required_keys = ['device','hidden_sizes','optimizer','lr','lr_decay','momentum','num_epochs','batch_size','loss_fn','iB','ecl','backward_ecl','backward_iB','seed','sampling_scheme','approximation_method']
for k in required_keys:
    assert k in c, f'Missing key: {k}'
assert c['batch_size'] == 256, f'batch_size should be 256, got {c[\"batch_size\"]}'
assert c['iB'] == 10, f'iB should be 10, got {c[\"iB\"]}'
print('All assertions passed')
"</automated>
  </verify>
  <done>
    - nbe_sanity_check.py exists with BenchmarkSet class
    - neuro_be_sanity_check.py is deleted
    - __init__.py exports single nbe_sanity_check object
    - nbe_sanity_check.problems is list of 4 Model objects
    - nbe_sanity_check.configs['nbe'] is list of 4 fully-populated config dicts with all ~35 fields
  </done>
</task>

<task type="auto">
  <name>Task 2: Update test script and documentation references</name>
  <files>
    notebooks/March-2026/test_benchmark_configs.py
    CLAUDE.md
    docs/creating_benchmark_sets.md
  </files>
  <action>
1. Rewrite `notebooks/March-2026/test_benchmark_configs.py` to use the new import pattern:
```python
# %% Imports
from nce.benchmark_problems import nbe_sanity_check

# %% Print all models and their paired configs
print("=== nbe_sanity_check: Models + Configs ===\n")
for model, config in zip(nbe_sanity_check.problems, nbe_sanity_check.configs['nbe']):
    print(f"Model: {model.modelfile}")
    print(f"  num_vars: {model.num_vars}, width: {model.width}")
    print(f"  Config ({len(config)} keys):")
    for k, v in config.items():
        print(f"    {k}: {v}")
    print()

# %% Print config key summary
print("=== Config key summary ===\n")
sample_config = nbe_sanity_check.configs['nbe'][0]
print(f"Total config keys: {len(sample_config)}")
print(f"Keys: {list(sample_config.keys())}")

# %% Verify expected values
print("\n=== Verification ===\n")
expected_hs = {
    'pedigree/pedigree13': 'nbe,3',
    'grids/grid40x40.f10': 'nbe,1',
    'grids/grid20x20.f10': 'nbe,1',
    'dbn/rbm_20': 'nbe,3',
}
for model, config in zip(nbe_sanity_check.problems, nbe_sanity_check.configs['nbe']):
    # Find which key this model corresponds to
    hs = config['hidden_sizes']
    bs = config['batch_size']
    iB = config['iB']
    status = "PASS" if bs == 256 and iB == 10 else "FAIL"
    print(f"  [{status}] {model.modelfile}: hidden_sizes={hs}, batch_size={bs}, iB={iB}")

print("\nDone!")
```

2. Update `CLAUDE.md` benchmark section (around lines 116-123) to use the new import pattern:
```python
from nce.benchmark_problems import nbe_sanity_check
for model, config in zip(nbe_sanity_check.problems, nbe_sanity_check.configs['nbe']):
    fastgm = FastGM(model=model, nn_config=config, device=config['device'])
```

3. Update `docs/creating_benchmark_sets.md`:
- Change all references from `neuro_be_sanity_check` to `nbe_sanity_check`
- Update the example usage to show the new `.problems` / `.configs` pattern
- Update the "Register in __init__.py" section to show the new import style
- Update the filename reference from `neuro_be_sanity_check.py` to `nbe_sanity_check.py`
- Show the BenchmarkSet pattern as the way to create new sets
  </action>
  <verify>
    <automated>/home/cohenn1/NCE/venv/bin/python /home/cohenn1/NCE/notebooks/March-2026/test_benchmark_configs.py 2>&1 | head -30</automated>
  </verify>
  <done>
    - test_benchmark_configs.py runs successfully with new import pattern
    - Full config dicts are printed (all ~35 keys visible)
    - CLAUDE.md shows updated import pattern
    - docs/creating_benchmark_sets.md references nbe_sanity_check throughout
    - No remaining references to neuro_be_sanity_check in actively-used code
  </done>
</task>

</tasks>

<verification>
1. `from nce.benchmark_problems import nbe_sanity_check` imports without error
2. `nbe_sanity_check.problems` has 4 Model objects
3. `nbe_sanity_check.configs['nbe']` has 4 fully-populated config dicts
4. Each config dict has ~35 keys matching the reference get_config() template
5. `notebooks/March-2026/test_benchmark_configs.py` runs and prints all configs
6. Old `neuro_be_sanity_check.py` file no longer exists
7. No import errors from stale references
</verification>

<success_criteria>
- Single import `from nce.benchmark_problems import nbe_sanity_check` provides access to both problems and configs
- `nbe_sanity_check.problems` returns list of 4 Model objects
- `nbe_sanity_check.configs['nbe']` returns list of 4 config dicts, each with all ~35 fields fully populated
- Test script runs and prints full configs for user review
</success_criteria>

<output>
After completion, create `.planning/quick/4-restructure-benchmark-problems-rename-to/4-SUMMARY.md`
</output>
