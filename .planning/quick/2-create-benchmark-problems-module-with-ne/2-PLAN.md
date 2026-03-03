---
phase: quick
plan: 2
type: execute
wave: 1
depends_on: []
files_modified:
  - nce/benchmark_problems/__init__.py
  - nce/benchmark_problems/neuro_be_sanity_check.py
  - nce/benchmark_problems/catalog_utils.py
  - docs/creating_benchmark_sets.md
  - CLAUDE.md
  - notebooks/March-2026/claude_experiments/test_benchmark_problems.py
autonomous: true
requirements: [BENCH-01]

must_haves:
  truths:
    - "import nce.benchmark_problems works without error"
    - "nce.benchmark_problems.neuro_be_sanity_check is a list of 4 Model objects"
    - "Each model object has .modelfile and .num_vars attributes"
    - "The 4 models are pedigree13, grid40x40.f10, grid20x20.f10, rbm_20"
    - "CLAUDE.md points to docs/creating_benchmark_sets.md"
    - "docs/creating_benchmark_sets.md contains thorough instructions with code snippets"
  artifacts:
    - path: "nce/benchmark_problems/__init__.py"
      provides: "Module entry point exporting neuro_be_sanity_check"
      contains: "neuro_be_sanity_check"
    - path: "nce/benchmark_problems/catalog_utils.py"
      provides: "Shared catalogue initialization with cache and pedigree source"
      contains: "get_catalog"
    - path: "nce/benchmark_problems/neuro_be_sanity_check.py"
      provides: "Benchmark set definition with 4 models"
      contains: "pedigree13"
    - path: "docs/creating_benchmark_sets.md"
      provides: "Instructions for creating new benchmark sets"
    - path: "CLAUDE.md"
      provides: "Updated with pointer to benchmark set docs"
      contains: "creating_benchmark_sets"
  key_links:
    - from: "nce/benchmark_problems/__init__.py"
      to: "nce/benchmark_problems/neuro_be_sanity_check.py"
      via: "import at module level"
      pattern: "from .neuro_be_sanity_check import"
    - from: "nce/benchmark_problems/neuro_be_sanity_check.py"
      to: "nce/benchmark_problems/catalog_utils.py"
      via: "get_catalog() call"
      pattern: "from .catalog_utils import get_catalog"
---

<objective>
Create an `nce.benchmark_problems` module that provides named benchmark sets loaded via the
pyGMs UAI catalogue. The first benchmark set is `neuro_be_sanity_check` containing pedigree13,
grid40x40.f10, grid20x20.f10, and rbm_20. Then update CLAUDE.md to point to a thorough
instructions document explaining how to create new benchmark sets.

Purpose: Makes it trivial to reference standard benchmark problem sets across experiments
by just importing them, instead of manually constructing TestProblem dicts with hardcoded paths.

Output: Working benchmark_problems module, instructions doc, updated CLAUDE.md.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md

<interfaces>
<!-- pyGMs catalogue API (from pyGMs.data.catalog): -->

```python
from pyGMs.data.catalog import Catalog, Model

# Catalog usage:
c = Catalog(cache=cache_dir, source=source_url)
# source_url = 'https://ics.uci.edu/~ihler/uai-data/models/index.json'
# cache_dir = '/home/cohenn1/NCE/.model_cache' (already exists with many cached models)

# Access models via path:
model = c['grids/grid40x40.f10']   # returns Model object
model = c['pedigree/pedigree13']   # pedigree set (see note below)
model = c['dbn/rbm_20']            # RBM models in dbn set

# Model properties:
model.modelfile  # str: e.g. 'grid40x40.f10.uai'
model.num_vars   # int: e.g. 1600
model.file       # str: full path to cached .uai file (triggers lazy download if not cached)
model.width      # int: treewidth
model.evidence   # dict: evidence variables
model.order      # tuple: elimination order
```

**IMPORTANT: Pedigree is NOT in the default remote index.json.**
The .model_cache/index.json must be patched to include:
```json
"pedigree": {
    "name": "pedigree",
    "description": "Pedigree genetic network models from UAI competitions",
    "modelset": "https://ics.uci.edu/~ihler/uai-data/models/pedigree/statistics.csv",
    "types": ["uai"]
}
```
This is needed because pedigree was added as a separate source in pyGMs
(see /home/cohenn1/SDBE/PyGMs/pyGMs/data/pedigree_index.json).
The catalog_utils.py helper should handle this automatically (read the pedigree_index.json
and merge entries into the local cache index if missing).

<!-- Existing FastGM accepts model objects via model= parameter: -->
```python
from nce.inference.graphical_model import FastGM
# FastGM.__init__ checks: if model is not None, uai_file = model.file
# So: FastGM(model=catalogue_model, nn_config=config, device=device) works
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create benchmark_problems module with catalogue utilities and neuro_be_sanity_check set</name>
  <files>
    nce/benchmark_problems/__init__.py
    nce/benchmark_problems/catalog_utils.py
    nce/benchmark_problems/neuro_be_sanity_check.py
  </files>
  <action>
Create the `nce/benchmark_problems/` package with three files:

**catalog_utils.py** -- Shared catalogue initialization:
```python
"""Utilities for initializing and accessing the pyGMs UAI model catalogue."""
import os
import json
from pyGMs.data.catalog import Catalog

# Default cache location (project-level, gitignored)
_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.model_cache')
_SOURCE_URL = 'https://ics.uci.edu/~ihler/uai-data/models/index.json'

# Additional model sets not in the default remote index
_EXTRA_SETS = {
    "pedigree": {
        "name": "pedigree",
        "description": "Pedigree genetic network models from UAI competitions",
        "modelset": "https://ics.uci.edu/~ihler/uai-data/models/pedigree/statistics.csv",
        "types": ["uai"]
    }
}

def _ensure_extra_sets(cache_dir):
    """Ensure extra model sets (e.g. pedigree) are in the cache index."""
    index_path = os.path.join(cache_dir, 'index.json')
    if not os.path.exists(index_path):
        return  # Will be created by Catalog.set_cache
    with open(index_path) as f:
        idx = json.load(f)
    updated = False
    for key, entry in _EXTRA_SETS.items():
        if key not in idx:
            idx[key] = entry
            updated = True
    if updated:
        with open(index_path, 'w') as f:
            json.dump(idx, f, indent=4)

def get_catalog(cache_dir=None, refresh=False):
    """Get a configured Catalog instance with all model sets available.

    Args:
        cache_dir: Path to cache directory. Defaults to .model_cache in project root.
        refresh: If True, force re-download of the catalogue index.

    Returns:
        Catalog instance ready to access models.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE
    os.makedirs(cache_dir, exist_ok=True)
    c = Catalog(cache=cache_dir, source=_SOURCE_URL)
    if refresh:
        c.update_cache()
    _ensure_extra_sets(cache_dir)
    return c
```

**neuro_be_sanity_check.py** -- The first benchmark set:
```python
"""neuro_be_sanity_check benchmark set.

Contains 4 models for quick sanity-checking of neural bucket elimination:
- pedigree13: large pedigree network (1077 vars, width 32)
- grid40x40.f10: large grid model (1600 vars, width 54)
- grid20x20.f10: medium grid model (400 vars)
- rbm_20: restricted Boltzmann machine (40 vars, width 20)
"""
from .catalog_utils import get_catalog

def _load_benchmark_set():
    """Load the neuro_be_sanity_check benchmark models from the catalogue."""
    catalog = get_catalog()
    models = [
        catalog['pedigree/pedigree13'],
        catalog['grids/grid40x40.f10'],
        catalog['grids/grid20x20.f10'],
        catalog['dbn/rbm_20'],
    ]
    return models

# Load on import so users can do: benchmark_problems.neuro_be_sanity_check
neuro_be_sanity_check = _load_benchmark_set()
```

**__init__.py** -- Module entry point:
```python
"""Benchmark problem sets for NCE experiments.

Usage:
    from nce.benchmark_problems import neuro_be_sanity_check

    for model in neuro_be_sanity_check:
        print(model.modelfile, model.num_vars)
"""
from .neuro_be_sanity_check import neuro_be_sanity_check
from .catalog_utils import get_catalog
```

Important notes:
- The `.model_cache/` directory is already gitignored and already contains many cached files.
- The `_ensure_extra_sets` function is the key -- it patches the local index.json to include
  pedigree (and any future extra sets) that aren't in the default remote catalogue.
- Models are loaded at import time via the Catalog, which only reads the stats CSV (small).
  The actual .uai files are downloaded lazily only when `.file` is accessed.
- Use `os.path.abspath(__file__)` to compute project root for the default cache path.
  </action>
  <verify>
cd /home/cohenn1/NCE && /home/cohenn1/NCE/venv/bin/python -c "
from nce.benchmark_problems import neuro_be_sanity_check
assert len(neuro_be_sanity_check) == 4, f'Expected 4 models, got {len(neuro_be_sanity_check)}'
names = [m.modelfile for m in neuro_be_sanity_check]
print('Models:', names)
assert 'pedigree13.uai' in names
assert 'grid40x40.f10.uai' in names
assert 'grid20x20.f10.uai' in names
assert 'rbm_20.uai' in names
for m in neuro_be_sanity_check:
    print(f'  {m.modelfile}: num_vars={m.num_vars}')
print('All assertions passed')
"
  </verify>
  <done>
    nce.benchmark_problems.neuro_be_sanity_check is a list of 4 Model objects.
    Each has .modelfile and .num_vars. The 4 models are pedigree13, grid40x40.f10,
    grid20x20.f10, and rbm_20.
  </done>
</task>

<task type="auto">
  <name>Task 2: Create benchmark set documentation and update CLAUDE.md</name>
  <files>
    docs/creating_benchmark_sets.md
    CLAUDE.md
    notebooks/March-2026/claude_experiments/test_benchmark_problems.py
  </files>
  <action>
**1. Create `docs/creating_benchmark_sets.md`:**

Write thorough instructions for creating new benchmark sets. This should be written AFTER
Task 1 is complete, based on the actual implementation. The document should include:

- Overview of the benchmark_problems module architecture (catalog_utils.py, per-set .py files, __init__.py)
- Step-by-step guide to creating a new benchmark set, including:
  1. How to browse the available catalogue categories and models (using `get_catalog().keys()` and `get_catalog()['category'].keys()`)
  2. How to create a new .py file in `nce/benchmark_problems/` following the neuro_be_sanity_check.py pattern
  3. How to register it in `__init__.py`
  4. How to handle models not in the default catalogue (like pedigree -- add to `_EXTRA_SETS` in catalog_utils.py)
- Full code snippets for:
  - Browsing the catalogue interactively
  - Creating a benchmark set file
  - Using a benchmark set in experiments (showing how model.file gives the .uai path for FastGM)
  - Refreshing the cache if needed
- Available model attributes (.modelfile, .num_vars, .file, .width, .evidence, .order)
- Available catalogue categories: alchemy, bn, csp, dbn, grids, objdetect, pedigree, ppi, promedas, protein, relational, segmentation

**2. Update CLAUDE.md:**

Add a new section after "### Data/Statistics Handling" (around line 112) titled:

```markdown
### Benchmark Problems

Benchmark problem sets are defined in `nce/benchmark_problems/`. Each set is a list of
pyGMs `Model` objects retrieved via the UAI catalogue.

```python
from nce.benchmark_problems import neuro_be_sanity_check
for model in neuro_be_sanity_check:
    fastgm = FastGM(model=model, nn_config=config, device=device)
```

For instructions on creating new benchmark sets, see `docs/creating_benchmark_sets.md`.
```

**3. Create a test script** at `notebooks/March-2026/claude_experiments/test_benchmark_problems.py`:
```python
"""Test that benchmark_problems module works correctly."""
import sys
sys.path.insert(0, '/home/cohenn1/NCE')
from nce.benchmark_problems import neuro_be_sanity_check

print("neuro_be_sanity_check benchmark set:")
print(f"  Number of models: {len(neuro_be_sanity_check)}")
print()
for model in neuro_be_sanity_check:
    print(f"  {model.modelfile}")
    print(f"    num_vars: {model.num_vars}")
    print(f"    width:    {model.width}")
    print()

print("SUCCESS: All models loaded correctly")
```

Run this test script to verify everything works end-to-end.
  </action>
  <verify>
cd /home/cohenn1/NCE && /home/cohenn1/NCE/venv/bin/python notebooks/March-2026/claude_experiments/test_benchmark_problems.py && test -f docs/creating_benchmark_sets.md && grep -q "creating_benchmark_sets" CLAUDE.md && echo "All checks passed"
  </verify>
  <done>
    Test script prints all 4 models with modelfile, num_vars, and width.
    docs/creating_benchmark_sets.md exists with thorough instructions and code snippets.
    CLAUDE.md contains a Benchmark Problems section pointing to the docs file.
  </done>
</task>

</tasks>

<verification>
1. `from nce.benchmark_problems import neuro_be_sanity_check` works without error
2. `len(neuro_be_sanity_check) == 4` with correct model names
3. Each model has `.modelfile` and `.num_vars` attributes
4. CLAUDE.md references docs/creating_benchmark_sets.md
5. docs/creating_benchmark_sets.md contains code snippets and step-by-step instructions
6. Test script runs successfully
</verification>

<success_criteria>
- `nce.benchmark_problems.neuro_be_sanity_check` is importable and contains 4 Model objects
- Models are pedigree13 (1077 vars), grid40x40.f10 (1600 vars), grid20x20.f10, rbm_20 (40 vars)
- Each model has .modelfile and .num_vars attributes accessible without downloading .uai files
- CLAUDE.md has a Benchmark Problems section with pointer to docs/creating_benchmark_sets.md
- docs/creating_benchmark_sets.md is thorough with code snippets based on actual implementation
</success_criteria>

<output>
After completion, create `.planning/quick/2-create-benchmark-problems-module-with-ne/2-SUMMARY.md`
</output>
