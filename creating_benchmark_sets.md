# Creating Benchmark Problem Sets

This guide explains how to create new benchmark problem sets in the `nce.benchmark_problems` module.

## Overview

Benchmark sets are named collections of pyGMs `Model` objects loaded from the UAI model catalogue, wrapped in a `BenchmarkSet` object that pairs models with their config dicts. They make it easy to reference standard problems across experiments without manually constructing paths or config dicts.

### Module Architecture

```
nce/benchmark_problems/
    __init__.py                  # Exports all benchmark sets + get_catalog
    catalog_utils.py             # Shared catalogue initialization (cache, pedigree patching)
    nbe_sanity_check.py          # Example benchmark set (4 models with neuroBE configs)
    your_new_set.py              # Your new benchmark set goes here
```

Each benchmark set file defines a `BenchmarkSet` instance that gets loaded at import time.

## Step-by-Step: Creating a New Benchmark Set

### 1. Browse Available Models

Use `get_catalog()` to explore what models are available:

```python
from nce.benchmark_problems import get_catalog

catalog = get_catalog()

# See all model categories
print(list(catalog.keys()))
# ['alchemy', 'bn', 'csp', 'dbn', 'grids', 'objdetect', 'pedigree',
#  'promedas', 'protein', 'ppi', 'relational', 'segmentation']

# Browse models in a category
print(list(catalog['grids'].keys()))
# ['grid10x10.f2', 'grid10x10.f5', 'grid20x20.f2', 'grid20x20.f10', ...]

# Inspect a specific model
model = catalog['grids/grid10x10.f5']
print(model.modelfile)   # 'grid10x10.f5.uai'
print(model.num_vars)    # 100
print(model.width)       # 14
```

### 2. Create the Benchmark Set File

Create a new Python file in `nce/benchmark_problems/`. Follow the pattern from `nbe_sanity_check.py`:

```python
# nce/benchmark_problems/my_new_set.py
"""my_new_set benchmark set.

Contains N models for [describe purpose]:
- model1: [brief description]
- model2: [brief description]
"""
from .catalog_utils import get_catalog
from .nbe_sanity_check import BenchmarkSet


_MODEL_KEYS = [
    'grids/grid10x10.f5',
    'dbn/rbm_10',
    # ... add more models
]


def _load_benchmark_set():
    """Load the my_new_set benchmark models from the catalogue."""
    catalog = get_catalog()
    return [catalog[key] for key in _MODEL_KEYS]


def _build_my_configs():
    """Build config dicts for each model (one per model, same order)."""
    configs = []
    for key in _MODEL_KEYS:
        configs.append({
            'device': 'cuda',
            'hidden_sizes': [64, 64],
            'iB': 10,
            'ecl': 2**22,
            # ... all other config fields
        })
    return configs


# Module-level instance
my_new_set = BenchmarkSet(
    problems=_load_benchmark_set(),
    configs={'default': _build_my_configs()},
)
```

Key points:
- Import `BenchmarkSet` from `nbe_sanity_check` (or define your own)
- The `_load_benchmark_set()` function uses `get_catalog()` to access models
- Models are accessed via `catalog['category/modelname']` syntax
- The `BenchmarkSet` wraps both models and configs into a single object
- Config dicts should be fully populated with all fields needed for experiments
- Model metadata (num_vars, width, etc.) is loaded from the cached statistics CSV -- no large file downloads happen at import

### 3. Register in `__init__.py`

Add your new set to `nce/benchmark_problems/__init__.py`:

```python
from .nbe_sanity_check import nbe_sanity_check
from .nbe_sanity_check import BenchmarkSet
from .my_new_set import my_new_set          # Add this line
from .catalog_utils import get_catalog
```

### 4. Test the New Set

```python
from nce.benchmark_problems import my_new_set

print(f"Number of models: {len(my_new_set.problems)}")
for model, config in zip(my_new_set.problems, my_new_set.configs['default']):
    print(f"  {model.modelfile}: {model.num_vars} vars, width={model.width}")
    print(f"  Config keys: {len(config)}")
```

## Using Benchmark Sets in Experiments

Benchmark models integrate directly with `FastGM`:

```python
from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM

for model, config in zip(nbe_sanity_check.problems, nbe_sanity_check.configs['nbe']):
    print(f"Running {model.modelfile} ({model.num_vars} vars)...")
    gm = FastGM(model=model, nn_config=config, device=config['device'])
    # ... run inference
```

The `model.file` property provides the full path to the cached `.uai` file. When you pass `model=model` to `FastGM`, it uses `model.file` internally to load the graphical model. The `.uai` file is downloaded lazily on first access if not already cached.

## Model Attributes

Each `Model` object from the catalogue has these attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.modelfile` | `str` | Model filename, e.g. `'grid40x40.f10.uai'` |
| `.num_vars` | `int` | Number of variables in the model |
| `.file` | `str` | Full path to cached `.uai` file (triggers download if not cached) |
| `.width` | `int` | Treewidth of the model |
| `.evidence` | `dict` | Evidence variables (if any) |
| `.order` | `tuple` | Pre-computed elimination order |

## Available Catalogue Categories

| Category | Description | Example Models |
|----------|-------------|----------------|
| `alchemy` | Alchemy MLN models | Various |
| `bn` | Bayesian networks | Various |
| `csp` | Constraint satisfaction problems | Various |
| `dbn` | Dynamic Bayes nets / RBMs | `rbm_10`, `rbm_20`, `rbm_ferromagnet_20` |
| `grids` | Grid Markov random fields | `grid10x10.f5`, `grid20x20.f10`, `grid40x40.f10` |
| `objdetect` | Object detection models | Various |
| `pedigree` | Pedigree genetic networks | `pedigree1` through `pedigree41` |
| `promedas` | Medical diagnosis models | Various |
| `protein` | Protein structure models | Various |
| `ppi` | Protein-protein interaction | Various |
| `relational` | Relational models | Various |
| `segmentation` | Image segmentation models | Various |

## Handling Models Not in the Default Catalogue

Some model sets (like `pedigree`) are not in the default remote `index.json`. If you need a model set that isn't available, add it to `_EXTRA_SETS` in `catalog_utils.py`:

```python
_EXTRA_SETS = {
    "pedigree": {
        "name": "pedigree",
        "description": "Pedigree genetic network models from UAI competitions",
        "modelset": "https://ics.uci.edu/~ihler/uai-data/models/pedigree/statistics.csv",
        "types": ["uai"]
    },
    # Add your new model set here:
    "new_set": {
        "name": "new_set",
        "description": "Description of the model set",
        "modelset": "https://url/to/statistics.csv",
        "types": ["uai"]
    }
}
```

The `_ensure_extra_sets()` function patches the local cache `index.json` automatically to include these entries.

## Refreshing the Cache

If the remote catalogue has been updated with new models:

```python
from nce.benchmark_problems import get_catalog

# Force refresh from remote
catalog = get_catalog(refresh=True)
```

This re-downloads the index and statistics CSVs. Cached `.uai` model files are not re-downloaded.
