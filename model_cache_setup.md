# Model Cache Setup for pyGMs Catalog Models

## Problem

`playground.py` (and any script calling `FastGM` on a pyGMs catalog model) hung indefinitely on
`FastGM` creation. No error was raised, no output was produced, and the process never timed out —
it simply froze.

## Root Cause

The pyGMs catalog uses lazy-loading: when a `Model` object is instantiated from the catalog, it
checks a local cache directory (`.model_cache/`) for three files per model:

1. `{model_name}.uai` — the factor graph in UAI format
2. `{model_name}.uai.ord` — the variable elimination order (one integer per line)
3. `{model_name}.uai.evid` — the evidence file

If any file is missing or has incorrect content, pyGMs calls `requests.get()` to download the
missing file from `sli.ics.uci.edu`. That server is unreachable from this machine, and
`requests.get()` has no timeout configured — causing an indefinite hang at the network call.

The hang happens silently during `Model` construction, before `FastGM` even runs, making the
root cause difficult to identify without tracing into pyGMs internals.

## Fix Applied (2026-03-09)

Three files were missing or incorrect in `.model_cache/grids/` for `grid10x10.f10.wrap`:

**1. `grid10x10.f10.wrap.uai` — was entirely absent.**
- Fix: Copied from `/home/cohenn1/SDBE/benchmark_problems/grid10x10.f10.wrap.uai`.

**2. `grid10x10.f10.wrap.uai.ord` — was absent.**
- Fix: Computed by running `wtminfill_order()` on the model and writing the variable list to disk
  (one integer per line). The resulting file contains 100 integers (the elimination order for the
  100-variable grid).

**3. `grid10x10.f10.wrap.uai.evid` — had wrong content.**
- The file contained an elimination order (a list of variable indices), not evidence.
- Fix: Replaced with a file containing just `0` — the correct format for an empty evidence set
  (0 observed variables).

After these three fixes, `FastGM(model=model, nn_config=nn_config, device='cuda')` completed
successfully in seconds.

**Note:** `.model_cache/` is gitignored, so these files are not tracked by git. They must be
present on disk for catalog-based code to work without network access.

## Pre-Caching Any Model

Before running any script that loads a model from the pyGMs catalog, verify the three required
files exist in `.model_cache/{category}/{model_name}`. Steps:

**Step 1 — Identify model name and category.**
Determine the model name (e.g., `grid10x10.f10.wrap`) and its category subdirectory
(e.g., `grids/`). The category matches the subdirectory used by the pyGMs catalog.

**Step 2 — Check or create the UAI file.**
```bash
ls .model_cache/{category}/{model_name}.uai
```
If absent, copy from an external source or download manually from
`https://sli.ics.uci.edu/PGT/Models` using a machine with network access.

**Step 3 — Check or create the elimination order file.**
```bash
ls .model_cache/{category}/{model_name}.uai.ord
```
If absent, compute it:
```python
import pyGMs as gm
from nce.inference.elimination_order import wtminfill_order

# Load model from UAI file directly (bypassing catalog to avoid the hang)
model = gm.GraphModel()
model.load('.model_cache/{category}/{model_name}.uai')

# Compute elimination order
elim_order = wtminfill_order(model)

# Write to disk, one integer per line
with open('.model_cache/{category}/{model_name}.uai.ord', 'w') as f:
    f.write('\n'.join(str(v) for v in elim_order))
```

**Step 4 — Check or create the evidence file.**
```bash
ls .model_cache/{category}/{model_name}.uai.evid
```
If absent or wrong, create with content `0` (no evidence, which is the correct format for an
empty evidence set):
```bash
echo "0" > .model_cache/{category}/{model_name}.uai.evid
```

**Step 5 — Verify by running a smoke test.**
```python
import pyGMs as gm
catalog = gm.CatalogModel()
model = catalog.load('{model_name}')  # Should complete in seconds, not hang
print(f"Loaded model with {model.num_vars} variables")
```

## Current Cache Status

As of 2026-03-09, the following models are pre-cached and working:

| Model | Category | UAI | ORD | EVID |
|-------|----------|-----|-----|------|
| grid10x10.f10.wrap | grids/ | yes | yes | yes (fixed) |
| grid10x10.f5.wrap | grids/ | yes | yes | (none needed) |
| grid10x10.f10 | grids/ | yes | yes | (none needed) |

Other models in `.model_cache/` (BN_*, or_chain_*, rbm_*, etc.) were pre-populated and are not
affected by this issue.

## Long-Term Recommendation

Consider adding a pre-flight check to `playground.py` that verifies all three cache files exist
before constructing the `Model` object. If any are missing, print a clear error message pointing
to this document rather than hanging on a network call.

Example check:
```python
import os

CACHE_DIR = '.model_cache'

def check_model_cache(category: str, model_name: str) -> None:
    """Raise an informative error if required cache files are missing."""
    base = os.path.join(CACHE_DIR, category, model_name)
    required = [base + '.uai', base + '.uai.ord', base + '.uai.evid']
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"Missing model cache files (would cause hang on download attempt):\n"
            + '\n'.join(f"  {f}" for f in missing)
            + f"\n\nSee docs/model_cache_setup.md for pre-caching instructions."
        )

# Call before loading from catalog:
check_model_cache('grids', 'grid10x10.f10.wrap')
model = catalog.load('grid10x10.f10.wrap')
```

This converts a silent hang into a clear, actionable error message.
