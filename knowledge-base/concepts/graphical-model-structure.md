---
type: concept
title: Graphical Model Structure (FastGM)
created: 2026-03-01
tags: [inference, graphical-models, data-structure, architecture]
---

# Graphical Model Structure (FastGM)

`FastGM` (`nce/inference/graphical_model.py`) is the top-level container for a probabilistic graphical model in NCE. It owns all variables, buckets, factors, and the elimination order, and orchestrates the full inference pipeline.

## What a FastGM Contains

- `vars` (list[Var]): All variables (from pyGMs), each with a label (string) and a domain size (`states`).
- `buckets` (dict[str, FastBucket]): Maps each variable label to its `FastBucket`.
- `elim_order` (list[str]): The variable elimination order (from wtminfill or provided externally).
- `config` (dict): Configuration dictionary (nn config, device, ecl, iB, etc.). Each FastGM gets its own **copy** of the config (not shared — critical to avoid bugs when multiple GMs exist simultaneously).
- `logSS`: Log state space size (used to decide when exact computation is feasible).
- `populate_bw_factors` (bool): If True, pre-compute WMB backward factors for all buckets.
- `wmb_fw_partitions` (int): Total WMB partitions accumulated during backward factor population.

## Initialization Paths

FastGM can be initialized from:
1. **UAI file** (`uai_file=`): Standard graphical model file format. Loads factors via pyGMs, creates buckets via `_load_from_uai()`.
2. **Model object** (`model=`): A `TestProblem` object that wraps a UAI file with an elimination order, evidence, and logSS.
3. **Buckets** (`buckets=`, `elim_order=`): Directly from existing bucket list (used internally for downstream GMs in backward message computation).
4. **Factors** (`factors=`, `elim_order=`, `reference_fastgm=`): From a list of `FastFactor` objects (used for downstream GMs).

## Key Methods

### `eliminate_variables(up_to=None, exact=False)`
Main inference routine. Processes buckets in elimination order:
1. For each bucket (up to `up_to` if specified):
   - Decides exact vs. approximate based on EC vs. ECL.
   - Calls the appropriate `FastBucket.compute_message_*()` method.
   - Routes the resulting message to the correct next bucket.
2. If `exact=True`, forces exact computation for all buckets (overrides ECL check).

### `process_bucket(bucket, exact=False)`
Dispatches a single bucket to the appropriate computation method based on:
- `EC <= ecl` → `compute_message_exact()`
- `EC > ecl and approximation_method='nn'` → `compute_message_nn()`
- `EC > ecl and approximation_method='wmb'` → `compute_wmb_message()` (then multiplies all WMB messages into subsequent buckets)
- `EC > ecl and approximation_method='decision_tree'` → `compute_message_decision_tree()`

### `get_log_partition_function()`
Returns the log partition function accumulated in the root bucket (scalar factor with empty labels).

### `get_bucket(var_label)`
Returns the `FastBucket` for the given variable.

### `matching_var(label)`
Finds the `Var` object with the given label.

### `_wmb_eliminate(bucket)`
Runs WMB for a bucket: partitions factors, eliminates from each mini-bucket, routes all resulting messages.

## Evidence Handling

When evidence is provided (observed variable assignments), the FastGM conditions on evidence by:
1. For each evidence variable, finding all factors containing it.
2. Taking the slice of each factor at the observed value (reducing the factor's scope by one).
3. Removing the evidence variable from the elimination order.

## Backward Factor Population

When `populate_bw_factors=True`, FastGM pre-computes WMB backward factors for all buckets during `eliminate_variables()`. This caches the downstream WMB factors in `bucket.approximate_downstream_factors`, avoiding repeated backward message computation during training.

## pyGMs Integration

FastGM is initialized using pyGMs for:
- File loading (`pyGMs.filetypes.readEvidence14`, `pyGMs.graphmodel.GraphModel`).
- Variable representation (`pyGMs.Var`).
- Alternative WMB computation (`pyGMs.wmb`).
- Alternative elimination ordering (`pyGMs.graphmodel.eliminationOrder`).

## Test Problems

`nce/problems/test_problems.py` defines ~30 `TestProblem` instances that each specify a UAI file, elimination order, evidence, and logSS. These are used in notebooks as standardized benchmarks.

## Related

- [[variable-elimination]]
- [[bucket-structure]]
- [[factor-operations]]
- [[elimination-ordering]]
- [[weighted-mini-bucket]]
- [[backward-messages]]
- [[three-layer-architecture]]
