---
type: concept
title: Bucket Structure (FastBucket)
created: 2026-03-01
tags: [inference, data-structure, graphical-models, variable-elimination]
---

# Bucket Structure (FastBucket)

A bucket is the fundamental organizational unit in variable elimination. Each variable in the elimination order has an associated bucket that collects all factors whose highest-priority variable (in the elimination order) is that variable's position.

## What a Bucket Contains

A `FastBucket` in NCE (`nce/inference/bucket.py`) has:

- `label` (str): The variable name being eliminated by this bucket.
- `factors` (list[FastFactor|FactorNN]): All factors assigned to this bucket.
- `elim_vars` (list[str]): Variables to eliminate (normally just `[label]`, but extensible).
- `isRoot` (bool): Whether this is the root bucket (accumulates the partition function).
- `gm` (FastGM): Reference to the parent graphical model.
- `wmb_stats` (dict): Tracks partitioning statistics for the forward and backward passes.
- `approximate_downstream_factors` (list[FastFactor]|None): Pre-computed WMB backward factors (when `populate_bw_factors=True`).

## Bucket Assignment

During graphical model initialization (`FastGM._load_from_uai`), each factor is assigned to the **earliest bucket** in the elimination order whose variable appears in the factor's scope. This is the standard variable elimination bucket assignment rule.

## Computation Modes

A bucket can compute its outgoing message in four ways:

### 1. Exact (`compute_message_exact`)
- Multiplies all factors together (log-space addition).
- Eliminates the bucket variable via logsumexp.
- Used when the bucket's exact complexity (EC) is within the ECL threshold.

### 2. Neural Network (`compute_message_nn`)
- Trains a neural network to approximate the exact message.
- Requires generating samples, computing targets, and running the Trainer.
- Produces a `FactorNN` that lazily evaluates the NN.
- Used when EC > ECL and `approximation_method='nn'`.

### 3. WMB (`compute_wmb_message`)
- Partitions factors into mini-buckets (groups of factors fitting within iB variables).
- Eliminates the bucket variable from each mini-bucket independently.
- Returns multiple messages (one per mini-bucket) that are multiplied into the next bucket.
- Used when `approximation_method='wmb'`.

### 4. Decision Tree (`compute_message_decision_tree`)
- Alternative to NN: fits a `DecisionTreeLossOptimizer` to approximate the message.
- Produces a `FastFactor` directly (no lazy evaluation).
- Used when `approximation_method='decision_tree'`.

## Key Properties

### Message Scope
The message output by a bucket has scope = (union of all factor scopes) minus the eliminated variable. `get_message_scope()` computes this.

### Exact Complexity (EC)
`get_exact_complexity()` returns the product of state counts over all variables in the bucket's combined scope. This determines whether exact computation is feasible.

### Message Complexity
`get_message_complexity()` returns the product of state counts in the **message scope** (after elimination). This is the size of the output tensor.

### Message Size
`get_message_size()` returns the total number of states in the message (i.e., message complexity).

### WMB Partitioning
`_get_wmb_partitions(iB)` partitions the bucket's factors into mini-buckets such that each mini-bucket's combined scope has at most `iB` variables. Returns a list of lists of factors.

## Key Role in Training

Buckets are the nexus of the training pipeline:
1. A bucket is identified as needing approximation (EC > ECL).
2. The bucket's factors and scope define the training problem.
3. `SampleGenerator` draws assignments in the message scope.
4. `compute_message_values()` evaluates the exact message at those assignments (via the bucket's factors).
5. The trained model (NN or decision tree) is wrapped in `FactorNN` and stored as the bucket's message.

## WMB Statistics Tracking

Each bucket tracks:
- `wmb_stats['fw_partitions']`: Number of WMB partitions in the forward pass.
- `wmb_stats['bw_partitions']`: Number of WMB partitions when computing backward message.
- `wmb_stats['num_mini_buckets']`: Number of mini-buckets (1 = exact, >1 = WMB).

## Related

- [[variable-elimination]]
- [[factor-operations]]
- [[weighted-mini-bucket]]
- [[iB-parameter]]
- [[elimination-ordering]]
- [[neural-network-factors]]
