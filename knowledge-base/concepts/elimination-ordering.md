---
type: concept
title: Elimination Ordering (wtminfill Heuristic)
created: 2026-03-01
tags: [inference, graphical-models, algorithm, heuristic]
---

# Elimination Ordering (wtminfill Heuristic)

The elimination order determines the sequence in which variables are removed from a graphical model during variable elimination. The order has a dramatic effect on computational complexity — a bad order can make exact inference exponentially harder, while a good order keeps intermediate factors (messages) small.

## Why Ordering Matters

When a variable is eliminated, a new factor is created over all of its current neighbors. This new factor becomes part of subsequent buckets. A good elimination order minimizes the scope of these intermediate factors, keeping the **induced width** (maximum message scope size) small.

The induced width of an elimination order determines the worst-case memory and time complexity of exact inference. Minimizing the induced width is NP-hard, so heuristics are used in practice.

## The Min-Fill Heuristic

The standard **min-fill** heuristic greedily eliminates the variable that adds the fewest edges to the interaction graph. When a variable is eliminated, fill edges are added between all pairs of its current neighbors that are not already connected.

Choosing the variable with fewest fill edges keeps the graph sparser, which tends to produce a lower induced width.

## The Weighted Min-Fill (wtminfill) Heuristic

NCE uses the **weighted min-fill** heuristic, implemented in `nce/inference/elimination_order.py` as `wtminfill_order()`.

The weight for each candidate variable is:

```
fill_weight = fill_edges * num_neighbors
```

This weights the fill count by the **cluster size** (number of neighbors), giving higher priority to variables that have fewer fill interactions relative to their cluster. The variable with the **lowest** fill weight is eliminated first.

### Algorithm

1. Build an adjacency matrix from all factors (connected variables share a factor).
2. For each candidate variable, compute `fill_weight = fill_edges * len(neighbors)`.
3. Sort candidates by ascending fill weight (lowest weight = eliminated first).
4. Eliminate the best candidate, update the adjacency matrix (add fill edges), recompute weights.
5. Repeat until all variables are eliminated.

### Implementation Details

```python
def wtminfill_order(factors_or_buckets, variables_not_eliminated=None):
    # ...
    pq.sort(reverse=True)  # Higher weight = higher priority means NOT to eliminate
    _, var = pq.pop()      # Pop last = lowest weight = eliminate this one
```

- Accepts either `FastFactor` objects or `FastBucket` objects as input.
- Optional `variables_not_eliminated` argument: specifies variables to **exclude** from elimination (e.g., query variables in backward message computation). These are appended at the end of the order.

## Usage in NCE

- **Forward pass**: `FastGM._load_from_uai()` calls `wtminfill_order()` to compute the elimination order for the full graphical model.
- **Backward message computation**: `get_backward_message()` calls `wtminfill_order()` on the downstream sub-model with the message scope variables pinned as `variables_not_eliminated`.
- **pyGMs interface**: pyGMs provides its own `eliminationOrder()` function (accessed via `pyGMs.graphmodel.eliminationOrder`), used as an alternative.

## Relationship to iB and ECL

A good elimination order keeps bucket scopes small, which reduces the need for WMB approximation. If the induced width is small enough that all bucket scopes fit within the ECL, exact inference is feasible. The iB parameter must be set relative to the actual induced width of the chosen order.

## Related

- [[variable-elimination]]
- [[weighted-mini-bucket]]
- [[bucket-structure]]
- [[iB-parameter]]
