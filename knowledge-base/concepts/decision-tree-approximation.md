---
type: concept
title: Decision Tree Approximation (DecisionTreeLossOptimizer)
created: 2026-03-01
tags: [approximation, neural-networks, decision-tree, inference]
---

# Decision Tree Approximation (DecisionTreeLossOptimizer)

The `DecisionTreeLossOptimizer` is an alternative to neural networks for approximating bucket messages. It fits a decision tree and then iteratively updates leaf values via gradient descent, allowing it to optimize arbitrary differentiable loss functions.

## Architecture

In `nce/neural_networks/decision_tree.py`:

1. **Fit phase**: A scikit-learn `DecisionTreeRegressor` is fit to the training data with a maximum of `num_leaves` (K) leaf nodes.
2. **Optimization phase**: The leaf values of the fitted tree are treated as parameters and updated iteratively via gradient-based optimization (PyTorch gradients through leaf value assignments).

The tree structure (splitting rules) is fixed after the fit phase — only the leaf values are updated. This makes the optimization convex with respect to the leaf values for MSE-type losses.

## Key Parameters

- `num_leaves` / `K`: Maximum number of leaf nodes. Higher K = more expressive but slower.
- `num_iterations`: Maximum gradient descent iterations.
- `learning_rate` (`dt_lr`): Step size for leaf value updates (default 0.25).
- `momentum` (`dt_momentum`): Momentum toward exact solution (default 0.25).
- `convergence_threshold` (`dt_convergence_threshold`): Stop when max change in leaf values is below this.

## Loss Function Support

Like the neural network Trainer, the decision tree optimizer reads `config['loss_fn']` and dispatches to the same loss functions defined in `losses.py`. This allows fair comparison between NN and decision tree approximations under identical loss objectives.

For sampled message gradient (SMG) losses, special handling extracts `sigma_f`, `sigma_g`, and `rho` from the loss function closure.

## Conversion to FastFactor

`decision_tree_to_FastFactor(dt_optimizer, gm)` converts a fitted `DecisionTreeLossOptimizer` into a `FastFactor` directly (not a `FactorNN`):
1. Enumerates all assignments in the message scope.
2. Evaluates the decision tree at each assignment.
3. Returns a `FastFactor` with the leaf values as tensor entries.

This is simpler than `FactorNN` (no lazy evaluation) because the decision tree produces a full materialized factor.

## Comparison to Neural Networks

| Property | Neural Network (Net) | Decision Tree (DTLossOptimizer) |
|---|---|---|
| Expressiveness | Arbitrary with enough depth | Limited to K leaf regions |
| Training cost | Many SGD iterations | One fit + few iterations |
| Generalization | Smooth function | Piecewise constant |
| Loss function support | Full | Full |
| Output type | FactorNN (lazy) | FastFactor (materialized) |
| Memory | Weights only | Full factor tensor |

## Related Files

- `nce/neural_networks/decision_tree.py` — main implementation
- `nce/neural_networks/dt2.py` — near-duplicate (likely dead code; see dead code report)

## Status

The decision tree approximation is implemented but its comparative performance vs. neural networks has not been systematically tested. It may be more efficient for small message scopes where the number of leaves K can cover most of the space.

## Sources

- [[@breiman1984cart]] — CART, the recursive-partitioning regression-tree method NCE fits
  before optimizing leaf values against the configured loss. (NCE uses scikit-learn's
  `DecisionTreeRegressor`, a CART implementation.)

## Related

- [[neural-network-factors]]
- [[loss-functions]]
- [[factor-operations]]
- [[sample-generation]]
- [[quantization]]
