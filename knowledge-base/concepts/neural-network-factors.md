---
type: concept
title: Neural Network Factors
created: 2026-03-01
tags: [neural-networks, approximation, inference]
---

# Neural Network Factors

Neural network factors are the primary approximation mechanism in NCE. When a bucket's exact complexity exceeds the ECL threshold, a neural network is trained to approximate the exact message, producing a `FactorNN` -- a lazy factor that evaluates the NN on demand.

## Training Pipeline

1. **Sampling**: `SampleGenerator` produces random variable assignments in the message scope.
2. **Target Computation**: For each assignment, the exact message value is computed via `compute_message_values()` (product of all bucket factors, marginalized).
3. **Backward Values** (optional): `compute_backward_values()` provides downstream weights for loss weighting.
4. **Preprocessing**: `DataPreprocessor` normalizes inputs and targets for NN training.
5. **Training**: `Trainer.train_model()` optimizes the NN using the selected loss function.
6. **Wrapping**: The trained NN is wrapped in a `FactorNN` object that implements the `FastFactor` interface.

## Neural Network Architecture

### Net (`nce/neural_networks/net.py`)
- Fully connected network with configurable hidden layers
- Tanh activation functions
- Xavier weight initialization
- Optional `bias_only` mode (no input weights -- biases serve as a lookup table)
- Optional `linspace_bias` mode (logsumexp combination of learned basis functions)

### Memorizer (extends Net)
- Maintains a dictionary of memorized (input, output) pairs
- Stores the top-N assignments by value seen during training
- Falls back to NN prediction when input is not memorized
- Combines NN generalization with exact recall of important assignments

### BitVectorLookup (extends Net)
- Uses `nn.Embedding` for binary-variable inputs
- Sums variable embeddings then passes through MLP
- Only works when all variables have 2 states

### SimpleNet
- Minimal network with a single scalar parameter
- Used for testing and baseline comparisons

## Input Encoding

Variable assignments are encoded as one-hot vectors:
- Each variable's assignment is a one-hot vector of size equal to the number of states
- All one-hot vectors are concatenated into a single input vector
- Optional `lower_dim` mode: uses n-1 dimensions (drops last category, implicit from others being zero)

Example: Variables [X0(3 states), X1(2 states)] with assignment [1, 0]:
- Standard encoding: [0,1,0, 1,0] (5-dimensional)
- Lower-dim encoding: [0,1, 1] (3-dimensional)

## FactorNN Interface

`FactorNN` (in `nce/inference/factor_nn.py`) wraps a trained NN to implement the `FastFactor` interface:
- `_get_slices(assignments, ...)`: Encodes assignments as one-hot, queries NN, returns predictions.
- `to_exact()`: Materializes the full tensor by evaluating the NN at all possible assignments.
- `get_factor_complexity()`: Returns the theoretical size without materializing.
- Participates in factor multiplication and elimination like any other factor.

## Decision Tree Alternative

`DecisionTreeLossOptimizer` (`nce/neural_networks/decision_tree.py`) provides an alternative to neural networks using sklearn's `DecisionTreeRegressor`:
- Fits a decision tree, then iteratively optimizes leaf values via gradient descent
- Supports multiple loss functions (log MSE, linear MSE, sampled message gradient)
- Can be converted directly to a `FastFactor` via `decision_tree_to_FastFactor()`

## Related

- [[factor-operations]]
- [[loss-functions]]
- [[variable-elimination]]
- [[backward-messages]]
