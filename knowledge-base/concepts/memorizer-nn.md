---
type: concept
title: Memorizer Neural Network
created: 2026-03-01
tags: [neural-networks, approximation, memorization, inference]
---

# Memorizer Neural Network

The Memorizer is a neural network variant in NCE that achieves perfect recall for all training samples while still generalizing to unseen inputs. It combines **exact lookup** for known assignments with a **linear fallback** for unseen inputs.

## Architecture

The `Memorizer` class (in `nce/neural_networks/net.py`) extends `Net` and has:

- `memory` (dict): Maps `tuple(input_one_hot_vector) -> target_value` for all training samples.
- `linear` (nn.Linear): A simple linear layer used for inputs not seen during training.

```python
class Memorizer(Net):
    def __init__(self, bucket, all_x, all_y):
        # Populate memory from all training samples
        for i in range(len(all_x)):
            input_vector = tuple(all_x[i].tolist())
            value = all_y[i].item()
            self.memory[input_vector] = value
        self.linear = nn.Linear(len(all_x[0]), 1)
```

## Forward Pass

```python
def forward(self, x):
    outputs = []
    for input_vector in x:
        input_tuple = tuple(input_vector.tolist())
        if input_tuple in self.memory:
            outputs.append(self.memory[input_tuple])  # Exact recall
        else:
            outputs.append(self.linear(input_vector).item())  # Linear fallback
    return torch.tensor(outputs, ...).view(-1, 1)
```

This is a simple per-sample dictionary lookup — O(1) per sample for memorized inputs.

## Training

`train_model()` is a no-op for the Memorizer:
```python
def train_model(self, X, Y, batch_size=None, ...):
    pass  # No training needed for memorization
```

The Memorizer stores all samples it was given at construction time; there is nothing to learn.

## When to Use (config flag)

Enabled via `config['use_memorizer'] = True`. The `compute_message_nn()` method in `FastBucket` detects this flag and:
1. Creates a `Net` for initialization (needed by Trainer/DataLoader).
2. Generates all samples (`all=True`).
3. Constructs a `Memorizer` from `(x_all, y_all)`.
4. Wraps it in a `FactorNN` for use as a factor.

## Use Case

The Memorizer is useful when:
- The message scope is small enough to enumerate all assignments exhaustively (`sampling_scheme='all'`).
- You want a **perfect upper bound** on what a NN can achieve for a given bucket (the Memorizer has zero training error by construction).
- Testing loss functions without NN approximation error as a confounder.

Note: The Memorizer only generalizes via the linear fallback for unseen inputs. For a message with a small discrete scope, all inputs may be seen during training, making the linear layer irrelevant.

## Limitations and Status

Per `prompt.txt`, the Memorizer has not yet been thoroughly tested. Potential issues to check:
1. Does the linear fallback initialize meaningfully (Xavier init in parent `Net`)? Since it is never trained, its initial random weights may produce poor predictions for unseen inputs.
2. Does FactorNN correctly wrap a Memorizer? The `_get_slices()` call chain should work since Memorizer extends Net and exposes the same `forward()` interface.
3. What happens when `to_exact()` is called? The Memorizer will be queried at all assignments — if some are unseen (e.g., due to batched training), the linear fallback is invoked.

## Related

- [[neural-network-factors]]
- [[sample-generation]]
- [[loss-functions]]
- [[factor-operations]]
