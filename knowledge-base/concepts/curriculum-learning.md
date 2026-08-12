---
type: concept
title: Curriculum Learning
created: 2026-03-01
tags: [training, neural-networks, optimization, research-direction]
---

# Curriculum Learning

Curriculum learning is a training strategy where the order in which examples are shown to a neural network is chosen deliberately to improve learning — typically by starting with "easier" examples and gradually introducing harder ones. This mimics how humans learn skills progressively.

## Core Idea

Standard training: present all training examples randomly (uniform shuffling each epoch).

Curriculum learning: sort or schedule examples by some difficulty measure, then present them in an order that improves convergence and generalization.

Two problems must be solved:
1. **Difficulty scoring**: Assign a "difficulty" score to each example.
2. **Pacing**: Determine how to gradually increase difficulty across training.

Reference: [Curriculum Learning (Bengio et al., 2009)](https://dl.acm.org/doi/10.1145/1553374.1553380)

## Benefits

- Faster convergence in early training (easy examples first builds a good initial representation).
- Better generalization, especially for non-convex losses with many local minima.
- Curriculum learning can be viewed as a **continuation method**: start with a simplified version of the problem, then gradually increase complexity.

Reference: [On The Power of Curriculum Learning in Training Deep Networks](https://arxiv.org/abs/1904.03626)

## Relevance to NCE

In NCE, each bucket presents a training problem: approximate a bucket message function. The difficulty varies across buckets:
- **Easy buckets**: Small message scope, simple factor structure, low variance in target values.
- **Hard buckets**: Large message scope, complex factor interactions, high variance or multimodal targets.

### Potential Curriculum Strategies

1. **By message scope size**: Train smaller-scope buckets first. Smaller scopes have smaller input spaces; the NN training problem is simpler.
2. **By target variance**: Train buckets with low-variance targets first (the function is smoother and easier to fit). Use `FastFactor.get_variance()` to assess target variance.
3. **By bucket position in elimination order**: Early buckets in the order often have smaller scopes (they eliminate simple variables first under wtminfill); process them before harder late buckets.
4. **By WMB partition count**: Buckets with more WMB partitions have more approximation error; training for these may benefit from curriculum over sampling difficulty.
5. **By backward message informativeness**: Use backward sensitivity analysis to prioritize buckets where the backward message has high variance (high sensitivity) — these contribute most to the final error.

### Sample-Level Curriculum

Within a single bucket's training, curriculum learning could order individual training samples:
- **Easy samples**: Assignments near the mode (high probability region) of the message.
- **Hard samples**: Low-probability or boundary assignments where the NN is likely to be inaccurate.
- Or the reverse: start with uniform samples, then focus on high-error samples (hard mining).

### Connection to Learning Rate Curriculum

The 2024 Learning Rate Curriculum (LeRaC) approach assigns different learning rates to different layers, creating a built-in curriculum without explicit example ordering. This is compatible with NCE's existing training setup and could be tried by varying layer-wise LR during training.

Reference: [Learning Rate Curriculum (LeRaC)](https://link.springer.com/article/10.1007/s11263-024-02186-5)

## Implementation Notes

The NCE Trainer currently trains each bucket independently, with i.i.d. shuffled samples across epochs. To implement curriculum learning:
- Modify `DataLoader.load_batches()` or `Trainer._make_dataloader()` to accept an ordering function.
- Add a difficulty scoring method to `SampleGenerator` (e.g., based on target variance or backward message weighting).
- Schedule the pacing in `Trainer.train_model()` (e.g., epoch-based or loss-based).

## Status

Curriculum learning is an open research direction for NCE, identified in `prompt.txt`. No implementation currently exists. The most natural starting point is a **scope-size curriculum** across buckets (eliminate easy buckets first in training order) since this requires no additional computation beyond the existing elimination order.

## Sources

- [[@bengio2009curriculum]] — the canonical curriculum-learning reference (easy→hard
  ordering as a continuation method). The other links above (LeRaC, "Power of Curriculum
  Learning") are inline URLs, not yet promoted to literature notes.

## Related

- [[neural-network-factors]]
- [[sample-generation]]
- [[loss-functions]]
- [[backward-sensitivity]]
- [[elimination-ordering]]
- [[adversarial-learning]]
