---
type: concept
title: Adversarial Learning for Message Approximation
created: 2026-03-01
tags: [training, neural-networks, adversarial, research-direction, optimization]
---

# Adversarial Learning for Message Approximation

Adversarial learning in the NCE context refers to a training setup where a **generator** model produces difficult training examples (inputs where the predictor is expected to have high loss), while the **predictor** (the message approximation network) learns to handle those examples. This is conceptually related to GANs but adapted to the regression problem of learning a mapping from discrete variable assignments to real-valued message outputs.

## Problem Setup

The NCE learning task for each bucket is:
- **Input**: discrete variable assignment vector (encoded as one-hot), dimension = sum of one-hot sizes for message scope variables.
- **Output**: a real number (the log-space message value at that assignment).
- **Training distribution**: currently uniform random sampling, or exhaustive enumeration.

Standard training uses a fixed, non-adaptive training distribution. Adversarial learning would make the training distribution **adaptive**: focus on assignments where the current predictor is worst.

## Adversarial Training Formulation

### Inner Maximization (Generator)
The generator maximizes the predictor's loss:
```
max_{x in X} L(predictor(x), target(x))
```
For discrete inputs, exact maximization requires evaluating the loss at all assignments (exponential in scope size). Approximations:
- **Stochastic search**: Sample candidates, evaluate loss, keep high-loss ones.
- **Gradient-based (via relaxation)**: Relax the discrete inputs to continuous, compute gradients w.r.t. inputs, project back.
- **Iterative quantized local search** (Guo et al.): Flip individual variable assignments greedily to increase loss.

### Outer Minimization (Predictor)
The predictor minimizes the loss on generator-produced examples:
```
min_{theta} E_{x ~ Generator} [ L(predictor_theta(x), target(x)) ]
```

## Relevance to NCE

The NCE message approximation NN takes discrete (one-hot) inputs and outputs real numbers. Key considerations:

### Why Adversarial Learning Might Help
- Uniform sampling concentrates training data randomly. For high-dimensional message scopes, most of the probability mass (high-value assignments) may be missed.
- An adversarial generator that focuses on high-loss regions could improve approximation where it matters most (especially relevant for the unnormalized KL loss, which weights errors by probability mass).
- Combined with backward messages: target assignments where `backward_message(x) * loss(x)` is large — these have the highest impact on the partition function error.

### Discrete Input Challenge
Unlike image-domain adversarial examples (continuous inputs), NCE's inputs are discrete one-hot vectors. The generator cannot use continuous gradient ascent directly. Options:
1. **Importance-weighted sampling**: Sample uniformly, reweight by current loss, use high-loss samples more.
2. **Greedy local search**: For a given assignment, try flipping each variable's value and keep the highest-loss variant. Repeat for K iterations.
3. **Genetic/evolutionary search**: Maintain a population of high-loss assignments, mutate and select.
4. **Separate generator network**: Train a generative model (e.g., small autoregressive network over the discrete space) to produce assignments that maximize the predictor's loss.

### Similarity to Active Learning
Adversarial learning in this context is closely related to **active learning** — querying the most informative data points. The key difference:
- Active learning: query oracle (expensive) for labels of selected inputs.
- NCE adversarial: compute target (exact message value) is expensive but feasible for any assignment. The "oracle" is `compute_message_values(assignments)`.

## Connection to Existing NCE Infrastructure

The `SampleGenerator.compute_message_values(assignments)` method can compute the exact message value at any assignment, not just random ones. An adversarial loop would:
1. Use the current trained NN to score all candidates.
2. Select high-loss assignments.
3. Call `compute_message_values()` to get targets.
4. Add these to the training set and retrain.

This requires modifying `DataLoader` to support an iterative adversarial loop, but the underlying computation is already implemented.

## Potential Concerns

- **Mode collapse**: The generator may fixate on a small set of assignments, causing the predictor to overfit to them while ignoring others.
- **Computational cost**: Adversarial search over a large discrete space is expensive. Needs efficient implementation (e.g., random restarts, limited search budget).
- **Convergence**: Standard GAN-style training can be unstable. For NCE, using the exact target (not a learned discriminator) avoids some instability, but the minimax dynamic still applies.

## Status

Adversarial learning is an open research direction for NCE, noted in `prompt.txt`. No current implementation exists. The simplest starting point is **loss-weighted importance sampling**: after initial training, compute losses on a validation set, and oversample high-loss assignments in subsequent training epochs.

## Sources

- [[@goodfellow2014gan]] — the GAN minimax framework this direction adapts (generator
  proposes high-loss assignments; the message network adapts). NCE's twist: the "oracle" is
  the *exact* message value, not a learned discriminator — closer to active learning.

## Related

- [[neural-network-factors]]
- [[sample-generation]]
- [[loss-functions]]
- [[curriculum-learning]]
- [[backward-messages]]
- [[importance-sampling]]
