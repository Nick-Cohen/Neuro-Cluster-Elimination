---
type: pattern
title: Three-Layer Architecture
created: 2026-03-01
tags: [architecture, design]
---

# Pattern: Three-Layer Architecture

## Description

The NCE package is organized into three main layers that separate concerns:

1. **Inference Layer** (`nce/inference/`): Core graphical model operations.
   - `FastFactor`: Data structure for factors in log space.
   - `FactorNN`: Neural network-backed lazy factor.
   - `FastBucket`: Variable elimination bucket.
   - `FastGM`: Graphical model container and elimination orchestrator.
   - `elimination_order.py`: Heuristic for computing elimination orders.
   - `message_gradient_factors.py`: Backward factor population.

2. **Neural Network Layer** (`nce/neural_networks/`): Approximation models and training.
   - `Net`, `Memorizer`, `BitVectorLookup`: Network architectures.
   - `Trainer`: Training orchestration.
   - `losses.py`: Loss function library.
   - `decision_tree.py`: Alternative decision tree approximation.
   - `quantization.py`: Optimal quantization approximation.
   - `linear_mse_solver.py`: Closed-form linear model solver.

3. **Data/Sampling Layer** (`nce/data/`, `nce/sampling/`): Training data generation.
   - `SampleGenerator`: Generates variable assignments and computes targets.
   - `DataLoader`: Orchestrates data loading pipeline.
   - `DataPreprocessor`: Normalizes data for network training.

Supporting layers:
- **Utils** (`nce/utils/`): Backward messages, sensitivity analysis, plotting, pyGMs integration, statistics.
- **Problems** (`nce/problems/`): Test problem definitions and benchmarks.

## Example

```
User code / Notebooks
       |
       v
  FastGM (inference layer)
       |
       |--- exact elimination (FastBucket.compute_message_exact)
       |
       |--- approximate elimination:
       |        |
       |        v
       |    SampleGenerator (data layer)  -->  DataLoader  -->  DataPreprocessor
       |        |
       |        v
       |    Trainer (neural network layer)  -->  Loss Functions
       |        |
       |        v
       |    FactorNN (inference layer - wraps trained NN)
       |
       v
  Messages flow forward through buckets
```
