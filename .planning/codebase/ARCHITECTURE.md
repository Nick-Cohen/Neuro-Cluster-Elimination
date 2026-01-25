# Architecture

**Analysis Date:** 2026-01-25

## Pattern Overview

**Overall:** Hybrid Inference with Neural Network Approximation

NCE (Neural Computational Elimination) implements a hybrid approach to probabilistic inference on graphical models. The core pattern combines exact elimination (bucket elimination / junction tree) with learned neural network approximations for high-complexity factors. The architecture uses a bucket-based elimination scheme where buckets that exceed complexity thresholds are approximated using neural networks trained to learn message-passing functions.

**Key Characteristics:**
- Bucket elimination framework with pluggable approximation methods
- Neural networks learn to approximate complex message computations
- Hybrid exact/approximate inference with configurable complexity thresholds
- Factor-based representation with automatic variable elimination
- Support for multiple approximation strategies: exact, neural network, weighted mini-bucket
- PyTorch integration for gradient-based learning and automatic differentiation

## Layers

**Inference Layer:**
- Purpose: Core probabilistic inference engine implementing bucket elimination
- Location: `nce/inference/`
- Contains: GraphicalModel, Buckets, Factors, Elimination schemes
- Depends on: PyTorch tensors, PyGMs library for model loading
- Used by: Neural network training, message passing, gradient computation

**Neural Network Layer:**
- Purpose: Learned approximations for complex message computations
- Location: `nce/neural_networks/`
- Contains: Network architectures, training routines, loss functions
- Depends on: Inference layer (queries buckets for training data), PyTorch
- Used by: Inference layer for approximation when buckets exceed thresholds

**Data/Sampling Layer:**
- Purpose: Training data generation and preprocessing for neural networks
- Location: `nce/data/` and `nce/sampling/`
- Contains: Sample generation, data preprocessing, one-hot encoding
- Depends on: Inference layer (gets factor values, message scope)
- Used by: Neural network training pipeline

**Utilities Layer:**
- Purpose: Cross-cutting concerns and helper functions
- Location: `nce/utils/`
- Contains: Message gradient computation, statistics, plotting, backward message functions
- Depends on: All layers
- Used by: Loss functions, training, analysis

## Data Flow

**Forward Elimination (Message Passing):**

1. User creates FastGM with factors and elimination order
2. eliminate_variables() iterates through elimination order
3. For each bucket:
   - Check if bucket complexity exceeds iB (width) or ecl (element count) thresholds
   - If below threshold: compute_message_exact() multiplies factors and eliminates variable
   - If above threshold (based on approximation_method config):
     - compute_message_nn(): Creates/trains neural network to approximate message
     - compute_message_wmb(): Uses weighted mini-bucket approximation
     - compute_message_dt(): Uses decision tree approximation
4. Output message passed to next bucket in elimination order
5. Final result: Partition function or marginal probability

**Neural Network Training (for high-complexity buckets):**

1. Trainer receives bucket to approximate
2. SampleGenerator creates training samples from message scope:
   - Uniformly samples assignments from message scope variables
   - Computes forward message values at sampled points (factor multiplication + elimination)
3. DataPreprocessor normalizes samples (handles log-space computations)
4. Net processes normalized assignments through hidden layers
5. Loss function (MSE, KL divergence, etc.) compares network output to true values
6. Backpropagation updates network weights
7. FactorNN wraps trained net, becomes usable as a factor in further eliminations

**Backward Message Computation (for loss functions):**

1. get_backward_message() or get_wmb_message_gradient() invoked
2. Creates new FastGM with remaining factors (not yet eliminated)
3. Eliminates all variables except those in the bucket's message scope
4. Returns backward message factor for use in loss computation

**State Management:**

- Config dict passed through all layers, contains training parameters and approximation thresholds
- FastGM maintains buckets dict mapping variables to FastBucket instances
- Each bucket maintains its factors and tracks whether it's been processed
- FactorNN holds reference to trained neural network, can be queried like any other factor
- Message scopes precalculated at FastGM initialization for routing

## Key Abstractions

**FastFactor:**
- Purpose: Represents probability factors in log space with labeled dimensions
- Examples: `nce/inference/factor.py`
- Pattern: Tensor-based with permutation-aware multiplication and elimination
- Operations: `__mul__` combines factors, `eliminate()` sums/maxes out variables, `to_exact()` converts NN to exact

**FastGM (Graphical Model):**
- Purpose: Orchestrates inference over entire model
- Examples: `nce/inference/graphical_model.py` (1800+ lines)
- Pattern: Manages buckets, elimination order, message routing
- Key methods: `eliminate_variables()`, `process_bucket()`, `_create_buckets_from_factors()`
- Extensibility: Supports UAI file loading, custom elimination orders, various approximation methods

**FastBucket:**
- Purpose: Container for factors targeting a single variable elimination
- Examples: `nce/inference/bucket.py`
- Pattern: Multiplies contained factors, computes outgoing messages
- Key methods: `compute_message_exact()`, `compute_message_nn()`, `compute_message_wmb()`
- Tracks: Complexity metrics, WMB statistics, downstream factors for backward messages

**FactorNN:**
- Purpose: Neural network-based factor that approximates complex messages
- Examples: `nce/inference/factor_nn.py`
- Pattern: Extends FastFactor with network evaluation capability
- Key method: `eliminate()` runs network in batches on all assignments, sums results
- Integration: Seamlessly replaces exact factors in further eliminations

**Net (Neural Network):**
- Purpose: Learns to approximate log-space message values
- Examples: `nce/neural_networks/net.py`
- Pattern: Configurable MLP or linear model with one-hot encoded inputs
- Features: Support for bias_only mode, linspace bias, Xavier initialization
- Forward: Accepts one-hot assignments, outputs log message estimate

**Trainer:**
- Purpose: Orchestrates neural network training for a bucket
- Examples: `nce/neural_networks/train.py`
- Pattern: Sets up dataloader, loss function, optimizer, runs training loop
- Extensibility: Supports convex early stopping, multiple loss functions, learning rate decay

**SampleGenerator:**
- Purpose: Generates training samples from message scope and computes their values
- Examples: `nce/sampling/sample_generator.py`
- Pattern: Deterministic seeding for reproducibility, supports uniform and exhaustive sampling
- Integration: Queries bucket factors for forward message values, supports backward factors

## Entry Points

**FastGM Initialization:**
- Location: `nce/inference/graphical_model.py` class FastGM.__init__()
- Triggers: User creates model from UAI file, raw factors, or pre-computed buckets
- Responsibilities: Loads model, computes elimination order if not provided, initializes buckets, calculates message scopes, optionally populates backward factors

**eliminate_variables():**
- Location: `nce/inference/graphical_model.py` method eliminate_variables()
- Triggers: User calls to perform inference (marginal computation, partition function)
- Responsibilities: Iterates elimination order, processes each bucket, routes messages, accumulates results

**process_bucket():**
- Location: `nce/inference/graphical_model.py` method process_bucket()
- Triggers: Called for each bucket during eliminate_variables()
- Responsibilities: Decides between exact/NN/WMB based on complexity, invokes appropriate computation, tracks statistics

**Trainer.train():**
- Location: `nce/neural_networks/train.py` method Trainer.train()
- Triggers: Called by bucket.compute_message_nn()
- Responsibilities: Loads training data, runs training loop with loss computation, early stopping, learning rate scheduling

## Error Handling

**Strategy:** Exception-based with detailed context

**Patterns:**
- Assertions validate tensor shapes, factor labels match expected elimination order
- Try-except blocks in eliminate_variables() with detailed error reporting showing bucket state, thresholds, factor complexities
- Tensor validity checks (NaN detection in output messages)
- Device consistency checks (factors match bucket device type)
- Graceful fallback in backward factor population: WMB fallback if exact computation fails

## Cross-Cutting Concerns

**Logging:**
- Uses print statements with descriptive messages
- Progress bars via tqdm for elimination and training
- Optional debug flag in config (config['debug']) gates extra output
- Message statistics collection when config['gather_message_stats'] = True

**Validation:**
- Factor label validation during multiplication (common labels alignment)
- Domain size extraction from tensor shapes or matched variables
- Elimination scope validation (only eliminable variables removed from buckets)

**Authentication:**
- Not applicable (pure computational library)

## Configuration Flow

All configuration is centralized in `nn_config` dict passed through:
- FastGM receives as parameter, stores in self.config
- Propagated to buckets via gm.config reference
- Buckets pass to Trainer
- Trainer passes to Net, SampleGenerator, loss functions
- Key config parameters:
  - `approximation_method`: 'exact', 'nn', 'wmb', 'dt'
  - `iB`: Maximum bucket width for exact computation
  - `ecl`: Exact computation limit (max tensor size in elements)
  - `hidden_sizes`: Neural network architecture
  - `loss_fn`: Loss function name ('mse', 'kl_div', etc.)
  - `num_epochs`, `batch_size`, `lr`: Training hyperparameters
  - `device`: 'cuda' or 'cpu'

---

*Architecture analysis: 2026-01-25*
