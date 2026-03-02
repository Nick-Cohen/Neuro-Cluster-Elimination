# Architecture

**Analysis Date:** 2026-02-21

## Pattern Overview

**Overall:** Multi-layered inference framework for neural approximation of probabilistic graphical models using message-passing elimination and neural network learning.

**Key Characteristics:**
- Graph-based elimination order computation with mini-bucket approximation (WMB)
- Bucket elimination algorithm with message-passing semantics
- Neural network approximation of intractable messages via supervised learning
- Data-driven training with sampling, preprocessing, and loss function composition
- Backward message computation for gradient-informed approximation
- Support for decision tree and linear solver alternatives to neural networks

## Layers

**Inference Core:**
- Purpose: Represents graphical models and orchestrates probabilistic inference via bucket elimination
- Location: `nce/inference/`
- Contains: `FastGM` (graphical model orchestrator), `FastBucket` (bucket abstraction), `FastFactor` (factor/tensor wrapper)
- Depends on: pyGMs library (external), torch for tensor operations
- Used by: All training and inference workflows

**Neural Network Approximation:**
- Purpose: Learns message approximations through supervised training with configurable architectures
- Location: `nce/neural_networks/`
- Contains: `Net` (PyTorch module), `Trainer` (training loop manager), loss functions
- Depends on: Inference core (buckets), data loaders, sampling generators
- Used by: Bucket elimination when `compute_message_nn()` is called

**Data & Sampling Pipeline:**
- Purpose: Generates training samples, computes message values, normalizes data
- Location: `nce/data/` and `nce/sampling/`
- Contains: `SampleGenerator` (deterministic sampling with seed control), `DataLoader` (message value computation), `DataPreprocessor` (normalization)
- Depends on: Inference core for message computation
- Used by: Neural network training, validation data generation

**Utility & Support:**
- Purpose: Message gradient computation, backward message approximation, visualization
- Location: `nce/utils/`
- Contains: `get_message_gradient()`, `get_backward_message()`, stats collection
- Depends on: Inference core, data layer
- Used by: Advanced training scenarios with backward approximation

**Test Problems & Benchmarks:**
- Purpose: Predefined UAI format graphical models for evaluation
- Location: `nce/problems/`
- Contains: TestProblem class, UAI files grouped by tree width, benchmark metadata
- Depends on: Inference core (FastGM)
- Used by: Notebooks and experiments

## Data Flow

**Inference & Elimination:**

1. User creates `FastGM` from UAI file or raw factors with elimination order
2. `FastGM` builds `FastBucket` instances, one per elimination variable
3. User calls `eliminate_variables()` to process buckets in order
4. Each bucket multiplies factors (including NN approximations if available) and marginalizes scope

**Neural Network Training Workflow:**

1. User specifies `nn_config` dict with loss function, optimizer, architecture, and data parameters
2. `FastGM` receives `nn_config` at init (stored as `config` dict)
3. When `bucket.compute_message_nn()` is called:
   - `Trainer` instantiates with bucket and config
   - `SampleGenerator` creates training samples with deterministic seeding
   - `DataLoader` loads samples and computes forward message values
   - Optional: `get_backward_message()` computes gradient factors, passes to `DataLoader`
   - `DataPreprocessor` normalizes targets using logsumexp-based centering
   - `Net` (neural network) trains with chosen loss function
   - Trained net approximates the message as a `FactorNN` object
4. Subsequent bucket elimination uses trained NN instead of exact computation

**Backward Message Flow (Optional):**

1. User calls `get_backward_message(gm, bucket_var)`
2. Creates downstream `FastGM` with remaining factors
3. Computes exact backward message via elimination
4. `SampleGenerator` evaluates backward factors at training samples
5. Loss function weights forward message error by backward message importance
6. Gradient signal focuses on high-importance configurations

**State Management:**

- **FastGM state**: Buckets, elimination order, configuration (immutable after init)
- **FastBucket state**: Factors list, message scope, cached message (mutable during elimination)
- **Trainer state**: Network parameters (evolve during training), optimizer state
- **SampleGenerator state**: Training/validation sample counters (for reproducibility)
- **DataPreprocessor state**: Normalizing constants computed once from full training data (used across all batches)

## Key Abstractions

**FastFactor:**
- Purpose: Wrapper for probabilistic factors (in log space) with tensor operations
- Examples: `nce/inference/factor.py`
- Pattern: Lazy tensors with label-based multiplication and marginalization
- Key methods: `__mul__()` (addition in log space), `eliminate()` (marginalization), `__matmul__()` (true multiplication)

**FastBucket:**
- Purpose: Represents a bucket in the elimination tree, manages message computation
- Examples: `nce/inference/bucket.py`
- Pattern: Can compute message exactly, via NN, via decision tree, or via linear solver (configurable)
- Key methods: `compute_message_exact()`, `compute_message_nn()`, `_get_nn_input_size()`

**FactorNN:**
- Purpose: Neural network that approximates a factor/message
- Examples: `nce/inference/factor_nn.py`
- Pattern: Wraps `Net` module, inherits from `FastFactor` interface
- Integration: Acts as normal factor during further elimination

**FastGM:**
- Purpose: Central orchestrator for graphical model operations
- Examples: `nce/inference/graphical_model.py`
- Pattern: Factory for buckets, maintains config dict passed to all downstream objects
- Key methods: `eliminate_variables()`, `process_bucket()`, `get_bucket()`

**Trainer:**
- Purpose: Manages single bucket training loop
- Examples: `nce/neural_networks/train.py`
- Pattern: Stateful - holds net, optimizer, learning rate scheduler; executes training epochs
- Loss functions: MSE, KL divergence, message gradient (mg) variants

**SampleGenerator:**
- Purpose: Deterministic sampling with reproducible seeds
- Examples: `nce/sampling/sample_generator.py`
- Pattern: Samples from message scope, computes message/backward values via factor evaluation
- Seeding: Bucket label + global seed + counter ensures reproducibility

**DataPreprocessor:**
- Purpose: Normalizes targets for stable learning
- Examples: `nce/data/data_preprocessor.py`
- Pattern: Lazy initialization from first batch, caches normalizing constant for all subsequent batches
- Backward message support: Weighted normalization when bw approximation enabled

## Entry Points

**Model Loading & Initialization:**
- Location: `nce/inference/graphical_model.py::FastGM.__init__()`
- Triggers: `FastGM(uai_file=path)` or `FastGM(factors=list, elim_order=list)`
- Responsibilities: Load UAI file via pyGMs, parse variables and factors, create buckets, validate device placement

**Training Workflow:**
- Location: `nce/neural_networks/train.py::Trainer` and `nce/inference/bucket.py::FastBucket.compute_message_nn()`
- Triggers: `bucket.compute_message_nn(loss_fn=...)`
- Responsibilities: Create trainer, generate data, train network, return trained NN factor

**Inference Execution:**
- Location: `nce/inference/graphical_model.py::FastGM.eliminate_variables()`
- Triggers: `gm.eliminate_variables()` or `gm.eliminate_max()`
- Responsibilities: Process buckets in elimination order, compute/multiply messages, marginalize

**Gradient Computation:**
- Location: `nce/utils/backward_message.py::get_backward_message()`
- Triggers: `get_backward_message(gm, bucket_var)`
- Responsibilities: Create downstream GM, compute exact backward factors, return as list

**Experimentation & Benchmarking:**
- Location: `nce/problems/test_problems.py`
- Triggers: `problem = test_problems["grid10x10.f10"]`
- Responsibilities: Load predefined test problem, provide ground truth partition function and metadata

## Error Handling

**Strategy:** Try-catch with logging at critical junctures; raises exceptions for fatal errors.

**Patterns:**

- **Factor multiplication**: Logs warnings if elimination fails in a bucket (mismatched scopes)
- **Device mismatches**: Asserts factor device matches bucket device at construction
- **Tensor shape errors**: Caught during permutation/reshaping in factor operations, re-raised with context
- **Numerical stability**: Uses logsumexp for log-space operations, max-shift in unnormalized KL loss
- **Missing config keys**: Uses `.get()` with defaults rather than KeyError

Example from `nce/inference/bucket.py`:
```python
try:
    message = message.eliminate(self.elim_vars)
except Exception as e:
    print(f"Warning: Elimination failed in bucket {self.label} with size {message.tensor.shape if message.tensor is not None else 'None'}: {e}")
    raise e
```

## Cross-Cutting Concerns

**Logging:**
- Console output via `print()` for progress (in Trainer loops, data generation)
- Statistics gathered in `FastGM.message_stats` list if `gather_message_stats=True`
- Error tracking in `FastGM.error_tracking_data` if `track_errors=True`
- No centralized logging framework; structured around FastGM attributes

**Validation:**
- Message scope validation: `SampleGenerator.get_message_scope_and_dims()` validates that sampled variables form valid scope
- Data normalization: `DataPreprocessor` ensures normalizing constant is computed from full dataset, not per-batch
- Factor device consistency: `FastBucket.__init__()` asserts all factors match device

**Authentication/Secrets:**
- Not applicable (no external services or credentials)

**Threading/Concurrency:**
- Not explicitly handled; assumes single-threaded use or manual synchronization at notebook level
- Torch CUDA operations are synchronous by default
- RNG seeding is deterministic per-bucket (see `SampleGenerator._compute_seed()`)

---

*Architecture analysis: 2026-02-21*
