# Codebase Concerns

**Analysis Date:** 2026-01-25

## Tech Debt

**Bare Exception Handlers:**
- Issue: Multiple bare `except:` clauses catch all exceptions including system exits and keyboard interrupts, masking errors and preventing debugging
- Files:
  - `nce/inference/factor.py` (lines 48, 86, 343)
  - `nce/inference/fastElim.py` (lines 67, 809, 948)
  - `nce/inference/message_gradient_factors.py` (line 129)
  - `nce/inference/graphical_model.py` (lines 798, 1464)
  - `nce/sampling/sample_generator.py` (line 298)
- Impact: Errors silently swallowed, followed by potentially invalid data flowing through computation. Difficult to trace bugs. Lines like `except:` followed by `print("got here")` and `raise()` indicate incomplete error handling.
- Fix approach: Replace all bare `except:` with specific exception types (e.g., `except ValueError as e:`, `except RuntimeError as e:`). Implement proper error logging and propagation.

**Duplicated/Dead Code:**
- Issue: Multiple outdated copies of training files exist in codebase
- Files:
  - `nce/neural_networks/NN_Train_copy.py` (867 lines) - appears to be a copy of train.py from earlier development phase
  - `nce/neural_networks/train_old.py` (363 lines) - superseded by train.py
- Impact: Maintenance burden, confusion about which version to modify, increased codebase size
- Fix approach: Delete NN_Train_copy.py and train_old.py. Archive old versions in git history if needed.

**Module-Level Global State:**
- Issue: `_flag0 = False` at top of `nce/neural_networks/train.py` suggests incomplete refactoring or debugging flag left in place
- Files: `nce/neural_networks/train.py` (line 1)
- Impact: Unclear purpose, potential source of subtle bugs if modified
- Fix approach: Identify why this flag exists and either remove it or document its purpose with explanation.

**Large Monolithic Classes:**
- Issue: Several core classes exceed 1200 lines, handling multiple responsibilities
- Files:
  - `nce/inference/graphical_model.py` (1809 lines) - manages model creation, factor elimination, backward message computation, NN training coordination
  - `nce/neural_networks/train.py` (1526 lines) - trainer, optimizer selection, loss management, data loading, validation
  - `nce/inference/fastElim.py` (1283 lines) - bucket elimination, message computation, WMB approximations
  - `nce/neural_networks/linear_mse_solver.py` (1205 lines) - matrix condition diagnosis, multiple solver strategies
- Impact: Difficult to test individual components, high cyclomatic complexity, increased bug risk, harder to reason about state mutations
- Fix approach: Break into smaller classes using composition. For example, extract message computation strategies from FastElim, separate solver strategies in LinearMSEOptimalSolver.

**Commented-Out Debug Code:**
- Issue: Numerous debug print statements and commented-out code scattered throughout, especially in graphical_model.py
- Files:
  - `nce/inference/graphical_model.py` (lines 1676, 1722-1727, 1807, etc.)
  - `nce/neural_networks/train.py` (commented code in optimizer setup)
  - `nce/neural_networks/NN_Train_copy.py` (debugging code)
- Impact: Code harder to read, maintenance burden, risk of accidentally using/reverting
- Fix approach: Remove all debug comments. If needed for future, use proper logging levels. Use version control for history.

## Known Bugs

**Factor Multiplication Sign Error in Log Space:**
- Issue: `FastFactor.__mul__()` and `__matmul__()` use addition (`+`) instead of multiplication for log-space values, treating it as element-wise addition rather than multiplication in log-probability space
- Files: `nce/inference/factor.py` (lines 33-69, 71-107)
- Trigger: Any multiplication of FastFactors. Expected log-probability product should use `+` in log space, but logic treats both operations identically
- Impact: Incorrect factor products, wrong belief propagation results, invalid partition function estimates
- Workaround: None identified
- Fix approach: Review intention: if working in log-space, `__mul__()` should use `+` (correct). If `__matmul__()` is meant for something else (true multiplication), ensure tensor operations match intent. Add comments clarifying which operation does what.

**Bare Exception with Invalid Raise Pattern:**
- Issue: `except:` blocks followed by bare `raise(ValueError(...))` with incorrect syntax
- Files: `nce/inference/factor.py` (lines 48-50, 86-88)
- Trigger: When accessing non-existent FastFactor attributes (e.g., `.item()` on high-dimensional tensor)
- Example: `raise(ValueError(...))` should be `raise ValueError(...)`
- Impact: Syntax valid but unconventional, indicates incomplete error handling
- Fix approach: Use proper raise syntax `raise ValueError(...)` without parentheses around exception type.

**Dense Matrix Operations Without Singularity Checks in Some Paths:**
- Issue: Multiple exception handlers in `linear_mse_solver.py` with fallback to pseudoinverse, but some code paths may not handle edge cases
- Files: `nce/neural_networks/linear_mse_solver.py` (lines 116-139, 179-215, 243-248)
- Trigger: Singular or ill-conditioned matrices during linear model fitting
- Impact: Silent degradation to minimum-norm solution without always warning, potential loss of convergence quality
- Workaround: Check `self.solution_method_` attribute after fitting to understand what method was used
- Fix approach: Ensure all paths log warnings to user. Consider making fallback strategies configurable.

## Security Considerations

**Unconstrained Data Loading:**
- Risk: Data loading functions load arbitrary files from paths specified in config without validation
- Files: `nce/data/data_loader.py`, `nce/inference/graphical_model.py`
- Current mitigation: None detected (relies on config being from trusted source)
- Recommendations:
  - Validate file paths against allowlist before loading
  - Add file size limits to prevent memory exhaustion
  - Use absolute paths, reject relative paths with `../`

**Unconstrained Configuration:**
- Risk: `config` dict is freely modified throughout codebase without schema validation
- Files: `nce/inference/graphical_model.py` (lines 30, 50-66 show unlimited dict access via `.get()`)
- Current mitigation: None detected
- Recommendations:
  - Create a Config class with validated fields instead of dict
  - Document all expected config keys and types
  - Validate config on load before use

**No Input Validation on Tensor Shapes:**
- Risk: Tensor shape mismatches may propagate silently due to broad exception handling
- Files: `nce/inference/factor.py`, `nce/inference/fastElim.py`
- Current mitigation: Some asserts (e.g., line 331 in factor.py)
- Recommendations:
  - Replace asserts with explicit shape validation and meaningful errors
  - Add tensor shape contracts to function docstrings

## Performance Bottlenecks

**Inefficient Factor Multiplication Order:**
- Problem: `FastBucket` and `fastElim` compute factor products in arbitrary order rather than optimal order
- Files:
  - `nce/inference/bucket.py` (line 9 TODO comment)
  - `nce/inference/fastElim.py` (line 241 TODO comment)
- Cause: TODO comments indicate this optimization was never implemented. Multiplying smallest factors first minimizes intermediate tensor sizes
- Improvement path: Implement heuristic ordering (e.g., smallest product size first). Benchmark against current approach.

**Full Data Loading on Each Training Set:**
- Problem: `train.py` may reload all data multiple times during training
- Files: `nce/neural_networks/train.py` (lines 289-304)
- Cause: Data loaded per set in training loop
- Improvement path: Preload all data once at trainer initialization, cache batches

**Expensive Matrix Diagnostics:**
- Problem: `LinearMSEOptimalSolver._diagnose_matrix()` computes full eigenvalue decomposition for every fit, even when not strictly needed
- Files: `nce/neural_networks/linear_mse_solver.py` (lines 63-126)
- Cause: Defensive programming without caching
- Improvement path: Cache diagnostic results if fitting same data multiple times, defer eigenvalue computation until needed

**Repeated Normalization Computations:**
- Problem: Normalizing constants may be recomputed for each batch in some training paths
- Files: `nce/data/data_preprocessor.py` (lines 39-48 show caching mechanism exists, but inconsistent usage)
- Cause: Multiple code paths call `normalize()` without checking if constant already computed
- Improvement path: Enforce single computation of `global_max_targets` at preprocessor initialization, never recompute

## Fragile Areas

**Complex Config State Machine:**
- Files: `nce/inference/graphical_model.py` (entire init, especially lines 20-100)
- Why fragile: Config dict controls 30+ behaviors via loose `.get()` calls. No validation means typos silently enable wrong behavior. Example: `use_bw_approx` setting affects loss computation but is checked in multiple places without central authority
- Safe modification:
  - Create ConfigManager class that validates all keys on initialization
  - Document all config keys in one place
  - Use enums for categorical config values instead of strings
- Test coverage: No unit tests visible for config loading and interpretation

**Message Computation Paths:**
- Files: `nce/inference/bucket.py` (compute_message_exact, compute_wmb_message methods), `nce/inference/fastElim.py`
- Why fragile: Multiple message computation strategies (exact vs WMB) with complex fallback logic. Backward factor computation adds additional complexity. Hard to verify correctness.
- Safe modification:
  - Extract strategy pattern for message computation (ExactMessageComputer, WMBMessageComputer)
  - Add comprehensive assertions about output shape/device
  - Unit test each strategy independently
- Test coverage: Integration tests exist but unit tests for individual strategies missing

**Neural Network Training Loop:**
- Files: `nce/neural_networks/train.py` (lines 289-400+)
- Why fragile: Long training loop with multiple loss functions, early stopping strategies, validation logic all intertwined. Changes to one path may break others. Multiple conditional branches based on config flags.
- Safe modification:
  - Extract loss selection into separate module
  - Extract early stopping into separate class
  - Use composition for training strategies instead of conditional branches
- Test coverage: No visible unit tests for training loop components

**Device Management:**
- Files: Throughout codebase (tensor creation, computation)
- Why fragile: Inconsistent handling of CPU vs GPU devices. Some paths assume CUDA, others have fallbacks. Device specification mixed between config dict and function parameters.
- Safe modification:
  - Create DeviceManager singleton to own all device-related decisions
  - Ensure all tensor creation uses same device context
  - Add runtime checks for device mismatches (already exists in some places)
- Test coverage: Device-specific tests would be valuable but unlikely to exist without dedicated test infrastructure

**Backward Message Approximate Flag:**
- Files: `nce/data/data_preprocessor.py`, `nce/neural_networks/losses.py`, `nce/inference/graphical_model.py`
- Why fragile: `use_bw_approx` flag fundamentally changes preprocessing and loss computation. Setting is passed through multiple layers. Easy to forget to propagate to all dependent code.
- Safe modification:
  - Ensure all code that depends on `use_bw_approx` validates it's set correctly before running
  - Create compile-time configuration (constants) rather than runtime flags
- Test coverage: No visible tests comparing use_bw_approx=True vs False paths

## Scaling Limits

**Memory Usage with Large Graphical Models:**
- Current capacity: Observed successful inference on models with message size ~12 states (from todo.txt: "385s for Alex's code on a size 12 message with 20 epochs")
- Limit: No explicit limits enforced in code. Tensor operations will exhaust GPU/CPU memory on larger models. Factor multiplication creates intermediate tensors proportional to product of all variable domains.
- Scaling path:
  - Implement streaming batch elimination for large factors
  - Use low-rank approximations for large intermediate tensors
  - Profile memory usage and identify bottlenecks

**Training Time:**
- Current capacity: ~385 seconds for size-12 message, 20 epochs (from todo.txt)
- Limit: Training scales poorly with message size. No apparent optimization for multiple training iterations across many buckets.
- Scaling path:
  - Implement progressive training (warm-start from previous similar buckets)
  - Parallelize across buckets
  - Cache computations reused across buckets

## Dependencies at Risk

**Deprecated Notebook Dependencies:**
- Risk: Code imports from `tqdm.notebook` which requires Jupyter
- Files: `nce/inference/graphical_model.py` (line 16), `nce/neural_networks/train.py` (line 16)
- Impact: Code won't work outside Jupyter notebooks. Error messages will be unhelpful.
- Migration plan:
  - Create wrapper that uses `tqdm` if not in notebook, `tqdm.notebook` if available
  - Or remove notebook-specific dependencies for core inference engine

**Custom pyGMs Library:**
- Risk: Dependency on `pyGMs` package not in standard package repos. Installation mechanism unclear.
- Files: `nce/inference/graphical_model.py` (lines 5-10 heavy usage)
- Impact: Hard to install, unclear compatibility with Python versions
- Migration plan: Document installation requirements, consider pinning version in setup.py

## Missing Critical Features

**No Distributed Training Support:**
- Problem: Training loop runs serially on single GPU/CPU. No support for multi-GPU or distributed computing.
- Blocks: Scaling to larger problems, using multiple available hardware resources
- Priority: Medium (depends on use cases)

**No Model Checkpointing:**
- Problem: No save/load functionality for trained neural network factors visible in code
- Blocks: Long training runs not resumable, can't persist trained models, can't checkpoint mid-training
- Priority: High (critical for practical use)

**No Hyperparameter Tuning Framework:**
- Problem: No systematic hyperparameter search (grid search, Bayesian optimization)
- Blocks: Automated experimentation, reproducible results
- Priority: Medium

## Test Coverage Gaps

**No Unit Tests for Core Inference:**
- What's not tested:
  - Factor multiplication correctness (factor.py `__mul__`, `__matmul__`)
  - Message computation (bucket.py exact message computation)
  - Elimination order generation
  - Device management across different tensor operations
- Files: `nce/inference/factor.py`, `nce/inference/bucket.py`, `nce/inference/fastElim.py`
- Risk: Bugs in factor computation silently propagate to final partition estimates. High impact but not caught.
- Priority: High

**No Tests for Loss Functions:**
- What's not tested:
  - Different loss function modes (unnormalized_kl, scaled_mse, etc.)
  - Backward message weighting correctness
  - Numerical stability with extreme values
- Files: `nce/neural_networks/losses.py`
- Risk: Loss computation errors during training cause divergence or invalid gradients
- Priority: High

**No Tests for Linear Solver:**
- What's not tested:
  - Singular matrix handling across all code paths
  - Regularization behavior
  - Intercept extraction with rank-deficient matrices
- Files: `nce/neural_networks/linear_mse_solver.py`
- Risk: Silent degradation to fallback methods without user awareness
- Priority: Medium

**No Data Preprocessing Tests:**
- What's not tested:
  - Normalization constant computation with edge cases (all same values, NaN inputs)
  - Backward message normalization
  - Numerical stability with extreme log values
- Files: `nce/data/data_preprocessor.py`
- Risk: Training instability traced back to preprocessing issues
- Priority: Medium

**No Integration Tests:**
- What's not tested:
  - End-to-end flow: graphical model → buckets → training → inference
  - Device consistency throughout pipeline
  - Config propagation to all components
- Files: All modules
- Risk: Integration issues discovered late in long training runs
- Priority: Medium

---

*Concerns audit: 2026-01-25*
