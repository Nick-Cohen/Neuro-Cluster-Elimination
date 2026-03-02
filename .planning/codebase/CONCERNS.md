# Codebase Concerns

**Analysis Date:** 2026-02-21

## Tech Debt

**Incomplete Factor Multiplication Ordering:**
- Issue: Factor multiplication order is arbitrary and not optimized
- Files: `nce/inference/bucket.py:9`, `nce/inference/fastElim.py:241`, `nce_package_files/bucket.py:13`
- Impact: Suboptimal computational efficiency; multiplying factors in poor order causes unnecessary large intermediate tensors
- Fix approach: Implement greedy elimination order minimizing intermediate tensor sizes during factor multiplication in `compute_message_exact()`

**Bare Exception Handlers:**
- Issue: Loose except clauses catch all exceptions without specificity
- Files: `nce/inference/factor.py:48`, `nce/inference/factor.py:86`, `nce/inference/message_gradient_factors.py:129`, `nce/inference/graphical_model.py:817`, `nce/inference/graphical_model.py:1483`
- Impact: Masks programming errors and makes debugging difficult; swallows unexpected exceptions without logging context
- Fix approach: Replace bare except with specific exception types; add logging context in except blocks

**Duplicate Copy Files in Version Control:**
- Issue: `nce/neural_networks/NN_Train_copy.py` and `nce/neural_networks/train_old.py` are outdated copies
- Files: `nce/neural_networks/NN_Train_copy.py` (867 lines), `nce/neural_networks/train_old.py` (363 lines)
- Impact: Code duplication creates maintenance burden; confuses which file is authoritative; unused code bloats repository
- Fix approach: Archive to separate directory or remove; keep single source of truth in `nce/neural_networks/train.py`

**Debug Print Statements Left in Production Code:**
- Issue: 746 print statements across 24 files; many are debug statements
- Files: `nce/neural_networks/linear_mse_solver.py:983`, `nce_package_files/linear_mse_solver.py:988` (bucket-specific debugging)
- Impact: Pollutes stdout; makes production runs verbose and hard to parse; wastes I/O cycles
- Fix approach: Replace print statements with proper logging module; conditionally enable debug logging only when needed

**Wildcard Imports:**
- Issue: Several modules use `from module import *` pattern
- Files: `nce/inference/graphical_model.py` (from pyGMs.neuro import *), `nce/inference/fastElim.py:26` (from pyGMs.neuro import *)
- Impact: Namespace pollution; unclear which symbols come from which module; breaks import tracking tools; makes code harder to understand
- Fix approach: Replace with explicit imports: `from pyGMs.neuro import [specific_symbols]`

**Inconsistent Module Imports:**
- Issue: Old naming patterns exist alongside new ones (NCE/ vs nce/)
- Files: `nce/neural_networks/train.py:8` (from NCE.inference.graphical_model commented out, uses nce instead)
- Impact: Legacy import patterns may be accidentally used; confusion about canonical import paths
- Fix approach: Clean up all imports to use consistent lowercase `nce` package; remove commented-out legacy imports

## Known Bugs

**Loss Function Gradient Computation Issue:**
- Symptoms: Batched training diverges when per-batch normalization used; documented with comment in `unnormalized_kl()`
- Files: `nce/neural_networks/losses.py:87-94`
- Trigger: Using per-batch max_val instead of global max_val computed from full dataset
- Current behavior: Code falls back to per-batch max when global max not provided, causing gradient inconsistency
- Workaround: Always pass `max_val` computed from full dataset; for batched training, compute once then pass to every batch

**FactorNN Backward Message Issue:**
- Symptoms: All backward_ecl values produce identical outputs during testing
- Files: `claude_files/test_backward_ecl_fix.py:124` (marked "BUG: All values are the same regardless of backward_ecl!")
- Trigger: Using backward message approximation (`use_bw_approx=True`)
- Impact: Backward message weighting is ineffective; NN not responding to backward message variations
- Current investigation: Test file suggests the approximation isn't properly influencing loss

**Bare Except Silencing Errors in Factor Operations:**
- Symptoms: "got here" printed to stdout; ValueError raised with minimal context
- Files: `nce/inference/factor.py:45-50`, `nce/inference/factor.py:83-88`
- Trigger: Calling factor multiplication operators with invalid operand types
- Workaround: Check operand type before calling multiplication; none
- Risk: Hard to debug when this silently fails in larger computation graphs

## Security Considerations

**No Input Validation on Configuration Dictionaries:**
- Risk: Arbitrary configuration values passed to nn_config can cause unexpected behavior
- Files: `nce/inference/graphical_model.py:30` (config dict created from untrusted input)
- Current mitigation: Config is created internally; no external untrusted input currently
- Recommendations: Add schema validation for nn_config if it becomes externally sourced; validate key-value types before use

**Tensor Device Mismatches Not Caught Early:**
- Risk: Silent device mismatches (CPU vs GPU) can cause cryptic PyTorch errors late in execution
- Files: `nce/inference/bucket.py:32-33` (asserts device match but only at bucket creation)
- Current mitigation: Assertion at bucket creation; catches most but not all mismatches
- Recommendations: Add device validation in tensor operations that combine tensors from different sources

**No Bounds Checking on Variable Domains:**
- Risk: Invalid state indices in assignments could cause array index out of bounds
- Files: `nce/inference/graphical_model.py:1612` (randint generates indices without validating against actual domain)
- Current mitigation: None; relies on correct domain size specification
- Recommendations: Validate var_dims before using in sampling; add assertions that generated indices are in valid range

## Performance Bottlenecks

**Large File Operations Without Streaming:**
- Problem: `FastGM` loads entire UAI files into memory at once
- Files: `nce/inference/graphical_model.py:71` (_load_from_uai call)
- Cause: No streaming parser; entire model structure materialized in memory
- Scaling limit: Files >2GB will cause OOM on typical hardware
- Improvement path: Implement streaming factor parser; load factors on-demand or in chunks

**Unoptimized Factor Multiplication Order:**
- Problem: Factors multiplied in arbitrary order during bucket elimination
- Files: `nce/inference/bucket.py:49-54` (sequential factor multiplication)
- Cause: No complexity calculation before multiplication
- Current complexity: O(n) factors multiplied sequentially can create intermediate tensors of exponential size
- Improvement path: Compute intermediate tensor sizes; use min-max ordering to keep intermediates small

**WMB Backward Message Computation Redundancy:**
- Problem: Backward messages recomputed for every epoch despite not changing between epochs
- Files: `nce/inference/bucket.py:144-150` (bw_wmb computed in compute_message_nn)
- Cause: Backward factors tied to training loop instead of model setup
- Impact: On large models with 1000+ epochs, backward factors computed 1000+ times unnecessarily
- Improvement path: Compute backward factors once during model initialization; cache and reuse across epochs

**Dense Tensor Operations for Sparse Scopes:**
- Problem: Messages with few variables still use dense tensors
- Files: `nce/data/data_preprocessor.py:758-760` (one-hot encoding creates dense tensors)
- Cause: No sparse tensor support in message representation
- Impact: 2-3 variable messages allocate unnecessarily large tensors
- Improvement path: Detect small-scope factors; use sparse representation or explicit enumeration

**Memory Not Released Between Batches:**
- Problem: Explicit cuda cache clearing per batch
- Files: `nce/neural_networks/train.py:905-906`, `nce/neural_networks/train.py:918`
- Cause: PyTorch's automatic memory management insufficient; manual clearing needed
- Impact: Training slower by ~10-15% due to memory fragmentation
- Improvement path: Use torch.cuda.empty_cache() once per epoch instead of per-batch; profile fragmentation

## Fragile Areas

**GraphicalModel Config Dictionary Sharing:**
- Files: `nce/inference/graphical_model.py:20-30`
- Why fragile: Config dict passed as reference; modifications in one GM affect copies if using dict assignment
- Safe modification: Always create new dict with dict(nn_config) constructor (code does this correctly at line 30, but easy to accidentally use = instead)
- Test coverage: No tests for config isolation between GM instances
- Risk: If someone refactors line 30 to use assignment instead of dict(), all GMs will share config

**Trainer.config Modification During Training:**
- Files: `nce/neural_networks/train.py:70-73`
- Why fragile: Config dict modified by trainer (learning rate schedule, scheduler state)
- Safe modification: Only modify config keys for training state; validate before modification
- Test coverage: No tests validating config state consistency
- Risk: Concurrent trainers sharing config dict will interfere with each other's learning rate schedules

**Loss Function Type Checking Based on String Matching:**
- Files: `nce/neural_networks/train.py:57-68`, `nce/neural_networks/train.py:928-930`
- Why fragile: Loss function names matched as strings; no validation that loss_fn actually exists
- Safe modification: Create enum or mapping of valid loss functions; validate loss_fn at initialization
- Test coverage: No tests for invalid loss_fn values
- Risk: Typos in loss_fn config string silently use wrong behavior or skip functionality

**WMB Approximation Method Conditional Logic:**
- Files: `nce/inference/bucket.py:79-143`
- Why fragile: Large conditional block checking multiple configuration flags; logic hard to follow
- Safe modification: Extract conditional logic to separate method; add comments explaining control flow
- Test coverage: Limited testing of different config combinations
- Risk: Changes to one config flag (e.g., use_bw_approx) interact unpredictably with others (sampling_scheme, backward_ecl)

**Message Gradient Factor Extraction Error Handling:**
- Files: `nce/inference/message_gradient_factors.py:127-130` (bare except catching all errors)
- Why fragile: Bare except masks any error; unclear what's expected to fail
- Safe modification: Specify expected exception type; add logging showing why extraction failed
- Test coverage: No tests for invalid factor extraction
- Risk: If pyGMS API changes, error is silently swallowed and empty factor list returned

## Scaling Limits

**Linear Elimination Order Computation:**
- Current capacity: Works well for models with <50 variables
- Limit: Elimination order computation becomes O(n^3+); fails to complete for >100 variable models
- Files: `nce/inference/elimination_order.py`, `nce/inference/graphical_model.py:71`
- Scaling path: Implement incremental/approximation-based ordering; cache orders for repeated models

**Single-GPU Memory for Exact Inference:**
- Current capacity: Exact inference feasible up to treewidth ~15 (message size ~10^6)
- Limit: Messages larger than GPU memory (24-80GB typical) cause OOM
- Files: `nce/inference/bucket.py:35-64` (exact message computation materializes full tensor)
- Scaling path: Implement streaming message operations; support multi-GPU sharding; add off-chip swapping

**Neural Network Training Batch Size:**
- Current capacity: Typical batch sizes 256-2048
- Limit: Batch size limited by GPU memory; larger batches don't improve with current architecture
- Files: `nce/neural_networks/train.py:910-918` (batch processing loop)
- Scaling path: Implement gradient accumulation; support distributed training across multiple GPUs

**WMB Statistics Tracking Memory:**
- Current capacity: Tracks ~5000 messages without issue
- Limit: Statistics dictionaries accumulate; no pruning for long-running experiments
- Files: `nce/inference/bucket.py:25-29` (wmb_stats dict grows unbounded)
- Scaling path: Implement ring buffer for statistics; archive old results; add max size limit

## Dependencies at Risk

**PyGMs Package Integration:**
- Risk: Heavy reliance on external pyGMs library; fork used locally
- Impact: If pyGMs API changes or project abandoned, codebase breaks
- Files: `nce/inference/graphical_model.py:5-10` (imports from pyGMs), `nce/neural_networks/train.py:6`
- Current version tracking: No version pinning visible; no compatibility tests
- Migration plan: Document pyGMs API usage; consider vendoring critical components; maintain compatibility shim layer

**PyTorch AMP (Automatic Mixed Precision) Reliance:**
- Risk: Muon optimizer requires amp.autocast; behavior changes across PyTorch versions
- Impact: Training fails silently with unexpected NaN if autocast disabled on older PyTorch
- Files: `nce/neural_networks/train.py:830-848` (conditional autocast only for muon)
- Current version tracking: No PyTorch version specified
- Migration plan: Test with PyTorch 2.0+; add version check at initialization; implement fallback without AMP

**Pandas Dependency for Data Loading:**
- Risk: Data preprocessing imports pandas but not listed explicitly
- Impact: Code fails silently if pandas not installed
- Files: Likely `nce/data/data_loader.py` (imports not fully visible)
- Current version tracking: No version requirements documented
- Migration plan: Create requirements.txt; use pandas only where necessary; consider lighter alternatives for CSV parsing

## Missing Critical Features

**No Experiment Tracking or Logging:**
- Problem: No structured logging of training runs; results scattered across print statements
- Blocks: Cannot reproduce experiments; hard to compare configurations; lost hyperparameter values
- Impact: Research reproduction impossible; experiments un-traceable
- Files: `nce/neural_networks/train.py` (uses print for all logging)
- Recommendation: Integrate WandB or MLflow; log config, metrics, artifacts systematically

**No Checkpointing During Training:**
- Problem: Training crashes lose all progress; no intermediate model snapshots
- Blocks: Cannot resume long-running experiments; cannot pick best epoch automatically
- Impact: Days of training lost to single crash; manual best-epoch selection
- Files: `nce/neural_networks/train.py` (no checkpoint save logic)
- Recommendation: Add periodic checkpoint saving; implement early stopping with best-model restoration

**No Configuration Schema or Validation:**
- Problem: nn_config is bare dict; no specification of required keys
- Blocks: Configuration errors caught only at runtime, deep in training loop
- Impact: Silent wrong behavior when config keys missing; hard to debug
- Files: `nce/inference/graphical_model.py:30`, `nce/neural_networks/train.py:72`
- Recommendation: Use dataclass or pydantic for config schema; validate at initialization

**No Unit Tests for Core Inference:**
- Problem: Factor multiplication, elimination, message computation untested
- Blocks: Refactoring impossible without breaking inference; bugs not caught
- Impact: Silent correctness issues in inference results; hard to verify against reference implementations
- Files: `nce/inference/` (no test files present)
- Recommendation: Create test suite comparing against pyGMs reference; test factor operations against known results

**No Performance Benchmarking Suite:**
- Problem: No standard benchmarks; hard to detect performance regressions
- Blocks: Cannot validate optimization benefits; unclear which changes improve speed
- Impact: Performance optimization attempts done blind; regressions not noticed
- Files: No benchmark files found
- Recommendation: Create benchmark suite with standard models; track performance across versions

## Test Coverage Gaps

**Factor Multiplication Never Tested:**
- What's not tested: `__mul__` and `__matmul__` operators in FastFactor
- Files: `nce/inference/factor.py:30-88`
- Risk: Silent errors in factor arithmetic; cascading errors in bucket elimination
- Priority: High - core inference depends on this

**Message Gradient Computation Edge Cases:**
- What's not tested: Empty factor lists, single variable messages, numerical edge cases
- Files: `nce/inference/message_gradient_factors.py`
- Risk: Gradient computation fails silently with bare except; unknown loss behavior
- Priority: High - used for all gradient-based training

**Bucket Elimination Order Sensitivity:**
- What's not tested: Different elimination orders on same problem; sensitivity analysis
- Files: `nce/inference/elimination_order.py`, `nce/inference/graphical_model.py`
- Risk: Hidden dependence on elimination order; results not reproducible
- Priority: Medium - affects efficiency, not correctness

**WMB Approximation Validation:**
- What's not tested: Accuracy of WMB approximations; comparison against exact inference
- Files: `nce/inference/graphical_model.py` (wmb path), `nce/utils/pygms_wmb_interface.py`
- Risk: Unknown error bounds in approximations; no validation of quality
- Priority: High - WMB used for most inference

**Loss Function Numerical Stability:**
- What's not tested: Extreme values (very large/small numbers), gradient flow through loss
- Files: `nce/neural_networks/losses.py` (all loss functions)
- Risk: NaN/Inf propagation; training divergence on edge cases
- Priority: High - primary training objective

**Data Preprocessing Reproducibility:**
- What's not tested: Determinism of preprocessing with different seeds; scaling factor consistency
- Files: `nce/data/data_preprocessor.py` (normalization logic)
- Risk: Training non-reproducible; scaling factors vary between runs
- Priority: Medium - affects training stability

---

*Concerns audit: 2026-02-21*
