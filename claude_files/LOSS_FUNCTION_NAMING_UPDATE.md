# Loss Function Naming Update

## Summary of Changes

This document describes the major refactoring of loss function names in the NCE codebase.

## Key Changes

### 1. `elp_least_squares` - New Implementation

**Old behavior:** Simple partition function error minimization
```python
# Old: E[(log Z(f + b) - log Z(s + b))²]
```

**New behavior:** Weighted least squares with second derivative weights
```python
# New: Σ w_i * (target_i - output_i)²
# where w_i = E[d²L/ds_i²] computed from partition function Hessian
```

**Key formula:**
```
d²L/ds_i² = 2 * E[psb_i² - DP*(psb_i - psb_i²)]

where:
- psb_i = exp(s_i + b_i - log Z(s + b))  # softmax probability
- DP = log Z(f + b) - log Z(s + b)       # partition function difference
- b = sampled backward messages
```

**Usage:**
- `elp_least_squares,50` - Use 50 backward samples for weight computation
- `elp_least_squares,100` - Use 100 backward samples (more accurate, slower)

**Location:** `nce/neural_networks/losses.py:664-769`

### 2. `mg_sampled_loss_fdb` → `elp`

**Old names:** `mg_sampled_loss_fdb`, `mg_sampled_loss_fdb_recompute`

**New names:** `elp`, `elp_recompute`

**Usage examples:**
- Old: `mg_sampled_loss_fdb_recompute,50`
- New: `elp_recompute,50`

**Location:** `nce/neural_networks/losses.py:54-89`

### 3. `elp_cancellation` (formerly `mg_sampled_loss_fdb_cancellation`)

Minor name change for consistency.

**Location:** `nce/neural_networks/losses.py:91+`

## Updated Code Locations

### Losses (`nce/neural_networks/losses.py`)
- `elp_least_squares()` - line 664 (NEW implementation with 2nd derivative weights)
- `elp()` - line 54 (formerly mg_sampled_loss_fdb)
- `elp_cancellation()` - line 91 (formerly mg_sampled_loss_fdb_cancellation)

### Trainer (`nce/neural_networks/train.py`)
- `elp_least_squares` handler - lines 530-540
- `elp_recompute` handler - lines 447-470
- `elp` handler (deprecated format) - lines 471-476

### Decision Tree (`nce/neural_networks/decision_tree.py`)
- Updated checks for 'elp' instead of 'mg_sampled' - lines 48, 75

## Migration Guide

### For Existing Code

**If you were using:**
```python
config = {'loss_fn': 'mg_sampled_loss_fdb_recompute,50'}
```

**Change to:**
```python
config = {'loss_fn': 'elp_recompute,50'}
```

**If you were using:**
```python
config = {'loss_fn': 'elp_least_squares,50'}  # old version
```

**No change needed!** The name stayed the same, but the implementation is now better (uses second derivative weights).

### For New Code

**Recommended loss functions:**
1. `elp_least_squares,50` - Best for partition function approximation (NEW implementation)
2. `unnormalized_kl` - General-purpose baseline
3. `logspace_mse_fdb` - Simple MSE in log space
4. `elp_recompute,50` - Alternative ELP formulation

## Test Scripts

- `test_elp_8_5_benchmarks_ibound10.py` - Full benchmark suite
- `test_weighted_ls_single_problem.py` - Single problem test
- `test_weighted_ls_simple.py` - Unit test for loss function

## Technical Details

### Why Second Derivative Weights?

The second derivative `d²L/ds²` tells us how sensitive the partition function error is to errors at each assignment. Assignments with higher second derivatives get more weight in the loss, focusing learning on the most important values.

This matches the approach used in the decision tree optimizer (`decision_tree.py` lines 119-124).

### Backward Message Sampling

Both `elp_least_squares` and `elp` use correlation-adjusted backward message sampling:
```python
alpha = 1 + (rho * sigma_f * sigma_g) / sigma_f²
sb' = sigma_g * sqrt(1 - rho²)
b ~ N(sb' * noise + (alpha - 1) * (f - mean(f)), I)
```

This ensures the backward samples have the correct correlation structure with forward messages.

## Configuration Requirements

To use `elp_least_squares` or `elp_recompute`, you must set:
```python
config = {
    'gather_message_stats': True,  # Required for sigma_f, sigma_g, rho
    'loss_fn': 'elp_least_squares,50',
    # ... other config
}
```

If statistics are not available, the loss function will use empirical estimates from the current batch.
