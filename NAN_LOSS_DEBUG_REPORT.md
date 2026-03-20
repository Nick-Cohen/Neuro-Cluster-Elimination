# NaN Loss Debugging Report

## Summary

**Problem:** 3 out of 10 hard buckets (all from `or_chain_10.fg.uai`) produce `loss=inf` at epoch 0, then `loss=nan` at epoch 1 onwards. The local error tracking also becomes NaN after epoch 1.

**Affected Buckets:**
- `or_chain_10.fg.uai` bucket 154 (selection error: 0.199)
- `or_chain_10.fg.uai` bucket 88 (selection error: 0.167)
- `or_chain_10.fg.uai` bucket 60 (selection error: 0.082)

**Working Buckets:**
- All BN_* buckets (BN_8, BN_11)
- All *_wcsp buckets (29, 404)
- grid10x10 bucket (partial training)

## Root Cause Analysis

### 1. Initial Loss = Inf at Epoch 0

**Location:** `nce/benchmark/training.py:337-343`

```python
if 0 in checkpoint_epochs:
    with torch.no_grad():
        initial_loss = trainer.compute_epoch_loss(batches, loss_fn).item()  # ← Returns inf
        approx_factor = FactorNN(net, trainer.data_preprocessor)
        approx_exact = approx_factor.to_exact()
        approx_contribution = (approx_exact * exact_bw).sum_all_entries()
        log_z_err = approx_contribution - exact_contribution
```

**Why inf occurs:**

The network is randomly initialized at epoch 0. For `logspace_mse_fdb` loss:

```python
# nce/neural_networks/losses.py:271-275
def logspace_mse_fdb(outputs, targets, bw_hat=None):
    difs = outputs - targets
    sqr_difs = difs ** 2
    avg_sqr_difs = torch.mean(sqr_difs)
    return avg_sqr_difs
```

If random initialization produces outputs that are extremely far from targets (e.g., outputs=100, targets=-100), then:
- `difs = 200`
- `sqr_difs = 40000`
- This is valid and **not the source of inf**

The `inf` likely comes from **extremely large random weights** in the randomly initialized network for these specific buckets. The or_chain buckets have 18-variable scopes (2^18 = 262,144 assignments), which creates a very large input dimension to the NN.

**Hypothesis 1:** Random initialization with Xavier/He initialization on high-dimensional inputs (262K features) can produce outputs with extreme magnitudes.

### 2. Loss Becomes NaN at Epoch 1

**Location:** `nce/neural_networks/train.py:905-910`

```python
else:
    loss = self.loss_fn(outputs.reshape(-1), y_batch, bw_hat_batch)
loss.backward()  # ← Gradient computation on inf loss produces NaN gradients
```

**Why NaN occurs:**

1. At epoch 0, loss = inf
2. `loss.backward()` computes gradients of inf
3. Gradients of inf are **NaN** (undefined in floating point)
4. Optimizer step applies NaN gradients → network weights become NaN
5. At epoch 1, network forward pass with NaN weights produces NaN outputs
6. NaN outputs → NaN loss
7. NaN loss → NaN gradients → permanent NaN state

**Hypothesis 2:** Once loss becomes inf, the gradient computation produces NaN gradients that poison all subsequent training.

### 3. Why or_chain Buckets Specifically?

Looking at the bucket properties:

| Bucket | Problem | Scope Size | Domain Product | Auto ECL | Status |
|--------|---------|------------|----------------|----------|--------|
| bucket 88 | or_chain_10 | 18 vars | 2^18 = 262,144 | 262,143 | NaN |
| bucket 154 | or_chain_10 | 18 vars | 2^18 = 262,144 | 262,143 | NaN |
| bucket 60 | or_chain_10 | 18 vars | 2^18 = 262,144 | 262,143 | NaN |
| bucket 58 | BN_11 | ~10 vars | ~1,000-10,000 | 131,071 | ✓ Works |
| bucket 75 | 29.wcsp | ~15 vars | ~32,768 | 2,097,151 | ✓ Works |

**Hypothesis 3:** The or_chain buckets have:
- Very large input dimension (262K)
- Possibly extreme target values in log-space
- Random initialization creates weights that produce outputs orders of magnitude away from targets

### 4. Loss Function Behavior

The `logspace_mse_fdb` loss has **no safeguards** against extreme values:

```python
def logspace_mse_fdb(outputs, targets, bw_hat=None):
    difs = outputs - targets
    sqr_difs = difs ** 2        # No clipping, no checks
    avg_sqr_difs = torch.mean(sqr_difs)
    return avg_sqr_difs
```

For comparison, many production loss functions include:
- Gradient clipping
- Loss value clipping
- Huber loss (quadratic near 0, linear far away)
- Numerical stability checks

## Evidence Supporting Hypotheses

### Evidence 1: Loss Progression Pattern
```
or_chain bucket 88:
  Epoch 0: loss=inf, log_Z_err=7.941845 (valid)
  Epoch 1: loss=nan, log_Z_err=nan
  → Confirms NaN starts after first backward pass
```

### Evidence 2: Working Buckets Have Smaller Scopes
All working buckets have auto_ecl < 262,143, suggesting smaller input dimensions.

### Evidence 3: BN_11 bucket 58 Success
```
BN_11 bucket 58:
  Epoch 0: loss=172.14 (finite and large, but not inf)
  Epoch 1: loss=165.05 (decreasing normally)
  → Confirms that finite initial loss allows training to proceed
```

## Recommended Fixes

### Fix 1: Add Gradient Clipping (REQUIRED)

**Location:** `nce/benchmark/training.py:296` (in `train_single_bucket`)

Add before creating Trainer:

```python
# Add gradient clipping to config if not present
if 'grad_clip_norm' not in config:
    config['grad_clip_norm'] = 1.0  # Conservative clipping
```

The Trainer already supports this at `nce/neural_networks/train.py:917-919`:

```python
grad_clip_norm = self.config.get('grad_clip_norm', None)
if grad_clip_norm is not None:
    torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip_norm)
```

**Effect:** Prevents NaN gradients from poisoning the network even if loss becomes inf.

### Fix 2: Add Loss Clipping

**Location:** `nce/neural_networks/losses.py:271` (modify `logspace_mse_fdb`)

```python
def logspace_mse_fdb(outputs, targets, bw_hat=None):
    difs = outputs - targets
    sqr_difs = difs ** 2
    avg_sqr_difs = torch.mean(sqr_difs)
    
    # Clip loss to prevent inf from poisoning gradients
    max_loss = 1e6  # Large but finite
    return torch.clamp(avg_sqr_difs, max=max_loss)
```

**Effect:** Ensures loss never becomes inf, preventing gradient computation issues.

### Fix 3: Better Weight Initialization for Large Input Dimensions

**Location:** `nce/neural_networks/net.py` (in Net.__init__)

Check the current initialization. If using default PyTorch initialization:

```python
# Current (likely):
self.layers = nn.ModuleList([
    nn.Linear(input_dim, hidden_sizes[0]),
    # ... more layers
])
# Uses default initialization: Xavier uniform
```

For very high-dimensional inputs, use smaller initialization scale:

```python
def __init__(self, bucket, hidden_sizes=[]):
    super().__init__()
    input_dim = bucket.get_input_dim()
    
    # ... layer creation ...
    
    # After creating layers, re-initialize with smaller scale for large inputs
    if input_dim > 100000:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Smaller initialization for high-dimensional inputs
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
```

**Effect:** Prevents random initialization from producing extreme output values.

### Fix 4: Disable Early Stopping in Benchmark (REQUIRED - User Request)

**Location:** `nce/benchmark/training.py:284` (in `train_single_bucket`)

Currently, early stopping config is not explicitly set, so it inherits from defaults in Trainer.

Add explicit disabling:

```python
# Benchmark-enforced settings
config['error_tracking'] = False       # Benchmark handles error tracking externally
config['sampling_scheme'] = 'all'      # Required for full-assignment training
config['device'] = device
config['convex_early_stopping'] = False        # NEW: Disable convex early stopping
config['nbe_early_stopping'] = False           # NEW: Disable NBE early stopping
config['neurobe_early_stopping'] = False       # NEW: Disable NeuroBE early stopping
config['use_validation_early_stopping'] = False  # NEW: Disable validation early stopping
```

**Current early stopping locations in Trainer:**

1. `nce/neural_networks/train.py:174-182` - Convex early stopping
```python
early_stopper = None
if should_use_convex_early_stopping(self.config):
    early_stopper = SimpleConvexEarlyStopping(
        patience=self.config.get('convex_patience', 50),
        delta=self.config.get('convex_delta', 0.01),
        warmup=self.config.get('convex_warmup', 10)
    )
```

2. `nce/neural_networks/train.py:188-242` - NBE and NeuroBE early stopping
```python
use_nbe_early_stopping = self.config.get('nbe_early_stopping', False)
use_neurobe_early_stopping = self.config.get('neurobe_early_stopping', False)
```

3. `nce/neural_networks/train.py:248-249` - Validation early stopping
```python
use_validation_early_stopping = False
if self.config.get('use_validation_early_stopping', False):
```

**Effect:** Training runs for full `num_epochs` or until `time_limit_seconds`, never stops early based on convergence heuristics.

## Minimal Reproduction Test

Create a test that isolates the issue:

```python
# test_nan_loss.py
import torch
from nce.neural_networks.net import Net
from nce.neural_networks.losses import logspace_mse_fdb

# Simulate or_chain bucket dimensions
input_dim = 262144  # 2^18
output_dim = 262144

# Create network
class MockBucket:
    def get_input_dim(self):
        return input_dim
    def get_output_dim(self):
        return output_dim

bucket = MockBucket()
net = Net(bucket, hidden_sizes=[20])

# Create random batch
x = torch.randn(100, input_dim)
y_target = torch.randn(100, output_dim) * 10  # Larger targets

# Forward pass
outputs = net(x)

# Compute loss
loss = logspace_mse_fdb(outputs.reshape(-1), y_target.reshape(-1))

print(f"Input dim: {input_dim}")
print(f"Output range: [{outputs.min():.2f}, {outputs.max():.2f}]")
print(f"Target range: [{y_target.min():.2f}, {y_target.max():.2f}]")
print(f"Loss: {loss.item()}")
print(f"Loss is inf: {torch.isinf(loss)}")
print(f"Loss is nan: {torch.isnan(loss)}")
```

## Priority Order for Fixes

1. **FIX 4** (disable early stopping) - REQUIRED by user, easy to implement
2. **FIX 1** (gradient clipping) - Most important for stability
3. **FIX 2** (loss clipping) - Prevents inf loss
4. **FIX 3** (better initialization) - Prevents the root cause

Implementing FIX 1 + FIX 2 should make all buckets trainable. FIX 3 is nice-to-have for cleaner convergence.

## Code Changes Required

### File 1: `nce/benchmark/training.py`

**Line 284 - Add early stopping disable + gradient clipping:**

```python
# OLD:
config['error_tracking'] = False
config['sampling_scheme'] = 'all'
config['device'] = device

# NEW:
config['error_tracking'] = False
config['sampling_scheme'] = 'all'
config['device'] = device
# Disable all early stopping mechanisms
config['convex_early_stopping'] = False
config['nbe_early_stopping'] = False
config['neurobe_early_stopping'] = False
config['use_validation_early_stopping'] = False
# Add gradient clipping for numerical stability
if 'grad_clip_norm' not in config:
    config['grad_clip_norm'] = 1.0
```

### File 2: `nce/neural_networks/losses.py`

**Line 271 - Add loss clipping to logspace_mse_fdb:**

```python
# OLD:
def logspace_mse_fdb(outputs, targets, bw_hat=None):
    difs = outputs - targets
    sqr_difs = difs ** 2
    avg_sqr_difs = torch.mean(sqr_difs)
    return avg_sqr_difs

# NEW:
def logspace_mse_fdb(outputs, targets, bw_hat=None):
    difs = outputs - targets
    sqr_difs = difs ** 2
    avg_sqr_difs = torch.mean(sqr_difs)
    # Clip to prevent inf loss from causing NaN gradients
    max_loss = 1e6
    return torch.clamp(avg_sqr_difs, max=max_loss)
```

## Expected Outcome After Fixes

With gradient clipping + loss clipping:
- Epoch 0 loss for or_chain buckets: ~1e6 (clipped) instead of inf
- Epoch 1+: Loss decreases normally as gradients are bounded
- All 10 buckets should train successfully

## Additional Diagnostic Data Needed

To confirm the exact cause, we'd need:
1. Weight initialization statistics for working vs failing buckets
2. Output value ranges before and after first forward pass
3. Gradient norms at epoch 0 (before clipping)
4. Target value ranges for each bucket

This can be added with instrumentation at epoch 0.
