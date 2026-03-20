# Update to NaN Loss Debugging Report

## Test Results After Initial Fixes

**Fixes Applied:**
1. ✅ Loss clipping: `max_loss = 1e6` in `logspace_mse_fdb`
2. ✅ Gradient clipping: `grad_clip_norm = 1.0` added to config
3. ✅ Early stopping disabled

**Result:** or_chain bucket 88 still produces NaN at epoch 1

```
Epoch 0: loss=1.000000e+06 (clipped from inf), log_Z_err=7.941845 ✓
Epoch 1: loss=nan, log_Z_err=nan ✗
```

## Updated Root Cause

The loss clipping worked (epoch 0 is now 1e6 instead of inf), but NaN still appears. This means:

**The gradient clipping happens AFTER gradients are computed, but if the loss is 1e6, the gradients can still be extremely large before clipping.**

The issue is that even with loss = 1e6 (clipped), the **gradient magnitudes** before clipping can be:
- grad = ∂(1e6)/∂weight
- For MSE: grad ∝ 2 × difference × input_feature
- With 262K input features, some gradients can be >> 1e6
- Gradient norm clipping to 1.0 **rescales** all gradients proportionally
- But if any gradient is NaN (from numerical overflow during backprop), the rescaling produces NaN

**Hypothesis 4 (CONFIRMED):** The backward pass through a 1e6 loss value with 262K-dimensional inputs creates gradients that overflow to inf during computation, then become NaN when inf participates in arithmetic operations.

## Stronger Fix Required

### Additional Fix: Smaller Learning Rate for or_chain Buckets

The issue is that the default lr=0.001 with gradient clipping=1.0 still allows large weight updates:
- If gradient norm = 1000, it gets clipped to 1.0
- With lr=0.001, weight update = 0.001 × (scaled gradients)
- But the **scaled gradients** can still have individual components >> 1.0

**Solution:** Use a much smaller learning rate for high-dimensional buckets.

### Fix 5: Adaptive Learning Rate Based on Input Dimension

**Location:** `nce/benchmark/training.py:271` (after checking input dim)

Add before creating Trainer:

```python
# Adaptive learning rate for high-dimensional buckets
input_dim = bucket.get_input_dim()
if input_dim > 100000 and 'lr' not in nn_config:
    # Scale lr inversely with sqrt of input dimension
    base_lr = config.get('lr', 0.001)
    scaled_lr = base_lr / (input_dim / 100000) ** 0.5
    config['lr'] = max(scaled_lr, 1e-6)  # Minimum 1e-6
    print(f"[BenchmarkTraining] Large input dim ({input_dim}), "
          f"scaling lr: {base_lr} → {config['lr']:.2e}")
```

### Fix 6: Even More Aggressive Loss Clipping

Current: `max_loss = 1e6`
Try: `max_loss = 1e4` (much safer for gradient computation)

**Location:** `nce/neural_networks/losses.py:276`

```python
def logspace_mse_fdb(outputs, targets, bw_hat=None):
    difs = outputs - targets
    sqr_difs = difs ** 2
    avg_sqr_difs = torch.mean(sqr_difs)
    # Aggressive clipping to prevent gradient overflow
    max_loss = 1e4  # Reduced from 1e6
    return torch.clamp(avg_sqr_difs, max=max_loss)
```

### Fix 7: Check for NaN Weights and Reset

**Location:** `nce/benchmark/training.py` (in the training loop, after each epoch)

Add after `epochs_completed = epoch`:

```python
epochs_completed = epoch

# Check for NaN weights and abort if detected
nan_detected = False
for param in net.parameters():
    if torch.isnan(param).any():
        print(f"[BenchmarkTraining] WARNING: NaN detected in weights at epoch {epoch}")
        nan_detected = True
        break

if nan_detected:
    print(f"[BenchmarkTraining] Aborting training due to NaN weights")
    break
```

This won't fix the NaN but will provide cleaner output.

## Implementation Priority

1. **Fix 6** (more aggressive loss clipping) - Easiest, might be sufficient
2. **Fix 5** (adaptive learning rate) - Should prevent the issue entirely
3. **Fix 7** (NaN detection) - For clean failure reporting

Let me test Fix 6 first (changing 1e6 → 1e4).
