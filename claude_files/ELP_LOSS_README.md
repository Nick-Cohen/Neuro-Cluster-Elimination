# ELP Least Squares Loss Function

## Overview

The `elp_least_squares` (Expected Log Partition Least Squares) loss function implements the same objective used by the decision tree optimizer for neural network training. It directly minimizes expected partition function errors.

## Mathematical Background

### Objective

Minimize: `E[(log Z(f + b) - log Z(s + b))²]`

Where:
- `f` = true message values (targets)
- `s` = approximated message values (outputs)
- `b` = sampled backward messages (message gradients)
- `Z = exp(logsumexp(...))` = partition function

### Backward Message Sampling

Backward messages are sampled using the correlation structure:

```
b ~ N((alpha - 1) * (f - mean(f)), sigma_b'^2)
```

Where:
- `alpha = 1 + (rho * sigma_f * sigma_g) / sigma_f^2`
- `sigma_b' = sigma_g * sqrt(1 - rho^2)`
- `rho` = correlation between forward and backward messages
- `sigma_f` = standard deviation of forward messages
- `sigma_g` = standard deviation of backward messages

These parameters are automatically computed from bucket statistics.

## Usage

### Basic Usage

```python
gm_config = {
    'loss_fn': 'elp_least_squares,100',  # 100 backward samples
    'approximation_method': 'nn',
    # ... other config
}

fastgm = FastGM(uai_file=problem.uai_file, nn_config=gm_config)
Z = fastgm.get_log_partition_function()
```

### Syntax

`elp_least_squares,<num_bw_samples>`

- `num_bw_samples`: Number of backward message samples to use (default: 100)
- Higher values = more accurate gradient estimates but slower training

### Recommended Values

- **Quick testing**: `elp_least_squares,10`
- **Standard**: `elp_least_squares,50` or `elp_least_squares,100`
- **High precision**: `elp_least_squares,200` or more

## Comparison with Other Losses

### vs. unnormalized_kl
- **elp_least_squares**: Directly targets partition function error
- **unnormalized_kl**: General-purpose KL divergence

### vs. mg_sampled_loss_fdb
- **elp_least_squares**: Uses correlation-adjusted backward sampling (matches decision tree)
- **mg_sampled_loss_fdb**: Uses simpler Gaussian correlation model

### vs. Decision Tree fit_smg
- **Decision tree**: Iterative Newton-style updates using the same objective
- **elp_least_squares**: Direct gradient descent on the same objective

## Implementation Details

### File Locations

- **Loss function**: `nce/neural_networks/losses.py:664-744`
- **Parser**: `nce/neural_networks/train.py:530-540`
- **Tests**:
  - `test_elp_simple.py` - Unit tests
  - `test_elp_partition_error.py` - Partition function comparison

### Key Parameters Retrieved Automatically

The loss function automatically retrieves from bucket statistics:
- `sigma_f`: Forward message standard deviation
- `sigma_g`: Backward message standard deviation
- `rho`: Forward-backward correlation

These are computed via `bucket.get_fw_bw_stats()`

### Reproducibility

When using decision trees (`approximation_method='dt'`), the loss uses a fixed seed for reproducibility. For neural networks, sampling is stochastic (no seed).

## Testing

Basic test:
```bash
python test_elp_simple.py
```

Partition function error comparison:
```bash
python test_elp_partition_error.py
```

## Expected Performance

The loss should provide competitive or better partition function errors compared to `unnormalized_kl`, especially for:
- Problems where backward messages are important
- Linear models (decision tree-like behavior)
- When sufficient backward samples are used (≥50)

## Notes

1. **Computational Cost**: Proportional to `num_bw_samples`
   - More samples = better gradient estimates but slower
   - Start with 10-50 for testing, increase if needed

2. **Statistics Required**: Needs `gather_message_stats` or statistics to be pre-computed
   - The framework automatically computes these during bucket processing

3. **Gradient Flow**: The loss is fully differentiable and supports backpropagation

4. **Memory Usage**: O(num_bw_samples × batch_size) for backward message samples
   - Usually not a concern for modern GPUs with typical batch sizes
