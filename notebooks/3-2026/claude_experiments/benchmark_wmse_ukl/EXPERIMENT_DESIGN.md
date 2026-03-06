# Benchmark Experiment: WMSE vs UKL Loss Functions

## Purpose

Compare weighted logspace MSE (WMSE) and unnormalized KL (UKL) loss functions for neural network-based approximate inference on the `small_problems` benchmark set (24 problems from benchmarks_12_4_2025, iB10 + iB15). The experiment tests 5 configurations varying loss function and backward information level.

**Key question:** Does backward message information improve neural network training quality, and how does UKL (with/without backward info) compare to WMSE (NeuroBE's loss)?

## Experiment Configurations

All 5 configurations share the same 24 problems and common settings (see below). They differ only in loss function and backward ECL level.

### Configuration 1: WMSE (no backward info)

| Parameter | Value |
|-----------|-------|
| loss_fn | `'weighted_logspace_mse'` |
| bw_ecl | `0` |
| use_bw_approx | `False` |
| populate_bw_factors | `False` |
| backward_ecl | `0` |

**Notes:**
- The loss function name in `_get_loss_fn()` is `'weighted_logspace_mse'` (not `'weighted_mse'` or `'wmse'`).
- `weighted_logspace_mse` accepts `bw_hat` in its signature but **ignores it** entirely. The loss only uses `outputs` and `targets`. With `bw_ecl=0`, no backward factors are populated, so `bw_hat` will be `None` during training. This is correct.
- Formula: normalizes targets to [0,1], creates weights proportional to normalized targets, computes weighted MSE in log-space.

### Configuration 2: UKL (no backward info)

| Parameter | Value |
|-----------|-------|
| loss_fn | `'unnormalized_kl'` |
| bw_ecl | `0` |
| use_bw_approx | `False` |
| populate_bw_factors | `False` |
| backward_ecl | `0` |

**Notes:**
- Baseline UKL without any backward message weighting.
- When `bw_hat=None`, UKL computes standard unnormalized KL divergence: `sum(p * (log_p - log_q) - p + q)`.

### Configuration 3: UKL + bw_ecl = 8

| Parameter | Value |
|-----------|-------|
| loss_fn | `'unnormalized_kl'` |
| bw_ecl | `8` |
| use_bw_approx | `True` |
| populate_bw_factors | `True` |
| backward_ecl | `8` |

**Notes:**
- Weak backward info: bw_ecl = 2^3 = 8 (very coarse backward approximation).
- UKL adds `bw_hat` (detached) to both outputs and targets before computing KL, effectively weighting the loss by approximate backward message.

### Configuration 4: UKL + bw_ecl = auto_ecl (same as forward)

| Parameter | Value |
|-----------|-------|
| loss_fn | `'unnormalized_kl'` |
| bw_ecl | **varies per problem** (see Problem Set table) |
| use_bw_approx | `True` |
| populate_bw_factors | `True` |
| backward_ecl | **varies per problem** (same as bw_ecl) |

**Notes:**
- Backward ECL matches the forward ECL (`auto_ecl`) for each problem. This means the backward message approximation uses the same granularity as the forward computation.
- **This is the tricky configuration**: `bw_ecl` varies per problem (range: 16,383 to 19,487,170). The standard YAML config format has a single `bw_ecl` list applied to all problems. This means Config 4 **cannot** use a single shared YAML file.
- See OPEN_QUESTIONS.md (Question 2) for execution approaches.

### Configuration 5: UKL + bw_ecl = 2^30 (effectively exact backward)

| Parameter | Value |
|-----------|-------|
| loss_fn | `'unnormalized_kl'` |
| bw_ecl | `1073741824` |
| use_bw_approx | `True` |
| populate_bw_factors | `True` |
| backward_ecl | `1073741824` |

**Notes:**
- Effectively exact backward message: bw_ecl = 2^30 = 1,073,741,824. With this threshold, backward elimination will compute nearly all messages exactly, providing the best possible backward approximation.

## Common Settings (all configurations)

| Parameter | Value | Notes |
|-----------|-------|-------|
| num_epochs | `5000` | User specified |
| skip_early_stopping | `True` | No early stopping |
| nbe_early_stopping | `False` | NeuroBE early stopping disabled |
| sampling_scheme | `'all'` | Enumerate full message space |
| batch_size | `10000000` | Very large int; exceeds all message sizes, so entire message is one batch. See OPEN_QUESTIONS.md Q1 |
| set_size | `100000` | Overridden by `sampling_scheme='all'` (train.py sets `set_size=message_size`) |
| num_samples | `100000` | Overridden by `sampling_scheme='all'` (train.py sets `num_samples=message_size`) |
| num_batches_per_set | `1` | Overridden; computed as `ceil(message_size / batch_size)` = 1 |
| hidden_sizes | `[3, 3]` | Default from small_problems config template |
| ecl | per-problem `auto_ecl` | See Problem Set table |
| iB | `100` | Effectively unlimited mini-bucket width |
| seed | `42` | Fixed for reproducibility |
| device | `'cuda'` | GPU training |
| optimizer | `'adam'` | |
| lr | `0.01` | |
| lr_decay | `1` | No decay |
| momentum | `0.9` | |
| inverse_time_decay_constant | `10` | |
| patience | `1` | |
| min_lr | `1e-8` | |
| num_epochs2 | `0` | No second training phase |
| val_set | `'all'` | Full enumeration for validation |
| fdb | `False` | |
| approximation_method | `'nn'` | Neural network approximation |
| backward_iB | `100` | |
| use_linspace_bias | `False` | |
| use_memorizer | `False` | |
| display_intermediate | `False` | |
| track_errors | `False` | |
| plot_messages | `False` | |
| debug | `False` | |
| lower_dim | `False` | |
| dope_factors | `False` | Consistent with small_problems defaults |
| gather_message_stats | `False` | |
| stratify_samples | `False` | |
| traced_losses | `[]` | |

## Problem Set

All 24 unique problems from `nce.benchmark_problems.small_problems`. Listed with catalog key, source iB set, modelfile, and auto_ecl value.

| # | Catalog Key | Source iB | Modelfile | auto_ecl |
|---|-------------|-----------|-----------|----------|
| 1 | alchemy/smokers_20 | 10 | smokers_20.uai | 262,143 |
| 2 | bn/BN_3 | 10 | BN_3.uai | 16,383 |
| 3 | bn/BN_5 | 10 | BN_5.uai | 16,383 |
| 4 | bn/BN_7 | 10 | BN_7.uai | 131,071 |
| 5 | bn/BN_10 | 10 | BN_10.uai | 32,767 |
| 6 | bn/BN_11 | 10 | BN_11.uai | 131,071 |
| 7 | segmentation/10_14_s.binary | 10 | 10_14_s.binary.uai | 32,767 |
| 8 | segmentation/10_16_s.binary | 10 | 10_16_s.binary.uai | 65,535 |
| 9 | segmentation/11_4_s.binary | 10 | 11_4_s.binary.uai | 131,071 |
| 10 | bn/BN_1 | 10 | BN_1.uai | 524,287 |
| 11 | promedas/or_chain_10.fg | 10 | or_chain_10.fg.uai | 262,143 |
| 12 | segmentation/11_17_s.binary | 10 | 11_17_s.binary.uai | 262,143 |
| 13 | objdetect/deer_rescaled_0034.K15.F1.5.model | 10 | deer_rescaled_0034.K15.F1.5.model.uai | 1,048,575 |
| 14 | objdetect/deer_rescaled_0294.K10.F1.75.model | 10 | deer_rescaled_0294.K10.F1.75.model.uai | 1,771,560 |
| 15 | grids/grid10x10.f5.wrap | 10 | grid10x10.f5.wrap.uai | 1,048,575 |
| 16 | bn/BN_2 | 15 | BN_2.uai | 2,097,151 |
| 17 | bn/BN_8 | 15 | BN_8.uai | 8,388,607 |
| 18 | bn/BN_9 | 15 | BN_9.uai | 4,194,303 |
| 19 | objdetect/deer_rescaled_0034.K10.F2.model | 15 | deer_rescaled_0034.K10.F2.model.uai | 19,487,170 |
| 20 | objdetect/deer_rescaled_0034.K15.F1.75.model | 15 | deer_rescaled_0034.K15.F1.75.model.uai | 16,777,215 |
| 21 | objdetect/deer_rescaled_0034.K20.F1.25.model | 15 | deer_rescaled_0034.K20.F1.25.model.uai | 4,084,100 |
| 22 | objdetect/deer_rescaled_0034.K20.F1.5.model | 15 | deer_rescaled_0034.K20.F1.5.model.uai | 4,084,100 |
| 23 | csp/29.wcsp | 15 | 29.wcsp.uai | 2,097,151 |
| 24 | csp/404.wcsp | 15 | 404.wcsp.uai | 4,194,303 |

## Total Experiments

- 5 configurations x 24 problems x 1 architecture x 1 run = **120 experiments**
- Each experiment trains all NN-eligible buckets for one problem under one configuration.

## Execution Strategy

### Recommended: Python Script (bypasses YAML/worker.py)

The most flexible approach is a Python script that directly uses `small_problems` and `FastGM`:

```python
import sys
sys.path.insert(0, '/home/cohenn1/NCE')
from nce.benchmark_problems.small_problems import small_problems, _AUTO_ECL, set_bw_ecl
from nce.inference.graphical_model import FastGM
import copy

# Base config from small_problems
base_configs = small_problems.configs['default']

# Define the 5 configurations
EXPERIMENTS = {
    'wmse_bw0': {
        'loss_fn': 'weighted_logspace_mse',
        'bw_ecl': 0,
    },
    'ukl_bw0': {
        'loss_fn': 'unnormalized_kl',
        'bw_ecl': 0,
    },
    'ukl_bw8': {
        'loss_fn': 'unnormalized_kl',
        'bw_ecl': 8,
    },
    'ukl_bw_ecl': {
        'loss_fn': 'unnormalized_kl',
        'bw_ecl': 'auto_ecl',  # per-problem
    },
    'ukl_bw30': {
        'loss_fn': 'unnormalized_kl',
        'bw_ecl': 2**30,
    },
}

for exp_name, exp_settings in EXPERIMENTS.items():
    for i, (model, base_cfg) in enumerate(zip(small_problems.problems, base_configs)):
        cfg = copy.deepcopy(base_cfg)
        cfg['loss_fn'] = exp_settings['loss_fn']
        cfg['num_epochs'] = 5000
        cfg['skip_early_stopping'] = True
        cfg['nbe_early_stopping'] = False
        cfg['batch_size'] = 10000000  # larger than any message_size

        bw_ecl = exp_settings['bw_ecl']
        if bw_ecl == 'auto_ecl':
            bw_ecl = cfg['ecl']  # use the per-problem auto_ecl
        cfg['bw_ecl'] = bw_ecl
        cfg['backward_ecl'] = bw_ecl
        cfg['populate_bw_factors'] = bw_ecl > 0
        cfg['use_bw_approx'] = bw_ecl > 0

        # Run inference
        fastgm = FastGM(model=model, nn_config=cfg, device=cfg['device'])
        log_z = fastgm.get_log_partition_function()
        # ... collect and save results
```

### Alternative: YAML-based (using worker.py)

For Configs 1, 2, 3, 5: Use per-problem YAML files.
For Config 4: Requires 24 individual YAML files (one per problem) since bw_ecl varies.

See OPEN_QUESTIONS.md (Q2) for tradeoffs.

## Expected Output

For each of the 120 experiments:
1. Log partition function estimate (`log_z`)
2. Per-bucket training loss curves (if `track_errors=True`)
3. Total inference wall-clock time

Results should be saved as JSON per experiment and aggregated across problems.

## Evaluation Metrics

1. **Accuracy**: Compare estimated `log_z` to known ground truth (available for small_problems)
2. **Per-bucket error**: MSE between trained NN output and exact message (for NN-eligible buckets)
3. **Training convergence**: Loss curve shape and final loss value
4. **Wall-clock time**: Total inference time per configuration

## Key Code References

| Component | File | Lines/Function |
|-----------|------|----------------|
| WMSE loss | `nce/neural_networks/losses.py` | `weighted_logspace_mse()` (line 751) |
| UKL loss | `nce/neural_networks/losses.py` | `unnormalized_kl()` (line 35) |
| Training loop | `nce/neural_networks/train.py` | `train()` (line 190+) |
| Batch/sampling logic | `nce/neural_networks/train.py` | Lines 254-275 |
| small_problems | `nce/benchmark_problems/small_problems.py` | Full file |
| set_bw_ecl helper | `nce/benchmark_problems/small_problems.py` | `set_bw_ecl()` (line 157) |
| FastGM | `nce/inference/graphical_model.py` | `FastGM` class |
| worker build_nn_config | `notebooks/_1-2026/worker.py` | `build_nn_config()` (line 115) |
