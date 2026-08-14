# NCE Config Reference

This is the authoritative reference for all configuration fields in the NCE package.
The single source of truth for field definitions is `NESTED_SECTIONS` in
[`nce/config_schema.py`](../nce/config_schema.py). This guide documents every field
with its type, default, accepted values, and purpose.

## Quick Start

NCE configs can be written in two formats: **flat** (legacy) or **nested** (recommended).
Both are accepted by `prepare_config()`, which validates, resolves aliases, and returns a
flat dict with internal key names that consumer code reads.

### Flat Config (Legacy)

```python
config = {
    'ecl': 14,
    'iB': 10,
    'approximation_method': 'nn',
    'device': 'cuda',
    'hidden_sizes': [32, 32],
    'loss_fn': 'logspace_mse_fdb',
    'num_epochs': 500,
    'lr': 0.001,
    'batch_size': 256,
    'optimizer': 'adam',
    'num_samples': 5000,
    'sampling_scheme': 'uniform',
    'val_set': True,
    'debug': False,
}
```

### Nested Config (Recommended)

```python
config = {
    'inference': {
        'exact_computation_limit': 14,
        'i_bound': 10,
        'approximation_method': 'nn',
        'device': 'cuda',
    },
    'nn': {
        'hidden_sizes': [32, 32],
    },
    'training': {
        'loss_fn': 'logspace_mse_fdb',
        'num_epochs': 500,
        'learning_rate': 0.001,
        'batch_size': 256,
        'optimizer': 'adam',
    },
    'sampling': {
        'num_samples': 5000,
        'sampling_scheme': 'uniform',
        'val_set': True,
    },
    'output': {
        'debug': False,
    },
}
```

### How `prepare_config` Works

1. **Detection**: Checks if any top-level key is a section name with a dict value → nested format.
2. **Validation**: Nested configs validate section names and field names. Flat configs check for dead fields.
3. **Alias resolution**: Readable names (e.g., `learning_rate`) are mapped to internal names (e.g., `lr`).
4. **Flattening**: Nested configs are flattened to a single dict with internal key names.
5. **Output**: A plain dict that consumer code reads using internal names like `config['lr']`.

Both the readable name and the internal name can be used in flat configs. In nested configs,
either name works within the appropriate section.

---

## Inference Section

Controls the core variable elimination process and how buckets decide between exact computation
and neural network approximation.

| Readable Name | Internal Name | Type | Default | Purpose |
|---|---|---|---|---|
| `exact_computation_limit` | `ecl` | `int` | `0` | Maximum bucket complexity (product of variable domain sizes) for exact computation. Buckets exceeding this limit use NN approximation instead. `0` means all buckets use NNs. |
| `i_bound` | `iB` | `int` | `0` | Mini-bucket i-bound for Weighted Mini-Bucket (WMB) elimination. Limits the width of mini-buckets. `0` means no WMB splitting. |
| `ib2` | `ib2` | `int` | `None` | Induced width threshold for binary domains. Sets both `iB` (width limit) and `ecl` (message size threshold = 2^ib2 - 1). Example: `ib2: 23` sets iB=23 and ecl=8388607. Mutually exclusive with `i_bound` and `exact_computation_limit`. |
| `approximation_method` | `approximation_method` | `str` | `'nn'` | Which approximation method to use for large buckets. See [Approximation Method Values](#approximation-method-values). |
| `dope_factors` | `dope_factors` | `bool` | `False` | Replace `-inf` values in factor tensors with a finite floor (`-5`). Prevents numerical issues from zero-probability entries in the original model. |
| `device` | `device` | `str` | `'cuda'` | PyTorch device for tensor computation. `'cuda'` for GPU, `'cpu'` for CPU. |
| `deterministic_guard` | `deterministic_guard` | `bool` | `False` | Enable `torch.use_deterministic_algorithms(True, warn_only=True)` for the process. Nondeterministic ops warn instead of raising, so a run completes and reports every offender. Costs roughly 2.5x wall time, so it is off by default. Equivalent env var: `NCE_DETERMINISM_GUARD=1`. See `nce/utils/determinism.py`. |
| `neurobe_mode` | `neurobe_mode` | `bool` | `False` | Master flag for NeuroBE-faithful training mode. When enabled, activates NeuroBE-specific defaults (min-max normalization, weighted MSE loss, ReLU activation, patience-based early stopping). |

## NN Section

Neural network architecture and configuration. These fields are only relevant when
`approximation_method='nn'`. Fields with the `dt_` prefix are only relevant when
`approximation_method='dt'` (decision tree).

| Readable Name | Internal Name | Type | Default | Purpose |
|---|---|---|---|---|
| `hidden_sizes` | `hidden_sizes` | `list[int]` \| `str` | `[]` | Hidden layer sizes for the neural network. Empty list `[]` = linear model (no hidden layers). See [hidden_sizes Polymorphic Values](#hidden_sizes). |
| `use_linspace_bias` | `use_linspace_bias` | `bool` | `False` | Add a learnable bias in linear-space (combined with network output via logsumexp). Useful for certain loss functions that operate in linear space. |
| `use_memorizer` | `use_memorizer` | `bool` | `False` | Use a lookup table (Memorizer) instead of a neural network. The memorizer stores exact values for all input combinations — no training, just enumeration. |
| `use_memorization_table` | `use_memorization_table` | `bool` | `False` | Hybrid NN + memorization table. After each NN cluster is trained, draw no-repeat samples from its WMB proposal tree, evaluate the TRUE message value there, keep the top-K as an exact lookup table, and splice those values over the trained network at inference (`HybridMemorizerNet`). Requires `populate_bw_factors` so the proposal tree exists. Unrelated to `use_memorizer`, which is all-or-nothing and has no NN fallback. |
| `memorize_num_samples` | `memorize_num_samples` | `int` | `0` | Absolute cap on the number of no-repeat proposal samples drawn per cluster when building the memorization table. `0` = no absolute cap (only `memorize_sample_frac` and the message size apply). |
| `memorize_sample_frac` | `memorize_sample_frac` | `float` | `0.0` | Cap on the no-repeat sample count expressed as a fraction of the cluster's full message size (`prod(domain sizes of the separator)`). E.g. `0.1` samples a tenth of the table. `0.0` = no fractional cap. |
| `memorize_top_k` | `memorize_top_k` | `int` | `0` | Absolute cap on the number of entries memorized per cluster (the top-K by the selection score). `0` = no absolute cap. |
| `memorize_frac` | `memorize_frac` | `float` | `0.0` | Cap on the memorized entry count expressed as a fraction of the cluster's full message size. E.g. `0.01` memorizes a hundredth of the table. `0.0` = no fractional cap. |
| `memorize_selection` | `memorize_selection` | `str` | `'fw_true'` | Which sampled assignments to keep. `'fw_true'` keeps the largest true forward message values. `'fw_bw'` keeps the largest `forward x backward` products (needs backward factors on the trainer's dataloader; falls back to `fw_true` if none are available). |
| `custom_hidden_sizes` | `custom_hidden_sizes` | `callable(bucket) -> list[int]` \| `None` | `None` | A function that takes a `FastBucket` and returns hidden layer sizes. Allows per-bucket architecture customization (e.g., scaling with bucket complexity). Overrides `hidden_sizes` when set. |
| `init_with_linear_optimum` | `init_with_linear_optimum` | `bool` | `False` | Initialize the neural network weights with the closed-form linear optimum before training. Can speed up convergence for networks with hidden layers by starting from a good linear solution. |
| `weight_decay` | `weight_decay` | `float` | `0.0` | L2 regularization coefficient. Applied to the optimizer. |
| `num_leaves` | `num_leaves` | `int` \| `None` | `None` | Maximum number of leaves per decision tree (only for `approximation_method='dt'`). Controls tree complexity. |
| `num_iterations` | `num_iterations` | `int` \| `None` | `None` | Maximum optimization iterations for decision tree training (only for `approximation_method='dt'`). |
| `dt_learning_rate` | `dt_lr` | `float` \| `None` | `None` | Learning rate (step size) for decision tree gradient updates (only for `approximation_method='dt'`). |
| `dt_momentum` | `dt_momentum` | `float` \| `None` | `None` | Momentum step size toward the exact solution during decision tree optimization (only for `approximation_method='dt'`). |
| `dt_random_seed` | `dt_random_seed` | `int` \| `None` | `None` | Random seed for decision tree training reproducibility. |
| `dt_convergence_threshold` | `dt_convergence_threshold` | `float` \| `None` | `None` | Stop decision tree optimization when the maximum change in predictions falls below this threshold. |
| `quantization_states` | `quantization_states` | `int` \| `None` | `None` | Number of quantization states for decision tree input discretization. |
| `activation` | `activation` | `str` | `'tanh'` | Activation function for hidden layers. `'tanh'` for Tanh (default), `'relu'` for ReLU. NeuroBE uses ReLU. |

### hidden_sizes

The `hidden_sizes` field accepts multiple forms:

| Value | Meaning |
|---|---|
| `[]` | Linear model — single linear layer, no hidden layers. |
| `[32, 32]` | Two hidden layers with 32 units each, Tanh activation. |
| `'bias_only'` | Learn only a single bias term. All weights frozen to zero. |
| `'nbe,<b>'` | NeuroBE-style adaptive sizing: two hidden layers of size `b * ceil(log2(message_size))`. Default `b=1` if omitted (i.e., `'nbe'`). |
| `'neurobe,<b>'` | NeuroBE-faithful adaptive sizing: two hidden layers of size `b * scope_size` (number of variables in message scope). Default `b=1` if omitted. Differs from `nbe` by using scope size instead of log2(message_size). |

## Training Section

Controls the optimization process: loss function, optimizer, learning rate scheduling,
early stopping, and phase-2 training.

| Readable Name | Internal Name | Type | Default | Purpose |
|---|---|---|---|---|
| `num_epochs` | `num_epochs` | `int` | **required** | Number of training epochs (phase 1). |
| `num_epochs_phase2` | `num_epochs2` | `int` | `0` | Number of additional training epochs in phase 2 (using `loss_fn2`). `0` disables phase 2. |
| `loss_fn` | `loss_fn` | `str` | **required** | Loss function for phase 1. See [Loss Function Values](#loss-function-values). |
| `loss_fn_phase2` | `loss_fn2` | `str` \| `None` | `None` | Loss function for phase 2 training. Same valid values as `loss_fn`. |
| `optimizer` | `optimizer` | `str` | `'adam'` | Optimization algorithm. See [Optimizer Values](#optimizer-values). |
| `learning_rate` | `lr` | `float` | `0.001` | Initial learning rate for the optimizer. |
| `learning_rate_decay` | `lr_decay` | `float` | `1.0` | Multiplicative learning rate decay factor applied per set. `1.0` = no decay. |
| `momentum` | `momentum` | `float` | `0.9` | Momentum for SGD optimizer. Ignored for Adam. |
| `batch_size` | `batch_size` | `int` \| `str` | `256` | Training batch size. See [batch_size Polymorphic Values](#batch_size). |
| `patience` | `patience` | `int` | `20` | Number of epochs with no improvement before learning rate reduction or early stopping. |
| `min_learning_rate` | `min_lr` | `float` | `1e-8` | Minimum learning rate floor. Training stops reducing LR below this value. |
| `seed` | `seed` | `int` | `42` | Random seed for reproducibility (weight initialization, sampling). |
| `skip_early_stopping` | `skip_early_stopping` | `bool` | `False` | Disable all early stopping — always train for the full `num_epochs`. |
| `nbe_early_stopping` | `nbe_early_stopping` | `bool` | `False` | Enable NeuroBE-style early stopping based on validation loss plateau detection. Requires a validation set. |
| `nbe_warmup_epochs` | `nbe_warmup_epochs` | `int` | `0` | Number of initial epochs to skip before checking NeuroBE early stopping. Allows the model to stabilize before plateau detection. |
| `convex_early_stopping` | `convex_early_stopping` | `bool` | `False` | Enable early stopping based on convex loss convergence. Only active for linear models with supported loss functions (`logspace_mse_fdb`, `linspace_mse_fdb`, `weighted_logspace_mse`, `ukf_sequential`). |
| `convex_patience` | `convex_patience` | `int` | `20` | Patience for convex early stopping (epochs without improvement). |
| `convex_min_delta` | `convex_min_delta` | `float` | `1e-8` | Minimum loss improvement to count as progress for convex early stopping. |
| `use_validation_early_stopping` | `use_validation_early_stopping` | `bool` | `False` | Enable validation-set-based early stopping. Checks every 10 epochs. |
| `inverse_time_decay_constant` | `inverse_time_decay_constant` | `int` | `100` | Constant `C` in inverse time decay schedule: `lr_multiplier = C / (C + step)`. Higher values = slower decay. |
| `lr_schedule` | `lr_schedule` | `str` | `'none'` | Learning rate schedule type. See [LR Schedule Values](#lr-schedule-values). |
| `lr_schedule_max_lr` | `lr_schedule_max_lr` | `float` \| `None` | `None` | Maximum learning rate for OneCycle schedule. Defaults to `lr * 10` if not set. |
| `lr_schedule_eta_min` | `lr_schedule_eta_min` | `float` | `1e-6` | Minimum learning rate for cosine annealing schedule. |
| `lr_schedule_pct_start` | `lr_schedule_pct_start` | `float` | `0.1` | Fraction of training spent in the warmup phase for OneCycle schedule. |
| `gradient_clip_norm` | `grad_clip_norm` | `float` \| `None` | `None` | Maximum gradient norm for gradient clipping. `None` disables clipping. |
| `nbe_plateau_threshold` | `nbe_plateau_threshold` | `float` | `0.1` | Threshold for detecting NeuroBE loss plateau (currently unused in active code). |
| `nbe_plateau_window` | `nbe_plateau_window` | `int` | `25` | Window size for NeuroBE plateau detection (currently unused in active code). |
| `nbe_plateau_min_improvement` | `nbe_plateau_min_improvement` | `float` | `0.01` | Minimum improvement within the plateau window (currently unused in active code). |
| `scaled_mse` | `scaled_mse` | `float` \| `None` | `None` | Scaling parameter for the `scaled_mse` loss function. When the `scaled_mse` loss is used, forward/backward statistics are incorporated to scale the MSE in linear space. |
| `normalization_mode` | `normalization_mode` | `str` | `'logspace_mean'` | Data normalization mode for the DataPreprocessor. `'logspace_mean'` subtracts the log-space mean (default NCE behavior). `'minmax_01'` applies min-max normalization to [0,1] range (NeuroBE-faithful). |
| `neurobe_early_stopping` | `neurobe_early_stopping` | `bool` | `False` | Enable NeuroBE-style patience-based early stopping. Distinct from `nbe_early_stopping` (which uses plateau detection). Stops training after `neurobe_stop_iter` consecutive non-improving epochs. |
| `neurobe_stop_iter` | `neurobe_stop_iter` | `int` | `2` | Number of consecutive non-improving epochs before NeuroBE early stopping triggers. Only used when `neurobe_early_stopping=True`. |
| `use_amp` | `use_amp` | `bool` | `True` | Enable automatic mixed precision (AMP) during training. Uses `torch.cuda.amp` for faster computation on supported GPUs. |

### batch_size

| Value | Meaning |
|---|---|
| `256` (int) | Standard mini-batch training with batches of this size. |
| `'all'` | Use the entire message as a single batch (full-batch training). |

## Sampling Section

Controls how training data is generated from the graphical model buckets.

| Readable Name | Internal Name | Type | Default | Purpose |
|---|---|---|---|---|
| `sampling_scheme` | `sampling_scheme` | `str` | `'uniform'` | How to generate training samples. See [Sampling Scheme Values](#sampling-scheme-values). |
| `num_samples` | `num_samples` | `int` \| `str` | **required** | Number of training samples to generate. See [num_samples Polymorphic Values](#num_samples). |
| `common_random_numbers` | `common_random_numbers` | `bool` | `True` | Common random numbers. Separator assignments, the uniform half of a mixed proposal draw, and the WMB proposal tree's own randomness are pure functions of (`seed`, separator, role, draw index) instead of of the global RNG, so two runs that produce the same separator draw byte-identical assignments in the same order regardless of merge strategy or elimination order, and a larger draw extends a smaller one. **Default-on since 2026-08-14** (Nick's directive); the determinism goldens were regenerated in the same commit, so numbers from before that date are not comparable. Set `False` to get the legacy global-RNG sampler back -- note that the legacy path's numbers also moved, because the collision-prone seed derivation it used was replaced. See `nce/sampling/crn.py` and `notebooks/_August-2026/claude_experiments/56-crn-complete.md`. |
| `set_size` | `set_size` | `int` \| `None` | `None` | Size of each sample set (training data is generated in sets). If `None`, all samples are generated as one set. `num_batches_per_set` is computed internally as `set_size // batch_size`. |
| `val_set` | `val_set` | `bool` \| `str` \| `None` | `True` | Validation set configuration. See [val_set Polymorphic Values](#val_set). |
| `stratify_samples` | `stratify_samples` | `bool` | `False` | Use stratified sampling to ensure coverage of the variable domain. Generates samples that cover all combinations of elimination variable values. |
| `lower_dim` | `lower_dim` | `bool` | `False` | Use lower-dimensional sample representation. Projects samples into a reduced feature space based on the message scope rather than the full bucket scope. |

### num_samples

| Value | Meaning |
|---|---|
| `5000` (int) | Generate exactly this many samples. |
| `'nbe,<epsilon>'` | NeuroBE-style adaptive sample count: computes samples based on message size and error tolerance `epsilon`. Default `epsilon=0.25` if omitted (i.e., `'nbe'`). |

### val_set

| Value | Meaning |
|---|---|
| `True` | Generate a validation set (separate from training data) for early stopping evaluation. |
| `False` | No validation set. |
| `None` | No validation set. |
| `'all'` | Use the full enumeration of all possible assignments as the validation set. |

## Backward Section

Controls backward pass approximation for message gradient computation. These fields
enable approximating the backward message (message gradient) which is used by certain
loss functions (e.g., `approx_smg`, `elp_recompute`).

| Readable Name | Internal Name | Type | Default | Purpose |
|---|---|---|---|---|
| `use_backward_approximation` | `use_bw_approx` | `bool` | `False` | Enable backward message approximation using WMB. Required for loss functions that use message gradients (`approx_smg`, `elp_recompute`, etc.). |
| `populate_backward_factors` | `populate_bw_factors` | `bool` | `False` | Pre-compute approximate backward factors using WMB during initialization. These are stored on each bucket and reused across training, avoiding repeated backward message computation. |
| `bw_ib2` | `bw_ib2` | `int` | `None` | Induced width threshold for backward approximation in binary domains. Sets both `bw_iB` and `bw_ecl` (2^bw_ib2 - 1). Mutually exclusive with `backward_i_bound` and `backward_ecl`. |
| `backward_i_bound` | `bw_iB` | `int` | `None` | Mini-bucket i-bound for backward messages. Limits the width of backward mini-buckets. Mutually exclusive with `bw_ib2` and `backward_ecl`. |
| `backward_ecl` | `bw_ecl` | `int` \| `None` | `None` | Exact computation limit for backward message computation. Falls back to the forward `ecl` if not set. Controls when backward messages use WMB vs exact computation. Mutually exclusive with `bw_ib2`. |
| `forward_diff_barrier` | `fdb` | `bool` | `False` | Apply a forward-difference barrier (stop-gradient) on normalizing constants in loss computation. Prevents gradients from flowing through the log-sum-exp normalization, stabilizing training. |

## Width-Based Thresholds

For binary-domain problems (all variables have domain size 2), the width-based parameters `ib2` and `bw_ib2` provide a cleaner interface than specifying message size thresholds directly.

**Equivalence for binary domains:**
- `ib2: N` sets `iB: N` and `ecl: 2^N - 1`
- `bw_ib2: N` sets `bw_iB: N` and `bw_ecl: 2^N - 1`

The off-by-one (`2^N - 1`) follows NCE's dispatch convention: buckets with `message_size > ecl` use approximation. For binary buckets with width > N to be approximated, we need `ecl = 2^N - 1`.

**Common binary width values:**

| ib2/bw_ib2 | ecl/bw_ecl | Message size (binary) |
|---|---|---|
| 18 | 262143 | 2^18 variables |
| 20 | 1048575 | 2^20 variables |
| 23 | 8388607 | 2^23 variables |
| 25 | 33554431 | 2^25 variables |

**Mutual exclusivity:**
- Forward thresholds: Use only ONE of `i_bound`, `ib2`, or `exact_computation_limit`.
- Backward thresholds: Use only ONE of `backward_i_bound`, `bw_ib2`, or `backward_ecl`.
- Configs specifying multiple thresholds in the same direction raise `ValueError` at preparation time.

**Non-binary domains:**
Width-based parameters assume binary domains (domain size = 2). For problems with domain size ≥ 3, use the direct threshold parameters (`ecl`, `bw_ecl`) instead. The schema accepts `ib2` for any problem, but the translation formula `ecl = 2^ib2 - 1` only matches the intended width-based semantics for binary variables.

## Output Section

Controls debugging output, error tracking, plotting, and diagnostic data collection.

| Readable Name | Internal Name | Type | Default | Purpose |
|---|---|---|---|---|
| `debug` | `debug` | `bool` | `False` | Enable verbose debug output during training (prints detailed per-epoch information). |
| `display_intermediate` | `display_intermediate` | `bool` \| `int` | `False` | Display intermediate message plots during training. If `int`, display every N epochs. If `False`/`0`, disabled. |
| `track_errors` | `track_errors` | `bool` | `False` | Track per-bucket NN approximation errors (log-Z error) during elimination. Results stored in `FastGM.nn_errors`. |
| `error_tracking` | `error_tracking` | `bool` | `False` | Enable detailed error tracking with checkpoints during training. Records loss and log-Z error at logarithmically-spaced epoch intervals. Results stored in `FastGM.error_tracking_data`. |
| `plot_messages` | `plot_messages` | `bool` | `False` | Plot exact vs approximate messages after each bucket's NN training completes. |
| `traced_losses` | `traced_losses` | `list[str]` | `[]` | List of additional loss function names to evaluate (but not train with) at each epoch. Results stored in `FastGM.traced_losses_data`. Useful for comparing loss landscapes. |
| `gather_message_stats` | `gather_message_stats` | `bool` | `False` | Collect forward/backward message statistics (variance, correlation) during elimination. Required for `approx_smg` loss functions that use global statistics. Results stored in `FastGM.message_stats`. |
| `complexity_limit` | `complexity_limit` | `int` | `0` | Skip training for buckets whose message complexity exceeds this limit. `0` disables the limit (no buckets skipped). |
| `log_file` | `log_file` | `str` \| `None` | `None` | Path to a JSONL file for structured training event logging. When set, emits per-bucket events (`bucket_training_start`, `epoch_loss`, `bucket_training_end`, `early_stopping`, `val_loss`) with timestamps. `None` disables logging (default). |
| `time_sample_gen` | `time_sample_gen` | `bool` | `False` | Print a `[GammaTiming] bucket=... T_gen=... m=... r=... e=... k=... w_scope=...` line for each cluster's NBE validation-set generation. Human-readable only; prefer `gamma_trace_path` for analysis. |
| `gamma_trace_path` | `gamma_trace_path` | `str` \| `None` | `None` | **Master switch for gamma-v2 instrumentation.** Path to a JSONL file receiving one structured row per cluster's sample generation: `T_gen_s`, phase split, `m`, `r`, **`n_nn`**, the true `prod k_v` over eliminated vars, separator sizes, streaming path/chunking, and the git revision that produced the row. `None` (default) disables tracing entirely — no file is opened and the traced code path is identical to the untraced one. See `nce/utils/gamma_trace.py`. |
| `gamma_trace_per_factor` | `gamma_trace_per_factor` | `bool` | `False` | Additionally time each factor in the cluster individually, yielding `t_nn_factors_s` / `t_table_factors_s` and a per-factor list. Requires a CUDA synchronize between factors, which inflates `T_gen_s` by roughly 10%; run a separate untraced-per-factor pass if you need a clean `T_gen`. No effect unless `gamma_trace_path` is set. |
| `gamma_trace_sync` | `gamma_trace_sync` | `bool` | `True` | Call `torch.cuda.synchronize()` at timing boundaries so wall times reflect completed GPU work rather than queue depth. Set `False` only if you specifically want async-dispatch timings. |
| `gamma_trace_run_id` | `gamma_trace_run_id` | `str` \| `None` | `None` | Tag stamped on every row so rows from one sweep can be selected. Defaults to a per-process random id. |
| `gamma_trace_strategy` | `gamma_trace_strategy` | `str` \| `None` | `None` | Explicit merge-strategy tag (`rnn`, `sub`, `nomerge`, ...) recorded on each row. When unset it is inferred from `experiment_name`, which is only best-effort. |

---

## Valid Values for Enum-Like Fields

### Loss Function Values

The `loss_fn` and `loss_fn2` fields accept any of the following values:

**Log-space losses** (operate on log-probabilities directly):

| Value | Description |
|---|---|
| `logspace_mse_fdb` | MSE on log-values with forward-diff barrier on the normalizer. Primary recommended loss. |
| `logspace_mse` | MSE on log-values without forward-diff barrier. |
| `weighted_logspace_mse` | Weighted MSE on log-values — weights by exponentiated target magnitude. |
| `weighted_logspace_mse_pedigree` | Variant of weighted log-space MSE for pedigree models. |
| `logspace_mse_pathIS` | Log-space MSE with path importance sampling weights. |

**Linear-space losses** (convert to probabilities before computing loss):

| Value | Description |
|---|---|
| `linspace_mse_fdb` | MSE on normalized probabilities (linear space) with forward-diff barrier. |
| `mse` / `MSE` | Standard MSE (operates on raw network outputs). |
| `scaled_mse` | MSE in linear space with forward/backward statistic scaling. |

**KL divergence variants:**

| Value | Description |
|---|---|
| `unnormalized_kl` | KL divergence on unnormalized distributions. Uses data preprocessor's normalizing constant. |
| `scaled_ukl` | Unnormalized KL with forward/backward statistic scaling. |

**Geometric losses** (L1-family, based on geometric error measures):

| Value | Description |
|---|---|
| `l1` | L1 loss on log-values. |
| `l1c` | Centered L1 loss. |
| `logspace_l1` | L1 loss in log-space (alias for `from_logspace_l1`). |
| `gil1` | Geometric-inspired L1 loss. |
| `gil1c` | Centered geometric-inspired L1 loss. |
| `w_gil1c` | Weighted centered geometric-inspired L1 loss. |
| `gil1c_linear` | Linear variant of centered geometric-inspired L1. |
| `gil2` | Geometric-inspired L2 loss. |
| `gil2c` | Centered geometric-inspired L2 loss. |
| `huber_gil1c` | Huber-smoothed variant of centered geometric-inspired L1. |

**From-logspace losses** (convert from log-space before computing):

| Value | Description |
|---|---|
| `from_logspace_mse` / `from_logspace_l2` | MSE after converting from log-space to linear-space. |
| `from_logspace_l1` | L1 after converting from log-space. |
| `from_logspace_gil2` | Geometric-inspired L2 after converting from log-space. |

**Combination and special losses:**

| Value | Description |
|---|---|
| `combined_gil1_ls_mse` | Combined geometric-inspired L1 + log-space MSE. |
| `z_err` | Partition function error loss. |

**Message gradient losses** (require backward approximation):

| Value | Description |
|---|---|
| `approx_smg,<N>` | Approximate sampled message gradient loss with `N` backward samples. Requires `use_bw_approx=True` and global stats (`sigma_g_global`, `rho_global`). |
| `elp_recompute,<N>` | Expected log-partition loss that recomputes forward/backward stats per bucket, with `N` backward samples. |
| `elp_loo,<N>` | Leave-one-out variant (deprecated). |

**Parameterized losses:**

| Value | Description |
|---|---|
| `power_exponential,<alpha>` | Power-exponential loss with exponent `alpha`. |
| `ukf_sequential` | Unscented Kalman filter sequential loss. Supports extended syntax: `ukf_sequential,<resample_period>,<m_per>`. |

### Optimizer Values

| Value | Description |
|---|---|
| `adam` / `Adam` | Adam optimizer (default). Uses `lr` config field. |
| `sgd` / `SGD` | Stochastic Gradient Descent with momentum. Uses `lr` and `momentum` config fields. |
| `muon` | Muon optimizer — splits parameters into ≥2D (Muon) and <2D (AdamW). Fixed LR settings. |

### LR Schedule Values

| Value | Description |
|---|---|
| `none` | No learning rate schedule (default). LR decays only via `lr_decay` or patience-based reduction. |
| `cosine` | Cosine annealing. Smoothly decays LR from initial value to `lr_schedule_eta_min` over `num_epochs`. |
| `onecycle` | OneCycle policy. Warms up to `lr_schedule_max_lr`, then decays. Good for escaping local minima. Uses `lr_schedule_pct_start` for warmup fraction. |

### Sampling Scheme Values

| Value | Description |
|---|---|
| `uniform` | Uniform random sampling over the variable domains (default). |
| `all` | Enumerate all possible assignments (full Cartesian product). Use only for small messages. |

### Approximation Method Values

| Value | Description |
|---|---|
| `nn` | Neural network approximation (default). Uses fields from the NN and Training sections. |
| `dt` | Decision tree approximation. Uses `dt_`-prefixed fields from the NN section. |

---

## Dead Fields

These field names appear in some legacy configs but are **never read by consumer code**.
Using them in a config will raise `ValueError` in strict mode or emit a warning in
non-strict mode.

| Dead Field | Migration |
|---|---|
| `backward_ecl` | Use `bw_ecl` instead (in the backward section). Note: this is the flat-level dead field, not the backward section's `backward_ecl` readable name which correctly maps to `bw_ecl`. |
| `num_batches_per_set` | Remove from config. This value is computed internally as `set_size // batch_size`. |

## Runtime-Injected Fields

These fields are **set programmatically** by `FastGM` during initialization — they should
**not** be set by the user in config dicts. They are present in `config` at read-time during
training because the graphical model populates them after computing global statistics.

| Field | Type | Purpose |
|---|---|---|
| `sigma_g_global` | `float` | Global standard deviation of backward message gradients across all buckets. Used by `approx_smg` loss functions. |
| `rho_global` | `float` | Global correlation between forward and backward messages. Used by `approx_smg` loss functions. |

## Legacy Flat-Only Fields

These fields are recognized in flat configs but are **not part of any nested section**.
They exist in `_LEGACY_FLAT_FIELDS` in the schema and are used by specific consumer code
paths.

| Field | Type | Purpose |
|---|---|---|
| `exact` | `bool` | When `True`, forces exact computation for all buckets and tracks numel (number of tensor elements) per factor. Used in `FastBucket.compute_message_exact()` for complexity measurement. Not applicable to NN-based inference. |
| `memorizer` | object | A pre-built memorizer object passed directly to the `Net` constructor. Used when `use_memorizer=True` with a pre-constructed lookup table. Rarely set manually. |

---

## Notes

- **Source of truth**: `NESTED_SECTIONS` in `nce/config_schema.py` defines every recognized
  field. If a field is not in the schema, `prepare_config()` will reject it in nested mode.
- **Alias pairs**: Fields with both a readable name and an internal name (e.g.,
  `learning_rate` → `lr`) are shown together in each table. Either name works in configs;
  consumer code always reads the internal name.
- **Required fields**: When `approximation_method='nn'`, the fields `loss_fn`, `num_epochs`,
  and `num_samples` are required. All other fields have defaults.
- **Phase 2 training**: Set `num_epochs2 > 0` and `loss_fn2` to enable a second training
  phase with a different loss function (e.g., fine-tuning with a message-gradient loss after
  initial MSE training).
- **Decision tree fields**: Fields prefixed with `dt_` are only used when
  `approximation_method='dt'`. They default to `None` and are passed to
  `DecisionTreeLossOptimizer`.
