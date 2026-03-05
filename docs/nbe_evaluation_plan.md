# NBE Algorithm Evaluation Plan

## Goal

Systematically evaluate the NeuroBE (NBE) algorithm implementation in this codebase by running pre-smoke tests, smoke tests, ablation studies, and comparison against the original NeuroBE paper results.

## Current State

### What We Have
- **Benchmark set:** `nbe_sanity_check` with 5 models (pedigree13, grid40x40.f10, grid20x20.f10, rbm_20, grid10x10.f5.wrap)
- **NBE-specific features implemented:**
  - `compute_nbe_num_samples(w, l, epsilon)` -- per-bucket sample count from the paper's formula
  - `hidden_sizes='nbe,<b>'` -- per-bucket NN sizing: one hidden layer of size `h = b * ceil(log2(message_size))`
  - `nbe_early_stopping` -- validation-based early stopping (currently disabled in configs)
  - `weighted_logspace_mse` -- weighted MSE loss function in log-space
- **Baseline configs:** `small_problems` benchmark set with standard NN configs (fixed hidden sizes, fixed num_samples=100k, unnormalized_kl loss)
- **Experiment runner:** YAML-config-based, multi-GPU, multi-run, auto-plotting

### NeuroBE Paper Parameters (from Config.h)
- `num_epochs = 500`
- `lr = 0.001`
- `dope_factors = True` (replace -inf with -5 for stable training)
- `loss_fn = 'weighted_logspace_mse'`
- Log Z comparison values can come from the NeuroBE paper (prompts/NeuroBE.pdf) but exact match is not expected due to different num_trained counts and implementation differences.

### Pre-Requisites Checklist

- [x] Fix loss_fn name: `'weighted_mse'` -> `'weighted_logspace_mse'` (DONE - quick task 7, Task 1)
- [x] Fix backward_iB: matches per-model iB via `_IB_MAP[key]` (DONE - quick task 7, Task 1)
- [x] Fix dope_factors: `False` -> `True` (DONE - quick task 7, Task 1)
- [x] Fix num_epochs: `10000` -> `500` (DONE - quick task 7, Task 1)
- [x] Fix lr: `0.01` -> `0.001` (DONE - quick task 7, Task 1)
- [x] Fix backward_ecl: `2**22` -> `None` (irrelevant for NBE since use_bw_approx=False) (DONE - quick task 7, Task 1)
- [ ] Verify `'nbe,<epsilon>'` num_samples resolution works end-to-end in compute_message_nn
- [x] Verify `'nbe,<b>'` hidden_sizes resolution works (already tested)
- [ ] Confirm all 5 benchmark models can be loaded (catalogue + cache refresh)
- [ ] Confirm `weighted_logspace_mse` loss function works correctly in training loop

**Note:** `bw_ib`, `bw_ecl`, and `backward_ecl` are irrelevant for NBE since `use_bw_approx=False` is the default. Setting `backward_ecl=None` makes this explicit.

---

## Evaluation Phases

### Phase 0a: Exact Computation Baseline

**Purpose:** Verify that exact bucket elimination works on grid10x10.f5.wrap by using very high ecl and iB values (forcing all buckets to be computed exactly, no NN training).

**Setup:**
```python
from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM

model = nbe_sanity_check.problems[4]  # grid10x10.f5.wrap
config = dict(nbe_sanity_check.configs['nbe'][4])
config['ecl'] = 2**30       # force exact computation everywhere
config['iB'] = 30           # effectively exact (no mini-buckets)
config['device'] = 'cpu'

fastgm = FastGM(model=model, nn_config=config, device='cpu')
# dope_factors is called automatically by constructor since config['dope_factors']=True
log_z = fastgm.get_log_partition_function()
print(f"Log Z estimate: {log_z}")
```

**What to check:**
- Does `FastGM` construct without errors?
- Does `get_log_partition_function()` complete?
- Is the log Z estimate reasonable (compare to `model.ln_z` if available)?

**Output:** Single log Z estimate, confirmation that exact mode works.

**Script:** `notebooks/March-2026/claude_experiments/nbe_eval_phase0a_exact.py`

---

### Phase 0b: Single Bucket NN Test

**Purpose:** Test NN training on ONE large bucket in isolation, verifying that the NN training pipeline (sampling, loss function, optimizer) works for a single bucket before running full inference.

**Setup:**
```python
from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM

model = nbe_sanity_check.problems[0]  # pedigree13 (1077 vars, width 32)
config = dict(nbe_sanity_check.configs['nbe'][0])
config['ecl'] = 2**30       # exact mode for all preceding buckets
config['iB'] = 30
config['num_epochs'] = 1    # just 1 epoch to test pipeline
config['device'] = 'cpu'

fastgm = FastGM(model=model, nn_config=config, device='cpu')
# dope_factors called automatically by constructor

# Find large buckets -- returns INTEGER labels (not Var objects)
large_labels = fastgm.get_large_message_buckets(iB=15, debug=True)
target_label = large_labels[0]  # integer

# CRITICAL: Convert int label to Var object before passing to eliminate_variables
target_var = fastgm.matching_var(target_label)
fastgm.eliminate_variables(up_to=target_var)

# get_bucket accepts int or Var
bucket = fastgm.get_bucket(target_label)
bucket.compute_message_nn()
```

**Key API notes:**
- `get_large_message_buckets()` returns **integer labels** (var.label values)
- `eliminate_variables(up_to=...)` expects a **Var object**
- Use `fastgm.matching_var(int_label)` to convert integer -> Var
- `get_bucket()` accepts either int or Var

**Output:** Confirmation that a single bucket can be trained, message shape printed.

**Script:** `notebooks/March-2026/claude_experiments/nbe_eval_phase0b_single_bucket.py`

---

### Phase 1: Smoke Test (grid10x10.f5.wrap, full NBE pipeline)

**Purpose:** Verify the full NBE pipeline works end-to-end on the smallest problem. Run with num_epochs=1 first (practice), then with the correct num_epochs=500.

**Practice run (1 epoch):**
```python
model = nbe_sanity_check.problems[4]  # grid10x10.f5.wrap
config = dict(nbe_sanity_check.configs['nbe'][4])
config['num_epochs'] = 1
config['device'] = 'cpu'
# Use benchmark defaults for ecl and iB (ecl=2**22, iB=10)

fastgm = FastGM(model=model, nn_config=config, device='cpu')
log_z = fastgm.get_log_partition_function()
```

**Full run (500 epochs):**
```python
config = dict(nbe_sanity_check.configs['nbe'][4])
config['device'] = 'cpu'
# ecl=2**22 and iB=10 are already set by benchmark config

fastgm = FastGM(model=model, nn_config=config, device='cpu')
log_z = fastgm.get_log_partition_function()
```

**What to check:**
- Does the `'nbe,0.35'` num_samples string resolve to an integer per bucket?
- Does the `'nbe,1'` hidden_sizes string resolve to a single hidden layer of size `h = 1 * ceil(log2(message_size))`?
- Is the log Z estimate reasonable (compare to exact value from Phase 0a)?
- How many buckets used NN training? (`fastgm.num_trained`)

**Output:** Log Z estimate, per-bucket sample counts, number of trained buckets, timing.

**Scripts:**
- Practice: `notebooks/March-2026/claude_experiments/nbe_eval_practice_1epoch.py`
- Full: `notebooks/March-2026/claude_experiments/nbe_eval_phase1_smoke.py` (to be created later)

---

### Phase 2: Single-Problem Deep Dive (rbm_20)

**Purpose:** rbm_20 has the highest width (20) among the sanity check set, so NBE's adaptive sampling matters most here. (Note: pedigree13 has width 32 and grid40x40.f10 has width 54, but those are larger problems better suited for Phase 3.)

**Setup:**
```python
model = nbe_sanity_check.problems[3]  # rbm_20
config = dict(nbe_sanity_check.configs['nbe'][3])
config['track_errors'] = True
config['device'] = 'cuda'
```

**What to measure:**
- Log Z estimate vs exact value
- Per-bucket: actual num_samples used (from nbe formula), hidden sizes chosen, training loss curve
- Total training time
- Compare to baseline: same problem with fixed num_samples=50000, hidden_sizes=[3,3], loss='unnormalized_kl'

**Ablation variants (run each separately):**
| Variant | num_samples | hidden_sizes | loss_fn | Description |
|---------|-------------|--------------|---------|-------------|
| NBE-full | nbe,0.1 | nbe,3 | weighted_logspace_mse | Full NBE config |
| NBE-fixed-samples | 50000 | nbe,3 | weighted_logspace_mse | Remove adaptive sampling |
| NBE-fixed-arch | nbe,0.1 | [3,3] | weighted_logspace_mse | Remove adaptive architecture |
| NBE-fixed-loss | nbe,0.1 | nbe,3 | unnormalized_kl | Remove weighted loss |
| Baseline | 50000 | [3,3] | unnormalized_kl | Our standard approach |

**hidden_sizes explanation:**
- `'nbe,1'` means scaling factor b=1, so `h = 1 * ceil(log2(message_size))`, giving ONE hidden layer of that size
- `'nbe,3'` means `h = 3 * ceil(log2(message_size))`, giving ONE hidden layer of that size
- `[3,3]` means two hidden layers of size 3 each (fixed, not adaptive)

**Output:** Table of log Z errors, timing, and per-bucket stats for each variant.

**Script location:** `notebooks/March-2026/claude_experiments/nbe_eval_phase2_rbm20.py`

---

### Phase 3: Full Benchmark Evaluation

**Purpose:** Run NBE on all 5 sanity check problems.

**Setup:** Use the experiment runner framework with YAML configs, or run scripts directly.

**Experiment matrix:**

| Problem | iB | ecl | hidden_sizes | num_samples | Runs |
|---------|-----|------|-------------|-------------|------|
| pedigree13 | 20 | 2^22 | nbe,3 | nbe,0.1 | 3 |
| grid40x40.f10 | 20 | 2^22 | nbe,1 | nbe,0.35 | 3 |
| grid20x20.f10 | 10 | 2^22 | nbe,1 | nbe,0.35 | 3 |
| rbm_20 | 20 | 2^22 | nbe,3 | nbe,0.1 | 3 |
| grid10x10.f5.wrap | 10 | 2^22 | nbe,1 | nbe,0.35 | 3 |

**Output table:**

| Problem | num_trained | num_vars | total_time | log_Z_true | log_Z_estimate | abs_err |
|---------|-------------|----------|------------|------------|----------------|---------|
| pedigree13 | | 1077 | | | | |
| grid40x40.f10 | | 1600 | | | | |
| grid20x20.f10 | | 400 | | | | |
| rbm_20 | | 40 | | | | |
| grid10x10.f5.wrap | | 100 | | | | |

**Script location:** `notebooks/March-2026/claude_experiments/nbe_eval_phase3_full.py`

---

### Phase 4: Early Stopping Evaluation

**Purpose:** Test whether `nbe_early_stopping` can reduce training time without sacrificing accuracy.

**Setup:** Re-run Phase 3 NBE configs with early stopping enabled:
```python
config['nbe_early_stopping'] = True
config['nbe_warmup_epochs'] = 50   # try different warmup values
config['num_epochs'] = 10000       # cap (early stopping should terminate before this)
```

**Variants:**
| Warmup Epochs | Expected Behavior |
|---------------|-------------------|
| 0 | Aggressive -- may stop too early |
| 10 | Moderate |
| 50 | Conservative |
| 100 | Very conservative |

**Metrics:** Same as Phase 3, plus:
- Actual epochs trained per bucket
- Early stopping trigger rate (% of buckets that stopped early)

**Script location:** `notebooks/March-2026/claude_experiments/nbe_eval_phase4_earlystop.py`

---

### Phase 5: Paper Comparison

**Purpose:** Compare our NBE implementation against reported results from the NeuroBE paper (Agarwal et al.).

**Paper benchmarks to replicate:**
- Pedigree instances (paper reports h=3w, N_avg 149k-350k)
- DBN/RBM instances (paper reports h={3w,5w}, N_avg 80k-180k)
- Grid instances (paper reports h=w, N_avg 12k-209k)

**What to compare:**
- Average sample count per bucket (ours vs paper)
- Log Z accuracy (if paper reports it)
- Hidden layer sizes (our `nbe,b` gives one layer of `b * ceil(log2(message_size))` vs paper's scaling)

**Note:** The paper uses a different notation (eta instead of epsilon) and a slightly different formula. The mapping is documented in `docs/neurobe_epsilon_values.md`.

**Script location:** `notebooks/March-2026/claude_experiments/nbe_eval_phase5_paper.py`

---

## Execution Order

```
Phase 0a (Exact Computation Baseline)
  |
  v
Phase 0b (Single Bucket NN Test)
  |
  v
Phase 1 - Practice (1-epoch full run on grid10x10)
  |
  v
Phase 1 - Full (500-epoch run on grid10x10)
  |
  v
Phase 2 (Deep Dive + Ablation on rbm_20)
  |
  v
Phase 3 (Full Benchmark - all 5 problems)
  |
  +---> Phase 4 (Early Stopping)  [can run in parallel with Phase 5]
  |
  +---> Phase 5 (Paper Comparison)
```

## Success Criteria

1. **Phase 0a:** grid10x10.f5.wrap exact elimination produces a log Z value
2. **Phase 0b:** Single bucket NN trains for 1 epoch without errors on pedigree13
3. **Phase 1:** grid10x10.f5.wrap runs end-to-end with NBE config, produces a log Z estimate
4. **Phase 2:** rbm_20 ablation shows which NBE component contributes most to accuracy
5. **Phase 3:** All 5 problems produce results; output table populated
6. **Phase 4:** Early stopping reduces training time by >20% with <5% accuracy degradation
7. **Phase 5:** Our sample counts and hidden sizes are within 2x of paper's reported values

## File Structure

```
notebooks/March-2026/claude_experiments/
├── nbe_eval_phase0a_exact.py           # Phase 0a: exact computation baseline
├── nbe_eval_phase0b_single_bucket.py   # Phase 0b: single bucket NN test
├── nbe_eval_practice_1epoch.py         # Phase 1 practice: 1-epoch full run
├── nbe_eval_phase1_smoke.py            # Phase 1: full 500-epoch smoke test
├── nbe_eval_phase2_rbm20.py            # Phase 2: rbm_20 deep dive + ablation
├── nbe_eval_phase3_full.py             # Phase 3: full benchmark comparison
├── nbe_eval_phase4_earlystop.py        # Phase 4: early stopping evaluation
├── nbe_eval_phase5_paper.py            # Phase 5: paper comparison
└── nbe_eval_results/                   # Output directory for all results
    ├── phase0a/
    ├── phase0b/
    ├── phase1/
    ├── phase2/
    ├── phase3/
    ├── phase4/
    └── phase5/
```
