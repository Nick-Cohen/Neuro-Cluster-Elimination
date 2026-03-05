# NBE Algorithm Evaluation Plan

## Goal

Systematically evaluate the NeuroBE (NBE) algorithm implementation in this codebase against both our standard NN baselines and, where possible, against results from the original NeuroBE paper.

## Current State

### What We Have
- **Benchmark set:** `nbe_sanity_check` with 5 models (pedigree13, grid40x40.f10, grid20x20.f10, rbm_20, grid10x10.f5.wrap)
- **NBE-specific features implemented:**
  - `compute_nbe_num_samples(w, l, epsilon)` — per-bucket sample count from the paper's formula
  - `hidden_sizes='nbe,<b>'` — per-bucket NN sizing: `h = b * ceil(log2(message_size))`
  - `nbe_early_stopping` — validation-based early stopping (currently disabled in configs)
  - `weighted_logspace_mse` — weighted MSE loss function
- **Baseline configs:** `small_problems` benchmark set with standard NN configs (fixed hidden sizes, fixed num_samples=100k, unnormalized_kl loss)
- **Experiment runner:** YAML-config-based, multi-GPU, multi-run, auto-plotting

### Known Issues to Fix Before Evaluation

1. **Loss function name mismatch:** `nbe_sanity_check` configs use `'weighted_mse'` but the registered loss function is `'weighted_logspace_mse'`. Fix the config to use the correct name, or add an alias.
2. **`backward_iB` is hardcoded to 10** in all nbe_sanity_check configs, but `iB` varies per model (10 or 20). `backward_iB` should match `iB`.
3. **`nbe,<epsilon>` resolution in compute_message_nn** — verify this is wired up correctly end-to-end (the string → int conversion must happen before Trainer sees it).

---

## Evaluation Phases

### Phase 1: Smoke Test (grid10x10.f5.wrap, exact mode)

**Purpose:** Verify the full NBE pipeline works end-to-end on the smallest problem before scaling up.

**Setup:**
```python
from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM

model = nbe_sanity_check.problems[4]  # grid10x10.f5.wrap
config = dict(nbe_sanity_check.configs['nbe'][4])
config['ecl'] = 2**30       # exact computation everywhere
config['iB'] = 30           # effectively exact
config['device'] = 'cpu'    # no GPU needed for small problem
```

**What to check:**
- Does `FastGM(model=model, nn_config=config)` construct without errors?
- Does `fastgm.run()` complete?
- Does the `'nbe,0.35'` num_samples string resolve to an integer per bucket?
- Does the `'nbe,1'` hidden_sizes string resolve to `[h, h]` per bucket?
- Is the log Z estimate reasonable (compare to exact value from `model.logZ` or `model.ln_z`)?

**Output:** Single log Z estimate, per-bucket sample counts printed, confirmation of end-to-end flow.

**Script location:** `notebooks/March-2026/claude_experiments/nbe_eval_phase1_smoke.py`

---

### Phase 2: Single-Problem Deep Dive (rbm_20)

**Purpose:** rbm_20 is the problem with the largest bucket widths (up to 20) in our sanity check set, so NBE's adaptive sampling matters most here. Run it and analyze per-bucket behavior.

**Setup:**
```python
model = nbe_sanity_check.problems[3]  # rbm_20
config = dict(nbe_sanity_check.configs['nbe'][3])
config['track_errors'] = True   # record per-bucket errors
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

**Output:** Table of log Z errors, timing, and per-bucket stats for each variant.

**Script location:** `notebooks/March-2026/claude_experiments/nbe_eval_phase2_rbm20.py`

---

### Phase 3: Full Benchmark Evaluation

**Purpose:** Run NBE on all 5 sanity check problems and compare to baselines.

**Setup:** Use the experiment runner framework with YAML configs.

**Experiment matrix:**

| Problem | NBE Config | Baseline Config | Runs |
|---------|-----------|----------------|------|
| pedigree13 | nbe,0.1 / nbe,3 / weighted_logspace_mse / iB=20 | 50k / [3,3] / ukl / iB=20 | 3 |
| grid40x40.f10 | nbe,0.35 / nbe,1 / weighted_logspace_mse / iB=20 | 50k / [3,3] / ukl / iB=20 | 3 |
| grid20x20.f10 | nbe,0.35 / nbe,1 / weighted_logspace_mse / iB=10 | 50k / [3,3] / ukl / iB=10 | 3 |
| rbm_20 | nbe,0.1 / nbe,3 / weighted_logspace_mse / iB=20 | 50k / [3,3] / ukl / iB=20 | 3 |
| grid10x10.f5.wrap | nbe,0.35 / nbe,1 / weighted_logspace_mse / iB=10 | 50k / [3,3] / ukl / iB=10 | 3 |

**Metrics to compare:**
- Log Z error (estimate - exact)
- Total wall-clock time
- Total samples generated across all buckets
- Number of NN-trained buckets (vs exact computation)

**Expected output:**
```
| Problem         | NBE LogZ Error | Baseline LogZ Error | NBE Time | Baseline Time | NBE Samples | Baseline Samples |
|-----------------|----------------|---------------------|----------|---------------|-------------|------------------|
| pedigree13      |                |                     |          |               |             |                  |
| grid40x40.f10   |                |                     |          |               |             |                  |
| ...             |                |                     |          |               |             |                  |
```

**Script location:** `notebooks/March-2026/claude_experiments/nbe_eval_phase3_full.py`
**Config files:** `notebooks/March-2026/claude_experiments/nbe_eval_nbe.yaml`, `nbe_eval_baseline.yaml`

---

### Phase 4: Early Stopping Evaluation

**Purpose:** Test whether `nbe_early_stopping` can reduce training time without sacrificing accuracy.

**Setup:** Re-run the Phase 3 NBE configs with early stopping enabled:
```python
config['nbe_early_stopping'] = True
config['nbe_warmup_epochs'] = 50   # try different warmup values
config['num_epochs'] = 10000       # cap (early stopping should terminate before this)
```

**Variants:**
| Warmup Epochs | Expected Behavior |
|---------------|-------------------|
| 0 | Aggressive — may stop too early |
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
- Hidden layer sizes (our `nbe,b` vs paper's scaling)

**Note:** The paper uses a different notation (eta/η instead of epsilon/ε) and a slightly different formula. The mapping is documented in `docs/neurobe_epsilon_values.md`.

**Script location:** `notebooks/March-2026/claude_experiments/nbe_eval_phase5_paper.py`

---

## Execution Order

```
Phase 1 (Smoke Test)
  └─ Fix bugs found
      └─ Phase 2 (Deep Dive on rbm_20)
          └─ Phase 3 (Full Benchmark)
              ├─ Phase 4 (Early Stopping)  [can run in parallel with Phase 5]
              └─ Phase 5 (Paper Comparison)
```

## Pre-Requisites Checklist

- [ ] Fix loss_fn name: change `'weighted_mse'` to `'weighted_logspace_mse'` in nbe_sanity_check configs
- [ ] Fix backward_iB: should match per-model iB (currently hardcoded to 10)
- [ ] Verify `'nbe,<epsilon>'` num_samples resolution works end-to-end in compute_message_nn
- [ ] Verify `'nbe,<b>'` hidden_sizes resolution works (already tested)
- [ ] Confirm all 5 benchmark models can be loaded (catalogue + cache refresh)
- [ ] Confirm `weighted_logspace_mse` loss function works correctly in training loop

## Success Criteria

1. **Phase 1:** grid10x10.f5.wrap runs end-to-end with NBE config, produces a log Z estimate
2. **Phase 2:** rbm_20 ablation shows which NBE component contributes most to accuracy
3. **Phase 3:** All 5 problems produce results; NBE vs baseline comparison table populated
4. **Phase 4:** Early stopping reduces training time by >20% with <5% accuracy degradation
5. **Phase 5:** Our sample counts and hidden sizes are within 2x of paper's reported values

## File Structure

```
notebooks/March-2026/claude_experiments/
├── nbe_eval_phase1_smoke.py           # Phase 1: smoke test
├── nbe_eval_phase2_rbm20.py           # Phase 2: rbm_20 deep dive + ablation
├── nbe_eval_phase3_full.py            # Phase 3: full benchmark comparison
├── nbe_eval_phase4_earlystop.py       # Phase 4: early stopping evaluation
├── nbe_eval_phase5_paper.py           # Phase 5: paper comparison
├── nbe_eval_nbe.yaml                  # YAML config for NBE experiments
├── nbe_eval_baseline.yaml             # YAML config for baseline experiments
└── nbe_eval_results/                  # Output directory for all results
    ├── phase1/
    ├── phase2/
    ├── phase3/
    ├── phase4/
    └── phase5/
```
