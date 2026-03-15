# M003: NeuroBE Reproduction Mode — Research

**Date:** 2026-03-12

## Summary

The NeuroBE reproduction mode requires changes across three layers: **data normalization** (min-max [0,1] in DataPreprocessor + corresponding denormalization in FactorNN), **training control** (patience-2 early stopping on validation loss), and **config integration** (neurobe_mode flag that sets a bundle of defaults). The existing codebase is well-structured for these changes — DataPreprocessor already handles normalization/denormalization as a swappable pipeline, Trainer already has multiple early stopping modes, and config_schema has a clean extension pattern.

The riskiest part isn't the code changes — it's **verifying that the NN counts match**. Research confirmed that variable orderings are identical between NCE and NeuroBE (verified for all 15 problems), which eliminates the #1 risk from the context doc. The ecl-to-width mapping for binary-domain problems is straightforward: `ecl = 2^(width_problem)` where `width_problem = MaxWidth - 1`. The existing `auto_ecl` values in `small_problems.py` are *wrong* for NeuroBE matching (off by 1 or off by a factor of 2), but the correct values are directly derivable from the NeuroBE results CSV.

The recommended approach is a **two-slice structure**: S01 implements the normalization, early stopping, and config machinery with tests; S02 does the ecl tuning, runs all 15 problems, and produces the comparison table. S01 should be proven first because the normalization round-trip is the most critical correctness property — if normalize→train→denormalize doesn't work, all experiments are worthless.

## Recommendation

**Extend DataPreprocessor with a `normalization_mode` parameter** that selects between `'logspace_mean'` (current default) and `'minmax_01'` (NeuroBE). This keeps the existing interface clean and avoids a separate class. The `undo_normalization` method already exists and is called in exactly two places (`factor_nn.py` lines 121 and 272) — it just needs a mode-aware implementation.

**Add patience-based early stopping as a new mode** in the training loop, selected by config. The existing `nbe_early_stopping` flag already exists with 3-consecutive-increases logic — repurpose or extend it rather than adding a wholly new mechanism. NeuroBE uses `stop_iter=2` (stop after 2 consecutive non-improving epochs), which is simpler than the current 3-consecutive-increases check.

**Keep the weighted MSE loss function separate** rather than trying to modify the existing `weighted_logspace_mse`. The existing one operates on already-normalized logspace targets; the NeuroBE-faithful version needs to operate on [0,1]-normalized targets with weights derived from the raw values. A clean `neurobe_weighted_mse` function avoids entangling the two.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Config validation/defaults | `prepare_config()` in config_schema.py | Already handles flat/nested detection, alias resolution, defaults. Just add new fields for neurobe_mode. |
| Variable ordering match verification | `.vo` and `.ord.elim` files in .model_cache and small_problems_uai/ | Research confirmed orderings are identical — no custom code needed, just document the finding. |
| NN input representation for binary domains | `lower_dim=True` with binary domains | Produces identical input to NeuroBE's raw integer encoding (both yield nArgs-dimensional 0/1 vectors). |
| Batch data loading pipeline | DataLoader + SampleGenerator in nce/data/ | Already supports sampling, normalization, backward message computation. Min-max mode plugs into DataPreprocessor.normalize(). |
| ecl values for NeuroBE NN matching | NeuroBE results CSV (`binary_domain_results.csv`) | Contains MaxWidth and width_problem per problem. For binary domains, `ecl = 2^width_problem` directly. |

## Existing Code and Patterns

- `nce/data/data_preprocessor.py` — Central normalization/denormalization logic. `normalize()` and `undo_normalization()` are the two methods to extend. Currently stores `self.normalizing_constant` (log-space mean); needs to also store `self.ln_min` and `self.ln_max` for min-max mode. The `_initialize_normalizing_constant()` method computes stats from training data — same call site works for computing min/max.

- `nce/inference/factor_nn.py` lines 121, 272 — The only two places `undo_normalization` is called. Both call `data_processor.undo_normalization(net_output)`. No changes needed here if `undo_normalization` is mode-aware.

- `nce/neural_networks/train.py` — `Trainer.train()` method (900+ lines). The early stopping logic lives at lines ~320-430 (multiple modes). The NBE early stopping check at ~430-490 uses 3-consecutive-increases on validation loss. Adding patience-2 is a small change in this block. Validation set generation at line 228 (`_generate_validation_set_nbe`) already creates a validation set from uniform samples.

- `nce/neural_networks/losses.py` — `weighted_logspace_mse()` at line 751 is NCE's current version. Operates on normalized logspace targets. NeuroBE's version operates on [0,1]-normalized targets with `w = labels * (ln_max - ln_min) / sum_ln` where `sum_ln = sum(arr[i] - ln_min)`. The weight reduces to `w[i] = (value_i - ln_min) / sum(value_j - ln_min)` — the proportion of shifted value.

- `nce/config_schema.py` — Schema definition with `NESTED_SECTIONS` OrderedDict. Adding `neurobe_mode` as a training field follows the existing pattern. `prepare_config()` can expand `neurobe_mode=True` into the individual settings.

- `nce/benchmark_problems/small_problems.py` — 24-problem benchmark set with `_AUTO_ECL` values. These ecl values **do not match** NeuroBE's width-based dispatch. Need new per-problem ecl values: `2^width_problem` from the NeuroBE results CSV.

- `tests/conftest.py` — Shared fixtures including `nn_training_config` (CPU-based, validated config). New tests should follow this pattern (hand-built factors, known Z values, D021/D022 conventions).

- `Clean-NeuroBE/.../Function-NN.hxx` — NeuroBE's Train() function (lines 157-310). Key behaviors: `log_sum_exp()` computes `ln_min`, `ln_max`, `sum_ln` from training data only; `samples_to_data()` normalizes to [0,1]; IS weights are `w = labels * (ln_max - ln_min) / sum_ln`; early stopping is `count > stop_iter` where count increments on non-improving epochs and resets on improvement.

- `Clean-NeuroBE/.../MiniBucket-NN.cpp` — Sample generation: 80/20 train/val split + 50k test. Sample count from pseudo-dimension formula: `nSamples = (pd + log(1/δ)) / ε` where `pd = ((l-1)*w_in² + l*w_in + 4) * log(pd_temp/l)`, `w_in = nArgs * var_dim`.

## Constraints

- **Binary domains only.** All 15/16 problems are binary (domain size 2). The `net` architecture path is used (not `masked_net`). This means: no input normalization (raw 0/1 values), no exp-conversion, no mask branch, denorm = `ln_min + nn_out * (ln_max - ln_min)`.

- **NeuroBE uses ReLU; NCE uses Tanh.** NeuroBE's `Net` has ReLU activations. NCE's `Net` uses Tanh (line 52 in net.py). This is a known architectural difference. For a **faithful** reproduction, NCE should use ReLU in neurobe_mode — either by adding an activation config parameter or by accepting this as a documented difference. *Candidate for user decision.*

- **NeuroBE hidden dim = nArgs × var_dim; NCE hidden dim = user-specified list.** NeuroBE sizes the hidden layers as `nArgs * var_dim` (e.g., `nArgs * 3`). NCE's existing `'nbe,3'` hidden_sizes mode computes `h = 3 * ceil(log2(message_size))` which is close but not identical. For binary domains, `nArgs = scope_size` and `log2(message_size) = scope_size - 1` (since message excludes eliminated var), so `nbe,3` gives `3 * (nArgs - 1)` vs NeuroBE's `3 * nArgs`. Off by a factor of `nArgs/(nArgs-1)`. *Should use `nArgs * var_dim` directly in neurobe_mode.*

- **NeuroBE learning rate is hardcoded at 0.001.** CLI bug means the lr parameter is ignored. NCE's `small_problems` configs use `lr=0.01`. The neurobe_mode config must use 0.001.

- **All operations in log10 space internally.** NCE factors store values in log10 space. NeuroBE stores raw function values (natural log in some places). The min-max normalization in NeuroBE operates on these raw values. NCE's DataPreprocessor converts to natural log space for normalization. The min-max normalization should be applied **after** conversion to natural log — i.e., min/max of `y_ln = y_log10 * ln(10)`.

- **NeuroBE computes sum_ln from training data only.** `ln_min`, `ln_max`, `sum_ln` are all computed from the training split (not validation or test). NCE's lazy normalization computes from whatever data is first loaded. This is fine for `sampling_scheme='all'` (all data is training), but for NeuroBE-faithful sampling, the train/val split matters.

- **NeuroBE uses n_epochs=500, stop_iter=2, batch_size=256.** These are the universal defaults across all 4 benchmark configurations.

- **NeuroBE uses iB=25 for binary domain problems** (from the results CSV), not iB=999. NCE's `small_problems` uses iB=100.

## Common Pitfalls

- **Min-max normalization with constant targets.** If all targets in a bucket are identical (all entries have the same value), `ln_max - ln_min = 0`, causing division by zero. NeuroBE doesn't guard against this. NCE should add an epsilon guard or skip normalization when the range is zero. This occurs in practice for trivial buckets.

- **Denormalization base mismatch.** NCE works in log10 space; NeuroBE works in natural log. The denormalization formula `ln_min + nn_out * (ln_max - ln_min)` produces a value in natural log space. NCE must convert back to log10 after denormalization: `result_log10 = (ln_min + nn_out * (ln_max - ln_min)) / ln(10)`. Getting this conversion wrong silently produces wrong results that look plausible.

- **Early stopping patience semantics.** NeuroBE's `count > stop_iter` with `stop_iter=2` means training stops after **3** consecutive non-improving epochs (count goes 1, 2, 3 → breaks when count > 2). Not 2 as the context doc says. The actual behavior is: count starts at 0, increments on non-improvement, resets to 0 on improvement. When `count > stop_iter`, loop breaks. So with `stop_iter=2`, you need 3 non-improving epochs. Need to verify this interpretation against actual NeuroBE runs.

- **Weight formula depends on normalization.** The IS weights in NeuroBE Train() are `w = labels * (ln_max - ln_min) / sum_ln` where `labels` are [0,1]-normalized. If the normalization changes, the weights change. The loss function and normalization are coupled — can't test them independently.

- **ecl off-by-one errors.** NCE's dispatch condition is `message_size > ecl` (strictly greater). NeuroBE's is `_Width > width_problem` (also strictly greater). For binary domains, message_size = `2^(scope_size - 1)`. A bucket with `scope_size = width_problem + 1` has `message_size = 2^width_problem`. Using `ecl = 2^width_problem` means `message_size > ecl` is `2^width_problem > 2^width_problem` which is **false** — the bucket would NOT get NN treatment. Need `ecl = 2^width_problem - 1` to match NeuroBE's dispatch. This is the classic off-by-one.

  Wait — NeuroBE dispatches on `_Width > width_problem` where `_Width` is the bucket width (number of variables in scope), not message size. For a bucket with `_Width = width_problem + 1`, this triggers NN dispatch. The corresponding message_size for binary is `2^(_Width - 1) = 2^width_problem`. NCE dispatches when `message_size > ecl`. So we need `2^width_problem > ecl`, i.e., `ecl = 2^width_problem - 1` or equivalently `ecl < 2^width_problem`. Setting `ecl = 2^width_problem - 1` ensures any bucket with width > width_problem gets NN treatment. **This matches the auto_ecl pattern** — many auto_ecl values are `2^n - 1`.

- **NeuroBE early stopping uses the IS-weighted validation loss** when `s_method="is"`, not the unweighted MSE. The `loss_to_compare` variable selects `w_val_mse` when IS mode is active. NCE's early stopping evaluates whatever `self.loss_fn` is — so if the loss function is set to weighted MSE, the early stopping metric matches automatically.

## Open Risks

- **Activation function difference (Tanh vs ReLU).** NCE's Net uses Tanh, NeuroBE's Net uses ReLU. This affects optimization landscape and convergence speed. For a faithful reproduction, either add a config option for activation function, or document this as a known difference and accept result variance. This is low risk for comparison validity but high risk for exact number matching.

- **Sample generation differs.** NeuroBE uses a pseudo-dimension formula for sample count and splits 80/20 train/val with an additional 50k test set. NCE uses `num_samples` from config with `sampling_scheme='all'` or `'uniform'`. The NeuroBE-faithful sample count depends on bucket size and `epsilon` — it's per-bucket, not a global setting. NCE's `'nbe,0.1'` num_samples mode already implements this formula, but the train/val split differs (NCE uses `num_samples // 9` for validation vs NeuroBE's 20%).

- **Xavier init vs default PyTorch init.** NCE's Net uses `xavier_normal_` initialization. NeuroBE's libtorch Net uses PyTorch's default initialization (which is Kaiming uniform). Different initialization can affect convergence behavior for the same hyperparameters.

- **AMP (mixed precision) in NCE.** NCE's Trainer uses `torch.cuda.amp.GradScaler` with autocast enabled. NeuroBE doesn't use mixed precision. This could cause numerical differences. NeuroBE mode should probably disable AMP.

## Candidate Requirements (Advisory)

These emerged from research but are not in the current scope. Surface for user decision:

1. **ReLU activation option** — Add `activation='relu'` config option for neurobe_mode. Without this, results will differ due to optimization landscape differences. *Recommend: include in S01 as part of neurobe_mode preset.*

2. **NeuroBE-faithful hidden dimension sizing** — Use `nArgs * var_dim` instead of `nbe,3` (which gives `3 * ceil(log2(message_size))`). Difference is small for large problems but noticeable for small ones. *Recommend: add `hidden_sizes='neurobe,3'` mode that uses `nArgs * var_dim`.*

3. **Disable AMP in neurobe_mode** — NeuroBE doesn't use mixed precision. AMP can introduce numerical differences. *Recommend: include as part of neurobe_mode preset.*

4. **NeuroBE-faithful sample count and train/val split** — NCE's current val set is `num_samples // 9`; NeuroBE uses 80/20 split of pseudo-dimension-based count. *Recommend: defer to S02 or accept as known difference.*

5. **Default init vs Xavier init** — NeuroBE uses PyTorch default (Kaiming uniform); NCE uses Xavier normal. *Recommend: document as known difference; matching init is possible but low priority.*

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| PyTorch | letta-ai/skills@pytorch-model-cli | available (31 installs) — not relevant, this is a model management CLI |
| Python scientific computing | — | none found — custom domain-specific work |

No relevant skills for this milestone. The work is domain-specific (probabilistic inference + neural networks) with no external service integrations.

## Requirement Analysis

### Table Stakes (R033-R038 are well-scoped)

All six active requirements are necessary and well-defined. R033 (min-max normalization) and R034 (patience-2 early stopping) are the core algorithmic changes. R035 (neurobe_mode config preset) is the integration glue. R036 (matched NN counts) and R038 (comparison table) are the validation/deliverable. R037 (round-trip test) is the quality gate.

### Likely Omissions

- **No requirement for activation function matching.** NeuroBE uses ReLU; NCE uses Tanh. This will affect result comparison quality. Should be included in R035 (neurobe_mode preset).

- **No requirement for AMP disable.** NeuroBE doesn't use mixed precision. NCE's Trainer enables AMP by default. This introduces numerical differences. Should be part of R035.

- **No requirement for hidden dimension sizing match.** The `nbe,3` mode produces slightly different sizes than NeuroBE's `nArgs * var_dim`. Should be part of R035 or documented as known difference.

### Not Missing (Addressed by Existing Code)

- Variable ordering matching — verified identical for all 15 problems.
- One-hot encoding for binary domains with `lower_dim=True` — matches NeuroBE's raw integer input.
- Log base conversion — DataPreprocessor already handles log10 ↔ natural log.

## Slice Ordering Signal

**S01 (normalization + early stopping + config + tests) should come first** because:
1. Round-trip correctness is the prerequisite for all experiments
2. The config changes gate everything else
3. Tests verify correctness before running expensive GPU experiments
4. Risk is implementation correctness, not discovery

**S02 (ecl tuning + experiments + comparison table) should come second** because:
1. Depends on neurobe_mode being implemented and tested
2. Involves GPU-intensive experiments (hours of runtime)
3. ecl values are deterministic (computed from NeuroBE results CSV)
4. Results may reveal additional differences requiring S01 fixes

## Sources

- NeuroBE `Function-NN.hxx` Train() function — normalization, IS weights, early stopping logic
- NeuroBE `MiniBucket-NN.cpp` — sample generation, train/val/test split, NN dispatch threshold
- NeuroBE `Net.h` — network architecture (ReLU, `nArgs * var_dim` hidden dim)
- NeuroBE `HYPERPARAMETERS.md` — parameter reference including lr bug, stop_iter=2
- NeuroBE `binary_domain_results.csv` — ground truth results for 15 working problems
- NCE `data_preprocessor.py` — current normalization pipeline
- NCE `train.py` — current early stopping modes
- NCE `config_schema.py` — config validation and translation
- NCE `small_problems.py` — existing benchmark set with auto_ecl values
- NCE `losses.py` — existing weighted_logspace_mse implementation
- Verified: `.vo` and `.ord.elim` ordering files produce identical elimination orders for all tested problems
