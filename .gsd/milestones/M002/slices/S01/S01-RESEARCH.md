# S01: Inference & Training Test Suite — Research

**Date:** 2026-03-12

## Summary

The existing test infrastructure (110 tests, ~33s, all in `tests/`) is solid and easy to extend. Hand-built `FastFactor` problems work for exact inference tests — confirmed working with both binary and ternary domains, producing correct log10(Z) values within 1e-5 tolerance. NN training on hand-built problems works on CPU with a complete 40+ field config, producing clear loss decrease over 50 epochs (~5.46 → ~2.64 on the validated test case).

The main complexity is config completeness: `Trainer.__init__` does bare `self.config['lower_dim']`, `self.config['debug']`, etc. — no `.get()` fallbacks. `prepare_config()` only returns fields you provide (no defaults injection). The `nn_training_config` fixture must include all ~40 fields. The `reference_flat_config` fixture in `conftest.py` is the template, but needs modifications: `device='cpu'`, `num_epochs=50`, `ecl` and `iB` low enough to trigger NN training, `sampling_scheme='all'`, and `hidden_sizes=[8, 8]` instead of `'nbe,3'` (which is a string format resolved at runtime by bucket code).

Loss function edge-case behavior is confirmed: all-neg-inf targets produce inf (`logspace_mse_fdb`) or NaN (all others). No unhandled exceptions — these are "known behavior" tests. All-zero targets in log-space are handled without crash by all tested loss functions.

## Targeted Requirements

| Req | Description | Key Finding |
|-----|-------------|-------------|
| R018 | Exact inference correctness | Hand-built factors with `FastGM(factors=..., elim_order=..., ...)` → `eliminate_variables(all=True)` works. Binary chain Z=2.0, ternary Z=4.5 both confirmed. |
| R019 | Single bucket training | Works via `Net(bucket)` → `Trainer(net, bucket)` → `trainer.train()`. Requires complete config. `trainer.losses` is list of `(epoch, loss_value)` tuples. |
| R020 | Domain ≥3 handling | Ternary factors (3×3 tensors) confirmed working through exact inference. For NN training, need ternary problem that produces a wide enough bucket to trigger NN path. |
| R021 | Convergence test | 50 epochs on star-graph problem: loss 5.46→2.64 (CPU, seed 42). Clear decrease. |
| R022 | No-infinity output | `logspace_mse_fdb(inf_outputs, targets)` returns inf. `weighted_logspace_mse` returns NaN. Tests should verify no *unhandled exception*, document finite/inf/nan behavior. |
| R023 | All-neg-inf target handling | All loss functions produce inf or NaN — no crashes. This is expected for zero-probability messages. |
| R024 | Extensible pattern | File-per-concern (test_inference.py, test_nn_training.py, test_robustness.py) with shared conftest fixtures. Add PATTERN.md. |

## Recommendation

Build 4 tasks:

1. **T01: Fixtures** (~30 min) — Extend `conftest.py` with `nn_training_config` (complete 40-field config for CPU NN training), hand-built factor problem fixtures (binary chain, star graph, ternary chain), and helper to build `FastGM` from factors. Add GPU skip markers.

2. **T02: Exact inference + domain tests** (~30 min) — `test_inference.py` with R018 (binary chain exact Z, star graph exact Z) and R020 (ternary chain exact Z). 3-4 parametrized test cases.

3. **T03: NN training + convergence** (~45 min) — `test_nn_training.py` with R019 (single bucket trains without error, losses recorded) and R021 (loss decreases over 50 epochs). Uses star graph fixture (produces wide enough bucket for NN).

4. **T04: Robustness + pattern** (~30 min) — `test_robustness.py` with R022 (loss functions with inf inputs don't crash), R023 (all-neg-inf targets don't crash), and R024 (PATTERN.md documenting how to add regression tests).

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Complete NN config | `reference_flat_config` fixture in `conftest.py` | Has all 42 fields; modify for CPU + small epochs |
| GPU skip markers | `pytest.mark.skipif(not torch.cuda.is_available(), ...)` | Already used in `test_regression.py` |
| Factor construction | `FastFactor(torch.log10(probs), labels)` | Standard pattern throughout codebase |
| FastGM from factors | `FastGM(factors=[...], elim_order=[...], nn_config=config, device='cpu')` | Confirmed working; must provide explicit `elim_order` |
| Loss function testing | Direct function calls: `loss_fn(outputs, targets)` | No Trainer needed for edge-case tests |

## Existing Code and Patterns

- `tests/conftest.py` — Has `reference_flat_config` (42 fields), `equivalent_nested_config`, `minimal_flat_config`. Extend with `nn_training_config` fixture. **Pattern: fixtures return dicts, tests import them.**
- `tests/test_regression.py` — Uses `FastGM(model=..., nn_config=..., device=...)` → `eliminate_variables(all=True)` → `gm.log_partition_function`. **Reuse this pattern for R018.**
- `nce/inference/factor.py` — `FastFactor(tensor, labels)` where tensor is in log10 space. `eliminate('all')` returns scalar. **All test problems must use `torch.log10(probs)` for tensor values.**
- `nce/inference/graphical_model.py` — `FastGM(factors=..., elim_order=...)` builds from factor list. Calls `_load_vars_from_factors()` to infer variable domains from tensor shapes. **Must provide explicit `elim_order` (list of ints) — `wtminfill_order` crashes with `None` `variables_not_eliminated`.**
- `nce/neural_networks/train.py` — `Trainer.__init__` accesses config with bare `[]` (not `.get()`): `self.config['lower_dim']`, `self.config['debug']`, `self.config['optimizer']`, `self.config['seed']`, `self.config['loss_fn']`. **Config MUST include all these fields.**
- `nce/neural_networks/train.py` — `trainer.losses` is a list of `(epoch, loss_value)` tuples, not bare floats. **Extract loss values with `trainer.losses[i][1]`.**
- `nce/neural_networks/losses.py` — Loss functions take `(outputs, targets, bw_hat=None)`. Can be called standalone without Trainer. **Use for R022/R023 edge-case tests.**
- `nce/inference/bucket.py` — `get_large_message_buckets()` returns bucket **keys** (ints), not bucket objects. Use `gm.matching_var(key)` to get `Var`, then `gm.buckets[var]` to get the bucket. **Important for R019 test setup.**

## Constraints

- **Config completeness** — `Trainer.__init__` uses bare dict access for ~15+ fields. `prepare_config()` does NOT inject defaults for fields not provided. The `nn_training_config` fixture must include every field that Trainer, DataLoader, SampleGenerator, and DataPreprocessor access. The safest approach: copy `reference_flat_config` and modify device/epochs/ecl/iB/hidden_sizes.
- **`elim_order` must be explicit** — `wtminfill_order(factors, variables_not_eliminated=None)` crashes at line 72 with `TypeError` (tries to iterate `None`). Always provide `elim_order` as a list of ints to `FastGM(factors=..., elim_order=[...])`.
- **Log10 space** — All factor tensors are `log10(probability)`. `FastFactor.__mul__` performs addition in log-space. Partition function comparison: compute Z manually in linear space, then `math.log10(Z)` for comparison against `gm.log_partition_function`.
- **`hidden_sizes` string format** — `reference_flat_config` uses `'nbe,3'` which is resolved at runtime by `bucket.compute_message_nn()`. For direct `Net(bucket, hidden_sizes=...)` construction in tests, must pass an explicit list like `[8, 8]`.
- **`sampling_scheme='all'` for small problems** — For problems with message_size ≤ 256, use `'all'` to enumerate all assignments. Avoids sample generation overhead and ensures deterministic coverage.
- **`tqdm.notebook` import** — `train.py` imports `from tqdm.notebook import tqdm`. In pytest (non-notebook), this falls back to text tqdm but may produce formatting warnings. Not a blocker.
- **AMP scaler on CPU** — `Trainer.train()` creates `torch.cuda.amp.GradScaler(enabled=True)` unconditionally. On CPU this is a no-op (scaler disables itself) but may produce deprecation warnings. Not a blocker.
- **`num_samples` as string** — `reference_flat_config` uses `'nbe,0.1'` for `num_samples`. For direct training tests, use an integer value instead.
- **`dope_factors`** — When `True`, modifies factor tensors in-place (clamps small values). Set to `False` in test config to avoid modifying hand-built factors.

## Common Pitfalls

- **Losses are tuples** — `trainer.losses` contains `(epoch, loss_value)` tuples. Access the loss value with `trainer.losses[i][1]`, not `trainer.losses[i]`. Formatting with `:.6f` on a tuple causes `TypeError`.
- **Missing config fields crash Trainer** — `KeyError: 'lower_dim'` if config doesn't include it. Build `nn_training_config` from `reference_flat_config` with overrides, not from scratch.
- **`get_large_message_buckets` returns ints** — Returns bucket keys (int labels), not `FastBucket` objects. To get the actual bucket: `var = gm.matching_var(key); bucket = gm.buckets[var]`.
- **Star graph needed for NN training** — Simple chains produce only 1-variable messages (message_size=2 for binary). Need a star/triangle topology where eliminating a hub variable produces a multi-variable message exceeding `ecl`. The validated star graph: `f012(X0,X1,X2)` + `f03(X0,X3)` with `elim_order=[0,1,2,3]`, bucket 0 has message scope [1,2,3] with size 8.
- **`dope_factors=True` modifies tensors** — If `True` in config, FastGM.__init__ calls `self.dope_factors()` which clamps small values in all factor tensors. For tests with known exact Z values, set `dope_factors=False`.

## Open Risks

- **tqdm formatting on CPU** — `train.py` uses `from tqdm.notebook import tqdm` which may behave differently in pytest context. The `TypeError: unsupported format string passed to tuple.__format__` we saw during prototyping appears to come from tqdm rendering, not from actual training failure. Training completes successfully despite this. Tests should capture stdout/stderr or suppress tqdm output.
- **Domain≥3 NN training** — Confirmed exact inference works with ternary factors. NN training with ternary variables needs validation: the one-hot encoding in `_get_nn_input_size()` is `nstates` per variable (or `nstates-1` if `lower_dim=True`). Constructing a ternary star graph that triggers NN training path needs a factor with 3+ ternary variables (to exceed ecl). May need `ecl=8` or lower.
- **Convergence tolerance** — The 5.46→2.64 convergence ratio (51% decrease) was with seed=42 on one specific problem. Other problems/seeds may show different convergence magnitudes. Use tolerance-based assertions: `final_loss < 0.9 * initial_loss` rather than hard thresholds.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| pytest | none needed | Standard pytest knowledge sufficient; no specialized skill required |
| PyTorch | none found | No testing-specific skill available |

## Sources

- Direct codebase exploration of `nce/inference/`, `nce/neural_networks/`, `nce/data/`, `nce/sampling/`
- Live execution: hand-built factor problems (binary chain Z=2.0, ternary Z=4.5), NN training on star graph (50 epochs CPU), loss function edge cases
- Existing test suite: 110 tests in `tests/`, `conftest.py` fixtures, `test_regression.py` patterns
