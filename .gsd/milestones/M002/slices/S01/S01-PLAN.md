# S01: Inference & Training Test Suite

**Goal:** `pytest tests/` runs all new tests — exact inference correctness on hand-built problems, single-bucket NN training, domain≥3 handling, convergence over 50 epochs, infinity/edge-case robustness — alongside the 110 existing config tests, all green.
**Demo:** `pytest tests/ -v` shows 110+ existing tests + ~15 new tests, all passing, total runtime under 120s.

## Must-Haves

- Hand-built factor problem fixtures with analytically known partition function values (binary chain Z=2.0, ternary chain Z=4.5, star graph)
- Complete `nn_training_config` fixture derived from `reference_flat_config` with CPU/small-epoch overrides
- Exact inference correctness tests verifying log10(Z) within 1e-5 tolerance (R018)
- Domain≥3 exact inference test with ternary variables (R020)
- Single-bucket NN training functional test — trains without error, losses recorded (R019)
- Convergence test — loss decreases over 50 epochs with `final_loss < 0.9 * initial_loss` (R021)
- Loss function edge-case tests — inf inputs don't crash (R022), all-neg-inf targets don't crash (R023)
- PATTERN.md documenting how to add failure-mode regression tests (R024)
- All existing 110 tests still pass
- Total `pytest tests/` runtime under 120 seconds

## Proof Level

- This slice proves: integration
- Real runtime required: yes (real FastGM inference, real Trainer training on CPU)
- Human/UAT required: no

## Verification

- `pytest tests/ -v` — all tests pass (existing 110 + new ~15)
- `pytest tests/test_inference.py -v` — exact inference correctness (R018) and domain≥3 (R020)
- `pytest tests/test_nn_training.py -v` — single-bucket training (R019) and convergence (R021)
- `pytest tests/test_robustness.py -v` — inf outputs (R022), all-neg-inf targets (R023)
- `test -f tests/PATTERN.md` — extensibility documentation exists (R024)
- `pytest tests/ --co -q | tail -1` — total test count ≥ 125
- `timeout 120 pytest tests/` — completes within 120s

## Observability / Diagnostics

- Runtime signals: pytest output with `-v` shows per-test pass/fail; loss values printed in convergence test failure messages
- Inspection surfaces: `pytest tests/test_nn_training.py -v -s` shows training loss trajectory when debugging convergence
- Failure visibility: assertion messages include computed vs expected values (e.g., `f"log10(Z) = {actual}, expected {expected}"`)
- Redaction constraints: none (no secrets in test suite)

## Integration Closure

- Upstream surfaces consumed: `nce/inference/factor.py` (FastFactor), `nce/inference/graphical_model.py` (FastGM), `nce/neural_networks/train.py` (Trainer), `nce/neural_networks/net.py` (Net), `nce/neural_networks/losses.py` (loss functions), `nce/config_schema.py` (prepare_config), `tests/conftest.py` (existing fixtures)
- New wiring introduced in this slice: test files that import and exercise real inference/training code paths; no production code changes
- What remains before the milestone is truly usable end-to-end: nothing — this is the only slice in M002

## Tasks

- [x] **T01: Build shared fixtures and hand-built test problems** `est:30m`
  - Why: All subsequent tests depend on a complete NN training config fixture and hand-built factor problems with known partition functions. Fixtures first, tests after.
  - Files: `tests/conftest.py`
  - Do: Add `nn_training_config` fixture (copy `reference_flat_config`, override: `device='cpu'`, `num_epochs=50`, `ecl=4`, `iB=2`, `hidden_sizes=[8,8]`, `sampling_scheme='all'`, `num_samples=256`, `dope_factors=False`, `set_size=None`). Add `binary_chain_factors` fixture (X0–X1 chain, Z=2.0). Add `ternary_chain_factors` fixture (X0–X1 with domain 3, Z=4.5). Add `star_graph_factors` fixture (X0 hub connected to X1,X2,X3 — bucket 0 has message scope large enough to trigger NN with ecl=4). All factor tensors in log10 space. Include analytically computed log10(Z) as part of each fixture's return value.
  - Verify: `pytest tests/ --co -q` collects all existing tests without error (fixtures importable)
  - Done when: All 4 new fixtures exist in conftest.py, existing 110 tests still collected

- [x] **T02: Exact inference correctness and domain≥3 tests** `est:30m`
  - Why: R018 (exact inference correctness) and R020 (domain≥3) are the foundation — if exact inference is wrong, everything else is meaningless.
  - Files: `tests/test_inference.py`
  - Do: Create `test_inference.py` with: (1) `test_binary_chain_exact_z` — build FastGM from binary chain factors, eliminate all, assert `log_partition_function ≈ log10(2.0)` within 1e-5. (2) `test_ternary_chain_exact_z` — same with ternary chain, assert `log_partition_function ≈ log10(4.5)` within 1e-5. (3) `test_star_graph_exact_z` — star graph with high ecl (exact path), verify Z. Use `FastGM(factors=..., elim_order=..., nn_config=config, device='cpu')` with `ecl=2**30` to force exact computation. Set `dope_factors=False` in config.
  - Verify: `pytest tests/test_inference.py -v` — all 3 tests pass
  - Done when: R018 has 2+ passing tests (binary chain, star graph), R020 has 1+ passing test (ternary chain)

- [x] **T03: Single-bucket NN training and convergence tests** `est:45m`
  - Why: R019 (training runs without error) and R021 (loss actually decreases) are the core NN validation. Uses the star graph which produces a multi-variable message exceeding ecl=4.
  - Files: `tests/test_nn_training.py`
  - Do: Create `test_nn_training.py` with: (1) `test_single_bucket_trains_without_error` — build FastGM from star graph with `nn_training_config`, call `eliminate_variables(all=True)`, assert no exception, assert `gm.per_bucket_training_log` is non-empty. (2) `test_convergence_loss_decreases` — same setup, extract loss curves from training log, assert final loss < 0.9 × initial loss for at least one NN-trained bucket. (3) `test_losses_are_tuples` — verify `trainer.losses` entries are `(epoch, loss_value)` tuples (guard against format regression). Use `nn_training_config` fixture. Seed=42 for reproducibility.
  - Verify: `pytest tests/test_nn_training.py -v` — all tests pass
  - Done when: R019 has 1+ passing test (training completes), R021 has 1+ passing test (loss decrease verified)

- [x] **T04: Robustness edge-case tests and extensibility pattern** `est:30m`
  - Why: R022 (inf handling), R023 (all-neg-inf targets), R024 (extensible pattern). These are standalone loss function tests — no Trainer needed.
  - Files: `tests/test_robustness.py`, `tests/PATTERN.md`
  - Do: Create `test_robustness.py` with: (1) `test_loss_fn_inf_input_no_crash` — parametrize over `[logspace_mse_fdb, linspace_mse_fdb, weighted_logspace_mse]`, pass `outputs=tensor([inf, 0, -1])` and valid targets, assert no unhandled exception (R022). (2) `test_loss_fn_all_neg_inf_targets_no_crash` — same parametrization, pass `targets=tensor([-inf, -inf, -inf])` with valid outputs, assert no unhandled exception (R023). (3) `test_loss_fn_all_zero_logspace_targets_no_crash` — pass `targets=tensor([0, 0, 0])`, assert no crash. Create `tests/PATTERN.md` documenting: file-per-concern structure, how to add a new failure-mode test (copy template, parametrize, assert no crash or assert specific behavior), fixture usage patterns (R024).
  - Verify: `pytest tests/test_robustness.py -v` — all tests pass; `test -f tests/PATTERN.md`
  - Done when: R022 has 1+ passing test, R023 has 1+ passing test, PATTERN.md exists with clear instructions

## Files Likely Touched

- `tests/conftest.py` (extended with new fixtures)
- `tests/test_inference.py` (new)
- `tests/test_nn_training.py` (new)
- `tests/test_robustness.py` (new)
- `tests/PATTERN.md` (new)
