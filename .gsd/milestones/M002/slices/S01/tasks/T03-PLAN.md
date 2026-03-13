---
estimated_steps: 4
estimated_files: 1
---

# T03: Single-bucket NN training and convergence tests

**Slice:** S01 — Inference & Training Test Suite
**Milestone:** M002

## Description

Create `tests/test_nn_training.py` with tests that verify single-bucket NN training runs without error (R019) and that loss measurably decreases over 50 epochs (R021). Uses the star graph fixture which produces a multi-variable message exceeding ecl=4, triggering the NN training path.

## Steps

1. Create `tests/test_nn_training.py`. Import `FastGM`, `prepare_config`, torch, pytest, copy. Import fixtures from conftest.
2. Write `test_nn_training_completes(star_graph_factors, nn_training_config)`: build FastGM with `nn_training_config` (ecl=4, 50 epochs, CPU), call `eliminate_variables(all=True)`. Assert no exception raised. Assert `gm.per_bucket_training_log` is non-empty (at least one bucket trained via NN). Assert the training log entries contain loss curve data.
3. Write `test_convergence_loss_decreases(star_graph_factors, nn_training_config)`: same setup. Extract loss curves from `per_bucket_training_log`. For at least one NN-trained bucket, compare first 5 epochs' average loss to last 5 epochs' average loss. Assert `final_avg < 0.9 * initial_avg`. Use seed=42 for reproducibility. Include diagnostic message showing the loss trajectory if assertion fails.
4. Write `test_training_log_format(star_graph_factors, nn_training_config)`: verify the structure of `per_bucket_training_log` — entries are dicts with expected keys (`label`, `epochs`, `hidden_sizes`, `loss_curve`). This guards against format regressions that would break downstream visualization.

## Must-Haves

- [ ] NN training completes without exception on CPU star graph problem (R019)
- [ ] At least one bucket is trained via NN path (per_bucket_training_log non-empty) (R019)
- [ ] Loss decreases: final avg loss < 0.9 × initial avg loss for ≥1 bucket (R021)
- [ ] Tests use real Trainer training (not mocks), real FastGM inference
- [ ] Seed fixed at 42 for reproducibility

## Verification

- `source venv/bin/activate && python -m pytest tests/test_nn_training.py -v` — all tests pass
- `source venv/bin/activate && python -m pytest tests/ --tb=short` — all tests (existing + new) pass
- Runtime check: `time pytest tests/test_nn_training.py` — completes in under 60s

## Observability Impact

- Signals added/changed: Test failure messages include loss trajectory (first 5 and last 5 epoch losses)
- How a future agent inspects this: Run with `-v -s` to see training progress; failure messages show exact loss values
- Failure state exposed: If convergence fails, the assertion message shows initial_avg, final_avg, and the ratio

## Inputs

- `tests/conftest.py` — T01 fixtures: `star_graph_factors`, `nn_training_config`
- `nce/inference/graphical_model.py` — `FastGM` with NN training path
- `nce/neural_networks/train.py` — `Trainer` internals (losses format, training log)
- S01-RESEARCH.md — confirmed convergence: 5.46→2.64 over 50 epochs on star graph, seed 42

## Expected Output

- `tests/test_nn_training.py` — 3 passing tests covering R019 (2 tests) and R021 (1 test)
