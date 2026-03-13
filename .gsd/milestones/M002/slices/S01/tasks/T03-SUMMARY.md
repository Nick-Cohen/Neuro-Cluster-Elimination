---
id: T03
parent: S01
milestone: M002
provides:
  - test_nn_training.py with 3 passing tests covering R019 (NN training completes, log format) and R021 (convergence)
key_files:
  - tests/test_nn_training.py
key_decisions:
  - Training log 'label' field is int (bucket index), not str — type check accepts both to match actual FastBucket.label type
patterns_established:
  - _run_star_graph_nn() helper encapsulates build+eliminate pattern for NN training tests, returns FastGM for inspection
  - Convergence test compares first-5 vs last-5 epoch average losses with diagnostic output on failure
observability_surfaces:
  - Convergence test failure messages show per-bucket loss trajectory (initial_avg, final_avg, ratio)
  - Run with `pytest tests/test_nn_training.py -v -s` to see training progress output
duration: 10m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T03: Single-bucket NN training and convergence tests

**Added 3 NN training tests verifying star graph NN path completes, training log format is stable, and loss converges over 50 epochs.**

## What Happened

Created `tests/test_nn_training.py` with two test classes:

- `TestNNTrainingCompletes`: two tests — `test_nn_training_completes` verifies NN training on the star graph (ecl=4, message_size=8 triggers NN path) runs without exception and produces non-empty `per_bucket_training_log` with loss data; `test_training_log_format` checks the structure of log entries (required keys: label, epochs_trained, hidden_sizes, losses, val_losses) and validates field types and loss tuple format.

- `TestConvergence`: one test — `test_convergence_loss_decreases` compares average loss over first 5 epochs vs last 5 epochs, asserts `final_avg < 0.9 * initial_avg` for at least one NN-trained bucket. Includes per-bucket diagnostic output on failure.

Hit one minor surprise: `FastBucket.label` is an int (bucket index), not a string. Updated the format test to accept `(str, int)`.

## Verification

- `pytest tests/test_nn_training.py -v` — 3/3 passed (8.99s)
- `pytest tests/ --tb=short` — 116/116 passed, no regressions (27.55s)
- Runtime well under 60s for the NN training tests alone

Slice-level checks (intermediate — T04 still pending):
- ✅ `pytest tests/test_nn_training.py -v` — R019 and R021 covered
- ✅ `pytest tests/ -v` — all 116 tests pass
- ⬜ `pytest tests/test_robustness.py -v` — not yet created (T04)
- ⬜ `test -f tests/PATTERN.md` — not yet created (T04)
- ⬜ `pytest tests/ --co -q | tail -1` — 116 tests, need ≥125 (T04 will add more)
- ✅ `timeout 120 pytest tests/` — completes in ~28s

## Diagnostics

- Run `pytest tests/test_nn_training.py -v -s` to see training output (bucket label, normalizing constant, epoch progress)
- Convergence failure messages show: per-bucket initial_avg, final_avg, ratio, and YES/NO verdict
- Training log format test enumerates missing keys and present keys on failure

## Deviations

- Plan said to check for key `loss_curve` — actual key in `per_bucket_training_log` is `losses` (list of `(epoch, loss_value)` tuples). Used real key names from bucket.py source.
- Plan said label is a string — it's actually an int (bucket index). Adjusted type assertion accordingly.

## Known Issues

None.

## Files Created/Modified

- `tests/test_nn_training.py` — 3 tests covering R019 (NN training completes, log format) and R021 (convergence)
