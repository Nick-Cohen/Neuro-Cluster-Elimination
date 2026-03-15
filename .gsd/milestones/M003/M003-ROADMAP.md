# M003: NeuroBE Reproduction Mode

**Vision:** A single `neurobe_mode: true` config flag switches NCE into NeuroBE-faithful training behavior, enabling direct comparison of inference results on 15 binary-domain problems.

## Success Criteria

- Setting `neurobe_mode: true` in a config produces NeuroBE-faithful training: min-max [0,1] normalization, patience-based early stopping, batch_size=256, lr=0.001, ReLU activation, no AMP, weighted MSE, no backward messages
- NN counts match NeuroBE for all 15 working binary-domain problems (same buckets get NN-trained)
- Min-max normalization round-trip is verified by automated tests: normalize → train → denormalize produces correct values
- Combined comparison table shows NCE neurobe_mode results alongside NeuroBE C++ results for all 15 problems
- `pytest tests/` passes with all existing + new tests green

## Key Risks / Unknowns

- **Normalization round-trip correctness:** Min-max [0,1] normalization + denormalization must produce values in the correct log10 scale. The log-base conversion (log10 ↔ ln) during normalize/denormalize is easy to get wrong silently — outputs look plausible but are off by a constant factor.
- **Early stopping semantics ambiguity:** NeuroBE's `count > stop_iter` with `stop_iter=2` could mean 2 or 3 non-improving epochs depending on counter initialization. Wrong interpretation affects training duration and results.
- **Weighted MSE coupling to normalization:** The IS weights in NeuroBE depend on the [0,1]-normalized targets (`w = labels * (ln_max - ln_min) / sum_ln`). The loss function and normalization are coupled — the existing `weighted_logspace_mse` in NCE operates on un-normalized logspace targets, so a new loss function path is needed.
- **ecl off-by-one:** NCE dispatches on `message_size > ecl` (strictly greater). For binary domains, the correct ecl is `2^width_problem - 1` to match NeuroBE's `_Width > width_problem`. Off-by-one here silently produces different NN counts.

## Proof Strategy

- **Normalization round-trip** → retire in S01 by building the real min-max normalization mode in DataPreprocessor, running a hand-built problem through NN training with it, and verifying denormalized outputs match expected values in automated tests
- **Early stopping semantics** → retire in S01 by implementing patience-based stopping matched to NeuroBE's counter logic, with a test that verifies it triggers at the correct epoch
- **ecl matching** → retire in S02 by computing ecl values from the NeuroBE results CSV and verifying NN counts match for all 15 problems before running full experiments

## Verification Classes

- Contract verification: `pytest tests/` — normalization round-trip, early stopping trigger, config expansion, plus all existing M001/M002 tests
- Integration verification: Full inference runs on 15 problems with neurobe_mode producing a comparison table against NeuroBE C++ results
- Operational verification: none
- UAT / human verification: Comparison table reviewed for result plausibility (log_Z values within reasonable tolerance of NeuroBE)

## Milestone Definition of Done

This milestone is complete only when all are true:

- Min-max [0,1] normalization and denormalization work correctly in DataPreprocessor (tested)
- Patience-based early stopping halts training at the correct epoch (tested)
- `neurobe_mode: true` expands to the full preset of NeuroBE-faithful settings in prepare_config
- ecl values per problem produce matching NN counts for all 15 working binary-domain problems
- 15 problems have been run through NCE neurobe_mode and results collected
- Combined comparison table exists with log_Z, error, NNs, and runtime for both codebases
- `pytest tests/` passes with all existing + new tests green
- Success criteria above are re-checked against actual run outputs

## Requirement Coverage

- Covers: R033, R034, R035, R036, R037, R038
- Partially covers: none
- Leaves for later: none
- Orphan risks: none — all 6 active M003 requirements are mapped

## Slices

- [ ] **S01: NeuroBE Training Mode** `risk:high` `depends:[]`
  > After this: A hand-built test problem trains with min-max [0,1] normalization, patience-based early stopping, ReLU activation, and neurobe_mode config preset — verified by `pytest tests/` including normalization round-trip and early stopping tests.
- [ ] **S02: ECL Tuning & Comparison Experiments** `risk:medium` `depends:[S01]`
  > After this: All 15 binary-domain problems have been run through NCE neurobe_mode with matched NN counts, and a combined comparison table shows results alongside NeuroBE C++ ground truth.

## Boundary Map

### S01 → S02

Produces:
- `DataPreprocessor` with `normalization_mode='minmax_01'` — stores `ln_min`, `ln_max`; `normalize()` returns [0,1]-scaled targets; `undo_normalization()` maps NN output back to log10 space
- `Trainer` with patience-based early stopping mode — `neurobe_early_stopping: true` with configurable `neurobe_stop_iter` (default 2)
- `prepare_config()` expansion of `neurobe_mode: true` → flat config with batch_size=256, lr=0.001, loss_fn='neurobe_weighted_mse', use_bw_approx=False, normalization_mode='minmax_01', neurobe_early_stopping=True, neurobe_stop_iter=2, activation='relu', use_amp=False
- `Net` with configurable activation function (ReLU vs Tanh via `activation` config field)
- `neurobe_weighted_mse` loss function operating on [0,1]-normalized targets with IS weights
- Passing tests: normalization round-trip, early stopping trigger, config preset expansion, existing M001/M002 suite

Consumes:
- nothing (first slice)

### S02

Produces:
- Per-problem ecl values for neurobe_mode (15 values computed from NeuroBE results CSV)
- neurobe benchmark config builder in `small_problems.py` or separate module
- Experiment runner script for 15 problems
- Combined comparison results CSV/table with columns: Problem, NCE_log_Z, NeuroBE_log_Z, NCE_error, NeuroBE_error, NCE_NNs, NeuroBE_NNs, NCE_time, NeuroBE_time

Consumes:
- S01's neurobe_mode config preset and all training machinery
- NeuroBE results CSV (`Clean-NeuroBE/results/binary_min_nn/binary_domain_results.csv`)
