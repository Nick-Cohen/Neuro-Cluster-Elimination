# M003: NeuroBE Reproduction Mode — Context

**Gathered:** 2026-03-12
**Status:** Queued — pending auto-mode execution

## Project Description

NCE is a Python package for neural network-based approximate inference on probabilistic graphical models. This milestone adds a durable `neurobe_mode` that faithfully reproduces the NeuroBE algorithm's training pipeline within the NCE codebase, enabling direct comparison of results between the two implementations on the same problems.

## Why This Milestone

The NeuroBE C++ codebase has been run on 16 binary-domain problems (results in `Clean-NeuroBE/results/binary_min_nn/binary_domain_results.csv`). NCE has also been run on these problems but with different algorithm details — different normalization, different early stopping, different loss function, and different NN dispatch criteria. The result discrepancies make it impossible to know whether differences come from algorithm choices or implementation bugs. A faithful reproduction mode isolates the comparison: if NeuroBE mode in NCE produces matching results, the codebase is correct; if not, the divergence points to specific implementation differences that need investigation.

Current NCE vs NeuroBE differences:
- **NN dispatch:** NCE uses `message_size > ecl`; NeuroBE uses `bucket_width > width_problem`. These produce different NN counts (9/15 problems mismatch).
- **Target normalization:** NCE subtracts a log-space mean (normalizing constant); NeuroBE normalizes targets to [0,1] via min-max: `(value - ln_min) / (ln_max - ln_min)`.
- **Denormalization:** NCE adds back normalizing constant and converts log base; NeuroBE uses `ln_min + nn_out * (ln_max - ln_min)`.
- **Early stopping:** NCE uses 3-consecutive-increases or convex early stopping; NeuroBE uses patience=2 (stop after 2 consecutive non-improving epochs on validation loss).
- **Loss function:** The comparison should use weighted MSE (importance-sampling weighted), matching NeuroBE's `s_method="is"`.
- **Batch size:** NCE defaults vary; NeuroBE uses 256.
- **Backward messages:** NCE supports bw_ecl; NeuroBE reproduction should use no backward info.

## User-Visible Outcome

### When this milestone is complete, the user can:

- Set `neurobe_mode: true` in a config and get NeuroBE-faithful training behavior (min-max [0,1] normalization, patience-2 early stopping, batch_size=256, weighted MSE, no bw)
- Run the 15 working binary-domain problems through NCE with matched ecl values that produce the same NN counts as NeuroBE
- Compare NCE neurobe_mode results directly against NeuroBE results in a combined results table
- Trust that any remaining result differences are due to numerical precision / RNG, not algorithm differences

### Entry point / environment

- Entry point: Python API (`FastGM` with neurobe_mode config), experiment scripts
- Environment: local dev with CUDA GPUs (4× NVIDIA TITAN RTX)
- Live dependencies involved: none

## Completion Class

- Contract complete means: tests verify min-max normalization round-trip (normalize → train → denormalize produces correct values), neurobe early stopping triggers at the right time, ecl values produce matching NN counts
- Integration complete means: full inference runs on 15 problems produce results comparable to NeuroBE
- Operational complete means: none

## Final Integrated Acceptance

To call this milestone complete, we must prove:

- NN counts match NeuroBE for all 15 working problems (same buckets get NN-trained)
- Min-max [0,1] normalization → training → denormalization round-trip is verified by tests
- NeuroBE early stopping (patience=2 on validation loss) is implemented and tested
- Results for 15 problems are collected in a combined comparison table alongside NeuroBE results
- `pytest tests/` still passes (all existing + new tests green)

## Risks and Unknowns

- **ECL matching:** Finding ecl values that produce the same NN counts as NeuroBE's width-based dispatch may require trial-and-error per problem. The relationship between ecl (message table size) and bucket width is problem-dependent.
- **Numerical divergence:** Even with matched algorithms, PyTorch (Python) vs libtorch (C++) may produce different floating-point results due to different RNG, different CUDA kernels, or different operation ordering. Results should be "comparable" not "identical."
- **Variable ordering:** NCE and NeuroBE may use different variable orderings for the same problem. If orderings differ, bucket structure differs, making NN count matching impossible. Need to verify orderings match.
- **Sample generation:** NeuroBE generates samples via a pseudo-dimension formula; NCE uses `sampling_scheme='all'` or other schemes. Sample distribution differences could cause result divergence even with matched training.

## Existing Codebase / Prior Art

- `nce/data/data_preprocessor.py` — Current DataPreprocessor with log-space mean normalization. Must be extended or wrapped for min-max [0,1] mode.
- `nce/neural_networks/train.py` — Trainer class with existing nbe_early_stopping (3-consecutive-increases). Must add NeuroBE's patience-2 early stopping.
- `nce/neural_networks/losses.py` — Loss functions including `weighted_logspace_mse`. Verify this matches NeuroBE's IS-weighted MSE.
- `nce/inference/graphical_model.py` — FastGM with ecl-based NN dispatch. ecl values must be tuned per problem.
- `nce/benchmark_problems/small_problems.py` — BenchmarkSet with 24 problems and auto_ecl values. Need new config builder for neurobe_mode.
- `Clean-NeuroBE/NeuroBE/BE-sampling-project/ARP/Problem/Function-NN.hxx` — NeuroBE's training code. Lines 157-310: `Train()` function with min-max normalization, IS-weighted MSE, patience-2 early stopping. Lines 910-920: `samples_to_data()` with `(value - ln_min) / (ln_max - ln_min)` normalization. Lines 112-114: denormalization `ln_min + nn_out * (ln_max - ln_min)`.
- `Clean-NeuroBE/results/binary_min_nn/binary_domain_results.csv` — NeuroBE results for 16 binary-domain problems (15 working, or_chain_10.fg failed).
- `Clean-NeuroBE/docs/HYPERPARAMETERS.md` — Detailed hyperparameter reference including early stopping (stop_iter=2), batch_size=256, lr=0.001 (hardcoded due to bug).

> See `.gsd/DECISIONS.md` for all architectural and pattern decisions — it is an append-only register; read it during planning, append to it during execution.

## Relevant Requirements

This milestone introduces new requirements:

- R033 — NeuroBE min-max [0,1] target normalization mode with correct denormalization
- R034 — NeuroBE patience-2 early stopping on validation loss
- R035 — NeuroBE reproduction config preset (batch_size=256, weighted MSE, no bw, lr=0.001)
- R036 — Matched NN counts: ecl values per problem that produce same num_trained as NeuroBE
- R037 — Normalization round-trip test: normalize → train → denormalize produces correct values
- R038 — Combined comparison results table for 15 binary-domain problems

## Scope

### In Scope

- Min-max [0,1] target normalization as a config option (neurobe_mode or explicit flag)
- Corresponding denormalization at inference time
- NeuroBE patience-2 early stopping (stop after 2 consecutive non-improving validation epochs)
- neurobe_mode config preset: batch_size=256, weighted MSE loss, no backward messages, lr=0.001
- Finding ecl values per problem that match NeuroBE's NN counts
- Tests verifying normalization round-trip correctness
- Running 15 binary-domain problems and collecting results
- Combined comparison table (NCE neurobe_mode vs NeuroBE C++ results)

### Out of Scope / Non-Goals

- Implementing NeuroBE's `masked_net` architecture (only `net` path for binary-domain problems)
- Matching NeuroBE's pseudo-dimension sample count formula
- Implementing width-based NN dispatch (will find matching ecl values instead)
- Matching NeuroBE's input normalization for masked_net ([-1,1] scaling)
- Reproducing NeuroBE's exact RNG sequence

## Technical Constraints

- Python 3.11, PyTorch 2.0.1+cu117
- GPUs 0, 1, 2, 3 available
- NeuroBE results are the ground truth for comparison
- All problems use binary domains (domain size = 2)
- NeuroBE used `iB=25`, `width_problem = MaxWidth - 1`, `network=net`, `var_dim=3`, `epsilon=0.1`

## Integration Points

- `DataPreprocessor` — Must support min-max [0,1] normalization mode alongside existing log-space mean normalization
- `Trainer` — Must support NeuroBE patience-2 early stopping alongside existing early stopping modes
- `FastGM` — ecl-based dispatch must produce same bucket selection as NeuroBE's width-based dispatch for each problem
- `config_schema.py` — New neurobe_mode fields must be added to the schema and documented
- `tests/` — New tests for normalization round-trip and early stopping behavior

## Open Questions

- **Variable ordering match:** Do NCE and NeuroBE use the same variable orderings for these problems? If not, bucket structures will differ and NN count matching is impossible. Need to verify early.
- **Weighted MSE implementation:** Does NCE's existing `weighted_logspace_mse` match NeuroBE's IS-weighted MSE: `(w * (output - labels)^2).mean()` where `w = (labels * (ln_max - ln_min)) / sum_ln`? The weight formula may differ.
- **Validation set construction:** NeuroBE splits samples into train/val/test (80/20 + 50k test). NCE may construct validation sets differently. This could affect early stopping behavior.
- **Learning rate:** NeuroBE has lr=0.001 hardcoded (CLI bug). NCE's small_problems configs use lr=0.01. The neurobe_mode must use 0.001.
