# Requirements

This file is the explicit capability and coverage contract for the project.

## Validated

### R001 — Nested config sections
- Class: core-capability
- Status: validated
- Description: Config dict supports nested sections (inference, nn, training, sampling, backward, output) as input format
- Why it matters: 42-field flat dict is unreadable and error-prone; nested sections group related fields
- Source: user
- Primary owning slice: M001/S01
- Supporting slices: M001/S02
- Validation: M001
- Notes: Internal code continues using flat dict; translation happens at entry point

### R002 — Dead config field removal
- Class: core-capability
- Status: validated
- Description: All dead config fields from removed/unused code paths are absent and raise errors if present
- Why it matters: Dead fields confuse users and create false expectations about functionality
- Source: user
- Primary owning slice: M001/S01
- Supporting slices: none
- Validation: M001
- Notes: Fields like num_epochs2, loss_fn2, complexity_limit, exact, memorizer

### R003 — Config field name cleanup
- Class: core-capability
- Status: validated
- Description: Field names are consistent and self-describing (lr→learning_rate, ecl→exact_computation_limit, iB→i_bound, fdb→forward_diff_barrier)
- Why it matters: Cryptic abbreviations require looking up meanings; readable names are self-documenting
- Source: user
- Primary owning slice: M001/S01
- Supporting slices: M001/S02, M001/S03
- Validation: M001
- Notes: Old names still accepted via backward compat layer

### R004 — Config validation with section-specific errors
- Class: core-capability
- Status: validated
- Description: Config validation catches missing required fields and unexpected keys with messages that name the offending field and its section
- Why it matters: Generic "invalid config" errors waste debugging time
- Source: user
- Primary owning slice: M001/S01
- Supporting slices: none
- Validation: M001
- Notes: none

### R005 — Backward compat for flat configs
- Class: core-capability
- Status: validated
- Description: Old flat config dicts are auto-detected and accepted without modification
- Why it matters: Existing scripts, notebooks, and benchmark configs must not break
- Source: user
- Primary owning slice: M001/S01
- Supporting slices: none
- Validation: M001
- Notes: Detection logic lives in config_schema.py, not FastGM

### R006 — Translation logic separated from FastGM
- Class: quality-attribute
- Status: validated
- Description: All config detection, validation, and translation logic lives in config_schema.py, not in FastGM.__init__
- Why it matters: Clean separation — FastGM calls one function and gets back a flat dict
- Source: user
- Primary owning slice: M001/S01
- Supporting slices: none
- Validation: M001
- Notes: FastGM.__init__ imports and calls a single function from config_schema

### R007 — Config documentation guide
- Class: core-capability
- Status: validated
- Description: A documentation guide exists listing every config field with its type, default value, and purpose in plain language
- Why it matters: Config is the primary user interface for experiments; it must be documented
- Source: user
- Primary owning slice: M001/S03
- Supporting slices: none
- Validation: M001
- Notes: none

### R008 — Code-level doc-sync enforcement
- Class: quality-attribute
- Status: validated
- Description: Every field definition in the codebase has a comment linking to or reproducing its documentation entry; adding a new field without updating the guide is detectable
- Why it matters: Documentation drifts from code without enforcement
- Source: user
- Primary owning slice: M001/S03
- Supporting slices: none
- Validation: M001
- Notes: none

### R009 — FastGM picklable with training metadata
- Class: core-capability
- Status: validated
- Description: FastGM object can be pickled and unpickled with all per-bucket training logs (loss curves, epochs trained, hidden sizes) intact
- Why it matters: Training state is currently lost after inference; researchers need to inspect results after the fact
- Source: user
- Primary owning slice: M001/S04
- Supporting slices: M001/S05
- Validation: M001
- Notes: Default mode — lightweight, no NN weights

### R010 — Optional full NN weight preservation
- Class: core-capability
- Status: validated
- Description: An optional flag (default off) preserves trained NN weights alongside training metadata when saving FastGM state
- Why it matters: Enables re-evaluation of trained networks on new inputs without retraining
- Source: user
- Primary owning slice: M001/S04
- Supporting slices: none
- Validation: M001
- Notes: When enabled, NN state dicts saved per bucket

### R011 — Undo-normalization accessible from saved state
- Class: core-capability
- Status: validated
- Description: When NN weights are preserved, a function to convert NN output back to original scale (undo normalization) is accessible from the saved state
- Why it matters: NN outputs are trained on normalized targets; need to recover actual values
- Source: user
- Primary owning slice: M001/S04
- Supporting slices: none
- Validation: M001
- Notes: data_processor.undo_normalization() already exists; this makes it accessible from saved state

### R012 — Modular state preservation
- Class: quality-attribute
- Status: validated
- Description: State preservation is cleanly separated and modularized, not tangled into FastGM internals
- Why it matters: User explicitly requested separable/modular design
- Source: user
- Primary owning slice: M001/S04
- Supporting slices: none
- Validation: M001
- Notes: Likely a separate module (e.g. nce/state/) that FastGM delegates to

### R013 — Standalone plotting functions on FastGM objects
- Class: core-capability
- Status: validated
- Description: Standalone functions generate plots directly from FastGM objects (live or unpickled) without requiring separate scripts
- Why it matters: Current plotting requires ad-hoc scripts; integrated plotting enables quick inspection
- Source: inferred
- Primary owning slice: M001/S05
- Supporting slices: none
- Validation: M001
- Notes: New nce/visualization/ module

### R014 — Per-NN learning curve visualization
- Class: core-capability
- Status: validated
- Description: Each bucket's NN produces a distinct learning curve subplot showing its individual training trajectory (loss over epochs)
- Why it matters: Diagnosing training issues requires per-bucket inspection
- Source: user
- Primary owning slice: M001/S05
- Supporting slices: none
- Validation: M001
- Notes: none

### R015 — Cross-experiment comparison plotting
- Class: core-capability
- Status: validated
- Description: Comparison functions accept multiple FastGM objects/logs and plot side-by-side comparisons
- Why it matters: Research requires comparing configs, loss functions, architectures across experiments
- Source: user
- Primary owning slice: M001/S05
- Supporting slices: none
- Validation: M001
- Notes: Compare across configs on same problem, same config across problems, or both

### R016 — Comprehensive logging to configurable log file
- Class: core-capability
- Status: validated
- Description: When a log file path is set in config, all training events (epoch, loss, bucket id) are written line-by-line to that file during inference
- Why it matters: Post-hoc debugging of long training runs requires persistent logs
- Source: user
- Primary owning slice: M001/S06
- Supporting slices: none
- Validation: M001
- Notes: none

### R017 — Regression test for config restructure
- Class: quality-attribute
- Status: validated
- Description: A regression test script translates a flat config to nested format and runs inference on a reference problem, confirming identical partition function estimates within numerical tolerance
- Why it matters: Config restructure must not change inference behavior
- Source: user
- Primary owning slice: M001/S07
- Supporting slices: none
- Validation: M001
- Notes: One-command, pass/fail output

## Active

### R018 — Exact inference correctness test
- Class: core-capability
- Status: validated
- Description: Test that exact inference (no NN approximation) produces known-correct partition function values
- Why it matters: Baseline correctness — if exact inference is wrong, everything built on it is wrong
- Source: user
- Primary owning slice: M002/S01
- Supporting slices: none
- Validation: M002
- Notes: Use small problems where exact partition function is computable

### R019 — Single bucket training functional test
- Class: core-capability
- Status: validated
- Description: Test that a single bucket can be trained via the NN path without errors
- Why it matters: Bucket training is the atomic unit of NN inference; must work in isolation
- Source: user
- Primary owning slice: M002/S01
- Supporting slices: none
- Validation: M002
- Notes: none

### R020 — Domain size ≥3 variable handling test
- Class: core-capability
- Status: validated
- Description: Test that inference works correctly for problems with variables of domain size 3 or higher
- Why it matters: Most test problems are binary; multi-valued variables exercise different code paths (one-hot encoding, domain size handling)
- Source: user
- Primary owning slice: M002/S01
- Supporting slices: none
- Validation: M002
- Notes: none

### R021 — Convergence test
- Class: core-capability
- Status: validated
- Description: Test that loss decreases when training one bucket for ~50 epochs
- Why it matters: Sanity check that training actually learns, not just runs without error
- Source: user
- Primary owning slice: M002/S01
- Supporting slices: none
- Validation: M002
- Notes: none

### R022 — No-infinity output robustness test
- Class: core-capability
- Status: validated
- Description: Test that loss functions never output infinity during training
- Why it matters: Infinity propagation silently corrupts results
- Source: user
- Primary owning slice: M002/S01
- Supporting slices: none
- Validation: M002
- Notes: none

### R023 — All-neg-inf / all-zero target handling test
- Class: core-capability
- Status: validated
- Description: Test that loss functions handle target batches of all -inf (logspace) or all 0 (linspace) without crashing
- Why it matters: These edge cases occur in real problems when bucket messages are deterministic
- Source: user
- Primary owning slice: M002/S01
- Supporting slices: none
- Validation: M002
- Notes: none

### R024 — Extensible failure-mode regression pattern
- Class: quality-attribute
- Status: validated
- Description: Test suite has a clear pattern for adding new failure-mode regression tests as they appear
- Why it matters: User wants to add test scripts for common failure modes as they encounter them
- Source: user
- Primary owning slice: M002/S01
- Supporting slices: none
- Validation: M002
- Notes: Convention/structure, not a feature

## Active

### R033 — NeuroBE min-max [0,1] target normalization mode
- Class: core-capability
- Status: validated
- Description: Min-max normalization of training targets to [0,1] range matching NeuroBE's `samples_to_data()`, with corresponding denormalization at inference via `ln_min + nn_out * (ln_max - ln_min)`
- Why it matters: Required to faithfully reproduce NeuroBE's training pipeline; different normalization produces different convergence behavior
- Source: user
- Primary owning slice: M003/S01
- Supporting slices: none
- Validation: M003
- Notes: DataPreprocessor normalization_mode='minmax_01'; coexists with existing logspace_mean mode

### R034 — NeuroBE patience-2 early stopping
- Class: core-capability
- Status: validated
- Description: Early stopping that halts training after 2 consecutive non-improving epochs on validation loss, matching NeuroBE's `stop_iter=2` behavior
- Why it matters: NeuroBE's early stopping is simpler than NCE's current 3-consecutive-increases logic; must match for faithful reproduction
- Source: user
- Primary owning slice: M003/S01
- Supporting slices: none
- Validation: M003
- Notes: count > stop_iter=2 means 3 non-improving epochs trigger stop; exercised in BN_1 (stopped at epoch 145)

### R035 — NeuroBE reproduction config preset
- Class: core-capability
- Status: validated
- Description: A `neurobe_mode` config flag that sets batch_size=256, weighted MSE loss, no backward messages, lr=0.001, min-max normalization, and patience-2 early stopping
- Why it matters: Single flag to switch NCE into NeuroBE-faithful mode for direct comparison experiments
- Source: user
- Primary owning slice: M003/S01
- Supporting slices: none
- Validation: M003
- Notes: NEUROBE_DEFAULTS dict with 20 keys; individual settings independently configurable via set-if-absent pattern

### R036 — Matched NN counts via ecl tuning
- Class: core-capability
- Status: validated
- Description: Per-problem ecl values that produce the same number of NN-trained buckets as NeuroBE's width-based dispatch for the 15 working binary-domain problems
- Why it matters: Apples-to-apples comparison requires the same buckets to be NN-trained in both codebases
- Source: user
- Primary owning slice: M003/S02
- Supporting slices: none
- Validation: M003
- Notes: ecl = 2^width_problem - 1; verified by scripts/verify_nn_counts.py (all 15 MATCH)

### R037 — Normalization round-trip test
- Class: quality-attribute
- Status: validated
- Description: Tests verify that min-max [0,1] normalize → train → denormalize produces correct output values (round-trip correctness)
- Why it matters: User explicitly requested tests to verify the normalization pipeline works end-to-end
- Source: user
- Primary owning slice: M003/S01
- Supporting slices: none
- Validation: M003
- Notes: test_round_trip_known_values verifies normalize → denormalize with known ln values; degenerate case also tested

### R038 — Combined NeuroBE comparison results table
- Class: core-capability
- Status: active
- Description: Run 15 binary-domain problems through NCE neurobe_mode and produce a combined comparison table with NeuroBE C++ results
- Why it matters: The whole point — direct comparison to validate the reproduction
- Source: user
- Primary owning slice: M003/S02
- Supporting slices: none
- Validation: unmapped
- Notes: Table should include log_Z, error, abs_error, time, NNs for both codebases

### R039 — Hard bucket selection precomputation
- Class: core-capability
- Status: active
- Description: One-time script that runs all 24 small_problems with UKL + bw + auto_ecl + 10000 epochs (full-batch), identifies buckets with local error > 0.1, and saves a curated list of up to 10 hard buckets to disk
- Why it matters: Provides the fixed evaluation set that all benchmark runs compare against
- Source: user
- Primary owning slice: M004/TBD
- Supporting slices: none
- Validation: unmapped
- Notes: Uses existing auto_ecl values from problem_ecl_values.csv; validates exact bw solvability per selected bucket

### R040 — Precomputed message caching
- Class: core-capability
- Status: active
- Description: Save exact forward messages, exact backward messages, and approximate backward messages at bw_ecl levels (2^2, 2^3, 2^5, 2^10, 2^15, 2^25) to disk per selected bucket
- Why it matters: Eliminates expensive message recomputation on every benchmark run
- Source: user
- Primary owning slice: M004/TBD
- Supporting slices: none
- Validation: unmapped
- Notes: Config specifies which bw_ecl to use for training; all levels cached for future comparison

### R041 — Time-limited single-bucket training
- Class: core-capability
- Status: active
- Description: Train a single bucket's NN with epoch-boundary timeout (fast=1min, slow=1h per bucket), save NN weights, compute local error at checkpoint epochs
- Why it matters: Enables fair time-controlled comparison across different configs
- Source: user
- Primary owning slice: M004/TBD
- Supporting slices: none
- Validation: unmapped
- Notes: Checkpoints at epoch 10, 20, 50, 100, 500, 1000, 2000, etc. or time-based fallback for slow epochs

### R042 — Multi-GPU parallel execution
- Class: core-capability
- Status: active
- Description: Run benchmark training across multiple GPUs (1 bucket per GPU), cycling through bucket list as GPUs become free
- Why it matters: 4× speedup on 4-GPU machine; benchmarks complete in practical time
- Source: user
- Primary owning slice: M004/TBD
- Supporting slices: none
- Validation: unmapped
- Notes: Default: all 4 GPUs; configurable via --gpus argument

### R043 — Per-bucket benchmark output
- Class: core-capability
- Status: active
- Description: Each benchmark bucket produces a folder with loss-over-epochs and local-error-over-epochs matplotlib PNG plots
- Why it matters: Visual inspection of learning quality per hard bucket
- Source: user
- Primary owning slice: M004/TBD
- Supporting slices: none
- Validation: unmapped
- Notes: Output folder also contains config, timing, and epoch metadata

### R044 — Historical comparison tracking
- Class: core-capability
- Status: active
- Description: JSONL history file records per-run metadata (config, timing, epochs, local errors); comparison chart shows current vs historical best for runs of equal or shorter duration
- Why it matters: Track whether config changes improve hard-bucket learning quality over time
- Source: user
- Primary owning slice: M004/TBD
- Supporting slices: none
- Validation: unmapped
- Notes: Best-ever comparison filtered by duration ≤ current run's duration for fair comparison

### R045 — CLI benchmark entry point
- Class: core-capability
- Status: active
- Description: `python scripts/bucket_benchmark.py <config.yaml> <fast|slow> [--gpus 0,1,2,3]` runs the full benchmark pipeline
- Why it matters: Single command to run, compare, and record benchmark results
- Source: user
- Primary owning slice: M004/TBD
- Supporting slices: none
- Validation: unmapped
- Notes: YAML config processed through prepare_config()

## Deferred

### R025 — Config inheritance (base + overrides)
- Class: core-capability
- Status: deferred
- Description: Support base config files with per-experiment overrides
- Why it matters: Reduces duplication across similar experiments
- Source: research
- Primary owning slice: none
- Supporting slices: none
- Validation: unmapped
- Notes: Deferred — current experiment scale doesn't demand it yet

### R026 — Named experiment presets
- Class: core-capability
- Status: deferred
- Description: Named presets for common experiment configurations
- Why it matters: Convenience for frequently-used setups
- Source: research
- Primary owning slice: none
- Supporting slices: none
- Validation: unmapped
- Notes: Deferred — benchmark configs serve this role for now

### R027 — Progress output during long experiments
- Class: core-capability
- Status: deferred
- Description: Structured progress output during long-running experiments
- Why it matters: Multi-hour runs with no output are anxiety-inducing
- Source: research
- Primary owning slice: none
- Supporting slices: none
- Validation: unmapped
- Notes: Deferred — logging (R016) partially addresses this

### R028 — Confidence intervals on mean plots
- Class: core-capability
- Status: deferred
- Description: Statistical confidence intervals on averaged plots
- Why it matters: Publication-quality results need error bars
- Source: research
- Primary owning slice: none
- Supporting slices: none
- Validation: unmapped
- Notes: Deferred — alpha=0.3 fill_between already used for confidence bands
## Out of Scope

### R029 — Interactive dashboard
- Class: anti-feature
- Status: out-of-scope
- Description: Real-time interactive monitoring dashboard
- Why it matters: Prevents scope creep into UI/web territory; post-hoc analysis is sufficient
- Source: prior
- Primary owning slice: none
- Supporting slices: none
- Validation: n/a
- Notes: Adds complexity not needed for research workflow

### R030 — Cloud/cluster submission
- Class: anti-feature
- Status: out-of-scope
- Description: Cloud or cluster job submission system
- Why it matters: Prevents over-engineering; local multi-GPU is sufficient
- Source: prior
- Primary owning slice: none
- Supporting slices: none
- Validation: n/a
- Notes: none

### R031 — Hyperparameter search
- Class: anti-feature
- Status: out-of-scope
- Description: Automatic hyperparameter search/optimization
- Why it matters: Manual control is preferred for research; auto-search obscures understanding
- Source: prior
- Primary owning slice: none
- Supporting slices: none
- Validation: n/a
- Notes: none

### R032 — Real-time plot updates
- Class: anti-feature
- Status: out-of-scope
- Description: Live-updating plots during training
- Why it matters: Post-hoc plotting is sufficient; real-time adds complexity
- Source: prior
- Primary owning slice: none
- Supporting slices: none
- Validation: n/a
- Notes: none
## Traceability

| ID | Class | Status | Primary owner | Supporting | Proof |
|---|---|---|---|---|---|
| R001 | core-capability | validated | M001/S01 | M001/S02 | M001 |
| R002 | core-capability | validated | M001/S01 | none | M001 |
| R003 | core-capability | validated | M001/S01 | M001/S02, M001/S03 | M001 |
| R004 | core-capability | validated | M001/S01 | none | M001 |
| R005 | core-capability | validated | M001/S01 | none | M001 |
| R006 | quality-attribute | validated | M001/S01 | none | M001 |
| R007 | core-capability | validated | M001/S03 | none | M001 |
| R008 | quality-attribute | validated | M001/S03 | none | M001 |
| R009 | core-capability | validated | M001/S04 | M001/S05 | M001 |
| R010 | core-capability | validated | M001/S04 | none | M001 |
| R011 | core-capability | validated | M001/S04 | none | M001 |
| R012 | quality-attribute | validated | M001/S04 | none | M001 |
| R013 | core-capability | validated | M001/S05 | none | M001 |
| R014 | core-capability | validated | M001/S05 | none | M001 |
| R015 | core-capability | validated | M001/S05 | none | M001 |
| R016 | core-capability | validated | M001/S06 | none | M001 |
| R017 | quality-attribute | validated | M001/S07 | none | M001 |
| R018 | core-capability | validated | M002/S01 | none | M002 |
| R019 | core-capability | validated | M002/S01 | none | M002 |
| R020 | core-capability | validated | M002/S01 | none | M002 |
| R021 | core-capability | validated | M002/S01 | none | M002 |
| R022 | core-capability | validated | M002/S01 | none | M002 |
| R023 | core-capability | validated | M002/S01 | none | M002 |
| R024 | quality-attribute | validated | M002/S01 | none | M002 |
| R025 | core-capability | deferred | none | none | unmapped |
| R026 | core-capability | deferred | none | none | unmapped |
| R027 | core-capability | deferred | none | none | unmapped |
| R028 | core-capability | deferred | none | none | unmapped |
| R029 | anti-feature | out-of-scope | none | none | n/a |
| R030 | anti-feature | out-of-scope | none | none | n/a |
| R031 | anti-feature | out-of-scope | none | none | n/a |
| R032 | anti-feature | out-of-scope | none | none | n/a |
| R033 | core-capability | validated | M003/S01 | none | M003 |
| R034 | core-capability | validated | M003/S01 | none | M003 |
| R035 | core-capability | validated | M003/S01 | none | M003 |
| R036 | core-capability | validated | M003/S02 | none | M003 |
| R037 | quality-attribute | validated | M003/S01 | none | M003 |
| R038 | core-capability | active | M003/S02 | none | unmapped |
| R039 | core-capability | active | M004/TBD | none | unmapped |
| R040 | core-capability | active | M004/TBD | none | unmapped |
| R041 | core-capability | active | M004/TBD | none | unmapped |
| R042 | core-capability | active | M004/TBD | none | unmapped |
| R043 | core-capability | active | M004/TBD | none | unmapped |
| R044 | core-capability | active | M004/TBD | none | unmapped |
| R045 | core-capability | active | M004/TBD | none | unmapped |
## Coverage Summary

- Active requirements: 37
- Mapped to slices: 30
- Validated: 29
- Unmapped active requirements: 7 (R039–R045, pending M004 planning)
