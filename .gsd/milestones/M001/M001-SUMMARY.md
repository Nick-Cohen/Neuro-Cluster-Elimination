---
id: M001
provides:
  - "nce/config_schema.py — nested config schema, validation, flat/nested translation (prepare_config)"
  - "nce/state/ — save_state/load_state for post-inference FastGM inspection with optional NN weights"
  - "nce/visualization/ — plot_learning_curves and compare_experiments from live or pickled FastGM"
  - "nce/training_logger.py — JSONL structured logging integrated into training loop"
  - "docs/config_reference.md — comprehensive config documentation guide (387 lines, all fields)"
  - "scripts/regression_test.py — one-command flat-vs-nested config regression test"
  - "Nested config builders in nbe_sanity_check and small_problems benchmark sets"
key_decisions:
  - "D001: Config translation lives in config_schema.py, not FastGM.__init__"
  - "D002: Auto-detect flat vs nested; flat passes through, nested translates"
  - "D003: Internal representation stays flat — zero consumer code changes"
  - "D010: Only backward_ecl and num_batches_per_set are dead fields (prior plan was wrong)"
  - "D011: Dead fields warn in flat mode, error in nested mode"
  - "D015: State dict is inspection-only, not reconstructable"
  - "D016: Weight capture at training time, stripping at save time"
  - "D017: JSONL format via Python stdlib logging"
  - "D018: nce.training logger namespace avoids root logger suppression"
patterns_established:
  - "Config schema as single source of truth — NESTED_SECTIONS defines all fields, defaults, and aliases"
  - "Benchmark sets offer both flat and nested config builders (configs['nbe'] + configs['nbe_nested'])"
  - "State preservation via thin serialization layer separate from FastGM internals"
  - "Per-bucket training log captures full loss curves before bucket deletion"
  - "Visualization functions accept either live FastGM or loaded state dict"
observability_surfaces:
  - "JSONL training log — per-epoch loss, bucket start/end, early stopping events (config: log_file)"
  - "per_bucket_training_log — in-memory and picklable loss curves, epochs, hidden sizes per bucket"
  - "Regression test script — pass/fail verification of config equivalence (scripts/regression_test.py)"
requirement_outcomes:
  - id: R001
    from_status: active
    to_status: validated
    proof: "Nested config with 6 sections (inference, nn, training, sampling, backward, output) accepted by prepare_config and FastGM.__init__. Verified by regression test (3/3 checks PASS)."
  - id: R002
    from_status: active
    to_status: validated
    proof: "DEAD_FIELDS contains backward_ecl and num_batches_per_set. Using them in nested config raises ValueError. Code audit (D010) corrected prior plan's wrong dead-field list."
  - id: R003
    from_status: active
    to_status: validated
    proof: "FIELD_ALIASES maps 14 readable names (learning_rate, exact_computation_limit, i_bound, forward_diff_barrier, etc.) to internal keys. Old names accepted as aliases."
  - id: R004
    from_status: active
    to_status: validated
    proof: "prepare_config raises ValueError with section name and field name for unexpected keys in nested configs."
  - id: R005
    from_status: active
    to_status: validated
    proof: "All 29 flat configs (5 nbe_sanity_check + 24 small_problems) auto-detected and accepted. Regression test confirms flat config produces identical inference results."
  - id: R006
    from_status: active
    to_status: validated
    proof: "FastGM.__init__ line 38: self.config = prepare_config(nn_config). All translation logic in config_schema.py."
  - id: R007
    from_status: active
    to_status: validated
    proof: "docs/config_reference.md — 387 lines covering all fields with type, default, accepted values, and purpose. Includes flat and nested examples."
  - id: R008
    from_status: active
    to_status: validated
    proof: "config_schema.py header references docs/config_reference.md. NESTED_SECTIONS is the single source of truth for field definitions; the doc guide references it as authoritative."
  - id: R009
    from_status: active
    to_status: validated
    proof: "save_state/load_state round-trip preserves per_bucket_training_log with full loss curves (134 entries on nbe_sanity_check model 0). Verified in fresh Python session."
  - id: R010
    from_status: active
    to_status: validated
    proof: "save_nn_weights=True in config captures nn_state_dict per bucket. save_state(save_weights=True) preserves them; save_state(save_weights=False) strips them."
  - id: R011
    from_status: active
    to_status: validated
    proof: "nce/state/state.py exports undo_normalization(outputs, normalizing_constant). normalizing_constant saved per-bucket when save_nn_weights=True."
  - id: R012
    from_status: active
    to_status: validated
    proof: "nce/state/ module is separate from FastGM. save_state/load_state are standalone functions that accept a FastGM and return a plain dict."
  - id: R013
    from_status: active
    to_status: validated
    proof: "plot_learning_curves(fastgm) and plot_learning_curves(state_dict) both produce PNG files from live or pickled state. Verified: 280KB output."
  - id: R014
    from_status: active
    to_status: validated
    proof: "plot_learning_curves produces subplots per bucket, each showing loss over epochs. Verified with 134-bucket model."
  - id: R015
    from_status: active
    to_status: validated
    proof: "compare_experiments([gm1, gm2], labels=['a','b']) produces side-by-side comparison plots. Verified: 328KB output."
  - id: R016
    from_status: active
    to_status: validated
    proof: "config log_file path -> JSONL output with bucket_training_start, epoch_loss, bucket_training_end events. 536 lines on 134-bucket model with 2 epochs."
  - id: R017
    from_status: active
    to_status: validated
    proof: "scripts/regression_test.py exits 0: config equality PASS, exact inference equality PASS (58.5306), NN inference equality PASS (129.8932)."
duration: "~3 hours"
verification_result: passed
completed_at: 2026-03-13T19:50:00Z
---

# M001: Config & Visualization

**Restructured configs into validated nested sections, added state preservation with loss curves and optional NN weights, built visualization and structured logging modules, and verified zero behavioral change via regression test.**

## What Happened

**S01 (Config Schema)** audited every `config[` reference in the codebase and built `config_schema.py` with `NESTED_SECTIONS` defining 6 sections, 40+ fields with defaults and aliases. `prepare_config()` auto-detects flat vs nested input, validates, resolves aliases, and returns a flat dict. Dead field inventory was corrected (D010) — only `backward_ecl` and `num_batches_per_set` are actually dead; 5 fields the prior plan listed as dead are live. FastGM.__init__ calls `prepare_config()` as its only config integration point.

**S02 (Benchmark Migration)** added nested config builders (`nbe_nested`, `default_nested`) to both benchmark sets (nbe_sanity_check: 5 configs, small_problems: 24 configs). All 29 flat configs continue to work unmodified.

**S03 (Config Documentation)** produced `docs/config_reference.md` — 387-line guide covering every field with type, default, accepted values, and purpose. The doc references `NESTED_SECTIONS` as the single source of truth.

**S04 (State Preservation)** built `nce/state/` as a thin serialization layer. `save_state()` extracts per_bucket_training_log (with full loss curves captured before bucket deletion via extended bucket.py), config, logZ, elim_order, and optional NN weights. `load_state()` returns a plain dict. `undo_normalization()` converts saved NN outputs back to original scale.

**S05 (Visualization)** built `nce/visualization/` with `plot_learning_curves()` (per-bucket loss subplots) and `compare_experiments()` (side-by-side multi-experiment comparison). Both accept live FastGM objects or loaded state dicts.

**S06 (Logging)** built `nce/training_logger.py` using JSONL format via Python stdlib logging with a dedicated `nce.training` namespace (avoiding root logger suppression). Integrated into `train.py` with epoch_loss, val_loss, early_stopping, bucket_training_start/end events. Activated by setting `log_file` in config.

**S07 (Regression)** built `scripts/regression_test.py` — verifies config equality, exact inference equality, and NN inference equality between flat and nested configs on rbm_20. All 3 checks pass.

## Cross-Slice Verification

| Success Criterion | Status | Evidence |
|---|---|---|
| Nested config sections accepted | ✅ | `prepare_config(nested)` returns valid flat dict; 6 sections with 40+ fields |
| Old flat configs work unchanged | ✅ | All 29 benchmark flat configs pass `prepare_config()`; regression test PASS |
| Dead fields raise errors | ✅ | `backward_ecl` and `num_batches_per_set` raise ValueError in nested mode |
| Pickled FastGM preserves metadata | ✅ | 134-entry training log with losses round-trips through save/load |
| Loss curves inspectable after pickle | ✅ | Each entry has `losses` and `val_losses` keys with per-epoch data |
| Optional NN weight preservation | ✅ | `save_nn_weights=True` captures state_dict + normalizing_constant per bucket |
| Undo-normalization accessible | ✅ | `nce.state.undo_normalization()` converts outputs using saved normalizing_constant |
| `plot_learning_curves` works | ✅ | Produces 280KB PNG with per-bucket subplots from live or loaded state |
| `compare_experiments` works | ✅ | Produces 328KB PNG comparing two experiments side-by-side |
| Config documentation complete | ✅ | 387-line guide covering every field with type, default, purpose |
| Structured log file output | ✅ | 536 JSONL lines on 134-bucket model; events: start, epoch_loss, end |
| Regression test passes | ✅ | 3/3 checks PASS: config equality, exact inference (58.53), NN inference (129.89) |

## Requirement Changes

- R001–R006: active → validated — Config schema, dead fields, aliases, validation, backward compat, separation
- R007–R008: active → validated — Config documentation guide and doc-sync enforcement
- R009–R012: active → validated — State preservation, optional weights, undo-normalization, modularity
- R013–R015: active → validated — Standalone plotting, per-NN learning curves, cross-experiment comparison
- R016: active → validated — Structured JSONL logging with configurable log file
- R017: active → validated — Regression test confirms identical inference results

## Forward Intelligence

### What the next milestone should know
- `prepare_config()` is now the single entry point for all config handling. Any new config field must be added to `NESTED_SECTIONS` in `config_schema.py` and documented in `docs/config_reference.md`.
- `per_bucket_training_log` is populated during `eliminate_variables()` — entries include `losses`, `val_losses`, `epochs_trained`, `hidden_sizes`, `label`, and optionally `nn_state_dict` and `normalizing_constant`.
- The `nce.training` logger namespace is isolated from root logger suppression in `stats.py`. Any new logging should use this namespace or a child of it.

### What's fragile
- `per_bucket_training_log` depends on bucket.py capturing data before bucket deletion — if the bucket lifecycle changes, loss curves will be lost silently.
- `save_nn_weights` config flag must be set before inference starts (capture happens at training time, not save time). There's no way to retroactively capture weights from a run that didn't set this flag.
- The `DEAD_FIELDS` set is small (2 fields). If consumer code is removed in the future, the corresponding config fields need to be added here manually.

### Authoritative diagnostics
- `scripts/regression_test.py` — run this after any change to config handling, bucket.py, or graphical_model.py to verify no behavioral change.
- `python -c "from nce.config_schema import prepare_config; print(prepare_config({...}))"` — quick smoke test for config translation.

### What assumptions changed
- Prior plan listed 5 dead fields (loss_fn2, num_epochs2, complexity_limit, exact, memorizer) — code audit found all 5 are live. Only backward_ecl and num_batches_per_set are actually dead (D010).
- State preservation is inspection-only (D015) — no FastGM reconstruction from saved state. This was a deliberate scope decision, not a limitation to fix later.

## Files Created/Modified

- `nce/config_schema.py` — nested config schema, validation, flat/nested translation (18KB, 420 lines)
- `nce/state/__init__.py` — state module public API exports
- `nce/state/state.py` — save_state, load_state, undo_normalization (3.6KB)
- `nce/visualization/__init__.py` — visualization module public API exports
- `nce/visualization/learning_curves.py` — plot_learning_curves (6.9KB)
- `nce/visualization/comparison.py` — compare_experiments (8.3KB)
- `nce/training_logger.py` — JSONL training event logger (3.5KB)
- `nce/inference/graphical_model.py` — integrated prepare_config and training logger setup
- `nce/inference/bucket.py` — extended per_bucket_training_log with loss curves and optional weights
- `nce/neural_networks/train.py` — integrated training logger event emission
- `nce/benchmark_problems/nbe_sanity_check.py` — added nbe_nested config builders
- `nce/benchmark_problems/small_problems.py` — added default_nested config builders
- `docs/config_reference.md` — comprehensive config documentation guide (387 lines)
- `scripts/regression_test.py` — flat-vs-nested regression test
- `tests/test_config_schema.py` — config schema unit tests
- `tests/test_benchmark_configs.py` — benchmark config validation tests
- `tests/test_config_docs.py` — config documentation sync tests
- `tests/test_regression.py` — pytest wrapper for regression test
- `scripts/verify_logging.py` — logging verification script
- `scripts/verify_s04_state_preservation.py` — state preservation verification
- `scripts/verify_s05_visualization.py` — visualization verification
