---
estimated_steps: 4
estimated_files: 3
---

# T03: Wire prepare_config into FastGM and verify end-to-end

**Slice:** S01 — Config Schema & Flat Translation
**Milestone:** M001

## Description

Close the integration loop: modify `FastGM.__init__` to call `prepare_config()` on the incoming `nn_config`, then run the full test suite including integration tests that verify backward compat with real benchmark configs and nested config equivalence. Handle the dead-field transition gracefully — existing benchmark configs include `backward_ecl` and `num_batches_per_set` which are dead but shouldn't crash during the S01→S02 transition.

## Steps

1. Edit `nce/inference/graphical_model.py`:
   - Add import: `from nce.config_schema import prepare_config`
   - Replace line `self.config = dict(nn_config) if nn_config else {}` with: `self.config = prepare_config(nn_config) if nn_config else {}`
   - No other changes needed — all downstream consumers read from `self.config` by reference
2. Handle dead-field transition for existing benchmark configs: The `nbe_sanity_check` configs include `backward_ecl` and `num_batches_per_set`. Rather than making `prepare_config` silently ignore dead fields (which defeats R002), adjust the approach: `prepare_config` strips known dead fields from flat configs with a `warnings.warn()`, and raises `ValueError` only when dead fields appear in nested configs (where the user is explicitly writing new-style config and should know better). This preserves backward compat (R005) while still enforcing dead-field detection for new configs (R002).
3. Add/update integration tests in `tests/test_config_schema.py`:
   - `test_fastgm_init_with_flat_config` — create FastGM with nbe_sanity_check flat config, verify `gm.config` has expected fields. This requires a model, so use a minimal test: verify `prepare_config` is called by checking the result dict has dead fields stripped.
   - `test_fastgm_init_with_nested_config` — same but with nested equivalent
   - `test_dead_fields_warn_in_flat` — flat config with dead fields produces warning, not error
   - `test_dead_fields_error_in_nested` — nested config with dead fields raises ValueError
   - `test_benchmark_config_survives_prepare` — actual nbe_sanity_check config passes through `prepare_config()` without error, all expected internal keys present
4. Run full test suite: `python -m pytest tests/test_config_schema.py -v` — ALL tests pass

## Must-Haves

- [ ] `FastGM.__init__` calls `prepare_config()` — the ONLY code change in graphical_model.py
- [ ] Real nbe_sanity_check benchmark config passes through without error
- [ ] Dead fields in flat configs produce warning + strip (backward compat)
- [ ] Dead fields in nested configs produce error (new config enforcement)
- [ ] All tests pass: `python -m pytest tests/test_config_schema.py -v`
- [ ] Nested and flat equivalent configs produce identical flat output

## Verification

- `python -m pytest tests/test_config_schema.py -v` — full suite green
- `python -c "from nce.config_schema import prepare_config; from nce.benchmark_problems.nbe_sanity_check import _build_nbe_configs; c = _build_nbe_configs()[0]; r = prepare_config(c); print('backward_ecl' not in r, 'ecl' in r)"` — prints `True True`

## Observability Impact

- Signals added/changed: `warnings.warn()` for dead fields in flat configs (visible in test output and during experiment runs)
- How a future agent inspects this: run `prepare_config(config)` on any config dict to see what it produces; check warnings with `-W all` flag
- Failure state exposed: if `prepare_config` raises, the full traceback shows which field in which section caused the problem

## Inputs

- `nce/config_schema.py` — complete module from T02
- `tests/test_config_schema.py` — test suite from T01 (integration tests may need additions)
- `nce/inference/graphical_model.py` — the file to modify (line ~36)
- `nce/benchmark_problems/nbe_sanity_check.py` — real benchmark config for integration testing

## Expected Output

- `nce/inference/graphical_model.py` — modified with 3-line change (import + replace config line)
- `tests/test_config_schema.py` — updated with integration tests
- `nce/config_schema.py` — minor update to dead-field handling (warn vs error based on config style)
- Full test suite passing: contract (schema validation) + integration (FastGM init, benchmark compat)
