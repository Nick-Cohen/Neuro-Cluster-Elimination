# S01: Config Schema & Flat Translation

**Goal:** `prepare_config()` in `nce/config_schema.py` accepts nested or flat config dicts, validates, flattens nested→flat, and returns a plain flat dict. FastGM calls it at init. Old flat configs auto-detected and unchanged. Dead fields error. Readable aliases accepted.
**Demo:** Pass a nested config dict to `FastGM(model=..., nn_config=nested)` — it validates, flattens, and runs inference identically to the old flat config. Pass an old flat config — auto-detected, no changes needed. Pass a dead field — clear error.

## Must-Haves

- Complete schema covering all 55+ live config fields across 6 sections (inference, nn, training, sampling, backward, output) plus decision_tree fields (R001)
- `DEAD_FIELDS` dict with clear error messages for `backward_ecl` and `num_batches_per_set` (R002)
- `FIELD_ALIASES` mapping readable names → internal names: `learning_rate`→`lr`, `exact_computation_limit`→`ecl`, `i_bound`→`iB`, `forward_diff_barrier`→`fdb`, etc. (R003)
- `validate_nested_config()` with section-specific error messages naming the offending field and section (R004)
- Auto-detection of flat vs nested configs via section-name + dict-value heuristic (R005)
- All config logic in `nce/config_schema.py`, not in `FastGM.__init__` — FastGM calls one function (R006)
- `prepare_config()` returns a plain mutable `dict` (not frozen/dataclass/wrapper) — consumer code mutates config at runtime
- No torch/pyGMs imports in `config_schema.py` — pure Python stdlib only
- Polymorphic types handled: `hidden_sizes` (list|str), `num_samples` (int|str), `batch_size` (int|`'all'`), `val_set` (bool|str|None)
- Internal flat key names preserved exactly (`ecl`, `lr`, `iB`, `fdb`, etc.) — all consumer code reads old names

## Proof Level

- This slice proves: contract + integration
- Real runtime required: yes (integration test runs FastGM with real benchmark config)
- Human/UAT required: no

## Verification

- `cd /home/cohenn1/NCE && source venv/bin/activate && python -m pytest tests/test_config_schema.py -v` — all tests pass
- Tests cover:
  - Nested config → flat translation produces correct internal keys
  - Flat config passthrough preserves all fields
  - Auto-detection correctly classifies flat and nested configs
  - Dead fields raise `ValueError` with descriptive message
  - Field aliases resolve to internal names
  - Unknown nested fields raise `ValueError` with section context
  - Missing required fields raise `ValueError` when `approximation_method='nn'`
  - Polymorphic types accepted without error
  - `prepare_config()` returns a plain `dict` (mutable)
  - Real benchmark flat config from `nbe_sanity_check` passes through `prepare_config()` unchanged (minus dead fields)
  - FastGM integration: `prepare_config()` called at init, config accessible as flat dict

## Observability / Diagnostics

- Runtime signals: `ValueError` exceptions with structured messages naming field, section, and suggestion (e.g., "Unknown field 'backward_ecl' — this field is dead. Use 'bw_ecl' instead.")
- Inspection surfaces: `prepare_config()` is a pure function — call it standalone to debug any config without instantiating FastGM
- Failure visibility: validation errors include field name, section name, expected type, and suggestion for common mistakes
- Redaction constraints: none (configs contain no secrets)

## Integration Closure

- Upstream surfaces consumed: none (first slice)
- New wiring introduced in this slice: `FastGM.__init__` calls `prepare_config(nn_config)` before storing `self.config`
- What remains before the milestone is truly usable end-to-end: S02 (benchmark config migration with nested builders), S03 (documentation), S07 (regression test proving identical inference results)

## Tasks

- [x] **T01: Set up pytest and write failing acceptance tests** `est:30m`
  - Why: Define the objective stopping condition before writing implementation. Tests encode every requirement (R001–R006) as executable assertions.
  - Files: `tests/__init__.py`, `tests/test_config_schema.py`, `tests/conftest.py`
  - Do: Install pytest in venv. Create `tests/` directory with test file covering: nested→flat translation, flat passthrough, auto-detection, dead field errors, alias resolution, unknown field errors, required field validation, polymorphic types, mutability, benchmark config passthrough. Use fixtures for reference configs. All tests import from `nce.config_schema` (which doesn't exist yet — tests will fail).
  - Verify: `python -m pytest tests/test_config_schema.py --co` lists all test cases (collection succeeds but tests fail on import)
  - Done when: Test file exists with 12+ test functions covering all 6 requirements, pytest collects them all

- [ ] **T02: Build config_schema.py — schema, validation, and translation** `est:1h30m`
  - Why: Core implementation — the schema definition, detection heuristic, validation, alias resolution, nested→flat flattening, and `prepare_config()` entry point.
  - Files: `nce/config_schema.py`
  - Do: Define `NESTED_SECTIONS` schema dict with all 55+ fields (types, defaults, required flags, internal names). Define `DEAD_FIELDS`, `FIELD_ALIASES`. Implement `_is_nested()`, `_resolve_aliases()`, `validate_nested_config()`, `flatten_config()`, `_validate_flat_config()`, `prepare_config()`. All pure Python — no torch/pyGMs. Return plain dict from `prepare_config()`. Follow field inventory from S01-RESEARCH exactly.
  - Verify: `python -m pytest tests/test_config_schema.py -v` — all unit tests pass (except FastGM integration test which needs T03)
  - Done when: All non-integration tests in `test_config_schema.py` pass

- [ ] **T03: Wire prepare_config into FastGM and verify end-to-end** `est:45m`
  - Why: Close the integration loop — FastGM actually uses `prepare_config()`. Verify backward compat with real benchmark config. Verify the full pipeline from nested config → FastGM init → flat dict accessible by all consumers.
  - Files: `nce/inference/graphical_model.py`, `tests/test_config_schema.py`
  - Do: Import `prepare_config` in `graphical_model.py`. Replace `self.config = dict(nn_config) if nn_config else {}` with `self.config = prepare_config(nn_config) if nn_config else {}`. Add integration test that creates FastGM with a real benchmark config and verifies `gm.config` is correct flat dict. Add integration test with nested config variant. Ensure dead fields in existing benchmark configs are handled gracefully (warn or strip, don't error — these are in-tree configs that S02 will clean up).
  - Verify: `python -m pytest tests/test_config_schema.py -v` — ALL tests pass including integration tests
  - Done when: Full test suite green. `FastGM.__init__` calls `prepare_config()`. Old benchmark configs still work. Nested config produces same flat dict as equivalent flat config.

## Files Likely Touched

- `nce/config_schema.py` (new)
- `nce/inference/graphical_model.py` (3-line edit)
- `tests/__init__.py` (new)
- `tests/conftest.py` (new)
- `tests/test_config_schema.py` (new)
