---
estimated_steps: 5
estimated_files: 1
---

# T02: Build config_schema.py — schema, validation, and translation

**Slice:** S01 — Config Schema & Flat Translation
**Milestone:** M001

## Description

Implement the complete `nce/config_schema.py` module: schema definition for all 55+ fields, flat/nested detection, alias resolution, validation with section-specific errors, nested→flat flattening, and the `prepare_config()` entry point. Pure Python — no torch/pyGMs imports. Returns a plain mutable dict.

## Steps

1. Define `NESTED_SECTIONS` ordered dict mapping section name → dict of field name → `{old_name, type, default, required}`. Follow the S01-RESEARCH field inventory exactly:
   - `inference`: 5 fields (ecl, iB, approximation_method, dope_factors, device)
   - `nn`: 6 fields (hidden_sizes, use_linspace_bias, use_memorizer, custom_hidden_sizes, init_with_linear_optimum, weight_decay) + 7 decision_tree fields with `dt_` prefix
   - `training`: 28 fields (num_epochs, loss_fn, optimizer, lr, batch_size, patience, etc.)
   - `sampling`: 6 fields (sampling_scheme, num_samples, set_size, val_set, stratify_samples, lower_dim)
   - `backward`: 5 fields (use_bw_approx, populate_bw_factors, bw_ecl, backward_iB, fdb)
   - `output`: 8 fields (debug, display_intermediate, track_errors, error_tracking, plot_messages, traced_losses, gather_message_stats, complexity_limit)
   - For each field: store the internal/old key name (what consumer code reads), accepted types, default value, and whether it's required
2. Define `DEAD_FIELDS` dict: `{'backward_ecl': "Dead field. Use 'bw_ecl' instead.", 'num_batches_per_set': "Dead field. Computed internally from set_size // batch_size."}`. Define `DEPRECATED_FIELDS` set for fields that warn but don't error (for benchmark config compat during transition — dead fields in existing in-tree configs should warn, not crash, until S02 cleans them up).
3. Define `FIELD_ALIASES` mapping readable names → internal names per D007: `{'learning_rate': 'lr', 'exact_computation_limit': 'ecl', 'i_bound': 'iB', 'forward_diff_barrier': 'fdb', 'backward_ecl_limit': 'bw_ecl', 'num_epochs_phase2': 'num_epochs2', 'loss_fn_phase2': 'loss_fn2', 'learning_rate_decay': 'lr_decay', 'min_learning_rate': 'min_lr', 'dt_learning_rate': 'dt_lr', 'gradient_clip_norm': 'grad_clip_norm', 'backward_i_bound': 'backward_iB'}`. Include reverse mapping for lookup.
4. Implement core functions:
   - `_is_nested(config)` — returns True if any key in SECTION_NAMES has a dict value
   - `_resolve_aliases(d)` — recursively resolve alias names to internal names in a flat or nested dict
   - `validate_nested_config(config)` — check each section exists in schema, each field within section is known, required fields present when applicable, raise `ValueError` with section and field context
   - `flatten_config(nested)` — iterate sections, map fields to internal names, merge to flat dict, apply defaults for missing optional fields
   - `_validate_flat_config(flat)` — check for dead fields (error or warn), check required fields when `approximation_method='nn'`
   - `prepare_config(config_dict)` — the single public entry point: if None/empty return `{}`; detect flat vs nested; if nested: validate, flatten; if flat: resolve aliases, validate; return plain dict
5. Run unit tests: `python -m pytest tests/test_config_schema.py -v -k "not integration and not fastgm"` — all non-integration tests should pass

**Key constraints from research:**
- Internal key names MUST match exactly what consumer code reads (e.g., `'ecl'` not `'exact_computation_limit'`)
- Return type MUST be plain `dict` — code mutates `self.config` at runtime
- No torch/pyGMs imports — module must be independently importable
- Dead fields in user configs should raise errors; dead fields coming from in-tree benchmark configs need a grace path (warn + strip) so existing code doesn't break before S02 cleans them up. Handle this by having `prepare_config` accept an optional `strict=True` parameter — strict mode errors on dead fields, non-strict warns. Default to strict. FastGM can pass `strict=False` if needed during transition, but tests should verify strict behavior.

## Must-Haves

- [ ] `NESTED_SECTIONS` covers all 55+ fields from research inventory
- [ ] `DEAD_FIELDS` includes `backward_ecl` and `num_batches_per_set` with clear messages
- [ ] `FIELD_ALIASES` maps all readable names from D007
- [ ] `prepare_config()` auto-detects flat vs nested
- [ ] `validate_nested_config()` errors name the offending field AND section
- [ ] `flatten_config()` produces correct internal key names
- [ ] No torch or pyGMs imports anywhere in the file
- [ ] Return type is always plain `dict`

## Verification

- `python -m pytest tests/test_config_schema.py -v -k "not integration and not fastgm"` — all non-integration tests pass
- `python -c "import nce.config_schema; print('OK')"` — module imports cleanly without torch

## Observability Impact

- Signals added/changed: `ValueError` exceptions with structured messages: field name, section name, suggestion
- How a future agent inspects this: call `prepare_config(config)` standalone — pure function, no side effects, inspectable return value
- Failure state exposed: validation errors include all context needed to fix the config

## Inputs

- `tests/test_config_schema.py` — the test suite defining acceptance criteria (from T01)
- `tests/conftest.py` — reference config fixtures (from T01)
- S01-RESEARCH field inventory — the authoritative field list

## Expected Output

- `nce/config_schema.py` — complete module with schema, detection, validation, flattening, and `prepare_config()` entry point
