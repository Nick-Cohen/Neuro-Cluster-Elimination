---
id: S01
milestone: M001
status: complete
---

# S01: Config Schema & Flat Translation — Context

<!-- Backfilled from completed work: task summaries, research, decisions D010–D014, and shipped code. -->

## Goal

Provide `prepare_config()` in `nce/config_schema.py` that auto-detects flat vs nested config dicts, validates, flattens nested→flat, and returns a plain flat dict — so FastGM can accept both legacy flat configs and new readable nested configs with identical inference behavior.

## Why this Slice

S01 is the foundation for the entire M001 milestone. Every downstream slice depends on its outputs:
- S02 (benchmark migration) needs the schema and `prepare_config()` to build nested config builders
- S03 (documentation) needs `NESTED_SECTIONS` metadata to generate the config reference guide
- S07 (regression verification) needs `prepare_config()` to translate flat→nested for comparison testing

Without a validated schema and translation layer, none of the config restructuring work can proceed.

## Scope

### In Scope

- Complete schema (`NESTED_SECTIONS`) covering all 68 known fields across 6 sections (inference, nn, training, sampling, backward, output), with decision tree fields folded into `nn` per D012
- `DEAD_FIELDS` dict for `backward_ecl` and `num_batches_per_set` (the only 2 genuinely dead fields per D010)
- `FIELD_ALIASES` mapping 14 readable names → internal names (learning_rate→lr, exact_computation_limit→ecl, etc.) per D007
- Auto-detection heuristic: section-name key + dict-value check
- `validate_nested_config()` with section-specific error messages
- `flatten_config()` for nested→flat translation
- `prepare_config()` as the single public entry point (D001)
- Dead field handling: warn+strip by default (`strict=False`), error on `strict=True` (D011, D014)
- Integration into `FastGM.__init__` (2-line change)
- pytest test suite with 26 acceptance tests covering R001–R006

### Out of Scope

- Changing any internal inference logic (bucket elimination, factor operations, training loops)
- Cleaning up dead fields from benchmark config files (deferred to S02)
- Config documentation guide (deferred to S03)
- Config inheritance, presets, or YAML config support
- Applying defaults during flatten (consumer code already uses `.get(key, default)` patterns)
- Handling dynamically-injected fields (`sigma_g_global`, `rho_global`) — these are set by FastGM after config processing

## Constraints

- **Pure Python only** — `config_schema.py` must not import torch or pyGMs, so it's independently importable for testing and doc generation
- **Internal flat key names preserved exactly** — all consumer code reads `self.config['ecl']`, `self.config['lr']`, etc. The translation layer maps readable names to these old internal names
- **Config dict must be mutable** — `prepare_config()` returns a plain `dict` because consumer code mutates config at runtime (num_samples override, loss_fn swap, sigma_g_global injection)
- **Backward compatibility is non-negotiable** — existing flat configs must pass through unchanged (D002, D003)
- **Polymorphic types** — `hidden_sizes` (list|str), `num_samples` (int|str), `batch_size` (int|'all'), `val_set` (bool|str|None) must all be accepted without type coercion
- **`lower_dim` lives in sampling section** — despite affecting NN input representation, it controls sample encoding (D013)

## Integration Points

### Consumes

- Nothing (first slice, no upstream dependencies)

### Produces

- `nce/config_schema.py` → `prepare_config(config_dict, strict=False) -> flat_dict` — auto-detects flat vs nested, validates, flattens, returns plain dict
- `nce/config_schema.py` → `validate_nested_config(config_dict)` — section-specific validation with clear error messages
- `nce/config_schema.py` → `flatten_config(nested_dict) -> flat_dict` — nested→flat translation
- `nce/config_schema.py` → `NESTED_SECTIONS` OrderedDict — complete field metadata (section→field→{old_name, default})
- `nce/config_schema.py` → `DEAD_FIELDS` dict — field→error message mapping
- `nce/config_schema.py` → `FIELD_ALIASES` dict — readable_name→internal_name mapping
- `nce/config_schema.py` → `SECTION_NAMES` set — for detection heuristic
- `tests/test_config_schema.py` → 26 acceptance tests covering R001–R006

## Open Questions

- **Benchmark model cache corruption** — pyGMs catalog has a Python 3 bug (`json.dump` to `'wb'` file). 2 integration tests skip when model files aren't cached. Not a config_schema issue but affects full test coverage. Current thinking: environment issue, will pass when cache is populated.
- **Strict mode flip timing** — D014 defaults to `strict=False` for the transition period. When S02 cleans benchmark configs, the default could flip to `strict=True`. Current thinking: revisit after S02 completion.
