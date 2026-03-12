# S01 Post-Slice Roadmap Assessment

## Verdict: Roadmap unchanged

S01 delivered exactly what the boundary map specified. No slice reordering, merging, splitting, or scope changes needed.

## What S01 Built

- `nce/config_schema.py`: `prepare_config()`, `validate_nested_config()`, `flatten_config()`, `NESTED_SECTIONS` (68 fields across 6 sections), `DEAD_FIELDS` (2), `FIELD_ALIASES` (14 mappings)
- `FastGM.__init__` wired to call `prepare_config()` — 2-line change in `graphical_model.py`
- 26 passing tests (2 skipped due to pyGMs model cache env issue, not code)
- Requirements R001–R006 addressed

## Risk Retirement

- **Config field inventory** (medium risk): Retired. Code audit found 68 fields, corrected prior plan's dead-field list (D010). Schema built from actual usage, not assumptions.
- **Pickle compatibility**: Still open → S04
- **Loss curve capture**: Still open → S04

## Boundary Contract Verification

All S01 exports match the boundary map exactly:
- `prepare_config(config_dict) -> flat_dict` ✅
- `validate_nested_config(config_dict)` ✅
- `flatten_config(nested_dict) -> flat_dict` ✅
- `NESTED_SECTIONS` schema definition ✅
- `DEAD_FIELDS` set ✅
- `FIELD_ALIASES` mapping ✅

## Success Criteria Coverage

- User can write experiment configs using readable nested sections → S01 ✅ (done)
- Old flat config dicts continue to work without any changes → S01 ✅ (done)
- Dead config fields raise clear errors when used → S01 ✅ (done)
- A pickled FastGM preserves per-bucket training metadata and can be inspected/plotted → S04, S05
- A one-command regression test proves nested and flat configs produce identical inference results → S07

All criteria have at least one owning slice. Coverage check passes.

## Requirement Coverage

R001–R006 addressed by S01. R007–R017 ownership unchanged (S02–S07). No requirement status changes needed.

## New Information

- D010: Dead field list corrected — only `backward_ecl` and `num_batches_per_set` are dead
- D011: Dual handling — warn+strip in flat, error in nested
- D012: DT fields in `nn` section with `dt_` prefix
- D013: `lower_dim` in sampling section
- D014: `strict=False` default, flippable after S02 cleans benchmark configs
- pyGMs model catalog has a Python 3 bug (`json.dump` to `'wb'` file). Affects benchmark model loading in some envs. Not a config_schema issue — S02 should account for this when working with benchmark configs.

## Next Slice

S02 (Benchmark Config Migration) is unblocked. S04 (State Preservation) is also independently unblocked.
