# M001: Config & Visualization

**Vision:** Make configs clean, documented, and well-organized. Add rich state capture and built-in plotting that works directly on FastGM objects. Verify nothing breaks.

## Success Criteria

- User can write experiment configs using readable nested sections (inference, nn, training, sampling, backward, output)
- Old flat config dicts continue to work without any changes
- Dead config fields raise clear errors when used
- A pickled FastGM preserves per-bucket training metadata (loss curves, epochs, hidden sizes) and can be inspected/plotted in a fresh session
- A one-command regression test proves nested and flat configs produce identical inference results

## Key Risks / Unknowns

- **Config field inventory may be incomplete** — the actual set of live fields must be audited from code, not assumed from prior planning docs. Wrong inventory means wrong schema.
- **Pickle compatibility with PyTorch/pyGMs objects** — NN state dicts, data preprocessors, and pyGMs Var objects may have pickle edge cases that only surface at save/load time.
- **Loss curve data not currently preserved** — `per_bucket_training_log` only stores label/epochs/hidden_sizes; actual loss curves live in `FactorNN.losses` which gets consumed during elimination. S04 must solve this capture problem.

## Proof Strategy

- **Config field inventory** → retire in S01 by auditing every `self.config[` and `config[` reference in the codebase and building the schema from actual usage
- **Pickle compatibility** → retire in S04 by round-trip pickling a real FastGM after inference on a reference problem
- **Loss curve capture** → retire in S04 by extending bucket.py to capture loss curves into per_bucket_training_log before bucket deletion

## Verification Classes

- Contract verification: Python scripts that validate config round-trips, pickle round-trips, plotting output existence
- Integration verification: FastGM.__init__ accepts both config formats and produces identical partition function estimates
- Operational verification: none (no services)
- UAT / human verification: visual inspection of generated plots

## Milestone Definition of Done

This milestone is complete only when all are true:

- Nested configs accepted everywhere flat configs were
- Old flat configs still work without modification across all benchmark sets
- FastGM preservable and inspectable after inference (metadata always, weights optionally)
- Plotting functions produce per-NN learning curves and cross-experiment comparisons from saved state
- Config documentation guide covers every active field
- Structured log file output works during inference
- Regression test passes: nested config produces identical partition function estimate to flat config on reference problem
- Success criteria re-checked against live behavior, not just artifacts

## Requirement Coverage

- Covers: R001, R002, R003, R004, R005, R006, R007, R008, R009, R010, R011, R012, R013, R014, R015, R016, R017
- Partially covers: none
- Leaves for later: R018–R024 (M002), R025–R028 (deferred)
- Orphan risks: none

## Slices

- [x] **S01: Config Schema & Flat Translation** `risk:medium` `depends:[]`
  > After this: Pass a nested config dict to FastGM, it validates, flattens, and runs inference identically to the old flat config. Old flat configs auto-detected and still work.

- [x] **S02: Benchmark Config Migration** `risk:low` `depends:[S01]`
  > After this: All benchmark sets (nbe_sanity_check, small_problems) offer nested config builders alongside flat; experiment_config.py validates nested configs via config_schema.

- [x] **S03: Config Documentation** `risk:low` `depends:[S01]`
  > After this: A markdown guide documents every config field (type, default, purpose). Code comments at config definition sites enforce doc-sync.

- [ ] **S04: FastGM State Preservation** `risk:medium` `depends:[]`
  > After this: Pickle a FastGM after inference, unpickle in a fresh session, inspect per-bucket loss curves. Optionally save full NN weights with undo-normalization function accessible.

- [ ] **S05: Visualization Module** `risk:low` `depends:[S04]`
  > After this: Call `plot_learning_curves(fastgm)` on a pickled FastGM and get per-NN loss-over-epoch subplots. Call `compare_experiments([gm1, gm2])` for side-by-side comparison.

- [ ] **S06: Logging System** `risk:low` `depends:[]`
  > After this: Set a log file path in config, run inference, find structured per-bucket training events (epoch, loss, bucket id) in the log file.

- [ ] **S07: Regression Verification** `risk:medium` `depends:[S01,S02]`
  > After this: A one-command script translates flat→nested config, runs inference on a reference problem, confirms identical partition function estimate. Reports pass/fail.

## Boundary Map

### S01 → S02

Produces:
- `nce/config_schema.py` → `prepare_config(config_dict) -> flat_dict` (auto-detects flat vs nested, validates, flattens)
- `nce/config_schema.py` → `validate_nested_config(config_dict)` (section-specific validation errors)
- `nce/config_schema.py` → `flatten_config(nested_dict) -> flat_dict` (nested→flat translation)
- `nce/config_schema.py` → `NESTED_SECTIONS` schema definition (section names, field names, types, defaults)
- `nce/config_schema.py` → `DEAD_FIELDS` set (fields that raise errors)
- `nce/config_schema.py` → `FIELD_ALIASES` mapping (old_name→new_name for backward compat)

Consumes:
- nothing (first slice)

### S01 → S03

Produces:
- `nce/config_schema.py` → `NESTED_SECTIONS` with complete field metadata (used to generate documentation)

Consumes:
- nothing (first slice)

### S01 → S07

Produces:
- `nce/config_schema.py` → `prepare_config()` (used to translate flat config for regression comparison)

Consumes:
- nothing (first slice)

### S02 → S07

Produces:
- Updated benchmark configs with nested builders (provides reference nested configs for regression testing)

Consumes from S01:
- `config_schema.py` → `prepare_config()`, `NESTED_SECTIONS`

### S04 → S05

Produces:
- `nce/state/` module → `save_state(fastgm, path, save_weights=False)` and `load_state(path) -> state_dict`
- State dict structure with `per_bucket_training_log` including full loss curves
- Optional NN weight preservation with `undo_normalization()` function accessible
- Extended `per_bucket_training_log` entries: `{label, epochs_trained, hidden_sizes, losses, val_losses, ...}`

Consumes:
- nothing (independent track)

### S06

Produces:
- Structured log output during inference (consumed by humans, not other slices)

Consumes:
- nothing (independent slice)
