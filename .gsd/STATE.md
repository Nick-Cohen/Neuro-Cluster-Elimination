# GSD State

**Active Milestone:** M003 — NeuroBE Reproduction Mode
**Active Slice:** None
**Phase:** planned
**Requirements Status:** 6 active (M003) · 24 validated · 4 deferred · 4 out of scope

## Milestone Registry
- ✅ **M001:** Config & Visualization
- ✅ **M002:** Test Suite
- 🔄 **M003:** NeuroBE Reproduction Mode — roadmap written, 2 slices (S01: training mode, S02: experiments)

## Recent Decisions
- D026: Extend DataPreprocessor with normalization_mode param, not separate class
- D027: New neurobe_weighted_mse loss function, not modifying existing
- D028: neurobe_mode as config expansion in prepare_config
- D029: Fold ReLU/AMP/hidden-dim into R035 preset
- D030: Two-slice structure — machinery+tests first, experiments second
- D031: ecl = 2^width_problem - 1 for binary-domain matching

## Blockers
- None

## Next Action
Execute S01: NeuroBE Training Mode (normalization, early stopping, config preset, activation, loss fn, tests)
