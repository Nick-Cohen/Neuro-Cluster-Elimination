---
id: T01
parent: S03
milestone: M001
provides:
  - docs/config_reference.md — comprehensive config reference guide for all NCE config fields
key_files:
  - docs/config_reference.md
key_decisions:
  - Organized guide by nested section (inference, nn, training, sampling, backward, output) with one markdown table per section showing readable name, internal name, type, default, and purpose
  - Polymorphic fields documented inline in the type column plus dedicated sub-tables for each polymorphic field
  - Loss functions organized by category (log-space, linear-space, KL, geometric, message gradient, parameterized) rather than alphabetically
patterns_established:
  - Section field tables use 5-column format: Readable Name | Internal Name | Type | Default | Purpose
  - Enum-like fields get their own subsection under "Valid Values for Enum-Like Fields" with 2-column tables
  - Polymorphic fields get sub-tables immediately below their section showing each accepted form
observability_surfaces:
  - none (documentation only)
duration: 35m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T01: Write comprehensive config reference guide

**Created `docs/config_reference.md` — authoritative reference for all 66 unique config fields across 6 sections, plus dead fields, runtime fields, legacy fields, and quick-start examples.**

## What Happened

Read `nce/config_schema.py` to extract the complete `NESTED_SECTIONS` field inventory (80 readable entries mapping to 66 unique internal names). Cross-referenced with consumer code (`graphical_model.py`, `bucket.py`, `train.py`, `net.py`, `decision_tree.py`, `sample_generator.py`, `losses.py`) to write accurate purpose descriptions for every field.

Wrote `docs/config_reference.md` with:
- Quick-start section showing both flat (legacy) and nested (recommended) config formats, plus how `prepare_config()` works
- 6 section tables (inference, nn, training, sampling, backward, output) — each field row shows readable name, internal alias, type with polymorphic variants, default, and purpose derived from codebase inspection
- Sub-tables for polymorphic fields: `hidden_sizes` (4 forms), `num_samples` (2 forms), `batch_size` (2 forms), `val_set` (4 forms)
- Complete loss function reference (29+ values) organized by category
- Valid values tables for optimizer (3), lr_schedule (3), sampling_scheme (2), approximation_method (2)
- Dead Fields section (2 fields with migration guidance)
- Runtime-Injected Fields section (2 fields: `sigma_g_global`, `rho_global`)
- Legacy Flat-Only Fields section (2 fields: `exact`, `memorizer`)

## Verification

1. **Field count match**: Programmatic cross-reference confirmed all 66 unique internal field names in `NESTED_SECTIONS` have entries in the guide, per section: inference (5), nn (13), training (29), sampling (6), backward (5), output (8).
2. **Alias pairs**: Verified readable + internal names appear together (e.g., `exact_computation_limit` | `ecl`).
3. **Polymorphic types**: All 5 polymorphic fields documented with all accepted forms.
4. **Enum values**: All valid values for loss_fn, optimizer, lr_schedule, sampling_scheme, approximation_method present.
5. **Structural sections**: Dead fields ✓, Runtime-injected fields ✓, Legacy flat-only fields ✓, Quick-start (flat + nested) ✓, NESTED_SECTIONS referenced as source of truth ✓.
6. **Schema importable**: `python -c "from nce.config_schema import NESTED_SECTIONS; print('schema importable')"` passes.

Slice-level verification status:
- `pytest tests/test_config_docs.py -v` — **cannot run yet** (test file is T02's deliverable)
- Schema importable — **passes**
- Visual inspection — guide is well-organized with accurate descriptions

## Diagnostics

Read `docs/config_reference.md`. To verify field completeness against schema, run:
```python
from nce.config_schema import NESTED_SECTIONS
# Count unique internal names
total = set()
for fields in NESTED_SECTIONS.values():
    for fdef in fields.values():
        total.add(fdef['old_name'])
print(len(total))  # Should be 66
```

## Deviations

None.

## Known Issues

- `nbe_plateau_threshold`, `nbe_plateau_window`, `nbe_plateau_min_improvement` are documented as "currently unused in active code" — the config schema defines them but the consumer code has them commented out. Documented as-is.

## Files Created/Modified

- `docs/config_reference.md` — new, comprehensive config reference guide (386 lines, 23.7KB)
