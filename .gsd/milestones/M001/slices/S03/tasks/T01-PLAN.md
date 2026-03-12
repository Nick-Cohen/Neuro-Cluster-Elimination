---
estimated_steps: 5
estimated_files: 1
---

# T01: Write comprehensive config reference guide

**Slice:** S03 — Config Documentation
**Milestone:** M001

## Description

Create `docs/config_reference.md` — the authoritative config documentation guide for NCE. This satisfies R007 (config documentation guide) and supports R003 (field name cleanup — by documenting both readable and internal names). The guide must cover every field in `NESTED_SECTIONS` organized by section, plus dead fields, runtime fields, legacy fields, and quick-start examples with both flat and nested formats.

## Steps

1. Read `nce/config_schema.py` to extract the complete field inventory from `NESTED_SECTIONS` — readable names, internal names (old_name), and defaults. Cross-reference with the S03-RESEARCH field inventory for types and polymorphic variants.
2. Read consumer code for each section to write accurate purpose descriptions:
   - inference: `nce/inference/graphical_model.py`, `nce/inference/bucket.py`
   - nn: `nce/neural_networks/net.py`, `nce/neural_networks/decision_tree.py`
   - training: `nce/neural_networks/train.py`, `nce/neural_networks/losses.py`
   - sampling: `nce/sampling/sample_generator.py`
   - backward: `nce/inference/graphical_model.py` (backward pass sections)
   - output: various debug/display consumers
3. Write `docs/config_reference.md` with these sections:
   - Overview and quick-start (flat example, nested example, how `prepare_config` works)
   - One section per nested config section (inference, nn, training, sampling, backward, output), each with a markdown table: Readable Name | Internal Name | Type | Default | Purpose
   - Valid values section for enum-like fields: loss_fn (all 29+ values), optimizer, lr_schedule, sampling_scheme, approximation_method
   - Dead Fields section (backward_ecl, num_batches_per_set) with migration guidance
   - Runtime-Injected Fields section (sigma_g_global, rho_global) — not user-settable
   - Legacy Flat-Only Fields section (exact, memorizer) with usage context
   - Note that `NESTED_SECTIONS` in config_schema.py is the single source of truth
4. For polymorphic fields, document all accepted forms in the Type column or a sub-table: `hidden_sizes` (list[int] | str), `num_samples` (int | str), `batch_size` (int | str), `val_set` (bool | str | None), `custom_hidden_sizes` (callable | None), `display_intermediate` (bool | int)
5. Review the guide against the `NESTED_SECTIONS` schema to verify every field has an entry. Deduplicate alias pairs (e.g., `learning_rate`/`lr` should be one row showing both names, not two separate rows).

## Must-Haves

- [ ] All 6 nested sections documented with field tables
- [ ] Every unique internal field name in `NESTED_SECTIONS` has a corresponding entry
- [ ] Alias pairs (readable + internal name) shown together, not as separate entries
- [ ] Polymorphic types documented with all accepted forms
- [ ] All valid values for enum-like fields listed
- [ ] Dead fields, runtime fields, and legacy fields each in their own section
- [ ] Quick-start examples for both flat and nested config formats
- [ ] Guide references `NESTED_SECTIONS` as source of truth

## Verification

- Count unique internal field names in NESTED_SECTIONS; count field entries in the guide; numbers match
- Every polymorphic field listed in S03-RESEARCH appears with all its type variants
- Guide renders correctly as markdown (headers, tables, code blocks)

## Observability Impact

- Signals added/changed: None (documentation only)
- How a future agent inspects this: read `docs/config_reference.md`; compare field list against `NESTED_SECTIONS` keys
- Failure state exposed: None

## Inputs

- `nce/config_schema.py` — `NESTED_SECTIONS`, `DEAD_FIELDS`, `FIELD_ALIASES`, `_LEGACY_FLAT_FIELDS`
- `.gsd/milestones/M001/slices/S03/S03-RESEARCH.md` — field inventory with types, polymorphic variants, enum values
- `.gsd/milestones/M001/slices/S01/S01-RESEARCH.md` — original field audit with source file references
- Consumer code: `train.py`, `net.py`, `losses.py`, `sample_generator.py`, `graphical_model.py`, `bucket.py`

## Expected Output

- `docs/config_reference.md` — complete config reference guide covering all fields, types, defaults, purposes, valid values, and usage examples
