# S03: Config Documentation

**Goal:** A markdown guide documents every config field (type, default, purpose) and a test enforces schema-doc sync.
**Demo:** `pytest tests/test_config_docs.py -v` passes, confirming every field in `NESTED_SECTIONS` appears in `docs/config_reference.md` and vice versa. The guide is readable, complete, and organized by section.

## Must-Haves

- `docs/config_reference.md` covers all 6 nested sections with every field's readable name, internal name, type, default, and purpose
- Polymorphic types fully documented (e.g., `hidden_sizes` accepts `list[int]` or `str`)
- Enum-like fields list all valid values (`loss_fn`, `optimizer`, `lr_schedule`, `sampling_scheme`, `approximation_method`)
- Dead fields, runtime-injected fields, and legacy flat-only fields documented in separate sections
- Quick-start examples showing both flat and nested config formats
- `tests/test_config_docs.py` catches drift in both directions: field in schema but not docs, field in docs but not schema
- Module-level comment in `config_schema.py` points to the guide
- `configs/example_nn_config.py` updated with nested config example alongside flat

## Proof Level

- This slice proves: contract (documentation completeness verified by automated test against schema)
- Real runtime required: no (test imports schema and parses markdown — no inference needed)
- Human/UAT required: yes (visual review that descriptions are accurate and guide is usable)

## Verification

- `pytest tests/test_config_docs.py -v` — all assertions pass (every schema field in docs, every doc field in schema)
- `python -c "from nce.config_schema import NESTED_SECTIONS; print('schema importable')"` — confirms schema pointer comment doesn't break import
- Visual inspection: `docs/config_reference.md` is well-organized with accurate field descriptions

## Observability / Diagnostics

- Runtime signals: none (documentation slice, no runtime behavior)
- Inspection surfaces: `pytest tests/test_config_docs.py -v` output shows exactly which fields are missing from either side
- Failure visibility: test assertion messages name the specific missing field(s) and direction of drift
- Redaction constraints: none

## Integration Closure

- Upstream surfaces consumed: `nce/config_schema.py` → `NESTED_SECTIONS`, `DEAD_FIELDS`, `FIELD_ALIASES`, `_LEGACY_FLAT_FIELDS` (all from S01)
- New wiring introduced in this slice: none (documentation + test, no runtime changes)
- What remains before the milestone is truly usable end-to-end: S04 (state preservation), S05 (visualization), S06 (logging), S07 (regression verification)

## Tasks

- [ ] **T01: Write comprehensive config reference guide** `est:1h`
  - Why: R007 — the config is the primary user interface; every field needs type, default, and purpose documented. This is the core deliverable of S03.
  - Files: `docs/config_reference.md`
  - Do: Write full markdown guide organized by nested section. For each field: readable name, internal alias, type (with polymorphic variants), default, purpose. Include sections for dead fields, runtime-injected fields, legacy flat-only fields. Add quick-start examples (flat and nested). Document all valid values for enum-like fields. Generate structural parts from schema; write purpose descriptions from codebase knowledge.
  - Verify: `python -c "open('docs/config_reference.md').read()"` confirms file exists and is non-empty; manual review of completeness against `NESTED_SECTIONS`
  - Done when: every field in `NESTED_SECTIONS` has an entry in the guide with type, default, and purpose

- [ ] **T02: Add doc-sync test, schema pointer, and update example config** `est:45m`
  - Why: R008 — documentation drifts without enforcement. The test catches additions to schema or docs that aren't mirrored. Also updates the stale example config.
  - Files: `tests/test_config_docs.py`, `nce/config_schema.py`, `configs/example_nn_config.py`
  - Do: Write pytest test that imports `NESTED_SECTIONS` and parses field tables from `docs/config_reference.md`. Test both directions: schema→docs and docs→schema. Add module-level docstring/comment in `config_schema.py` pointing to `docs/config_reference.md`. Update `configs/example_nn_config.py` with both flat and nested examples.
  - Verify: `pytest tests/test_config_docs.py -v` passes
  - Done when: doc-sync test passes, schema has doc pointer, example config shows both formats

## Files Likely Touched

- `docs/config_reference.md` (new)
- `tests/test_config_docs.py` (new)
- `nce/config_schema.py` (add doc pointer comment)
- `configs/example_nn_config.py` (update with nested example)
