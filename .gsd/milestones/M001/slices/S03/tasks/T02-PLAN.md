---
estimated_steps: 4
estimated_files: 3
---

# T02: Add doc-sync test, schema pointer, and update example config

**Slice:** S03 — Config Documentation
**Milestone:** M001

## Description

Create `tests/test_config_docs.py` to enforce that `docs/config_reference.md` and `nce/config_schema.py:NESTED_SECTIONS` stay in sync. This satisfies R008 (code-level doc-sync enforcement). Also add a module-level doc pointer in `config_schema.py` and update the stale `configs/example_nn_config.py` with a nested config example.

## Steps

1. Write `tests/test_config_docs.py` with these test cases:
   - `test_every_schema_field_documented`: Import `NESTED_SECTIONS`, collect all unique internal field names (deduped across alias pairs). Parse `docs/config_reference.md` field tables to extract documented internal names. Assert every schema field appears in docs. On failure, name the missing fields.
   - `test_every_documented_field_in_schema`: Parse documented internal names from the guide. Assert every documented field exists in the schema. On failure, name the orphan fields.
   - `test_dead_fields_documented`: Import `DEAD_FIELDS`, verify each appears in the Dead Fields section of the guide.
   - `test_legacy_fields_documented`: Import `_LEGACY_FLAT_FIELDS`, verify each appears in the Legacy Fields section.
   - Parsing strategy: regex on markdown table rows in the guide. Each section table has an "Internal Name" column — extract those values. Use `re.findall` on table rows (lines matching `| ... | ... |` pattern).
2. Add a module-level comment/docstring line in `nce/config_schema.py` near the top (after the existing docstring) pointing to the guide: `# Field documentation: docs/config_reference.md`
3. Update `configs/example_nn_config.py` to show both formats:
   - Keep the existing flat config (rename to `flat_config`)
   - Add a `nested_config` example using the same values in nested section format
   - Add a brief comment explaining both are valid and `prepare_config()` handles either
4. Run `pytest tests/test_config_docs.py -v` and confirm all tests pass. If any fail, fix the guide or test until green.

## Must-Haves

- [ ] Doc-sync test catches missing fields in both directions (schema→docs and docs→schema)
- [ ] Test assertion messages name the specific missing field(s)
- [ ] Dead fields and legacy fields have their own sync checks
- [ ] `config_schema.py` has a doc pointer comment near the top
- [ ] `configs/example_nn_config.py` shows both flat and nested config examples
- [ ] `pytest tests/test_config_docs.py -v` passes

## Verification

- `pytest tests/test_config_docs.py -v` — all tests pass
- Intentionally remove a field from the guide, re-run test — it fails naming the missing field (manual sanity check during development)
- `python -c "from nce.config_schema import NESTED_SECTIONS"` — import still works after adding comment

## Observability Impact

- Signals added/changed: pytest test output names specific missing fields on failure
- How a future agent inspects this: run `pytest tests/test_config_docs.py -v`; failure output says exactly which fields drifted
- Failure state exposed: assertion error messages list field names and drift direction

## Inputs

- `docs/config_reference.md` — the guide created in T01 (must exist before this test can pass)
- `nce/config_schema.py` — `NESTED_SECTIONS`, `DEAD_FIELDS`, `_LEGACY_FLAT_FIELDS`
- T01 output: complete guide with field tables containing Internal Name columns

## Expected Output

- `tests/test_config_docs.py` — doc-sync enforcement test (new file)
- `nce/config_schema.py` — module-level doc pointer added (minor edit)
- `configs/example_nn_config.py` — updated with both flat and nested examples
