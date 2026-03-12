# S03: Config Documentation — Research

**Date:** 2026-03-12

## Summary

S03 owns R007 (config documentation guide) and R008 (code-level doc-sync enforcement), and supports R003 (field name cleanup — the documentation aspect).

The schema source of truth already exists: `nce/config_schema.py` defines `NESTED_SECTIONS` with 80 readable entries mapping to 66 unique internal field names across 6 sections, plus 2 dead fields (`backward_ecl`, `num_batches_per_set`) and 2 legacy flat-only fields (`exact`, `memorizer`). The field inventory was audited in S01 against every `config[` and `config.get(` callsite in the codebase. The schema is machine-readable — documentation can be generated from it, not hand-maintained separately.

R008 (doc-sync enforcement) is the interesting design question. The requirement says "adding a new field without updating the guide is detectable." Two approaches: (1) inline comments on each `_field()` call in config_schema.py that duplicate the doc, or (2) a script/test that compares the fields in the schema to the fields documented in the guide. Option 2 is strictly better — it's verifiable, doesn't clutter the schema, and catches drift in either direction. A pytest test can parse the markdown guide's field tables and compare against `NESTED_SECTIONS` keys.

## Recommendation

**Two deliverables:**

1. **`docs/config_reference.md`** — comprehensive markdown guide documenting every config field, organized by nested section. Each field gets: readable name, internal name (alias), type, default, and plain-language purpose. Include a section for dead fields, dynamically-injected fields, and a nested vs flat quick-start example.

2. **`tests/test_config_docs.py`** — doc-sync test that imports `NESTED_SECTIONS` from config_schema and parses `docs/config_reference.md` to verify every schema field appears in the guide and every guide field exists in the schema. Fails fast on drift.

**Generate from schema, not by hand.** Write a small helper (can live in the test file or as a standalone script in `scripts/`) that programmatically generates the field tables from `NESTED_SECTIONS`. The manually-written part is the "purpose" description for each field — these must be written by understanding the codebase. The structural parts (name, alias, type, default) come from the schema.

**Doc-sync enforcement via test, not comments.** R008 says "Every field definition in the codebase has a comment linking to or reproducing its documentation entry." A test that fails when schema and docs diverge is a stronger enforcement mechanism than comments, and avoids polluting config_schema.py. We add a brief module-level comment in config_schema.py pointing to the docs ("Field documentation: docs/config_reference.md") to satisfy the spirit of R008 without duplicating content.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Markdown table generation | Python f-strings / `str.format` | Simple enough — no library needed for table formatting |
| Markdown parsing for doc-sync | `re` module (regex) | Parse field names from markdown tables; no markdown library needed for structured tables |
| Schema introspection | `nce.config_schema.NESTED_SECTIONS` | Already machine-readable; import directly in tests |

## Existing Code and Patterns

- `nce/config_schema.py` — **The single source of truth.** `NESTED_SECTIONS` OrderedDict contains all field metadata. `_REQUIRED` sentinel marks required fields. `DEAD_FIELDS` dict has dead field names and messages. `FIELD_ALIASES` maps readable→internal names. `_LEGACY_FLAT_FIELDS` = `{'exact', 'memorizer'}`.
- `nce/config_schema.py:_field()` — Each field is `{'old_name': str, 'default': value}`. Note: no `type` or `description` metadata in the schema currently. Types and descriptions must be documented manually based on the S01-RESEARCH field inventory.
- `docs/` — 9 existing docs, all markdown. Convention: lowercase with underscores, topic-specific guides. `config_reference.md` fits this pattern.
- `tests/test_config_schema.py` — 28 tests for config validation. Doc-sync tests should be a separate file (`test_config_docs.py`) since they test a different contract.
- `nce/benchmark_problems/nbe_sanity_check.py:_build_nbe_nested_configs()` — Good example of a nested config in practice. Can be referenced from the docs.
- `configs/example_nn_config.py` — Minimal flat config example (outdated — only 7 fields, uses old names). Should be updated or referenced with caveats.

## Schema Field Inventory (from S01 audit + current config_schema.py)

### By section (unique internal names):
- **inference**: 5 fields — `ecl`, `iB`, `approximation_method`, `dope_factors`, `device`
- **nn**: 13 fields — `hidden_sizes`, `use_linspace_bias`, `use_memorizer`, `custom_hidden_sizes`, `init_with_linear_optimum`, `weight_decay`, `num_leaves`, `num_iterations`, `dt_lr`, `dt_momentum`, `dt_random_seed`, `dt_convergence_threshold`, `quantization_states`
- **training**: 29 fields — `num_epochs`, `num_epochs2`, `loss_fn`, `loss_fn2`, `optimizer`, `lr`, `lr_decay`, `momentum`, `batch_size`, `patience`, `min_lr`, `seed`, `skip_early_stopping`, `nbe_early_stopping`, `nbe_warmup_epochs`, `convex_early_stopping`, `convex_patience`, `convex_min_delta`, `use_validation_early_stopping`, `inverse_time_decay_constant`, `lr_schedule`, `lr_schedule_max_lr`, `lr_schedule_eta_min`, `lr_schedule_pct_start`, `grad_clip_norm`, `nbe_plateau_threshold`, `nbe_plateau_window`, `nbe_plateau_min_improvement`, `scaled_mse`
- **sampling**: 6 fields — `sampling_scheme`, `num_samples`, `set_size`, `val_set`, `stratify_samples`, `lower_dim`
- **backward**: 5 fields — `use_bw_approx`, `populate_bw_factors`, `bw_ecl`, `backward_iB`, `fdb`
- **output**: 8 fields — `debug`, `display_intermediate`, `track_errors`, `error_tracking`, `plot_messages`, `traced_losses`, `gather_message_stats`, `complexity_limit`

### Dynamically-injected fields (set by FastGM, not user-configurable):
- `sigma_g_global` — global std-dev of backward message gradient, computed during init
- `rho_global` — global correlation between forward and backward messages, computed during init

### Dead fields:
- `backward_ecl` → use `bw_ecl` instead
- `num_batches_per_set` → computed internally from `set_size // batch_size`

### Legacy flat-only fields:
- `exact` — used in bucket.py for exact-only computation mode
- `memorizer` — used in net.py when `use_memorizer=True`

### Polymorphic types requiring documentation:
- `hidden_sizes`: `list[int]` | `str` (e.g., `'nbe,3'`, `'bias_only'`)
- `num_samples`: `int` | `str` (e.g., `'nbe,0.35'`)
- `batch_size`: `int` | `str` (`'all'`)
- `val_set`: `bool` | `str` (`'all'`) | `None`
- `custom_hidden_sizes`: `callable(bucket) -> list[int]` | `None`
- `display_intermediate`: `bool` | `int` (frequency)

### Loss function values for `loss_fn` field:
29 recognized values: `logspace_mse_fdb`, `linspace_mse_fdb`, `unnormalized_kl`, `scaled_ukl`, `mse`/`MSE`, `scaled_mse`, `logspace_mse`, `l1`, `l1c`, `logspace_l1`, `gil1`, `gil1c`, `w_gil1c`, `gil1c_linear`, `gil2`, `gil2c`, `huber_gil1c`, `from_logspace_mse`, `from_logspace_l1`, `from_logspace_l2`, `from_logspace_gil2`, `combined_gil1_ls_mse`, `weighted_logspace_mse`, `weighted_logspace_mse_pedigree`, `logspace_mse_pathIS`, `ukf_sequential`, `z_err`, plus pattern-based: `approx_smg,<N>`, `elp_recompute,<N>`, `elp_loo,<N>`, `power_exponential,<alpha>`

### Optimizer values: `adam`/`Adam`, `sgd`/`SGD`, `muon`
### LR schedule values: `none`, `cosine`, `onecycle`
### Sampling scheme values: `uniform`, `all`
### Approximation method values: `nn`, `dt`

## Constraints

- The guide must be complete enough to serve as the primary config reference — this is R007's deliverable
- Doc-sync test must catch both directions of drift: field added to schema but not docs, field in docs but removed from schema
- `nce/config_schema.py` currently has no `type` or `description` metadata per field — the schema only stores `old_name` and `default`. Types/descriptions must be derived from the S01-RESEARCH audit and codebase inspection, then written into the guide manually
- The guide should use the readable alias names as primary (e.g., `learning_rate` not `lr`) since the nested config format is the forward-looking API, but must also list internal names for users reading consumer code

## Common Pitfalls

- **Duplicating the schema in prose** — If the guide hard-codes field lists independent of the schema, it will drift. The doc-sync test catches this, but we should also mention in the guide that `NESTED_SECTIONS` is the source of truth.
- **Missing polymorphic type documentation** — Several fields accept multiple types (`hidden_sizes`, `num_samples`, `batch_size`, `val_set`). The guide must document all accepted forms, not just the primary type.
- **Confusing dead fields with deprecated aliases** — `backward_ecl` is dead (not read anywhere). `ecl` is an alias for the readable name `exact_computation_limit`. These are different concepts and should be documented differently.
- **Omitting dynamically-injected fields** — `sigma_g_global` and `rho_global` appear in config reads but should not be set by users. The guide needs a "Runtime Fields" section to prevent confusion.

## Open Risks

- **S01 summary is a doctor placeholder** — the authoritative record of what was delivered is in S01 task summaries, not the slice summary. Research verified directly against config_schema.py source, so this doesn't block S03.
- **configs/example_nn_config.py is stale** — uses old flat names and has only 7 fields. S03 should either update it or add a note. Decision: probably update it to show both flat and nested forms as part of the documentation effort.

## Task Decomposition Sketch

- **T01**: Write `docs/config_reference.md` with full field reference (all 6 sections + dead/runtime/legacy), quick-start examples (flat and nested), and valid values for enum-like fields (loss_fn, optimizer, lr_schedule, sampling_scheme)
- **T02**: Write `tests/test_config_docs.py` doc-sync test + add module-level docstring/comment in config_schema.py pointing to the guide. Update `configs/example_nn_config.py` with nested example.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| Documentation generation | github/awesome-copilot@create-oo-component-documentation | Not relevant — wrong domain (OO components, not research config) |
| Code documentation | jeffallan/claude-skills@code-documenter | Not relevant — generic code docs, not config reference |

No external skills needed. This is a domain-specific documentation task using Python stdlib.

## Sources

- `nce/config_schema.py` — schema source of truth (read directly)
- `.gsd/milestones/M001/slices/S01/S01-RESEARCH.md` — complete field inventory with source file references
- S01 task summaries (T01, T02, T03) — delivery details for config_schema module
- `nce/neural_networks/train.py:_get_loss_fn()` — loss function name registry (29 recognized values)
- `nce/neural_networks/train.py:set_optimizer()` — optimizer name options
- `nce/sampling/sample_generator.py` — sampling scheme options
