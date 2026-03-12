# S01: Config Schema & Flat Translation — Research

**Date:** 2026-03-12

## Summary

The config system is a flat dict of ~55 distinct fields passed to `FastGM.__init__` as `nn_config`, then shared by reference through `bucket.gm.config` to every consumer (Trainer, SampleGenerator, Net, DecisionTree, backward_message utils). This single-entry-point architecture makes S01 clean: we add `prepare_config()` at `FastGM.__init__` and every downstream consumer gets a validated flat dict without any code changes.

The field audit revealed 55 live config fields across 6 files, 2 genuinely dead fields (`backward_ecl` as a config key — code uses `bw_ecl` instead; `num_batches_per_set` — never read from config, computed locally from `set_size // batch_size`), and 2 dynamically-injected fields (`sigma_g_global`, `rho_global`) that are set by FastGM during init and should not appear in user-facing configs. The prior plan's "dead field" list was partially wrong: `loss_fn2`, `num_epochs2`, `complexity_limit`, `exact`, and `memorizer` are all live, used in active code paths. Only `backward_ecl` and `num_batches_per_set` are truly dead (present in benchmark config but never read).

The nested→flat translation is straightforward dict manipulation with no heavy dependencies. The main design challenge is the detection heuristic for flat vs nested configs, field alias mapping, and handling the ~10 fields with non-obvious types (e.g., `hidden_sizes` can be a list, string like `'nbe,3'`, or `'bias_only'`; `batch_size` can be int or `'all'`; `num_samples` can be int or string like `'nbe,0.35'`).

## Recommendation

Build `nce/config_schema.py` as a pure-Python module (no torch, no pyGMs) with:

1. **Schema dict** mapping section→field→{old_name, type, default, required} for all 55 live fields across 6 sections (inference, nn, training, sampling, backward, output)
2. **`prepare_config(config_dict)`** as the single public entry point: detects flat vs nested, validates, flattens nested to flat, passes flat through with validation, returns a new flat dict. This is the function FastGM calls.
3. **`DEAD_FIELDS`** dict mapping dead field names to error messages suggesting the correct alternative
4. **`FIELD_ALIASES`** mapping new readable names → old internal names (e.g., `learning_rate` → `lr`)

Integration: one 3-line change in `FastGM.__init__` — import `prepare_config`, call it on `nn_config`, store result as `self.config`. Zero changes to bucket.py, train.py, sample_generator.py, net.py, decision_tree.py.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Config validation | Python stdlib `typing` + manual checks | No external deps needed; pydantic/attrs would add dependency for simple dict validation |
| Type checking | `isinstance()` | Config types are simple (int, float, bool, str, list, None); no need for schema libraries |

## Existing Code and Patterns

- `nce/inference/graphical_model.py:36` — `self.config = dict(nn_config) if nn_config else {}` — the single entry point where `prepare_config()` will be inserted
- `nce/inference/graphical_model.py:38-74` — Fields extracted from config into instance attrs (iB, ecl, sampling_scheme, etc.) — these reads happen AFTER config is stored, so prepare_config runs first
- `nce/inference/bucket.py:14` — `self.config = gm.config` — bucket gets config by reference from gm; no separate config handling needed
- `nce/neural_networks/train.py:69` — `self.config = bucket.gm.config` — trainer gets config the same way
- `nce/sampling/sample_generator.py:13` — `self.config = gm.config` — same reference pattern
- `nce/neural_networks/net.py:22-27` — Net reads `nn_config['hidden_sizes']`, `nn_config['device']`, `nn_config.get('seed')`, `nn_config.get('use_linspace_bias')`, and checks `'memorizer' in nn_config` — this is the only consumer that receives config as a parameter (not via gm reference)
- `nce/benchmark_problems/nbe_sanity_check.py:83-127` — Reference flat config template with 42 fields; some dead fields included
- `notebooks/_1-2026/experiment_config.py` — YAML config loader with its own validation; operates at experiment level (problem, loss, architectures, bw_ecl, gpus) not nn_config level; may optionally integrate with config_schema later (S02)
- `.planning/phases/05-config-restructure/05-01-PLAN.md` — Prior detailed plan; useful reference but has incorrect dead-field analysis and some naming choices that need validation against the actual code audit

## Complete Config Field Inventory

### Section: inference (5 fields)

| New Name | Old/Internal Name | Type | Default | Source |
|----------|------------------|------|---------|--------|
| `exact_computation_limit` | `ecl` | int | 0 | graphical_model.py:39, bucket.py:127,241,450 |
| `i_bound` | `iB` | int | 0 | graphical_model.py:38, bucket.py:126,240,449 |
| `approximation_method` | `approximation_method` | str | `'nn'` | graphical_model.py:406-427 |
| `dope_factors` | `dope_factors` | bool | False | graphical_model.py:119 |
| `device` | `device` | str | `'cuda'` | train.py:148,705,884,917,1251,1289; net.py:23 |

### Section: nn (6 fields)

| New Name | Old/Internal Name | Type | Default | Source |
|----------|------------------|------|---------|--------|
| `hidden_sizes` | `hidden_sizes` | list\|str | `[]` | graphical_model.py:62, bucket.py:200,206, net.py:22 |
| `use_linspace_bias` | `use_linspace_bias` | bool | False | net.py:25 |
| `use_memorizer` | `use_memorizer` | bool | False | bucket.py:80 |
| `custom_hidden_sizes` | `custom_hidden_sizes` | callable\|None | None | bucket.py:200 |
| `init_with_linear_optimum` | `init_with_linear_optimum` | bool | False | bucket.py:706 |
| `weight_decay` | `weight_decay` | float | 0.0 | bucket.py:575,715 |

### Section: training (28 fields)

| New Name | Old/Internal Name | Type | Default | Source |
|----------|------------------|------|---------|--------|
| `num_epochs` | `num_epochs` | int | — (required) | train.py:250,120; bucket.py (via trainer) |
| `num_epochs_phase2` | `num_epochs2` | int | 0 | bucket.py:324-325 |
| `loss_fn` | `loss_fn` | str | — (required when approx_method=nn) | train.py:145,168-169,290; bucket.py:324 |
| `loss_fn_phase2` | `loss_fn2` | str\|None | None | bucket.py:324-325, train.py:290, graphical_model.py:121 |
| `optimizer` | `optimizer` | str | `'adam'` | train.py:92,112,802,849 |
| `learning_rate` | `lr` | float | 0.001 | train.py:130,1262,1280 |
| `learning_rate_decay` | `lr_decay` | float | 1.0 | graphical_model.py:66 |
| `momentum` | `momentum` | float | 0.9 | train.py:1281 |
| `batch_size` | `batch_size` | int\|str(`'all'`) | 256 | train.py:198,245-262,1287 |
| `patience` | `patience` | int | 20 | graphical_model.py:67 |
| `min_learning_rate` | `min_lr` | float | 1e-8 | graphical_model.py:68 |
| `seed` | `seed` | int | 42 | train.py:907; net.py:24 |
| `skip_early_stopping` | `skip_early_stopping` | bool | False | train.py:509,558 |
| `nbe_early_stopping` | `nbe_early_stopping` | bool | False | train.py:187 |
| `nbe_warmup_epochs` | `nbe_warmup_epochs` | int | 0 | train.py:231 |
| `convex_early_stopping` | `convex_early_stopping` | bool | False | train.py:63 |
| `convex_patience` | `convex_patience` | int | 20 | train.py:176 |
| `convex_min_delta` | `convex_min_delta` | float | 1e-8 | train.py:177 |
| `use_validation_early_stopping` | `use_validation_early_stopping` | bool | False | train.py:239 |
| `inverse_time_decay_constant` | `inverse_time_decay_constant` | int | 100 | train.py:107 |
| `lr_schedule` | `lr_schedule` | str | `'none'` | train.py:111 |
| `lr_schedule_max_lr` | `lr_schedule_max_lr` | float\|None | None | train.py:130 |
| `lr_schedule_eta_min` | `lr_schedule_eta_min` | float | 1e-6 | train.py:124 |
| `lr_schedule_pct_start` | `lr_schedule_pct_start` | float | 0.1 | train.py:131 |
| `gradient_clip_norm` | `grad_clip_norm` | float\|None | None | train.py:844,1309 |
| `scaled_mse` (no rename) | — | — | — | Special: train.py:290 checks if loss_fn is `'scaled_mse'` |
| `nbe_plateau_threshold` | `nbe_plateau_threshold` | float | 0.1 | train.py:644 (commented out but present) |
| `nbe_plateau_window` | `nbe_plateau_window` | int | 25 | train.py:645 (commented out) |
| `nbe_plateau_min_improvement` | `nbe_plateau_min_improvement` | float | 0.01 | train.py:646 (commented out) |

**Note:** The 3 `nbe_plateau_*` fields are in commented-out code (train.py:644-646). Include in schema with defaults for forward compat but mark as experimental.

### Section: sampling (6 fields)

| New Name | Old/Internal Name | Type | Default | Source |
|----------|------------------|------|---------|--------|
| `sampling_scheme` | `sampling_scheme` | str | `'uniform'` | graphical_model.py:59, sample_generator.py:20 |
| `num_samples` | `num_samples` | int\|str | — (required) | train.py:199,226,245; bucket.py:101,108,222,229 |
| `set_size` | `set_size` | int\|None | None | train.py:199,251; graphical_model.py:72 |
| `val_set` | `val_set` | bool\|str(`'all'`)\|None | True | train.py:924-926 |
| `stratify_samples` | `stratify_samples` | bool | False | train.py:201,352 |
| `lower_dim` | `lower_dim` | bool | False | graphical_model.py:61, train.py:74; decision_tree.py:385 |

**Note:** `lower_dim` controls how samples are encoded (lower-dimensional representation). Despite the name it's a sampling concern, not an output concern.

### Section: backward (5 fields)

| New Name | Old/Internal Name | Type | Default | Source |
|----------|------------------|------|---------|--------|
| `use_backward_approximation` | `use_bw_approx` | bool | False | bucket.py:122,236,448; train.py:909 |
| `populate_backward_factors` | `populate_bw_factors` | bool | False | graphical_model.py:57 |
| `backward_ecl` | `bw_ecl` | int\|None | None | bucket.py:127,241,450; graphical_model.py:1394; message_gradient_factors.py:116 |
| `backward_i_bound` | `backward_iB` | int\|None | None | bucket.py:126,240,449 |
| `forward_diff_barrier` | `fdb` | bool | False | sample_generator.py:155; decision_tree.py:384 |

### Section: output (8 fields)

| New Name | Old/Internal Name | Type | Default | Source |
|----------|------------------|------|---------|--------|
| `debug` | `debug` | bool | False | train.py:78,178,180,597 |
| `display_intermediate` | `display_intermediate` | bool\|int | False | train.py:305,391 |
| `track_errors` | `track_errors` | bool | False | graphical_model.py:53 |
| `error_tracking` | `error_tracking` | bool | False | train.py:332 |
| `plot_messages` | `plot_messages` | bool | False | bucket.py:83,401,488,532 |
| `traced_losses` | `traced_losses` | list | `[]` | graphical_model.py:60, train.py:184 |
| `gather_message_stats` | `gather_message_stats` | bool | False | graphical_model.py:74 |
| `complexity_limit` | `complexity_limit` | int | 0 | graphical_model.py:40,392 |

### Section: decision_tree (7 fields, only used when approximation_method='dt')

| New Name | Old/Internal Name | Type | Default | Source |
|----------|------------------|------|---------|--------|
| `num_leaves` | `num_leaves` | int | (computed) | decision_tree.py:57 |
| `num_iterations` | `num_iterations` | int | (computed) | decision_tree.py:59 |
| `dt_learning_rate` | `dt_lr` | float | (default in DT) | decision_tree.py:60 |
| `dt_momentum` | `dt_momentum` | float | (default in DT) | decision_tree.py:61 |
| `dt_random_seed` | `dt_random_seed` | int | (default in DT) | decision_tree.py:62 |
| `dt_convergence_threshold` | `dt_convergence_threshold` | float | (default in DT) | decision_tree.py:63 |
| `quantization_states` | `quantization_states` | int | ecl | graphical_model.py:413 |

**Design decision needed:** Should decision_tree fields be a subsection of `nn` or a separate 7th section? Since they're only relevant when `approximation_method='dt'`, I recommend putting them in the `nn` section with a `dt_` prefix, or as a flat subsection. The prior plan used 6 sections — keeping 6 and folding dt fields into `nn` is simpler.

### Dynamically-injected fields (NOT user-configurable)

These are set by FastGM during initialization, not by users:
- `sigma_g_global` — set by `populate_global_stats()` (graphical_model.py:1295)
- `rho_global` — set by `populate_global_stats()` (graphical_model.py:1296)

These should NOT appear in the schema or be accepted in user configs. `prepare_config` should ignore them (they're injected after config processing).

### Dead fields (in benchmark config but never read from config)

| Field | In Config | Actually Read? | Suggestion |
|-------|-----------|----------------|------------|
| `backward_ecl` | nbe_sanity_check.py:113 | No — code reads `bw_ecl` | Remove from benchmark configs; use `bw_ecl` |
| `num_batches_per_set` | nbe_sanity_check.py:102 | No — computed as local var from `set_size // batch_size` | Remove from benchmark configs |

### Legacy fields (live but niche)

| Field | Used Where | Notes |
|-------|------------|-------|
| `exact` | bucket.py:47,50 | Debug-only numel counter; defaults to False |
| `memorizer` | net.py:26-27 | Sets `self.memorizer` on Net; `use_memorizer` controls whether to USE it (bucket.py:80) |
| `complexity_limit` | graphical_model.py:40,392 | Alternative to ecl for exact computation threshold; defaults to 0 (disabled) |

## Constraints

- **No torch/pyGMs imports in config_schema.py** — it must be importable independently for testing and documentation generation
- **Internal flat key names must be preserved exactly** — all consumer code uses `self.config['ecl']`, `self.config['lr']`, etc. The internal names are the old names.
- **Config dict is mutated during execution** — `num_samples` is overwritten in bucket.py:108,229 when NBE sampling computes actual counts; `loss_fn` is temporarily swapped in train.py:169,696; `sigma_g_global`/`rho_global` are injected. `prepare_config` must return a regular dict (not a frozen/validated wrapper).
- **`nn_config` is also passed directly to `Net.__init__`** as `nn_config` parameter (not via `gm.config`) — net.py accesses `nn_config['hidden_sizes']`, `nn_config['device']`, `nn_config.get('seed')`. The flat dict from `prepare_config` will work here since Net reads old-name keys.
- **`hidden_sizes` has polymorphic type** — can be `list` (e.g., `[64, 64]`), `str` (e.g., `'nbe,3'` for NBE-computed sizes), or `'bias_only'` (special mode). Type validation must accommodate all three.
- **`num_samples` has polymorphic type** — can be `int` or `str` (e.g., `'nbe,0.35'` for NBE-computed sample count). 
- **`batch_size` has polymorphic type** — can be `int` or `'all'` string.
- **`val_set` has polymorphic type** — can be `bool`, `str` (`'all'`), or `None`.

## Common Pitfalls

- **Incorrect dead field identification** — Prior plan listed `num_epochs2`, `loss_fn2`, `complexity_limit`, `exact`, `memorizer` as dead. Code audit shows all 5 are live. Only `backward_ecl` (config key, not param) and `num_batches_per_set` are truly dead. Must audit from code, not from assumptions.
- **Breaking config mutation** — Several code paths write back to `self.config` (num_samples override, loss_fn swap, sigma_g_global injection). If `prepare_config` returns anything other than a plain dict, these writes will break. Do NOT use frozen dicts, dataclasses, or validated wrappers.
- **Flat detection false positives** — If a user has a flat config with a key called `'training'` or `'inference'`, the detector would misclassify it as nested. Mitigation: check if the value is a dict (nested sections have dict values; flat fields named `'training'` would have non-dict values). Edge case: no current flat config field overlaps with section names.
- **Net receives nn_config directly** — `Net.__init__` receives the config dict as `nn_config` parameter from `bucket.py`. This is the SAME dict object (`gm.config`), so it will already be flattened. But if someone constructs a Net directly with a nested config, it would fail. Acceptable — Net is internal API.
- **Decision tree fields share config space** — DT-specific fields (`dt_lr`, `num_leaves`, etc.) use the same flat config dict. Schema should include them even though they're only used with `approximation_method='dt'`.

## Open Risks

- **Benchmark config cleanup creates a diff** — Removing `backward_ecl` and `num_batches_per_set` from `nbe_sanity_check.py` is desirable but technically out of S01 scope (S02 handles benchmark migration). S01 should add them to DEAD_FIELDS with clear error messages so S02 can clean them up. Actually, since they're in benchmark configs that users currently pass to FastGM, marking them as dead would break existing usage. Better: add them to a DEPRECATED_FIELDS set that warns but doesn't error, or simply omit them from the schema (unknown flat fields pass through for backward compat).
- **Section count: 6 vs 7** — Decision tree fields don't fit cleanly into any of the 6 planned sections. Options: (a) fold into `nn` section, (b) add 7th `decision_tree` section, (c) leave as flat-only fields. Recommend (a) with `dt_` prefix since DT is an approximation method alternative to NN.
- **`lower_dim` placement** — Affects sample encoding (sampling concern) but semantically about the NN input representation. Could go in `nn` or `sampling`. Prior plan put it in `output` which is wrong. Recommend `sampling` since it's consumed in the sampling/preprocessing pipeline.

## Flat vs Nested Detection Strategy

```
is_nested = any(
    key in config and isinstance(config[key], dict)
    for key in SECTION_NAMES
)
```

This is safe because:
1. No existing flat config field has a name matching a section name (`inference`, `nn`, `training`, `sampling`, `backward`, `output`)
2. Even if a flat config had such a key, its value would be a scalar/list, not a dict
3. Checking `isinstance(config[key], dict)` adds a second guard

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| Python configuration | `wshobson/agents@python-configuration` (2.4K installs) | Available but unnecessary — work is simple dict manipulation |
| Pydantic | `bobmatnyc/claude-mpm-skills@pydantic` (1.6K installs) | Available but adds dependency; not worth it for this scope |

No skills recommended for installation. The work is pure Python dict validation — no external libraries needed.

## Sources

- Config field inventory: exhaustive `grep` audit of `self.config[`, `self.config.get(`, `config.get(`, `config[`, and `nn_config[` across all `nce/**/*.py` files
- Prior plans: `.planning/phases/05-config-restructure/05-01-PLAN.md` and `05-02-PLAN.md` — useful structure reference but dead-field analysis was incorrect
- Benchmark config template: `nce/benchmark_problems/nbe_sanity_check.py:83-127` — canonical 42-field flat config
- Decision D001-D008 in `.gsd/DECISIONS.md` — architectural choices for config translation boundary, backward compat, internal representation
