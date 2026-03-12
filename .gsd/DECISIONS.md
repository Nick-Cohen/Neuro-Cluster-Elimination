# Decisions Register

<!-- Append-only. Never edit or remove existing rows.
     To reverse a decision, add a new row that supersedes it.
     Read this file at the start of any planning or research phase. -->

| # | When | Scope | Decision | Choice | Rationale | Revisable? |
|---|------|-------|----------|--------|-----------|------------|
| D001 | M001 | arch | Config translation boundary | `prepare_config()` in config_schema.py, not FastGM.__init__ | Clean separation — FastGM calls one function, gets flat dict | No |
| D002 | M001 | arch | Backward compat strategy | Auto-detect flat vs nested, pass flat through, translate nested | Existing scripts/notebooks must not break | No |
| D003 | M001 | arch | Internal config representation | Keep flat dict internally, translate at entry point only | Zero consumer code changes (bucket.py, train.py, etc.) | Yes — if full codebase migration justified |
| D004 | M001 | arch | State preservation module | New `nce/state/` module, separate from FastGM internals | User requested modular/separable design | No |
| D005 | M001 | arch | Default save mode | Training metadata only (loss curves, epochs, hidden sizes), no NN weights | Lightweight default; weights optional via flag | Yes — if weights become commonly needed |
| D006 | M001 | arch | Visualization module location | New `nce/visualization/` module (not extending nce/utils/plots.py) | Distinct capability deserves distinct module; existing plots.py is factor-comparison focused | No |
| D007 | M001 | convention | Config field name mapping | lr→learning_rate, ecl→exact_computation_limit, iB→i_bound, fdb→forward_diff_barrier | Readable names; old names accepted as aliases | No |
| D008 | M001 | arch | Loss curve capture | Extend per_bucket_training_log to include full loss curves before bucket deletion | Currently only label/epochs/hidden_sizes; loss curves live in FactorNN.losses which gets consumed | No |
| D009 | M002 | convention | Test framework | pytest in tests/ directory | Standard Python testing; extensible fixture/parametrize pattern | No |
