# NCE Refactoring & Publication-Readiness Assessment

*Prepared 2026-06-12. Goal: make the NCE codebase more efficient, easier to read, and ready for
other researchers to understand and use — without disturbing the experiments currently running on the
machine.*

Companion file: [`claude_code_refactoring_research.md`](./claude_code_refactoring_research.md) — a
cited survey of how practitioners use Claude Code to refactor codebases safely. The recommendations
below follow that playbook (map → lock behavior → small atomic commits → verify).

---

## 0. How this was produced (and how to keep working safely)

**Constraint observed at start:** all 4 GPUs were running active experiments
(`run_experiment.py`, GPUs 0–3 at 47–100%). Per your instruction, I used **0% GPU** — every check
below ran CPU-only (`CUDA_VISIBLE_DEVICES=""`) and briefly.

**Clean room.** Experiments import the *live working tree* in `/home/cohenn1/NCE`, so editing files
there can change what a newly-launched experiment runs. I therefore did all inspection/testing in an
isolated mirror, **not** in the working tree:

```bash
# Faithful mirror of the working tree (INCLUDING untracked files), minus heavy/irrelevant dirs.
# This is what experiments actually run — a `git worktree` from HEAD would be STALE (see §1).
rsync -a --exclude='.git/' --exclude='venv/' --exclude='data/' --exclude='notebooks/' \
  --exclude='*.pkl' --exclude='*.pt' --exclude='*.png' --exclude='__pycache__/' \
  --exclude='nbe_eval_results/' --exclude='.gsd/' --exclude='.planning/' \
  /home/cohenn1/NCE/ /tmp/nce_clean/

# Run anything against the mirror, CPU-only, so it can't touch experiment GPUs:
cd /tmp/nce_clean && CUDA_VISIBLE_DEVICES="" PYTHONPATH=/tmp/nce_clean \
  /home/cohenn1/NCE/venv/bin/python -m pytest tests/ -q
```

> **Resource rule of thumb you gave:** ≤1% of a busy GPU is fine, ~50% is not. The safe path while
> experiments run is to stay on CPU entirely for refactor verification, and reserve GPU-touching
> numerical-equivalence tests for when the GPUs are idle.

**Test baseline (CPU-only, on the faithful mirror):** **145 passed, 7 failed, 2 skipped** in ~12 s.
The 7 failures are all in config/benchmark-config hygiene, not core math (details in §3). The core
inference / training / loss / sampling paths are green — a usable safety net to refactor against.

### Safety legend used throughout

| Tag | Meaning |
|-----|---------|
| 🟢 **SAFE** | No runtime behavior change. Comments, docstrings, dead-code/dead-import removal, packaging metadata. Do freely (you explicitly OK'd comments/usage docs). |
| 🟡 **LOW-RISK** | Mechanical behavior-adjacent change with an easy CPU check (e.g. `print`→`logging`, `tqdm.notebook`→`tqdm.auto`). Verify with the test suite. |
| 🟠 **NEEDS TESTING** | Touches numerics / control flow. Needs characterization tests + (eventually) GPU numerical-equivalence checks before/after. Do when GPUs are free. |
| 🔴 **NEEDS A DECISION** | Cannot be done correctly without your input (e.g. which sampler is canonical). Don't guess. |

---

## 1. 🔴 The #1 finding: the codebase has no single source of truth

This is the most important thing in this document, and it blocks both "publication-ready" and "safe
refactoring." **The committed code and the code experiments actually run are very different.**

Working tree vs committed `HEAD` (`176cb33`):

| | Count |
|---|---|
| Modified tracked files | 30 |
| Deleted tracked files | 19 |
| **Untracked files** | **154** |
| Untracked `.py` files **inside `nce/`** | **8** |

The 8 untracked source files live in the importable package but were **never committed**:

```
nce/sampling/no_replacement_sampler.py            (untracked)
nce/sampling/no_replacement_sampler_v2.py         (untracked)
nce/sampling/no_replacement_sampler_v3.py         (untracked)
nce/sampling/no_replacement_sampler_multilevel.py (untracked)
nce/sampling/no_replacement_sampler_multilevel_lean.py (untracked)
nce/sampling/proposal_sampler.py                  (untracked)
nce/benchmark/proposal_in_elim.py                 (untracked)
nce/utils/dtype_utils.py                          (untracked)
```

Plus 16 tracked `nce/` modules are modified beyond HEAD (e.g. `sample_generator.py`: +107/−12 lines).

**Consequences**
- A `git clone` (or `git worktree` from HEAD) yields code that **does not import the same way** and is
  missing functionality the experiments depend on. That's why my first clean checkout had only 2 files
  in `nce/sampling/` while the running code has 8.
- You cannot hand this to another researcher today — what they'd get is not what produces your results.
- Any refactor done against HEAD is refactoring stale code.

**Recommended first step (before *any* refactoring), do this when convenient:**
1. Decide, per untracked file, whether it's **canonical** (commit it) or **scratch** (move to a
   `scratch/`/`experiments/` dir outside the package, or delete).
2. Commit the 16 modified `nce/` modules (or revert intentionally).
3. After this, `git status` for `nce/` should be clean. *Then* the mirror == HEAD and refactoring is
   safe and reviewable. This is 🔴 because only you know which sampler variants are keepers.

> Tie-in to the research: practitioners' #1 safety rule is "small atomic commits on a clean tree you
> can diff and bisect." That's impossible until the tree is reconciled.

---

## 2. Publication-readiness gaps (mapped to the JOSS checklist)

These block "other researchers can understand and use this." Most are 🟢 (additive, no behavior risk).

| Item | Status | Tag | Action |
|---|---|---|---|
| **LICENSE** | ❌ missing | 🟢 | Add an OSI license (MIT/BSD-3/Apache-2.0). *Decision on which is 🔴 — your call.* |
| **`pyproject.toml`** | ❌ missing (only a 7-line `setup.py`) | 🟢 | Modern packaging w/ metadata, `[project.dependencies]`, tool config. |
| **Pinned dependencies** | ❌ none (`setup.py` has no `install_requires`) | 🟢 | List + pin `torch, numpy, pyGMs, scikit-learn, matplotlib, tqdm`. |
| **README** | ⚠️ exists (175 lines, example-focused) | 🟢 | Add: what NCE is, install, 10-line quickstart, citation. |
| **Hardcoded path to PyGMs** | ❌ `sys.path.insert(0, '/home/cohenn1/SDBE/PyGMs')` | 🟠 | **Hard blocker** — see §4.1. Won't run on any other machine. |
| **Examples** | ⚠️ partial | 🟢 | One runnable end-to-end example (small problem, CPU, fixed seed). |
| **API docs** | ❌ none | 🟢 | Sphinx + napoleon + autosummary (after docstrings, §5). |
| **CONTRIBUTING / CHANGELOG / CITATION.cff** | ❌ missing | 🟢 | Add stubs. |
| **CI** | ❌ none | 🟢 | GitHub Actions: `pytest` (CPU subset) + `ruff` + `mypy` on PRs. |
| **`src/` layout** | ❌ flat `nce/` | 🟠 | Optional but recommended (catches packaging bugs). Touches imports → test. |

---

## 3. The 7 failing tests (characterize before fixing)

All 7 failures are config-hygiene, not core algorithm:

- `test_benchmark_configs.py::TestWorkerConfigClean::*` (5 tests) — these do
  `importlib.import_module('worker')`, but **no `worker.py` exists anywhere in the repo** (tracked or
  untracked). The tests are non-hermetic; they depend on a top-level script that isn't present. 🟢 to
  fix (repair or skip the import shim), no math risk.
- `TestCleanBenchmarkConfigs::test_default_configs_no_backward_ecl` and
  `test_config_docs::test_every_schema_field_documented` — dead/undocumented config fields
  (`backward_ecl`, `num_batches_per_set` are flagged "DEAD" but still surface warnings). 🟢/🟡 to fix
  by removing dead fields + syncing `docs/config_reference.md`.

**Recommendation:** fix these *first* — a green suite is the prerequisite safety net for everything in
§4–§6. None require a GPU.

---

## 4. Code-quality findings, by safety tier

> File:line references point at the **live working tree** (what runs). A few were surfaced by an
> automated read of the working tree and tagged *"verify"* — confirm the exact line before acting.

### 4.1 🟠 Hardcoded machine paths (publication blocker)
- `nce/utils/pygms_conversion.py:60-61` — `sys.path.insert(0, '/home/cohenn1/SDBE/PyGMs')`
- `nce/utils/pygms_wmb_interface.py:14` — `PYGMS_PATH = '/home/cohenn1/SDBE/PyGMs'`
- `nce/neural_networks/decision_tree.py:191` — commented `/home/cohenn1/NCE/notebooks/...` path

**Fix:** make pyGMs a real installed dependency (it's already imported as `pyGMs`), or read the path
from an env var / config with a clear error if unset. 🟠 because import order can be subtle — test that
`import nce.utils.pygms_wmb_interface` still works on a machine without that path.

### 4.2 🟢 Dead imports & commented-out code (safe to remove)
- `nce/neural_networks/train.py:8` — `import torch.nn.functional as F` → **0 uses** (verified).
- `nce/neural_networks/train.py:13` — `import sys` → **0 uses** (verified).
- Large commented-out blocks: `train.py` (~18% comment lines), `graphical_model.py` (~10%). Removing
  pure comments/dead imports is behavior-preserving. Keep comments that explain *why*; delete
  commented-out *code*.

### 4.3 🟡 Debug `print()` → `logging` (420 occurrences in library code)
Verified **420** `print(` calls across `nce/` (excluding tests). Worst offenders (per the working-tree
scan, *verify counts*): `linear_mse_solver.py` (~101, incl. `print("🔍 BUCKET 34 SPECIFIC
DEBUGGING")`), `train.py` (~88), `benchmark/training.py` (~54, incl. a `pdb`-drop on NaN),
`graphical_model.py` (~81), `bucket.py` (~48).

**Fix:** introduce a module logger (`logger = logging.getLogger("nce.<module>")`) and convert prints to
`logger.debug/info`. 🟡 not 🟢 because a few prints may be load-bearing for downstream log parsing —
the test suite + a grep for any code that parses stdout will catch this. Do module-by-module, commit
each, run the CPU suite after each.

### 4.4 🟡 Notebook-only imports break headless use
- `nce/inference/graphical_model.py:18` and `nce/neural_networks/train.py:14` —
  `from tqdm.notebook import tqdm`. This raises/garbles output outside Jupyter.
- **Fix:** `from tqdm.auto import tqdm` (auto-selects notebook vs terminal). One-line, test-covered. 🟡.
- matplotlib imported at module top in ~6 files — move to function-level or guard for headless envs. 🟡.

### 4.5 🟠 `linear_mse_solver.py` (~1.2k LOC) function sprawl
~10 overlapping entry points: `solve_optimal_logspace_mse`, `enhanced_solve_optimal_logspace_mse`,
`fix_linear_solver_issues`, `diagnose_linear_solver_failure`, `solve_rank_deficient_system`,
`solve_with_strong_regularization`, `enhanced_linear_mse_solver_with_rank_handling`,
`debug_bucket_34_specifically` (hardcoded bucket id). Magic thresholds (`1e-10`, `1e-12`, `1e-6`,
`1e-3`) scattered inline.

**Fix:** identify the one path actually called in production, deprecate the rest, lift thresholds to
named constants. 🟠 — this is numerical code; needs characterization tests (capture solver output on a
few fixed inputs) and GPU equivalence checks before/after. **Do not** touch while experiments rely on it
without locking behavior first.

### 4.6 🔴 Five+ coexisting sampler implementations
`no_replacement_sampler{,_v2,_v3,_multilevel,_multilevel_lean}.py` + `proposal_sampler.py`. All are
*untracked* (§1). None are imported outside `nce/sampling/` by the package itself, yet the running
experiments evidently use them. **Which is canonical is a decision only you can make** → 🔴. Once
chosen: keep one (or a clearly-named small set), delete/relocate the rest, add a docstring explaining
the algorithm.

### 4.7 🟠 `graphical_model.py` (~2.1k LOC) monolith
`FastGM` is the main entry point but the file mixes orchestration, I/O, bucket construction, and
message logic. Splitting is valuable for readability but high-risk (it's the core). 🟠 — defer until
behavior is locked by tests; split along existing method clusters, one extraction per commit.

### 4.8 🟡 Misc smells
- `device='cuda'` defaulted in ~20 places (`config_schema.py`, `FastGM.__init__`, etc.). For
  publication, default to **auto-detect** (`'cuda' if torch.cuda.is_available() else 'cpu'`) — but
  **keep your benchmark configs' explicit `device` untouched** (CLAUDE.md: config fidelity). 🟡.
- `ukf_helpers.py` `FORCE_CPU` hardcoded `True` inside a function — make it a parameter. 🟡.
- 4 star-imports (`from .losses import *`, `from .stats import *`, …) — replace with explicit names for
  IDE/tooling support. 🟢 (mechanical) but run tests (could change what names are exported).
- Config alias sprawl (`ecl`/`exact_computation_limit`, `iB`/`i_bound`, …) — documented but worth a
  single canonical-name table in the README. 🟢.

---

## 5. Documentation & type hints (mostly 🟢 — you OK'd these)

- **Docstrings:** ~50% coverage on average; main classes documented, utilities sparse. Add
  **NumPy-style** docstrings to all public API (`nce/inference`, `nce/neural_networks`,
  `nce/data`). 🟢 — but **human-verify each generated docstring against the actual math** (the research
  report flags AI docstrings confidently misstating units/semantics).
- **Type hints:** ~10–15% coverage. Add to public signatures first. 🟢 if additive; wire `mypy` in CI
  so they stay honest.
- **Module-level "what/why" headers** on the big files (`graphical_model.py`, `bucket.py`,
  `train.py`, `losses.py`) — a 5–10 line orientation comment each. 🟢, high readability payoff.
- **Architecture doc** (`docs/architecture.md`): the 3-layer design + execution flow from `CLAUDE.md`
  belongs in user-facing docs. 🟢.

---

## 6. Recommended sequencing (safe-first)

Each step is independently shippable and CPU-verifiable except where noted.

**Phase 0 — make the tree real (🔴, prerequisite).** Reconcile untracked/modified `nce/` files (§1).
Decide canonical samplers (§4.6) and license (§2). *Nothing else is safe until this is done.*

**Phase 1 — green the suite (🟢/🟡).** Fix the 7 failing tests (§3); remove dead imports & commented
code (§4.2). Commit per module.

**Phase 2 — packaging & docs scaffolding (🟢).** `pyproject.toml`, pinned deps, LICENSE, README
quickstart, CONTRIBUTING/CHANGELOG/CITATION, CI running the CPU test subset + ruff. No code behavior
touched.

**Phase 3 — headless & hygiene (🟡).** `tqdm.auto`; `print`→`logging` module-by-module; device
auto-detect default (configs untouched); de-hardcode the PyGMs path (§4.1). CPU suite after each.

**Phase 4 — docstrings & type hints (🟢, human-reviewed).** Public API first; add `mypy` gate.

**Phase 5 — structural refactors (🟠, GPUs-idle only).** Before each: write **characterization tests**
that snapshot current numeric output, run them on GPU to confirm green, *then* refactor
`linear_mse_solver.py` (§4.5), split `graphical_model.py` (§4.7), consolidate samplers (§4.6). Verify
numeric equivalence before/after. These are the ones that can silently change results — do them last,
with a net, when nothing is running.

**Tooling that enforces this (from the research):** add a `Stop` hook running `ruff + mypy + pytest`
(CPU) so any refactor turn is blocked until checks pass; treat `tests/` as read-only to the agent; run
`/code-review` on each diff with "flag correctness gaps only."

---

## 7. Quick wins you can greenlight today (all 🟢/🟡, zero GPU)

1. Delete the 2 confirmed-dead imports in `train.py` (🟢).
2. `from tqdm.notebook import tqdm` → `from tqdm.auto import tqdm` in 2 files (🟡, test-covered).
3. Add `pyproject.toml` + pinned `requirements.txt` + a LICENSE of your choice (🟢).
4. Add module-header comments + a `docs/architecture.md` (🟢).
5. Fix the 5 `worker`-import tests so the suite is fully green (🟢).

Everything above was checked CPU-only against `/tmp/nce_clean`; no experiment GPU was touched.

---

*Open decisions needing your input (🔴): (a) which sampler variant(s) are canonical; (b) which license;
(c) whether to adopt a `src/` layout. I held off on all code edits pending your go-ahead — this is an
assessment, not an applied change.*
