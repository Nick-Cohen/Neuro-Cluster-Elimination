---
phase: quick-19
plan: 19
type: execute
wave: 1
depends_on: []
files_modified:
  - docs/model_cache_setup.md
autonomous: true
requirements: []

must_haves:
  truths:
    - "The root cause of the hang is documented with enough detail to reproduce the fix"
    - "Future agents know to pre-populate .model_cache/ before running playground.py or any FastGM on catalog models"
  artifacts:
    - path: "docs/model_cache_setup.md"
      provides: "Documented fix and pre-caching instructions for grids models"
  key_links:
    - from: "playground.py / FastGM"
      to: ".model_cache/grids/"
      via: "pyGMs catalog lazy-loader reads UAI, ORD, EVID files from cache"
      pattern: "model_cache.*grids"
---

<objective>
Document the operational fix for playground.py hanging indefinitely on FastGM creation.

Purpose: The hang was caused by pyGMs catalog lazy-loading model files from an unreachable server (sli.ics.uci.edu) with no request timeout. The fix was to manually populate .model_cache/grids/ with the three required files. This plan records that fix and creates a pre-caching guide so future agents do not re-encounter this issue.

Output: docs/model_cache_setup.md explaining root cause, fix applied, and how to pre-cache any future catalog model.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@/home/cohenn1/NCE/.planning/STATE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Document root cause and fix in docs/model_cache_setup.md</name>
  <files>docs/model_cache_setup.md</files>
  <action>
    Create docs/model_cache_setup.md documenting the following:

    **Section 1 — Problem**
    playground.py (and any script calling FastGM on a pyGMs catalog model) hung indefinitely on FastGM creation. No error, no output, no timeout — just a frozen process.

    **Section 2 — Root Cause**
    The pyGMs catalog uses lazy-loading: when a Model object is instantiated from the catalog, it checks a local cache directory (.model_cache/) for the UAI file, elimination order file (.uai.ord), and evidence file (.uai.evid). If any file is missing or empty, pyGMs calls requests.get() to download from sli.ics.uci.edu. That server is unreachable from this machine, and requests.get() has no timeout configured — causing an indefinite hang at the network call.

    **Section 3 — Fix Applied (2026-03-09)**
    Three files were missing or incorrect in .model_cache/grids/:

    1. grid10x10.f10.wrap.uai — was entirely absent. Copied from /home/cohenn1/SDBE/benchmark_problems/.
    2. grid10x10.f10.wrap.uai.ord — was absent. Computed and saved by running wtminfill_order() on the model and writing the variable order to disk (one integer per line).
    3. grid10x10.f10.wrap.uai.evid — had wrong content (contained an elimination order, not evidence). Replaced with a file containing just "0" (no evidence, which is the correct format for an empty evidence set).

    After these fixes, FastGM(model=model, nn_config=nn_config, device='cuda') completed successfully in seconds.

    Note: .model_cache/ is gitignored, so these files are not tracked by git. They must be present on disk for catalog-based code to work without network access.

    **Section 4 — Pre-Caching Any Model**
    Before running any script that loads a model from the pyGMs catalog, verify the three required files exist in .model_cache/{category}/{model_name}. Steps:

    1. Identify the model name (e.g., grid10x10.f10.wrap) and category subdirectory (e.g., grids/).
    2. Check .model_cache/{category}/{model_name}.uai — if absent, copy from an external source or generate.
    3. Check .model_cache/{category}/{model_name}.uai.ord — if absent, compute: load the UAI file via pyGMs, run wtminfill_order(), write the resulting variable list to disk one integer per line.
    4. Check .model_cache/{category}/{model_name}.uai.evid — if absent or wrong, create with content "0" (no evidence).

    Alternatively, set the pyGMs cache directory to a location that already has all needed files.

    **Section 5 — Long-Term Recommendation**
    Consider adding a pre-flight check to playground.py that verifies all three cache files exist before constructing the Model object. If any are missing, print a clear error message pointing to this document rather than hanging on a network call.
  </action>
  <verify>
    ls /home/cohenn1/NCE/docs/model_cache_setup.md && wc -l /home/cohenn1/NCE/docs/model_cache_setup.md
  </verify>
  <done>docs/model_cache_setup.md exists with content covering root cause, fix applied, pre-caching steps, and long-term recommendation.</done>
</task>

</tasks>

<verification>
- docs/model_cache_setup.md exists and is non-empty
- File covers: root cause (unreachable server + no timeout), specific files fixed, steps to pre-cache future models
</verification>

<success_criteria>
Future agents encountering a FastGM hang on catalog model loading can read docs/model_cache_setup.md and resolve the issue without investigation.
</success_criteria>

<output>
After completion, create .planning/quick/19-fix-playground-py-hanging-on-fastgm-crea/19-SUMMARY.md
</output>
