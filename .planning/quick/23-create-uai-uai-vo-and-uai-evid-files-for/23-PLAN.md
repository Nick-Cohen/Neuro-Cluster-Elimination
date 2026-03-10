---
phase: quick-23
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/export_small_problems_cache.py
autonomous: true
requirements: [QUICK-23]
must_haves:
  truths:
    - "All 24 small_problems models have .uai, .uai.vo, and .uai.evid files in .model_cache subdirectories"
    - "The .uai.vo files use the SDBE format: # header line followed by one variable per line"
    - "The .uai.evid files contain either real evidence (from catalog) or 0 (no evidence) for models without evidence"
  artifacts:
    - path: "scripts/export_small_problems_cache.py"
      provides: "Script to generate/verify all cache files for small_problems"
  key_links:
    - from: "scripts/export_small_problems_cache.py"
      to: ".model_cache/{category}/"
      via: "reads .ord files, writes .vo and .evid files"
      pattern: "open.*\\.uai\\.vo.*\\.uai\\.evid"
---

<objective>
Create .uai, .uai.vo, and .uai.evid files for all 24 small_problems benchmark instances.

Purpose: Ensure all models in the small_problems benchmark set have complete cache files
in the correct formats, including .uai.vo files (SDBE-style: # header + one variable per
line) alongside the existing .uai.ord files (pyGMs-style: count + space-separated on one
line). The .uai.evid files are needed for 14 models currently missing them.

Output: A script that generates all missing files, plus the generated files themselves in
.model_cache subdirectories.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@nce/benchmark_problems/small_problems.py
@nce/benchmark_problems/catalog_utils.py
@docs/model_cache_setup.md

<interfaces>
<!-- The 24 models are defined in small_problems.py as catalog keys -->
<!-- Format: (catalogue_key, source_iB) -->
_MODELS = [
    ('alchemy/smokers_20', 10),
    ('bn/BN_1', 10), ('bn/BN_2', 15), ('bn/BN_3', 10), ('bn/BN_5', 10),
    ('bn/BN_7', 10), ('bn/BN_8', 15), ('bn/BN_9', 15), ('bn/BN_10', 10), ('bn/BN_11', 10),
    ('segmentation/10_14_s.binary', 10), ('segmentation/10_16_s.binary', 10),
    ('segmentation/11_17_s.binary', 10), ('segmentation/11_4_s.binary', 10),
    ('promedas/or_chain_10.fg', 10),
    ('objdetect/deer_rescaled_0034.K15.F1.5.model', 10),
    ('objdetect/deer_rescaled_0294.K10.F1.75.model', 10),
    ('grids/grid10x10.f5.wrap', 10),
    ('objdetect/deer_rescaled_0034.K10.F2.model', 15),
    ('objdetect/deer_rescaled_0034.K15.F1.75.model', 15),
    ('objdetect/deer_rescaled_0034.K20.F1.25.model', 15),
    ('objdetect/deer_rescaled_0034.K20.F1.5.model', 15),
    ('csp/29.wcsp', 15),
    ('csp/404.wcsp', 15),
]

<!-- Current cache state analysis (as of 2026-03-10): -->
<!-- ALL 24 models: .uai EXISTS, .uai.ord EXISTS in .model_cache/{category}/ -->
<!-- 10 models with ALL 3 files (uai, ord, evid): bn/BN_1..BN_11, promedas/or_chain_10.fg -->
<!-- 14 models MISSING .uai.evid: alchemy/smokers_20, segmentation/*, objdetect/*, grids/grid10x10.f5.wrap, csp/* -->
<!-- 0 models have .uai.vo files (ALL need to be created) -->

<!-- .ord format (pyGMs): single line, "count var1 var2 ... varN" -->
<!-- .vo format (SDBE): line 1 is "#", then one variable per line -->
<!-- Example .ord: "100 48 74 66 20 21 ..." -->
<!-- Example .vo: -->
<!-- # -->
<!-- 48 -->
<!-- 74 -->
<!-- 66 -->
<!-- ... -->

<!-- .evid format: -->
<!-- No evidence: "0" -->
<!-- With evidence: "N var1 val1 var2 val2 ... varN valN" (N = number of observed vars) -->
<!-- The BN models have real evidence; models without evidence get "0" -->
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create export script and generate all .uai.vo and .uai.evid files</name>
  <files>scripts/export_small_problems_cache.py</files>
  <action>
Create `scripts/export_small_problems_cache.py` that does the following:

1. For each of the 24 models defined in `_MODELS` list in `nce/benchmark_problems/small_problems.py`:

   a. **Read the .ord file** from `.model_cache/{category}/{model_name}.uai.ord`
      - Format: single line with count followed by space-separated variable indices
      - Parse it: split on spaces, skip first element (count), rest are variable indices

   b. **Write the .vo file** to `.model_cache/{category}/{model_name}.uai.vo`
      - Format: First line is `#`, then one variable index per line
      - Convert from .ord format to .vo format
      - Skip if .vo file already exists (idempotent)

   c. **Write the .evid file** to `.model_cache/{category}/{model_name}.uai.evid` (only if missing)
      - For models that don't have a .evid file yet, write `0` (no evidence)
      - Do NOT overwrite existing .evid files (BN models and or_chain_10 have real evidence)
      - Skip if .evid file already exists (idempotent)

2. Print a summary table showing for each model: category/name, has_uai, has_ord, has_vo, has_evid (all should be True after running).

3. The script should be runnable standalone: `python scripts/export_small_problems_cache.py`
   - Use hardcoded model list (copy from small_problems.py) to avoid importing nce and triggering catalog loading
   - Use `os.path` for path resolution relative to project root (parent of scripts/)

The script must NOT import from nce (to avoid triggering pyGMs catalog network calls). Instead, hardcode the 24 model keys and work directly with the filesystem.

Run the script after creating it to generate all missing files.
  </action>
  <verify>
Run the script: `cd /home/cohenn1/NCE && /home/cohenn1/NCE/venv/bin/python scripts/export_small_problems_cache.py`

Then verify:
- All 24 models have .uai.vo files: `for pair in alchemy/smokers_20 bn/BN_1 bn/BN_2 bn/BN_3 bn/BN_5 bn/BN_7 bn/BN_8 bn/BN_9 bn/BN_10 bn/BN_11 segmentation/10_14_s.binary segmentation/10_16_s.binary segmentation/11_17_s.binary segmentation/11_4_s.binary promedas/or_chain_10.fg objdetect/deer_rescaled_0034.K15.F1.5.model objdetect/deer_rescaled_0294.K10.F1.75.model grids/grid10x10.f5.wrap objdetect/deer_rescaled_0034.K10.F2.model objdetect/deer_rescaled_0034.K15.F1.75.model objdetect/deer_rescaled_0034.K20.F1.25.model objdetect/deer_rescaled_0034.K20.F1.5.model csp/29.wcsp csp/404.wcsp; do test -f ".model_cache/${pair}.uai.vo" || echo "MISSING: $pair.uai.vo"; done`
- All 24 models have .uai.evid files: same loop with .uai.evid
- A .vo file starts with `#` and has one var per line: `head -5 .model_cache/bn/BN_1.uai.vo`
- Existing BN evidence not overwritten: `cat .model_cache/bn/BN_1.uai.evid` still has real evidence (18 vars)
  </verify>
  <done>
All 24 small_problems models have .uai, .uai.vo, and .uai.evid files in their .model_cache subdirectories. The .vo files use SDBE format (# header + one var per line). Existing .evid files with real evidence are preserved. Missing .evid files contain "0" (no evidence).
  </done>
</task>

</tasks>

<verification>
1. Count files: 24 .uai.vo files created, 14 new .uai.evid files created
2. Format check: .vo files have # header + one integer per line
3. Preservation check: BN_1.uai.evid still contains "18 3 0 6 0 ..." (real evidence)
4. Completeness: Every model in _MODELS has all 4 files (.uai, .ord, .vo, .evid)
</verification>

<success_criteria>
- All 24 .uai.vo files exist with correct SDBE format
- All 24 .uai.evid files exist (14 newly created with "0", 10 preserved with real evidence)
- Reusable script committed at scripts/export_small_problems_cache.py
</success_criteria>

<output>
After completion, create `.planning/quick/23-create-uai-uai-vo-and-uai-evid-files-for/23-SUMMARY.md`
</output>
