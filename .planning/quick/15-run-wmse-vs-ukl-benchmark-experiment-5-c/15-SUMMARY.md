---
phase: quick-15
plan: 01
subsystem: experiments
tags: [benchmark, wmse, ukl, loss-functions, multi-gpu, subprocess]

# Dependency graph
requires:
  - phase: quick-13
    provides: EXPERIMENT_DESIGN.md, verify_config_correctness.py, verify_full_data_training.py
provides:
  - run_benchmark.py: multi-GPU benchmark runner for 120 WMSE vs UKL experiments
  - results/: per-experiment JSON files + aggregate summary (when complete)
affects: [analysis, plotting, next-benchmarks]

# Tech tracking
tech-stack:
  added: []
  patterns: [subprocess GPU isolation, orchestrator/worker architecture, round-robin wave execution]

key-files:
  created:
    - notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/run_benchmark.py
  modified:
    - lab_notebook.txt

key-decisions:
  - "Subprocess spawning with CUDA_VISIBLE_DEVICES for GPU isolation (same as experiment_runner.py)"
  - "Round-robin wave distribution: 4 jobs per wave, 30 waves total"
  - "Copied build_experiment_config() from verify_config_correctness.py (already verified correct)"
  - "Created symlinks for model files in pyGMs catalog subdirectories (cache had files in root, catalog expected subdirs)"

patterns-established:
  - "Orchestrator/worker pattern: orchestrator builds jobs, spawns workers via subprocess.Popen with CUDA_VISIBLE_DEVICES"
  - "Per-experiment JSON results with aggregate summary.json at end"

requirements-completed: [BENCH-01]

# Metrics
duration: 13min
completed: 2026-03-06
---

# Quick Task 15: WMSE vs UKL Benchmark Summary

**120-experiment benchmark (5 configs x 24 problems x 5000 epochs) with subprocess-based 4-GPU parallel execution, launched and running**

## Performance

- **Duration:** 13 min (script creation + verification + launch)
- **Started:** 2026-03-06T22:21:11Z
- **Completed:** 2026-03-06T22:34:21Z
- **Tasks:** 2
- **Files created:** 1 (run_benchmark.py, 566 lines)

## Accomplishments
- Created run_benchmark.py with all 5 configs, 24 problems, orchestrator/worker subprocess architecture
- Both verification scripts pass: 99 config checks + 5 batch/sampling tests
- Fixed pyGMs catalog model file resolution (symlinks for 24 models + auxiliary files)
- Benchmark launched across 4 TITAN RTX GPUs, first wave of 4 jobs executing
- Discord notifications sent (start + running confirmation)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create run_benchmark.py with multi-GPU distribution** - `0f2bd32` (feat)
2. **Task 2: Run verification, launch benchmark, update lab notebook** - `00fc57c` (chore)

## Files Created/Modified
- `notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/run_benchmark.py` - Main benchmark script (566 lines): orchestrator/worker modes, 5 CONFIGS, build_all_jobs(), run_single_experiment(), wave-based GPU distribution, aggregate summary.json
- `lab_notebook.txt` - Added 2026-03-06 entry

## Decisions Made
- Reused build_experiment_config() verbatim from verify_config_correctness.py (already verified for all 5 configs)
- Subprocess spawning (not multiprocessing) for clean GPU context isolation
- Round-robin GPU assignment: jobs 0,1,2,3 to GPUs 0,1,2,3; jobs 4,5,6,7 to GPUs 0,1,2,3; etc.
- No timeout on worker subprocess.communicate() -- experiments run until complete per CLAUDE.md rules

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed pyGMs catalog model file resolution via symlinks**
- **Found during:** Task 2 (benchmark launch)
- **Issue:** pyGMs catalog `model.file` property looks for files in subdirectories (e.g., `.model_cache/bn/BN_3.uai`) but all 24 model files were in the cache root (`.model_cache/BN_3.uai`). This triggered network downloads that failed (no network access on this machine).
- **Fix:** Created symlinks from expected subdirectory paths to actual cache root locations for all 24 `.uai` files plus their `.ord` (elimination order) and `.evid` (evidence) auxiliary files.
- **Files modified:** `.model_cache/bn/`, `.model_cache/objdetect/`, `.model_cache/alchemy/`, etc. (symlinks, not tracked in git)
- **Verification:** `model.file` returns correct path without network access; benchmark successfully starts and spawns workers.
- **Committed in:** 00fc57c (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Symlink fix was necessary to allow experiment execution without network access. No scope creep.

## Issues Encountered
- First launch attempt failed because orchestrator sanity check called `FastGM(model=..., device='cpu')` which triggered `model.file` property -> pyGMs catalog download -> network timeout. Fixed by creating symlinks before relaunching.
- `results/benchmark.pid` write failed on first attempt due to shell quoting in nohup command chain; manually wrote PID file from `ps aux` output.

## Experiment Status

**The experiment is running in background and will take approximately 2-3 hours.**

- PID: saved in `results/benchmark.pid`
- Monitor: `tail -f notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/results/benchmark_stdout.log`
- Results will appear in `results/{config_name}/{modelfile}.json` as each experiment completes
- Aggregate `results/summary.json` written at end
- Discord ping will fire automatically when benchmark completes (built into script)

## Next Steps
- A follow-up task should check results after ~3 hours
- Analyze results: compare log_z estimates across 5 configs for each problem
- Generate comparison plots / tables

## User Setup Required
None - no external service configuration required.

## Self-Check: PASSED

- FOUND: run_benchmark.py
- FOUND: commit 0f2bd32
- FOUND: commit 00fc57c
- FOUND: benchmark process RUNNING

---
*Quick Task: 15*
*Completed: 2026-03-06*
