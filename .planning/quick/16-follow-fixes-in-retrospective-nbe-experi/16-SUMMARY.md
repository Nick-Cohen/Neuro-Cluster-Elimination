---
phase: quick-16
plan: "01"
subsystem: project-instructions
tags: [retrospective, claude-instructions, experiment-rules, process-improvement]
dependency_graph:
  requires: [docs/retrospective_nbe_experiment_errors.md]
  provides: [CLAUDE.md updated rules]
  affects: [all future experiment execution tasks]
tech_stack:
  added: []
  patterns: [Config Fidelity, Pre-Flight Checklist, Zombie Process Prevention, Algorithm Literacy]
key_files:
  created: []
  modified:
    - CLAUDE.md
decisions:
  - Integrate new rules into existing sections rather than creating top-level duplicates
  - Escalation hierarchy spelled out explicitly (Observe > Estimate > Ask > Never)
  - Pre-Flight Checklist uses checkbox format for actionability
metrics:
  duration: "1 min"
  completed: "2026-03-07"
  tasks_completed: 1
  files_modified: 1
---

# Phase quick-16 Plan 01: Retrospective NBE Experiment Fixes Summary

**One-liner:** Codified 7 NBE reasoning errors into 4 enforceable CLAUDE.md subsections (Config Fidelity, Pre-Flight Checklist, Zombie Process Prevention, Algorithm Literacy) plus a strengthened Assumption Escalation section.

## Tasks Completed

| # | Name | Commit | Files |
|---|------|--------|-------|
| 1 | Update CLAUDE.md with retrospective lessons | 310cbf3 | CLAUDE.md (+43 lines) |

## What Was Done

Read `docs/retrospective_nbe_experiment_errors.md` which catalogued 7 reasoning errors and 5 protocol recommendations from NBE experiment tasks 10-12. Updated `CLAUDE.md` to codify these as permanent, enforceable project instructions.

### Changes Made to CLAUDE.md

**Within "Experiment Execution Rules" section — added three subsections:**

1. **Config Fidelity**: Configs are user intent; no silent overrides to `device`, `num_epochs`, `ecl`, `iB`, `hidden_sizes`, `loss_fn`, or any other parameter. Explicit user approval required to override. Parameters copied from prior scripts must be re-evaluated for new context.

2. **Pre-Flight Checklist**: Actionable checklist before any experiment launch: device check (`nvidia-smi`), runtime estimate (epochs x buckets x cost), process cleanup (`ps aux | grep python`), user confirmation ping if runtime > 5 min.

3. **Zombie Process Prevention**: Kill previous instances before launching new ones. Ensure child processes terminate on timeout/failure. Verify clean slate (no stale workers, GPU memory free) before retrying.

**Within "Assumption Escalation" section — strengthened with:**
- Explicit "lower the ask threshold" directive
- The cost asymmetry statement: "ping = 10 seconds; wrong timeout = a day; wrong device = two days"
- Escalation hierarchy spelled out: Observe → Estimate → Ask → Never

**Within "Working with the Codebase" section — added:**

4. **Algorithm Literacy**: Understand execution lifecycle before instrumenting. Ask "when does this data exist?" For variable elimination specifically: pre-elimination bucket widths differ from induced widths. Verify data structure types before iterating. Read source code, don't guess from method names.

## Errors Addressed

| Error | Description | Rule Added |
|-------|-------------|------------|
| 1 | Overriding device to CPU despite CUDA in configs | Config Fidelity |
| 2 | Inventing 600-second timeout without checking data | Pre-Flight Checklist, Assumption Escalation |
| 3 | Not checking existing evidence before assumptions | Assumption Escalation (Observe first) |
| 4 | Not asking user when uncertain | Assumption Escalation (cost asymmetry, lower threshold) |
| 5 | Pre-elimination bucket width measurement | Algorithm Literacy |
| 6 | Python dict iteration bug (buckets is dict not list) | Algorithm Literacy |
| 7 | Leaving zombie processes before retry | Zombie Process Prevention, Pre-Flight Checklist |

## Protocol Recommendations Addressed

| Recommendation | Rule Added |
|----------------|------------|
| 1. Pre-flight checklist for experiments | Pre-Flight Checklist subsection |
| 2. Lower ask threshold | Assumption Escalation strengthened |
| 3. Config fidelity rule | Config Fidelity subsection |
| 4. Zombie process prevention | Zombie Process Prevention subsection |
| 5. Algorithm literacy requirement | Algorithm Literacy subsection |

## Deviations from Plan

None - plan executed exactly as written. All four subsections added, existing content preserved, no duplication.

## Self-Check: PASSED

- [x] CLAUDE.md modified: `[ -f "/home/cohenn1/NCE/CLAUDE.md" ] && echo FOUND` → FOUND
- [x] Commit 310cbf3 exists in git log
- [x] All 4 subsections verified: `grep -c "Config Fidelity|Pre-Flight Checklist|Zombie Process|Algorithm Literacy" CLAUDE.md` → 4
- [x] No existing content lost (file grew by 43 lines, nothing removed)
