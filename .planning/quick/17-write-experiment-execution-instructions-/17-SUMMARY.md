---
phase: quick-17
plan: "01"
subsystem: docs
tags: [documentation, experiments, best-practices, retrospective]
dependency_graph:
  requires: [docs/retrospective_nbe_experiment_errors.md, CLAUDE.md]
  provides: [docs/experiment_execution_guide.md]
  affects: []
tech_stack:
  added: []
  patterns: [experiment-execution, subprocess-architecture, pre-flight-checklist]
key_files:
  created: [docs/experiment_execution_guide.md]
  modified: []
decisions:
  - "Guide references CLAUDE.md rules by name/section without duplicating them (DRY principle)"
  - "All 7 retrospective errors mapped to numbered 'Pitfall' sections with detection advice"
  - "Quick reference card designed to fit on one screen for runtime use"
  - "Timing table uses actual measured data from project history (not estimates)"
metrics:
  duration: 8 min
  completed: 2026-03-09
  tasks_completed: 1
  files_created: 1
---

# Quick Task 17: Write Experiment Execution Instructions - Summary

**One-liner:** Comprehensive 723-line experiment execution guide for Claude Code agents, synthesizing all 7 NBE retrospective failures and WMSE benchmark patterns into a pre-flight-to-reporting reference.

## What Was Built

`docs/experiment_execution_guide.md` -- a step-by-step guide covering:

1. **Purpose and Scope** -- target audience, what it covers, when it applies
2. **Pre-Flight Checklist** -- GPU availability (`nvidia-smi`), process cleanup (`ps aux | grep python`), config verification, runtime estimation, user confirmation
3. **Config Fidelity Rules** -- reference to CLAUDE.md sections, key override prohibition
4. **Runtime Estimation Reference Table** -- actual measured times from project history (grid10x10 ~5 min, rbm_20 ~54 min, grid20x20 ~48 min, pedigree13 ~10 hrs, grid40x40 ~6 hrs, small_problems 4-5 min each)
5. **Execution Patterns** -- single experiment, orchestrator/worker subprocess architecture, sequential single-GPU, background execution with nohup + PID file
6. **Results and Output** -- directory structure, per-experiment JSON format, aggregate summary.json format
7. **Monitoring and Error Handling** -- live monitoring commands, orchestrator failure handling, diagnosis for common failure modes
8. **Ask vs. Proceed Decision Framework** -- explicit list of when to ping Discord vs. proceed autonomously
9. **Post-Experiment Reporting** -- result counting, Discord ping, lab_notebook.txt update, follow-up analysis
10. **Common Pitfalls (7 Retrospective Errors)** -- each error has: pattern to avoid, rule, detection step
11. **Quick Reference Card** -- four-phase checklist (before launch, script creation, after launch, after completion)
12. **NCE Package Reference** -- Python executable path, import paths, key APIs

## Verification

- `docs/experiment_execution_guide.md` exists: PASS
- Line count: 723 lines (minimum 200): PASS
- All 7 retrospective errors addressed as numbered pitfalls: PASS
- Pre-flight checklist includes all 5 required sections: PASS
- Execution patterns cover single, multi-GPU, and background: PASS
- CLAUDE.md referenced 14 times without duplicating content: PASS
- All commands use actual project paths (`/home/cohenn1/NCE/venv/bin/python`, etc.): PASS
- Key link from guide to CLAUDE.md present throughout: PASS

## Deviations from Plan

None -- plan executed exactly as written. The document structure followed the plan's 11-section outline exactly, with Section 12 (NCE Package Reference) added as a natural extension to give agents a single-page API reference.

## Commits

| Hash | Message |
|------|---------|
| 93dde3a | feat(quick-17): add comprehensive experiment execution guide |

## Self-Check

- File exists: `/home/cohenn1/NCE/docs/experiment_execution_guide.md` -- FOUND
- Commit 93dde3a exists in git log -- FOUND

## Self-Check: PASSED
