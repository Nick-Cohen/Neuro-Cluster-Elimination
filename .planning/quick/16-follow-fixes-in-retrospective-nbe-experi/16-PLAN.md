---
phase: quick-16
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - CLAUDE.md
autonomous: true
requirements: ["RETRO-UPDATE"]

must_haves:
  truths:
    - "CLAUDE.md contains a config fidelity rule stating configs are user intent and must not be overridden"
    - "CLAUDE.md contains a pre-flight checklist for experiment launches covering device, runtime estimate, process cleanup"
    - "CLAUDE.md contains a zombie process prevention rule requiring cleanup before retries"
    - "CLAUDE.md contains an algorithm literacy requirement before instrumenting code"
    - "Existing Experiment Execution Rules and Assumption Escalation sections are strengthened, not duplicated"
  artifacts:
    - path: "CLAUDE.md"
      provides: "Updated project instructions with retrospective lessons codified"
      contains: "Config Fidelity"
  key_links: []
---

<objective>
Update CLAUDE.md to codify the 7 errors and 5 protocol recommendations from docs/retrospective_nbe_experiment_errors.md as enforceable rules/guidelines.

Purpose: Prevent recurrence of the reasoning errors catalogued in the NBE experiment retrospective (device override, invented timeouts, zombie processes, etc.) by encoding them as permanent project instructions.

Output: Updated CLAUDE.md with strengthened existing sections and new subsections.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@docs/retrospective_nbe_experiment_errors.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Update CLAUDE.md with retrospective lessons</name>
  <files>CLAUDE.md</files>
  <action>
Read CLAUDE.md and docs/retrospective_nbe_experiment_errors.md. Update CLAUDE.md with the following changes. Preserve all existing content that is not being replaced. Do NOT create new top-level sections that duplicate existing ones -- integrate into existing sections where possible.

**1. Strengthen "Experiment Execution Rules" section:**

Add a "Config Fidelity" subsection (within Experiment Execution Rules) with these rules:
- Config values are user intent. Never override device, num_epochs, ecl, iB, hidden_sizes, loss_fn, or any other config parameter unless the user explicitly requests it.
- If you believe a config value is wrong, ASK the user -- do not silently change it.
- Overriding a config value requires explicit user approval AND must be logged in the task summary.
- Copying parameters from a prior script/task into a new one requires re-evaluating whether they still apply in the new context. A 1-epoch diagnostic script and a 500-epoch training run have different requirements.

Add a "Pre-Flight Checklist" subsection (within Experiment Execution Rules) as a checklist to run before ANY experiment launch:
- [ ] Device check: config says cuda? Verify GPUs available with nvidia-smi. Never downgrade to CPU without user approval.
- [ ] Runtime estimate: Compute expected runtime from num_epochs x num_NN_buckets x per-epoch cost. Check prior run data if available.
- [ ] Process cleanup: Run `ps aux | grep python` and `nvidia-smi` to check for zombie processes from prior runs. Kill stale workers before launching.
- [ ] User confirmation: If estimated runtime > 5 minutes, ping Discord with the estimate before launching.

Add a "Zombie Process Prevention" subsection (within Experiment Execution Rules):
- Before launching any new experiment subprocess, check for and kill previous instances of the same script.
- When a subprocess times out or fails, ensure all child processes are also terminated.
- Before retrying a failed experiment, always verify: no stale Python workers running, GPU memory is free (nvidia-smi), no orphaned processes from the previous attempt.

**2. Strengthen "Assumption Escalation" section:**

Add to the existing escalation hierarchy:
- Lower the ask threshold: ANY operational parameter not directly specified by the user AND not derivable from existing data in the project should trigger a Discord ping. Default to asking, not guessing.
- Add explicit statement: "The cost of a Discord ping is 10 seconds. The cost of a wrong timeout is a day. The cost of a wrong device is two days. Always ask."

**3. Add new "Algorithm Literacy" subsection** (as a new subsection under "Working with the Codebase"):

Before instrumenting, modifying, or measuring algorithm internals:
- Understand the algorithm's execution lifecycle, not just API signatures.
- Ask: "When does this data exist? When is it populated? When is it valid to read?"
- For variable elimination specifically: bucket state changes during elimination as messages arrive. Pre-elimination bucket widths are NOT the same as induced widths.
- Verify data structure types before writing iteration code (e.g., `fastgm.buckets` is a dict, not a list).
- If uncertain about algorithm behavior, read the source code and trace the data flow rather than guessing from method names.

**Formatting rules:**
- Match the existing CLAUDE.md style (bold headers, bullet lists, code blocks for commands).
- Keep the file well-organized -- sections should flow logically.
- Do not add a reference/link to the retrospective doc (the rules should stand on their own).
  </action>
  <verify>
    <automated>grep -c "Config Fidelity\|Pre-Flight Checklist\|Zombie Process\|Algorithm Literacy" /home/cohenn1/NCE/CLAUDE.md | grep -q "4" && echo "PASS: All 4 new subsections present" || echo "FAIL: Missing subsections"</automated>
  </verify>
  <done>
CLAUDE.md contains all four new subsections (Config Fidelity, Pre-Flight Checklist, Zombie Process Prevention, Algorithm Literacy). Existing sections (Experiment Execution Rules, Assumption Escalation) are strengthened with the retrospective lessons. No content is duplicated. The file reads as a cohesive set of project instructions.
  </done>
</task>

</tasks>

<verification>
- All 7 errors from the retrospective are addressed by at least one rule in CLAUDE.md
- All 5 protocol recommendations are codified
- Existing content is preserved (not deleted or duplicated)
- File is well-formatted and readable
</verification>

<success_criteria>
- CLAUDE.md contains Config Fidelity rules (addresses errors 1, 3)
- CLAUDE.md contains Pre-Flight Checklist (addresses errors 2, 3, 7 and recommendation 1)
- CLAUDE.md contains Zombie Process Prevention (addresses error 7 and recommendation 4)
- CLAUDE.md contains strengthened Assumption Escalation (addresses errors 2, 4 and recommendations 2, 3)
- CLAUDE.md contains Algorithm Literacy (addresses errors 5, 6 and recommendation 5)
- No existing CLAUDE.md content is lost
</success_criteria>

<output>
After completion, create `.planning/quick/16-follow-fixes-in-retrospective-nbe-experi/16-SUMMARY.md`
</output>
