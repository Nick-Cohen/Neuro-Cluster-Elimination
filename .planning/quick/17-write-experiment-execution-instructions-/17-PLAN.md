---
phase: 17-write-experiment-execution-instructions
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - docs/experiment_execution_guide.md
autonomous: true
requirements: [DOC-01]

must_haves:
  truths:
    - "A Claude Code agent reading only this guide can execute an NCE experiment correctly without referencing CLAUDE.md"
    - "Guide covers the full lifecycle: pre-flight, config handling, runtime estimation, execution, monitoring, post-experiment reporting"
    - "Every lesson from the 7 retrospective errors is addressed by a concrete instruction"
    - "Guide references actual project paths, commands, and APIs (not generic placeholders)"
    - "Guide distinguishes between rules (must follow) and patterns (recommended approaches)"
  artifacts:
    - path: "docs/experiment_execution_guide.md"
      provides: "Complete experiment execution guide for Claude Code agents"
      min_lines: 200
  key_links:
    - from: "docs/experiment_execution_guide.md"
      to: "CLAUDE.md"
      via: "References CLAUDE.md rules without duplicating them"
      pattern: "CLAUDE\\.md"
---

<objective>
Create a comprehensive experiment execution guide (docs/experiment_execution_guide.md) that codifies how Claude Code agents should conduct experiments in the NCE project.

Purpose: Synthesize all hard-won lessons from the NBE experiment failures (7 errors, 5 recommendations), the successful WMSE vs UKL benchmark execution, and the existing CLAUDE.md rules into a single actionable reference document. This guide should be the definitive "how to run experiments" resource for any Claude Code agent working on this project.

Output: docs/experiment_execution_guide.md -- a step-by-step guide covering pre-flight checks, config fidelity, runtime estimation, subprocess architecture, monitoring, error handling, and post-experiment reporting.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@docs/retrospective_nbe_experiment_errors.md
@docs/preventing_assumption_failures.md
@.planning/quick/15-run-wmse-vs-ukl-benchmark-experiment-5-c/15-PLAN.md
@.planning/quick/15-run-wmse-vs-ukl-benchmark-experiment-5-c/15-SUMMARY.md
@.planning/quick/12-run-full-nbe-experiment-on-all-5-problem/12-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Write the experiment execution guide</name>
  <files>docs/experiment_execution_guide.md</files>
  <action>
Create `/home/cohenn1/NCE/docs/experiment_execution_guide.md` -- a comprehensive, step-by-step guide for Claude Code agents executing experiments in the NCE project.

**Document structure and content:**

## 1. Purpose and Scope
- State that this is the authoritative guide for experiment execution by Claude Code agents
- Reference CLAUDE.md as the source of truth for project-level rules; this guide provides the HOW
- State the target audience: Claude Code agents asked to "run an experiment" or "launch a benchmark"

## 2. Pre-Flight Checklist (MANDATORY before ANY experiment)
Step-by-step checklist that must be completed before launching:

**2.1 GPU Availability**
- Run `nvidia-smi` to confirm GPUs are available and have free memory
- If config says `device='cuda'`, GPUs MUST be available. Never downgrade to CPU without explicit user approval
- Check GPU memory: if >50% used, investigate what's running before proceeding
- Command: `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv`

**2.2 Process Cleanup**
- Check for zombie/stale Python processes: `ps aux | grep python | grep -v grep`
- Check for existing experiment processes: look for nohup workers, orphaned subprocesses
- If stale processes found: kill them with `kill -9 <PID>` before proceeding
- Verify GPU memory freed after killing: re-run `nvidia-smi`
- This prevents the Error 7 pattern (zombie processes from prior failed runs)

**2.3 Config Verification**
- Read the experiment config(s) being used
- Verify `device` field matches available hardware
- Note `num_epochs` value -- this drives runtime estimation
- Note `ecl` and `iB` values -- these determine how many buckets use NNs
- NEVER override config values. Config is user intent. (Reference: Error 1 from retrospective)
- If you believe a config value is wrong, ASK the user via Discord ping

**2.4 Runtime Estimation**
Compute expected runtime BEFORE launching:

```
estimated_time = num_epochs x num_NN_buckets x per_epoch_cost
```

- `num_epochs`: from config (e.g., 500, 5000)
- `num_NN_buckets`: use `fastgm.get_large_message_buckets(iB=100, ecl=config['ecl'])` on a quick CPU run, OR check prior run data in `nbe_eval_results/`, lab_notebook.txt, or experiment SUMMARYs
- `per_epoch_cost`: ~0.01-0.1 seconds on GPU (varies by hidden size and scope width). Check prior run timing data if available.

If estimated runtime > 5 minutes: ping Discord with estimate before launching.
If you cannot estimate: ASK the user. Never guess. (Reference: Error 2, Error 3 from retrospective)

**2.5 User Confirmation**
- For experiments > 5 min estimated runtime: `~/.claude/ai-ops/scripts/ping_nick.sh "About to launch [description]. Estimated runtime: [X]. Proceed?"`
- For experiments with unclear runtime: ask before running
- The cost asymmetry is decisive: asking = 10 seconds, wrong assumption = hours/days

## 3. Config Fidelity Rules
- Config values are sacred. They represent user intent.
- NEVER override: `device`, `num_epochs`, `ecl`, `iB`, `hidden_sizes`, `loss_fn`, `batch_size`, `seed`, `sampling_scheme`
- If copying parameters from a prior script: re-evaluate whether they apply. A 1-epoch diagnostic and a 500-epoch training run have fundamentally different requirements. (Reference: Error 1 -- CPU device cargo-culted from diagnostic script)
- If config says `num_epochs=5000`, the experiment runs for exactly 5000 epochs. Period.
- Overriding ANY config value requires: (1) explicit user approval, (2) logging the override in the task summary

## 4. Runtime Estimation Reference
Provide a reference table of known runtimes from actual project data:

| Problem Set | Epochs | NN Buckets | Device | Approx Time/Problem |
|-------------|--------|------------|--------|---------------------|
| nbe_sanity_check (5 problems) | 500 | 33 (grid10x10) | CPU | >10 min (timed out) |
| nbe_sanity_check (5 problems) | 500 | 33 (grid10x10) | CUDA | ~2-5 min |
| small_problems (24 problems) | 5000 | varies | CUDA | ~4-5 min/problem |

- These are reference points. Actual times depend on model size, hidden sizes, ecl, and GPU model (TITAN RTX on this machine).
- When in doubt, run a single problem first to calibrate, then extrapolate.
- For multi-problem benchmarks: total_time ~ (num_problems / num_GPUs) x per_problem_time

## 5. Execution Patterns

**5.1 Single Experiment (Quick Test)**
```python
from nce.inference.graphical_model import FastGM
fastgm = FastGM(model=model, nn_config=config)
log_z = fastgm.get_log_partition_function()
```
- Use for: sanity checks, single-problem debugging, <5 min runs
- Run inline (not subprocess)

**5.2 Multi-Experiment Benchmark (Subprocess Architecture)**
The proven pattern from the WMSE vs UKL benchmark (quick-15):

- **Orchestrator/Worker pattern:**
  - Main script acts as orchestrator in default mode
  - Spawns workers via `subprocess.Popen` with `CUDA_VISIBLE_DEVICES=N`
  - Workers run the same script with `--mode worker --job-id N`
  - Round-robin GPU assignment: job 0 -> GPU 0, job 1 -> GPU 1, ..., job 4 -> GPU 0, etc.
  - Process in waves: one job per GPU per wave

- **Why subprocess, not multiprocessing:**
  - Clean GPU context isolation via `CUDA_VISIBLE_DEVICES`
  - `torch` imported inside worker function (respects env var set by parent)
  - Individual worker failures don't crash the orchestrator

- **Key implementation details:**
  - `sys.path.insert(0, '/home/cohenn1/NCE')` for imports
  - Lazy torch import inside main() / worker function
  - NO timeout on `subprocess.Popen.communicate()` -- experiments run until complete
  - argparse with `--mode`, `--job-id`, `--results-dir`, `--gpus`, `--dry-run`

**5.3 Background Execution for Long Runs**
```bash
cd /path/to/experiment/dir
nohup /home/cohenn1/NCE/venv/bin/python run_script.py > results/stdout.log 2>&1 &
echo $! > results/experiment.pid
```
- Use Bash tool with `run_in_background: true` for the launch
- Save PID file for monitoring
- Verify process started: `kill -0 $(cat results/experiment.pid)`
- Do NOT wait for completion -- return control to user
- Discord ping when launching AND when complete (build into script if possible)

## 6. Results and Output

**6.1 Directory Structure**
```
experiment_dir/
  run_script.py
  results/
    config_name_1/
      problem_1.json
      problem_2.json
    config_name_2/
      ...
    summary.json
    stdout.log
    experiment.pid
```

**6.2 Per-Experiment JSON Format**
```json
{
  "job_id": 0,
  "config_name": "wmse_bw0",
  "modelfile": "BN_1.uai",
  "loss_fn": "weighted_logspace_mse",
  "bw_ecl": 0,
  "ecl": 524287,
  "num_epochs": 5000,
  "hidden_sizes": [3, 3],
  "seed": 42,
  "log_z_estimate": -123.456,
  "num_buckets_trained": 5,
  "duration_seconds": 245.3,
  "status": "completed",
  "start_time": "ISO8601",
  "end_time": "ISO8601",
  "cuda_device": "NVIDIA TITAN RTX",
  "error": null,
  "traceback": null
}
```

**6.3 Aggregate Summary**
Write summary.json at end with: total_experiments, completed, failed, total_duration, per_config_summary, failed_experiments list.

## 7. Monitoring and Error Handling

**7.1 Monitoring Running Experiments**
- Check process alive: `kill -0 $(cat results/experiment.pid)`
- Follow logs: `tail -f results/stdout.log`
- Check GPU usage: `nvidia-smi`
- Count completed results: `find results/ -name "*.json" | wc -l`

**7.2 Error Handling in Scripts**
- Worker exceptions: catch in try/except, save error + traceback to result JSON, exit with code 1
- Orchestrator: NEVER crash on individual worker failure. Log error, mark as failed, continue to next job.
- At end: report total completed vs failed, list failed experiments with errors

**7.3 When Things Go Wrong**
- Worker hangs: check GPU memory with nvidia-smi, check if process is using CPU (wrong device?)
- OOM error: reduce batch_size -- but ONLY with user approval
- All workers fail: stop, diagnose root cause (import error? missing data file? wrong config?), fix, clean up processes, then retry

## 8. When to Ask vs Proceed

**Always ask (Discord ping):**
- Runtime estimate > 5 minutes and user hasn't pre-approved
- You need to override any config value
- You're uncertain about any operational parameter (timeout, batch size, device, etc.)
- The experiment failed and you want to retry with different parameters
- You're about to hardcode a value the user didn't specify

**Proceed autonomously:**
- Running pre-flight checks (GPU, processes, config verification)
- Runtime < 5 minutes with standard configs
- User explicitly said "run the full experiment" or "launch the benchmark"
- Experiment completed successfully -- report results
- Killing zombie processes from prior runs

**Escalation hierarchy:**
1. Observe: check existing data, configs, prior runs
2. Estimate: compute from known quantities
3. Ask: ping the user if still uncertain
4. NEVER: guess and hope

Reference: CLAUDE.md "Assumption Escalation" section. The cost of asking is always lower than the cost of a wrong assumption.

## 9. Post-Experiment Reporting

After an experiment completes:
1. Check results: count completed/failed experiments
2. Ping Discord with summary: "Benchmark complete: X/Y experiments succeeded. [brief findings]"
3. Update lab_notebook.txt with dated entry
4. If results warrant analysis: offer to create analysis/plots as a follow-up task

## 10. Common Pitfalls (Lessons from Retrospective)

List each of the 7 errors from `docs/retrospective_nbe_experiment_errors.md` as a numbered pitfall with the pattern to avoid:

1. **Device override** -- Never change `device='cuda'` to `device='cpu'`. Config is intent.
2. **Invented timeouts** -- Never add arbitrary timeouts. "Full" means full. No `timeout=600`.
3. **Not checking evidence** -- Always check existing run data before making assumptions about runtime.
4. **Not asking** -- When uncertain, ping Discord. Cost of asking << cost of wrong assumption.
5. **Pre-elimination measurement** -- Understand algorithm lifecycle. Bucket state changes during elimination. Measure at the right time.
6. **Dict iteration bug** -- `fastgm.buckets` is a dict. Use `.items()` to iterate, not bare `for x in dict`.
7. **Zombie processes** -- Always clean up before retrying. `ps aux | grep python`, `nvidia-smi`, kill stale workers.

## 11. Quick Reference Card
A condensed checklist format that fits on one "screen":

```
BEFORE LAUNCH:
  [ ] nvidia-smi -- GPUs available, memory free
  [ ] ps aux | grep python -- no zombies
  [ ] Config read -- device, num_epochs, ecl, iB noted
  [ ] Runtime estimated -- epochs x NN_buckets x per_epoch_cost
  [ ] User pinged if runtime > 5 min

DURING LAUNCH:
  [ ] Config values NOT overridden
  [ ] No timeouts added to subprocess.communicate()
  [ ] Background execution for long runs (nohup + PID file)
  [ ] Discord ping sent

AFTER LAUNCH:
  [ ] Process confirmed running (kill -0 PID)
  [ ] Log file being written
  [ ] Do NOT wait for completion of long experiments

AFTER COMPLETION:
  [ ] Results counted (completed/failed)
  [ ] Discord ping with summary
  [ ] lab_notebook.txt updated
```

**Writing style notes:**
- Write in imperative mood ("Run nvidia-smi", not "You should run nvidia-smi")
- Use concrete project paths: `/home/cohenn1/NCE/venv/bin/python`, not "the Python executable"
- Include actual command examples, not pseudocode
- Reference specific files and APIs from the NCE codebase
- Cross-reference CLAUDE.md sections by name but do not duplicate their content
- Use markdown headers, code blocks, and tables for scannability
  </action>
  <verify>
    <automated>test -f /home/cohenn1/NCE/docs/experiment_execution_guide.md && wc -l /home/cohenn1/NCE/docs/experiment_execution_guide.md | awk '{if ($1 >= 200) print "PASS: " $1 " lines"; else print "FAIL: only " $1 " lines"}'</automated>
  </verify>
  <done>
docs/experiment_execution_guide.md exists with 200+ lines. Covers all 11 sections: purpose, pre-flight checklist (GPU/processes/config/runtime/user confirmation), config fidelity rules, runtime estimation reference table, execution patterns (single/multi/background), results format, monitoring and error handling, ask-vs-proceed decision framework, post-experiment reporting, all 7 retrospective pitfalls, and quick reference card.
  </done>
</task>

</tasks>

<verification>
1. `docs/experiment_execution_guide.md` exists and is 200+ lines
2. All 7 retrospective errors are referenced in the "Common Pitfalls" section
3. Pre-flight checklist includes: GPU check, process cleanup, config verification, runtime estimation, user confirmation
4. Execution patterns section covers: single experiment, subprocess architecture, background execution
5. Document references CLAUDE.md sections by name without duplicating them
6. All commands use actual project paths (`/home/cohenn1/NCE/venv/bin/python`, etc.)
</verification>

<success_criteria>
- A Claude Code agent given this guide and asked "run an experiment" can execute the pre-flight checklist, estimate runtime, launch correctly with subprocess isolation, monitor progress, and report results -- without needing to reference any other document except CLAUDE.md for project-level rules
- Every lesson from the NBE retrospective (7 errors) is addressed by a concrete, actionable instruction
- The guide uses real project paths, APIs, and commands -- not generic placeholders
</success_criteria>

<output>
After completion, create `.planning/quick/17-write-experiment-execution-instructions-/17-SUMMARY.md`
</output>
