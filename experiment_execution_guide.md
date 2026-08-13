# NCE Experiment Execution Guide

**Audience:** Claude Code agents asked to "run an experiment," "launch a benchmark," or "train models" in the NCE project.

**Authority:** This guide is the definitive HOW for experiment execution. For project-level rules (config fidelity, device policy, assumption escalation, notification requirements), see `CLAUDE.md` — this guide references those rules but does not duplicate them.

**Key principle:** A Claude Code agent reading only this guide and `CLAUDE.md` should be able to execute any NCE experiment correctly without guessing about any operational parameter.

---

## 1. Purpose and Scope

This guide covers the full experiment lifecycle:

1. Pre-flight checks (before launching anything)
2. Config verification and runtime estimation
3. Execution patterns (single experiment, multi-GPU benchmark, background execution)
4. Results format and output directory structure
5. Monitoring and error handling
6. Post-experiment reporting

**When this guide applies:** Any task that calls `FastGM`, trains neural networks, runs benchmarks across problem sets, or launches subprocesses to run NCE experiments.

**When this guide does NOT apply:** Editing source code, running unit tests, creating benchmark configs, or analysis/plotting tasks that don't involve running inference.

---

## 2. Pre-Flight Checklist (MANDATORY before ANY experiment)

Run every item before launching. Do not skip steps. Each step takes under one minute.

### 2.1 GPU Availability

Run `nvidia-smi` to verify GPU state:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

Rules:
- If config says `device='cuda'`, GPUs MUST be available and have free memory. Never downgrade to CPU without explicit user approval. See CLAUDE.md "Config Fidelity" section.
- If GPU memory is more than 50% used: investigate what is running before proceeding. Something may have leaked from a prior run.
- 4 NVIDIA TITAN RTX GPUs (24 GB each) are available on this machine. If fewer appear, diagnose before proceeding.

Expected output for a clean state:

```
index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
0, NVIDIA TITAN RTX, 0 MiB, 24220 MiB, 0 %
1, NVIDIA TITAN RTX, 0 MiB, 24220 MiB, 0 %
2, NVIDIA TITAN RTX, 0 MiB, 24220 MiB, 0 %
3, NVIDIA TITAN RTX, 0 MiB, 24220 MiB, 0 %
```

### 2.2 Process Cleanup

Check for zombie or stale Python processes before launching:

```bash
ps aux | grep python | grep -v grep
```

If stale processes are found:
1. Identify them (look for prior experiment script names, nohup workers, etc.)
2. Kill them: `kill -9 <PID>`
3. Re-run `nvidia-smi` to verify GPU memory was freed

This prevents the Error 7 pattern: launching a new experiment while a prior failed run's workers are still alive and consuming GPU memory. See Section 10 (Common Pitfalls).

### 2.3 Config Verification

Read the experiment config(s) before running. Note these values — you need them for runtime estimation:

| Config Key | Why You Need It |
|------------|----------------|
| `device` | Must match available hardware |
| `num_epochs` | Primary runtime driver |
| `ecl` | Determines which buckets use NNs |
| `iB` | Mini-bucket i-bound |
| `hidden_sizes` | Affects per-epoch cost |
| `loss_fn` | Affects training cost |

**Never override config values.** Config is user intent. If you believe a value is wrong, ping Discord and ask. See CLAUDE.md "Config Fidelity" section for the full rule.

### 2.4 Runtime Estimation

Compute expected runtime before launching. This is not optional.

```
estimated_time = num_epochs x num_NN_buckets x per_epoch_cost
```

**Step 1: Get num_epochs** from config (e.g., `config['num_epochs']`).

**Step 2: Get num_NN_buckets.** Try these sources in order:
- Prior run data in `nbe_eval_results/`, `lab_notebook.txt`, or plan SUMMARYs
- Quick CPU dry run: `fastgm.get_large_message_buckets(iB=100, ecl=config['ecl'])`
- Note: The 5 `nbe_sanity_check` problems with iB=10, ecl=512 have approximately 33 NN-eligible buckets each for grid10x10

**Step 3: Estimate per_epoch_cost.** Per-epoch cost on a TITAN RTX GPU is approximately:
- Small hidden sizes ([3,3] or fewer): ~0.01-0.05 seconds per epoch per bucket
- Large hidden sizes ([64,64] or more): ~0.05-0.2 seconds per epoch per bucket
- Check prior run timing data if available (the timing reference table in Section 4)

**Step 4: Check against CLAUDE.md threshold.** Per CLAUDE.md "Experiment Execution Rules": if total training iterations > 10,000, warn the user. If estimated runtime > 5 minutes, ping Discord with the estimate before launching.

### 2.5 User Confirmation for Long Experiments

For experiments with estimated runtime > 5 minutes:

```bash
~/.claude/ai-ops/scripts/ping_nick.sh "About to launch [description]. Estimated runtime: [X hours/minutes] on [N GPUs]. Proceed?"
```

The cost asymmetry is decisive: asking takes 10 seconds. A wrong assumption wastes hours. See CLAUDE.md "Assumption Escalation" section for the full protocol.

---

## 3. Config Fidelity Rules

These rules are stated in CLAUDE.md "Config Fidelity" section. They are repeated here as an immediate reminder, not as a duplicate.

- **Config values are user intent.** Never override `device`, `num_epochs`, `ecl`, `iB`, `hidden_sizes`, `loss_fn`, `batch_size`, `seed`, or `sampling_scheme`.
- If copying parameters from a prior script, re-evaluate whether they apply. A 1-epoch diagnostic and a 500-epoch training run have fundamentally different requirements. Do not cargo-cult `device='cpu'` from a diagnostic into a production run.
- If a config says `num_epochs=5000`, the experiment runs for exactly 5000 epochs.
- Overriding any config value requires: (1) explicit user approval via Discord, (2) the override logged in the task summary with the reason.

---

## 4. Runtime Estimation Reference Table

Known runtimes from actual project runs (all on TITAN RTX):

| Problem | Epochs | NN Buckets | Config | Approx Time |
|---------|--------|------------|--------|-------------|
| grid10x10.f5.wrap | 500 | 33 | NBE (iB=10, ecl=512) | ~5 min on CUDA |
| grid10x10.f5.wrap | 500 | 33 | NBE | >10 min on CPU (timed out at 10 min) |
| rbm_20 | 500 | ~20-30 | NBE | ~54 min on CUDA |
| grid20x20 | 500 | ~40-60 | NBE | ~48 min on CUDA |
| pedigree13 | 500 | varies | NBE | ~10 hours on CUDA |
| grid40x40 | 500 | varies | NBE | ~6 hours on CUDA |
| small_problems (24 problems) | 5000 | varies | WMSE/UKL | ~4-5 min/problem on CUDA |
| small_problems (120 total) | 5000 | varies | 5 configs, 4 GPUs | ~2-3 hours total |

These are reference points. Actual times depend on model size, `hidden_sizes`, `ecl`, and whether early stopping fires. When running a new problem type without prior timing data, run a single problem first to calibrate, then extrapolate.

For multi-GPU benchmarks:
```
total_time ≈ (num_problems x num_configs / num_GPUs) x per_problem_time
```

---

## 5. Execution Patterns

### 5.1 Single Experiment (Quick Test or Debug)

For sanity checks, single-problem debugging, or runs expected to take less than 5 minutes:

```python
import sys
sys.path.insert(0, '/home/cohenn1/NCE')

from nce.inference.graphical_model import FastGM
from nce.benchmark_problems import nbe_sanity_check

model = nbe_sanity_check.problems[0]
config = nbe_sanity_check.configs['nbe'][0]

fastgm = FastGM(model=model, nn_config=config, device=config['device'])
log_z = fastgm.get_log_partition_function()
print(f"log Z = {log_z}")
```

Notes:
- Always import `torch` inside the function or script main block, not at module level, if `CUDA_VISIBLE_DEVICES` will be set later
- Use `config['device']` — do NOT hardcode `'cpu'` or `'cuda'`
- `get_log_partition_function()` is the correct inference API, not `run()`

### 5.2 Multi-Experiment Benchmark (Subprocess Architecture)

The proven pattern from the WMSE vs UKL benchmark (quick-15). Reference implementation:
`/home/cohenn1/NCE/notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/run_benchmark.py`

**Architecture: Orchestrator + Worker pattern**

The script runs in two modes controlled by `--mode`:

- **Orchestrator** (default): builds the full job list, spawns one worker subprocess per GPU per wave, waits for each wave, collects results, writes `summary.json`
- **Worker**: runs a single job identified by `--job-id`, saves result JSON, exits

**Why subprocess, not multiprocessing:**

- `CUDA_VISIBLE_DEVICES` is set per subprocess via `env=` in `subprocess.Popen`. This gives clean GPU context isolation.
- `torch` is imported inside the worker function AFTER `CUDA_VISIBLE_DEVICES` is set. Importing torch at module level before setting this env var causes PyTorch to see all GPUs instead of the assigned one.
- Individual worker failures do NOT crash the orchestrator. Failed workers log their errors and the orchestrator continues.

**Key implementation skeleton:**

```python
#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# CRITICAL: NCE package path
sys.path.insert(0, '/home/cohenn1/NCE')

def run_single_experiment(job, results_dir):
    """Worker function: import torch here, NOT at module level."""
    import torch  # lazy import respects CUDA_VISIBLE_DEVICES
    from nce.inference.graphical_model import FastGM

    start_time = datetime.now()
    result = {
        'job_id': job['job_id'],
        'config_name': job['config_name'],
        'modelfile': job['modelfile'],
        'status': None,
        'error': None,
        'traceback': None,
        'start_time': start_time.isoformat(),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    }

    try:
        fastgm = FastGM(model=job['model'], nn_config=job['nn_config'])
        log_z = fastgm.get_log_partition_function()
        end_time = datetime.now()
        result.update({
            'log_z_estimate': float(log_z),
            'num_buckets_trained': getattr(fastgm, 'num_trained', 0),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'end_time': end_time.isoformat(),
            'status': 'completed',
        })
    except Exception as e:
        import traceback
        end_time = datetime.now()
        result.update({
            'log_z_estimate': None,
            'num_buckets_trained': 0,
            'duration_seconds': (end_time - start_time).total_seconds(),
            'end_time': end_time.isoformat(),
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
        })

    # Save result JSON
    out_dir = Path(results_dir) / job['config_name']
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{job['modelfile']}.json", 'w') as f:
        json.dump(result, f, indent=2)

    return result


def run_orchestrator(args):
    """Distribute jobs across GPUs in waves."""
    import torch  # lazy import

    gpus = [int(g) for g in args.gpus.split(',')]
    jobs = build_all_jobs()
    total_jobs = len(jobs)
    script_path = str(Path(__file__).resolve())

    completed, failed, failed_jobs = 0, 0, []
    job_idx = 0

    while job_idx < total_jobs:
        wave_procs = []

        # Spawn one job per GPU
        for gpu_id in gpus:
            if job_idx >= total_jobs:
                break
            job = jobs[job_idx]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            proc = subprocess.Popen(
                [sys.executable, script_path,
                 '--mode', 'worker',
                 '--job-id', str(job['job_id']),
                 '--results-dir', args.results_dir],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            wave_procs.append((proc, job, gpu_id))
            job_idx += 1

        # Wait for wave -- NO timeout
        for proc, job, gpu_id in wave_procs:
            stdout, stderr = proc.communicate()  # blocks until process exits
            # check result JSON to determine success/failure
            ...

    # Write summary.json, ping Discord
    ...


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['orchestrator', 'worker'], default='orchestrator')
    parser.add_argument('--job-id', type=int, default=None)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.mode == 'worker':
        return run_worker(args)
    return run_orchestrator(args)

if __name__ == '__main__':
    sys.exit(main())
```

**Critical detail: `proc.communicate()` has NO timeout.** Experiments run until complete. Never add `timeout=` to `proc.communicate()` unless the user explicitly requests it. See CLAUDE.md "Experiment Execution Rules."

**GPU assignment:** Round-robin across requested GPUs. Job 0 -> GPU 0, Job 1 -> GPU 1, ..., Job N -> GPU (N % num_gpus). Set `CUDA_VISIBLE_DEVICES` in the subprocess env, not in the config.

**dict iteration for `fastgm.buckets`:** `fastgm.buckets` is a dict keyed by variable. Use `.items()`:
```python
for var, bucket in fastgm.buckets.items():  # CORRECT
    ...
# NOT: for bucket in fastgm.buckets:        # WRONG - iterates over dict keys
```

### 5.3 Sequential Single-GPU Execution

For experiments that run one problem at a time, sequentially. Reference implementation:
`/home/cohenn1/NCE/notebooks/March-2026/claude_experiments/nbe_full_experiment.py`

Pattern: spawn each problem as a subprocess with `PROBLEM_TIMEOUT_SECS = None` (no timeout). The subprocess captures output via stdout and parses `RESULT_JSON:` lines.

```python
PROBLEM_TIMEOUT_SECS = None  # Full experiment = no timeout

result = subprocess.run(
    ['/home/cohenn1/NCE/venv/bin/python', worker_file, str(problem_idx)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    timeout=PROBLEM_TIMEOUT_SECS,   # None = no timeout
)
```

### 5.4 Background Execution for Long Runs

For experiments expected to take more than a few minutes, use background execution so the agent is not blocking:

```bash
cd /home/cohenn1/NCE/notebooks/3-2026/claude_experiments/my_experiment/
nohup /home/cohenn1/NCE/venv/bin/python run_script.py --mode orchestrator --gpus 0,1,2,3 \
    > results/stdout.log 2>&1 &
echo $! > results/experiment.pid
```

Use the Bash tool with `run_in_background: true` for the launch command when using Claude Code.

After launching:
1. Save the PID: `echo $! > results/experiment.pid`
2. Verify the process started: `kill -0 $(cat results/experiment.pid) && echo "running"`
3. Do NOT wait for the process to complete — return control to the user
4. Discord ping: confirm the experiment launched and provide the PID and log path

Build the completion ping into the experiment script itself (see Section 9).

---

## 6. Results and Output

### 6.1 Directory Structure

Use this layout for all experiments:

```
notebooks/3-2026/claude_experiments/my_experiment/
  run_script.py
  results/
    config_name_1/
      BN_1.uai.json
      BN_2.uai.json
    config_name_2/
      BN_1.uai.json
      ...
    summary.json
    stdout.log
    benchmark.log
    experiment.pid
```

The `notebooks/<current-month>/claude_experiments/` directory is the standard location for experiment scripts. See CLAUDE.md "Running Experiments" section.

### 6.2 Per-Experiment JSON Format

Each completed or failed experiment writes one JSON result file:

```json
{
  "job_id": 0,
  "config_name": "wmse_bw0",
  "modelfile": "BN_1.uai",
  "problem_idx": 0,
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
  "start_time": "2026-03-06T14:22:00",
  "end_time": "2026-03-06T14:26:05",
  "cuda_device": "NVIDIA TITAN RTX",
  "error": null,
  "traceback": null
}
```

For failed experiments, `status` is `"failed"`, `log_z_estimate` is `null`, and `error`/`traceback` contain the exception details.

### 6.3 Aggregate Summary JSON

Write `summary.json` after all experiments complete:

```json
{
  "total_experiments": 120,
  "completed": 118,
  "failed": 2,
  "total_duration_seconds": 9823.4,
  "per_config_summary": {
    "wmse_bw0": {"completed": 24, "failed": 0, "mean_duration": 245.3},
    "ukl_bw0": {"completed": 24, "failed": 0, "mean_duration": 238.1}
  },
  "failed_experiments": [
    {"job_id": 47, "config_name": "ukl_bw30", "modelfile": "BN_22.uai",
     "returncode": 1, "stderr": "..."}
  ]
}
```

---

## 7. Monitoring and Error Handling

### 7.1 Monitoring Running Experiments

Check if the background process is alive:
```bash
kill -0 $(cat /path/to/results/experiment.pid) && echo "running" || echo "finished or crashed"
```

Follow the live log:
```bash
tail -f /path/to/results/stdout.log
```

Check GPU utilization:
```bash
nvidia-smi
```

Count completed result files:
```bash
find /path/to/results/ -name "*.json" -not -name "summary.json" | wc -l
```

### 7.2 Error Handling in Scripts

In worker code:
- Wrap the entire experiment in `try/except Exception`
- Save the error and full traceback to the result JSON
- Exit with code 1 on failure

In orchestrator code:
- NEVER crash on individual worker failure
- Read the result JSON to determine success/failure (not just returncode)
- Log each failure with the job identifier and error snippet
- Continue to the next job
- At the end: report total completed vs. failed, list failed experiments

Example orchestrator failure handling:
```python
stdout, stderr = proc.communicate()   # No timeout

result_path = Path(results_dir) / job['config_name'] / f"{job['modelfile']}.json"
if result_path.exists():
    with open(result_path) as f:
        result = json.load(f)
    status = result.get('status', 'unknown')
else:
    status = 'no_output_file'

if proc.returncode != 0 or status != 'completed':
    failed += 1
    failed_jobs.append({
        'job_id': job['job_id'],
        'config_name': job['config_name'],
        'modelfile': job['modelfile'],
        'returncode': proc.returncode,
        'stderr': stderr[-500:] if stderr else '',
        'status': status,
    })
```

### 7.3 When Things Go Wrong

**Worker hangs without GPU activity:** Check if it's using the wrong device. `nvidia-smi` will show zero GPU utilization. Root cause: `CUDA_VISIBLE_DEVICES` not set before `import torch`, or config was overridden to CPU.

**OOM (CUDA out of memory):** Do not automatically reduce batch size — that changes the config without user approval. Ping Discord with the error and ask for instructions.

**All workers in a wave fail immediately:** Stop. Do not continue. Diagnose root cause:
1. Import error? Run one worker manually: `CUDA_VISIBLE_DEVICES=0 /home/cohenn1/NCE/venv/bin/python run_script.py --mode worker --job-id 0 --results-dir results/`
2. Missing data file?
3. Wrong config structure?
Fix the root cause, clean up any stale processes, then retry.

**Process completed but result JSON missing:** Worker may have crashed before writing output. Check `stderr` from `proc.communicate()`.

---

## 8. When to Ask vs. When to Proceed

Per CLAUDE.md "Assumption Escalation" section, the escalation hierarchy is: Observe -> Estimate -> Ask -> Never guess.

### Always ping Discord before proceeding:

- Estimated runtime > 5 minutes (even if user said "run it")
- You need to override any config value for any reason
- You are uncertain about any operational parameter the user did not specify (timeout, batch size, number of retries, etc.)
- The experiment failed and you want to retry with different parameters
- You are about to hardcode a value you cannot derive from existing project data

### Proceed autonomously without asking:

- Running pre-flight checks (GPU check, process check, config read)
- Runtime < 5 minutes with standard configs and user has already approved the task
- User explicitly said "run the full experiment" or "launch the benchmark" — proceed, but ping when done
- Experiment completed successfully — report results, ping Discord with summary
- Killing zombie processes from a prior failed run before launching a new one

---

## 9. Post-Experiment Reporting

After an experiment completes:

**Step 1: Check results.**
```bash
find /path/to/results/ -name "*.json" -not -name "summary.json" | wc -l
# Compare to expected number of experiments
```

**Step 2: Ping Discord with summary.**
```bash
~/.claude/ai-ops/scripts/ping_nick.sh "Benchmark complete: X/Y experiments succeeded. [brief finding, e.g. WMSE mean log_z = -123.4]. Results in notebooks/3-2026/claude_experiments/my_experiment/results/"
```

**Step 3: Update lab_notebook.txt with a dated entry.**

Format:
```
2026-03-09
- Ran WMSE vs UKL benchmark: 118/120 experiments succeeded (2 OOM on BN_22)
- WMSE mean log_z = -123.4, UKL mean log_z = -124.1
- Results in notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/results/
...............
```

**Step 4: Offer follow-up analysis if warranted.** If the results contain interesting patterns (one config outperforms another, failures are concentrated in specific problems, etc.), offer to create analysis/plots as a separate follow-up task.

Build the Discord ping into the script itself (see `run_benchmark.py` lines 441-450 for the pattern) so it fires automatically when the background experiment finishes, even if the Claude Code session has ended.

---

## 10. Common Pitfalls (Lessons from Retrospective)

These are the 7 errors from `docs/retrospective_nbe_experiment_errors.md`. Every experiment execution should verify it is not repeating them.

### Pitfall 1: Device Override (Error 1 from retrospective)

**Pattern to avoid:** Copying `device='cpu'` from a diagnostic script (where CPU is intentional for a 1-epoch run) into a production experiment script (where the config says `device='cuda'`).

**Rule:** Config values are user intent. If `config['device']` says `'cuda'`, use `'cuda'`. If you believe it is wrong, ask — do not silently change it.

**Detection:** Before running, check: `print(config['device'])`. If it says 'cpu' but you have GPUs, re-read the config source to verify.

### Pitfall 2: Invented Timeouts (Error 2 from retrospective)

**Pattern to avoid:** Adding `timeout=600` (or any arbitrary number) to `subprocess.run()` or `proc.communicate()` without prior run data justifying that value.

**Rule:** See CLAUDE.md "Experiment Execution Rules." When the user says "full," they mean full. Use `timeout=None` unless the user specifies a timeout. A missing timeout is better than a wrong one.

**Detection:** Search your script for `timeout=` and verify every instance is either `None` or a value the user specified.

### Pitfall 3: Not Checking Existing Evidence (Error 3 from retrospective)

**Pattern to avoid:** Making assumptions about runtime without checking `lab_notebook.txt`, `nbe_eval_results/`, prior plan SUMMARYs, or the config itself.

**Rule:** Before setting any operational parameter, check: (1) existing run data, (2) compute from known quantities, (3) ask the user. See Section 2.4 and CLAUDE.md "Assumption Escalation."

**Detection:** Before launching, ask yourself: "Have I looked at any prior timing data for this problem/config?"

### Pitfall 4: Not Asking the User (Error 4 from retrospective)

**Pattern to avoid:** Proceeding with uncertain operational parameters (timeout, device, epochs) without pinging Discord.

**Rule:** The cost hierarchy is: asking (10 seconds) << wrong assumption (hours/days). Per CLAUDE.md: ping Discord when uncertain. The threshold for asking should be LOW.

**Detection:** If you are uncertain about anything and are about to make a choice, stop and ping.

### Pitfall 5: Pre-Elimination Measurement (Error 5 from retrospective)

**Pattern to avoid:** Checking bucket widths or bucket state BEFORE running variable elimination. At that point, buckets only contain their original factors. The induced width (and NN-eligibility) is only determined during elimination.

**Rule:** Understand the algorithm lifecycle before instrumenting it. For variable elimination: bucket state changes as messages propagate. Pre-elimination bucket widths are NOT the same as induced widths.

**Detection:** If you need per-bucket scope widths, measure them during elimination via a callback (see `custom_hidden_sizes` callback pattern in CLAUDE.md) or after `get_log_partition_function()` returns.

### Pitfall 6: Dict Iteration Bug (Error 6 from retrospective)

**Pattern to avoid:** `for bucket in fastgm.buckets` iterates over dictionary KEYS (variable labels), not bucket objects.

**Rule:** `fastgm.buckets` is a dict. Always use `.items()`:
```python
for var, bucket in fastgm.buckets.items():  # CORRECT
```

**Detection:** Before writing any loop over `fastgm.buckets`, verify the data structure type: `print(type(fastgm.buckets))`.

### Pitfall 7: Zombie Processes (Error 7 from retrospective)

**Pattern to avoid:** Launching a retry experiment while a prior failed run's worker subprocesses are still alive and using GPU memory.

**Rule:** Before launching any experiment, run the process cleanup check (Section 2.2): `ps aux | grep python | grep -v grep`. Kill stale workers. Verify GPU memory freed via `nvidia-smi`.

**Detection:** If a prior run failed, always assume it left orphaned processes until you verify otherwise.

---

## 11. Quick Reference Card

Copy this checklist for each experiment launch:

```
BEFORE LAUNCH:
  [ ] nvidia-smi -- confirm GPUs available, memory < 50% used
  [ ] ps aux | grep python -- no zombie workers; kill any found
  [ ] Read config -- note device, num_epochs, ecl, iB, hidden_sizes
  [ ] Compute runtime estimate: epochs x NN_buckets x per_epoch_cost
  [ ] Check prior run data (lab_notebook.txt, nbe_eval_results/, SUMMARYs)
  [ ] Ping Discord if estimated runtime > 5 minutes
  [ ] Confirm ZERO config values are being overridden

DURING SCRIPT CREATION:
  [ ] sys.path.insert(0, '/home/cohenn1/NCE') at top
  [ ] torch imported inside worker function (NOT at module level)
  [ ] proc.communicate() has NO timeout argument (or timeout=None)
  [ ] CUDA_VISIBLE_DEVICES set in subprocess env, not globally
  [ ] fastgm.buckets iterated with .items() (dict, not list)
  [ ] Results directory created with mkdir(parents=True, exist_ok=True)
  [ ] Each experiment saves result JSON on success AND on failure
  [ ] Orchestrator does NOT crash on worker failure

AFTER LAUNCH:
  [ ] Process confirmed running: kill -0 $(cat experiment.pid)
  [ ] Log file being written: tail -f results/stdout.log
  [ ] Do NOT block waiting for long experiments -- return control to user
  [ ] Ping Discord confirming launch (include PID and log path)

AFTER COMPLETION:
  [ ] Count results: find results/ -name "*.json" | wc -l
  [ ] Check for failures in summary.json
  [ ] Ping Discord with summary (X/Y succeeded, brief findings)
  [ ] Update lab_notebook.txt with dated entry
```

---

## 12. NCE Package Reference

Key APIs used in experiments:

```python
# Python executable
/home/cohenn1/NCE/venv/bin/python

# Package import path (in scripts)
sys.path.insert(0, '/home/cohenn1/NCE')

# Core inference API
from nce.inference.graphical_model import FastGM
fastgm = FastGM(model=model, nn_config=config, device=config['device'])
log_z = fastgm.get_log_partition_function()  # Returns float log Z

# Benchmark problem sets
from nce.benchmark_problems import nbe_sanity_check
from nce.benchmark_problems.small_problems import small_problems

# Count NN-eligible buckets (run AFTER elimination completes)
nn_buckets = fastgm.get_large_message_buckets(iB=100, ecl=config['ecl'])

# Iterate buckets AFTER inference
for var, bucket in fastgm.buckets.items():
    if hasattr(bucket, 'epochs_trained'):
        print(bucket.label, bucket.epochs_trained, bucket.trained_hidden_sizes)

# Discord notifications
~/.claude/ai-ops/scripts/ping_nick.sh "message here"
```
