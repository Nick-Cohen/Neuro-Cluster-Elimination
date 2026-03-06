---
phase: 15-run-wmse-vs-ukl-benchmark
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/run_benchmark.py
  - notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/results/  # output directory
autonomous: true
requirements: [BENCH-01]

must_haves:
  truths:
    - "run_benchmark.py runs all 120 experiments (5 configs x 24 problems) to completion"
    - "Each experiment trains for exactly 5000 epochs with no early stopping"
    - "Results are saved as per-experiment JSON files with log_z, duration, config info"
    - "Experiments distribute across 4 GPUs for parallel execution"
    - "Script recovers gracefully from single-experiment failures without aborting"
    - "Discord ping sent when experiment starts and when it finishes"
  artifacts:
    - path: "notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/run_benchmark.py"
      provides: "Main benchmark script"
      min_lines: 200
    - path: "notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/results/"
      provides: "120 per-experiment JSON result files + aggregate summary"
  key_links:
    - from: "run_benchmark.py"
      to: "nce.benchmark_problems.small_problems"
      via: "import small_problems, _AUTO_ECL"
      pattern: "from nce.benchmark_problems.small_problems import"
    - from: "run_benchmark.py"
      to: "nce.inference.graphical_model.FastGM"
      via: "FastGM(model=model, nn_config=cfg)"
      pattern: "FastGM\\(model="
---

<objective>
Create and run the WMSE vs UKL benchmark experiment: 5 configurations x 24 small_problems x 5000 epochs = 120 experiments across 4 GPUs.

Purpose: Compare weighted logspace MSE (NeuroBE's loss) against unnormalized KL with varying backward information levels to understand how loss function and backward message quality affect neural network-based approximate inference.

Output: 120 per-experiment JSON result files with log_z estimates, wall-clock times, and experiment metadata. Aggregate summary JSON for analysis.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/EXPERIMENT_DESIGN.md
@notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/OPEN_QUESTIONS.md
@notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/wmse_config_output.txt
@notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/verify_config_correctness.py
@notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/verify_full_data_training.py
@nce/benchmark_problems/small_problems.py
@notebooks/_1-2026/worker.py (results format reference)
@notebooks/_1-2026/experiment_runner.py (GPU distribution reference)

<interfaces>
<!-- Key types and contracts from the codebase that the executor needs -->

From nce/benchmark_problems/small_problems.py:
```python
from nce.benchmark_problems.small_problems import small_problems, _AUTO_ECL
# small_problems.problems -> list of 24 pyGMs Model objects
# small_problems.configs['default'] -> list of 24 nn_config dicts (one per problem)
# _AUTO_ECL -> dict mapping modelfile -> auto_ecl value
# model.modelfile -> str (e.g., 'BN_1.uai')
```

From nce/inference/graphical_model.py:
```python
from nce.inference.graphical_model import FastGM
# FastGM(model=model, nn_config=cfg) -> constructs inference engine
# fastgm.get_log_partition_function() -> float (log Z estimate)
# fastgm.num_trained -> int (number of NN-trained buckets)
# fastgm.get_large_message_buckets(iB=100, ecl=ecl) -> list of bucket keys
```

From verify_config_correctness.py:
```python
def build_experiment_config(base_cfg, loss_fn, bw_ecl, num_epochs=5000):
    """Reuse this function -- it's already verified correct for all 5 configs."""
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create run_benchmark.py with multi-GPU distribution</name>
  <files>notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/run_benchmark.py</files>
  <action>
Create `/home/cohenn1/NCE/notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/run_benchmark.py` -- a self-contained Python script that runs all 120 experiments.

**Script structure:**

1. **Imports and path setup:**
   - `sys.path.insert(0, '/home/cohenn1/NCE')`
   - Import `small_problems`, `_AUTO_ECL` from `nce.benchmark_problems.small_problems`
   - Import `copy`, `json`, `os`, `time`, `traceback`, `logging`, `subprocess`, `argparse`
   - Import `datetime`, `pathlib.Path`

2. **Reuse `build_experiment_config()` from verify_config_correctness.py** -- copy it directly (it's already verified correct). This function takes `base_cfg, loss_fn, bw_ecl, num_epochs=5000` and returns a complete nn_config dict. Key operations:
   - `copy.deepcopy(base_cfg)` to avoid mutation
   - Sets `loss_fn`, `num_epochs=5000`, `skip_early_stopping=True`, `nbe_early_stopping=False`, `batch_size=10000000`
   - Sets `bw_ecl`, `backward_ecl`, `populate_bw_factors=(bw_ecl > 0)`, `use_bw_approx=(bw_ecl > 0)`

3. **Define the 5 experiment configurations** as a list of dicts:
   ```python
   CONFIGS = [
       {'name': 'wmse_bw0', 'loss_fn': 'weighted_logspace_mse', 'bw_ecl': 0},
       {'name': 'ukl_bw0', 'loss_fn': 'unnormalized_kl', 'bw_ecl': 0},
       {'name': 'ukl_bw8', 'loss_fn': 'unnormalized_kl', 'bw_ecl': 8},
       {'name': 'ukl_bw_ecl', 'loss_fn': 'unnormalized_kl', 'bw_ecl': 'auto_ecl'},
       {'name': 'ukl_bw30', 'loss_fn': 'unnormalized_kl', 'bw_ecl': 2**30},
   ]
   ```

4. **Build all 120 experiment jobs** as a list of dicts:
   ```python
   jobs = []
   for config in CONFIGS:
       for i, (model, base_cfg) in enumerate(zip(small_problems.problems, base_configs)):
           bw_ecl = config['bw_ecl']
           if bw_ecl == 'auto_ecl':
               bw_ecl = base_cfg['ecl']  # per-problem auto_ecl
           nn_config = build_experiment_config(base_cfg, config['loss_fn'], bw_ecl)
           jobs.append({
               'job_id': len(jobs),
               'config_name': config['name'],
               'problem_idx': i,
               'modelfile': model.modelfile,
               'nn_config': nn_config,
               'model': model,  # NOT serialized -- only used in-process
           })
   ```

5. **Worker function** `run_single_experiment(job, results_dir)`:
   - Import torch lazily inside this function (for subprocess usage)
   - Create FastGM: `fastgm = FastGM(model=job['model'], nn_config=job['nn_config'])`
   - Call `log_z = fastgm.get_log_partition_function()`
   - Record: `log_z`, `num_trained` (from `fastgm.num_trained`), `duration_seconds`, `status`
   - On exception: catch, record error + traceback, set `status='failed'`
   - Save result as JSON to `results_dir/{config_name}/{modelfile}.json`
   - Return result dict

6. **GPU-parallel execution using subprocess spawning** (following experiment_runner.py pattern):
   - Detect available GPUs: `torch.cuda.device_count()` (expect 4)
   - The script should support two modes via argparse:
     - `--mode worker --job-id N` -- runs a single job (spawned by orchestrator)
     - `--mode orchestrator` (default) -- distributes jobs across GPUs
   - **Orchestrator mode:**
     - Builds all 120 jobs
     - Assigns jobs to GPUs round-robin
     - Processes in waves (one job per GPU per wave = 4 jobs per wave = 30 waves)
     - For each wave: spawn 4 `subprocess.Popen` processes, each with `CUDA_VISIBLE_DEVICES=N` and `--mode worker --job-id J`
     - Wait for all 4 to complete, log results
     - Continue to next wave
     - Log progress: `[job_id/120] config_name | modelfile | GPU N | duration | status`
   - **Worker mode:**
     - Reads `--job-id`, rebuilds the job list to get the specific job
     - Sets seed, imports torch (respecting CUDA_VISIBLE_DEVICES set by parent)
     - Runs the single experiment
     - Saves result JSON and exits

7. **Results directory structure:**
   ```
   results/
     wmse_bw0/
       smokers_20.uai.json
       BN_3.uai.json
       ...
     ukl_bw0/
       ...
     ukl_bw8/
       ...
     ukl_bw_ecl/
       ...
     ukl_bw30/
       ...
     summary.json    # aggregate across all experiments
     benchmark.log   # full console log
   ```

8. **Per-experiment JSON format** (compatible with worker.py results.json):
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
     "start_time": "2026-03-06T...",
     "end_time": "2026-03-06T...",
     "cuda_device": "NVIDIA TITAN RTX",
     "error": null,
     "traceback": null
   }
   ```

9. **Aggregate summary.json** written at end:
   ```json
   {
     "total_experiments": 120,
     "completed": 118,
     "failed": 2,
     "total_duration_seconds": 36000,
     "per_config_summary": {
       "wmse_bw0": {"completed": 24, "failed": 0, "mean_duration": 180.0},
       ...
     },
     "failed_experiments": [{"job_id": 45, "error": "..."}]
   }
   ```

10. **Logging:**
    - Set up Python `logging` to write to both console (stdout) and `results/benchmark.log`
    - Log: timestamps, progress, per-experiment results, errors, wave summaries
    - At start: log total experiments (120), GPUs available, estimated runtime

11. **Pre-run validation (in orchestrator mode before spawning):**
    - Verify `len(small_problems.problems) == 24`
    - Verify `len(jobs) == 120` (5 configs x 24 problems)
    - Verify GPU availability: `torch.cuda.device_count() >= 1`
    - Print config summary table (config_name, loss_fn, bw_ecl pattern)
    - For the first problem (BN_3, smallest ecl=16383), count NN-eligible buckets using `fastgm.get_large_message_buckets(iB=100, ecl=16383)` on CPU as sanity check
    - Log: "Starting 120 experiments across N GPUs. Estimated ~4-5 min/experiment, ~8-10 hours total (sequential), ~2-3 hours with 4 GPUs."

12. **Error handling:**
    - Worker subprocess failures: capture returncode, stdout, stderr. Log error. Mark job as failed. Continue to next wave.
    - Individual experiment exceptions: caught in worker, saved to result JSON with traceback. Worker exits with code 1.
    - Orchestrator: never crashes on individual job failure. Counts and reports failures at end.

13. **argparse CLI:**
    - `--mode`: 'orchestrator' (default) or 'worker'
    - `--job-id`: integer, required for worker mode
    - `--results-dir`: path to results directory (default: `./results` relative to script)
    - `--gpus`: comma-separated GPU IDs (default: '0,1,2,3')
    - `--dry-run`: print job list and exit without running

**CRITICAL constraints (from CLAUDE.md):**
- NO timeouts on experiments. Workers run until complete.
- NO timeout parameter on subprocess.Popen.communicate(). Let it run.
- num_epochs=5000 exactly. Do not reduce.
- batch_size=10000000 exactly. Do not change.
- seed=42 exactly. Do not change.
- Run worker subprocesses with NO timeout.
  </action>
  <verify>
    <automated>
cd /home/cohenn1/NCE/notebooks/3-2026/claude_experiments/benchmark_wmse_ukl && /home/cohenn1/NCE/venv/bin/python run_benchmark.py --dry-run 2>&1 | head -50
    </automated>
  </verify>
  <done>
run_benchmark.py exists, --dry-run prints all 120 jobs with correct config_name/modelfile/bw_ecl combinations, no import errors, pre-run validation passes (24 problems, 120 jobs, GPU count logged).
  </done>
</task>

<task type="auto">
  <name>Task 2: Run verification scripts, then launch full benchmark in background</name>
  <files>notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/results/</files>
  <action>
**Step 1: Run the two existing verification scripts to confirm setup is correct.**

```bash
cd /home/cohenn1/NCE/notebooks/3-2026/claude_experiments/benchmark_wmse_ukl
/home/cohenn1/NCE/venv/bin/python verify_config_correctness.py
/home/cohenn1/NCE/venv/bin/python verify_full_data_training.py
```

Both must output "PASS" on all checks. If any check fails, STOP and diagnose before proceeding.

**Step 2: Ping Discord that experiment is starting.**

```bash
~/.claude/ai-ops/scripts/ping_nick.sh "WMSE vs UKL benchmark starting: 120 experiments (5 configs x 24 problems x 5000 epochs) across 4 GPUs. Estimated ~2-3 hours."
```

**Step 3: Launch the full benchmark in background with NO timeout.**

```bash
cd /home/cohenn1/NCE/notebooks/3-2026/claude_experiments/benchmark_wmse_ukl
nohup /home/cohenn1/NCE/venv/bin/python run_benchmark.py --mode orchestrator --gpus 0,1,2,3 > results/benchmark_stdout.log 2>&1 &
echo $! > results/benchmark.pid
```

Use the Bash tool with `run_in_background: true` for the actual experiment launch. Do NOT set any timeout.

**Step 4: Verify the process started correctly.**

Check:
- PID file exists and process is running: `kill -0 $(cat results/benchmark.pid)`
- Log file is being written: `ls -la results/benchmark_stdout.log`
- First lines of log show correct job count: `head -20 results/benchmark_stdout.log`

**Step 5: Ping Discord confirming experiment is running.**

```bash
~/.claude/ai-ops/scripts/ping_nick.sh "Benchmark running (PID $(cat results/benchmark.pid)). 120 experiments across 4 GPUs. Results in notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/results/. Check progress: tail -f results/benchmark_stdout.log"
```

**CRITICAL: Do NOT wait for the experiment to finish. Launch in background and verify it started. The experiment will take hours.**

**Step 6: Update lab_notebook.txt with today's entry.**

Add entry for 2026-03-06:
```
2026-03-06
- Created run_benchmark.py for WMSE vs UKL benchmark (5 configs x 24 problems x 5000 epochs = 120 experiments)
- Launched benchmark across 4 TITAN RTX GPUs, estimated ~2-3 hours
- Script uses subprocess spawning for GPU isolation, round-robin distribution (30 waves of 4 jobs)
```
  </action>
  <verify>
    <automated>
kill -0 $(cat /home/cohenn1/NCE/notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/results/benchmark.pid 2>/dev/null) 2>/dev/null && echo "RUNNING" || echo "NOT RUNNING"
    </automated>
  </verify>
  <done>
Both verification scripts pass. Benchmark process is running in background (PID confirmed alive). Log file shows correct startup (120 jobs, 4 GPUs). Discord pinged with start notification. Lab notebook updated.
  </done>
</task>

</tasks>

<verification>
1. `run_benchmark.py --dry-run` prints all 120 experiment jobs with correct parameters
2. `verify_config_correctness.py` passes all checks across 5 configurations
3. `verify_full_data_training.py` passes all batch size and sampling scheme checks
4. Benchmark process is running (`kill -0 PID` succeeds)
5. Log file shows experiments executing on 4 GPUs
6. After completion: `results/summary.json` exists with 120 total experiments, check completed count
</verification>

<success_criteria>
- run_benchmark.py created with all 5 configs, 24 problems, 4-GPU parallel execution
- Both verification scripts pass
- Benchmark launched in background with no timeout
- Discord notified at start
- Process confirmed running
- Lab notebook updated
</success_criteria>

<output>
After completion, create `.planning/quick/15-run-wmse-vs-ukl-benchmark-experiment-5-c/15-SUMMARY.md`

Note: The experiment will run for ~2-3 hours in background. A separate follow-up task should check results and ping Discord when complete. The executor should set up a monitoring check or remind the user to check `tail -f results/benchmark_stdout.log`.
</output>
