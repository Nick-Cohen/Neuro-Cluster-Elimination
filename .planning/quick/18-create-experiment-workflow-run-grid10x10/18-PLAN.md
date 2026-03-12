---
phase: quick-18
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - notebooks/3-2026/claude_experiments/grid10x10_ukl/run_grid10x10_ukl.py
  - notebooks/3-2026/claude_experiments/grid10x10_ukl/analyze_results.py
autonomous: true
requirements: [quick-18]
must_haves:
  truths:
    - "grid10x10.f10.uai runs with UKL loss, bw_ecl=0, ecl=1024, num_epochs=500 on CUDA"
    - "CSV output contains all requested columns: problem_name, width, num_vars, log_Z_ground_truth, log_Z_hat, err, abs_err, num_trained, time, num_samples, architecture, loss_fn"
    - "Pickle file contains full experimental results (fastgm state, config, per-bucket data)"
    - "Analysis script produces bar chart and summary table from pickle"
  artifacts:
    - path: "notebooks/3-2026/claude_experiments/grid10x10_ukl/run_grid10x10_ukl.py"
      provides: "Experiment runner script"
    - path: "notebooks/3-2026/claude_experiments/grid10x10_ukl/analyze_results.py"
      provides: "Pickle-to-graphs-and-tables analysis script"
    - path: "notebooks/3-2026/claude_experiments/grid10x10_ukl/results/"
      provides: "Output directory with CSV, pickle, and plots"
  key_links:
    - from: "run_grid10x10_ukl.py"
      to: "nce.benchmark_problems.catalog_utils.get_catalog"
      via: "catalog['grids/grid10x10.f10'] to load model"
    - from: "run_grid10x10_ukl.py"
      to: "nce.inference.graphical_model.FastGM"
      via: "FastGM(model=model, nn_config=config) then get_log_partition_function()"
    - from: "analyze_results.py"
      to: "results/experiment_results.pkl"
      via: "pickle.load to reconstruct full experiment state"
---

<objective>
Create a self-contained experiment workflow that runs grid10x10.f10 with UKL loss on CUDA, produces a CSV of all metrics, pickles all outputs for reproducibility, and includes an analysis script for generating graphs and tables from the pickle.

Purpose: Establish a reproducible experiment pattern that future Claude Code agents can follow.
Output: Runner script, analysis script, CSV results, pickle file, and plots.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@docs/experiment_execution_guide.md
@nce/benchmark_problems/catalog_utils.py
@nce/benchmark_problems/nbe_sanity_check.py
@nce/inference/graphical_model.py
@notebooks/3-2026/claude_experiments/benchmark_wmse_ukl/run_benchmark.py

Key facts from codebase exploration:
- Catalog model attributes: model.PR (exact log Z in log10), model.num_vars, model.width, model.modelfile
- grid10x10.f10: width=12, num_vars=100, PR=303.085957 (log10 space)
- grid10x10.f10.wrap: width=21, num_vars=100, PR=333.321335
- FastGM.num_trained incremented in bucket.compute_message_nn() (in nce/inference/bucket.py line 339/494/725/737)
- FastGM.get_log_partition_function() is the correct API (not run())
- FastBucket.epochs_trained and FastBucket.trained_hidden_sizes available after compute_message_nn()
- Torch must be imported INSIDE worker function to respect CUDA_VISIBLE_DEVICES
- Config uses ecl=2**10=1024, bw_ecl=0, loss_fn='unnormalized_kl', num_epochs=500
- With iB=10 and ecl=1024, grid10x10.f10 (width=12) will have NN-eligible buckets
- Reference runtime: grid10x10.f5.wrap with 500 epochs/33 NN buckets takes ~5 min on CUDA
</context>

<interfaces>
From nce/inference/graphical_model.py:
```python
class FastGM:
    num_trained: int  # incremented in bucket.compute_message_nn()
    buckets: dict     # var -> FastBucket (available after elimination)
    log_partition_function: float  # set after eliminate_variables(all=True)

    def get_log_partition_function(self) -> float:
        """Returns log partition function, computing if needed."""

    def get_large_message_buckets(self, iB=None, ecl=None) -> list:
        """Returns list of bucket vars that exceed iB or ecl thresholds."""
```

From nce/inference/bucket.py (after compute_message_nn):
```python
class FastBucket:
    epochs_trained: int           # actual epochs run (set after training)
    trained_hidden_sizes: list    # resolved hidden sizes used
    label: int                    # bucket variable label
```

From pyGMs Catalog Model:
```python
class Model:
    modelfile: str    # e.g. 'grid10x10.f10.uai'
    num_vars: int     # e.g. 100
    width: int        # treewidth, e.g. 12
    PR: float         # exact log Z (log10), e.g. 303.085957
```
</interfaces>

<tasks>

<task type="auto">
  <name>Task 1: Create experiment runner and analysis scripts</name>
  <files>
    notebooks/3-2026/claude_experiments/grid10x10_ukl/run_grid10x10_ukl.py
    notebooks/3-2026/claude_experiments/grid10x10_ukl/analyze_results.py
  </files>
  <action>
Create `notebooks/3-2026/claude_experiments/grid10x10_ukl/run_grid10x10_ukl.py`:

A self-contained script that:

1. **Config construction** (do NOT use nbe_sanity_check configs -- build a fresh config dict):
   - loss_fn='unnormalized_kl'
   - bw_ecl=0, backward_ecl=0
   - ecl=2**10 (1024)
   - num_epochs=500
   - iB=10
   - device='cuda'
   - hidden_sizes='nbe,1' (NeuroBE adaptive hidden sizes)
   - num_samples='nbe,0.35' (NeuroBE adaptive)
   - sampling_scheme='uniform'
   - batch_size=256
   - dope_factors=True
   - approximation_method='nn'
   - seed=42
   - All other fields populated following the nbe_sanity_check config template (see nbe_sanity_check.py _build_nbe_configs for the full 42-field template)
   - Print the full config dict at start

2. **Model loading**: Use `get_catalog()['grids/grid10x10.f10']` to load the model. Extract model.PR, model.num_vars, model.width, model.modelfile before running.

3. **Pre-flight**: Check CUDA availability with torch. Print GPU name and memory. Do NOT run if CUDA not available (exit with error).

4. **Run inference**:
   - `fastgm = FastGM(model=model, nn_config=config, device='cuda')`
   - `log_z_hat = fastgm.get_log_partition_function()`
   - Time the full inference with time.time()

5. **Collect per-bucket data** after inference:
   - Iterate `fastgm.buckets.items()` (it's a dict, var -> bucket)
   - For each bucket with `hasattr(bucket, 'epochs_trained')`:
     - Record: bucket.label, bucket.epochs_trained, bucket.trained_hidden_sizes

6. **Compute metrics**:
   - log_Z_star = model.PR (exact log Z from catalog, log10 space)
   - log_Z_hat = float(log_z_hat) (the NCE estimate)
   - err = log_Z_hat - log_Z_star
   - abs_err = abs(err)
   - num_trained = fastgm.num_trained
   - num_samples = config['num_samples'] (the config string, e.g. 'nbe,0.35')

7. **Output CSV** to `results/grid10x10_f10_ukl_results.csv`:
   - Header: problem_name,width,num_vars,log_Z_star,log_Z_hat,err,abs_err,num_trained,time_seconds,num_samples,architecture,loss_fn
   - Single data row with all values
   - architecture = str(config['hidden_sizes'])

8. **Pickle all results** to `results/experiment_results.pkl`:
   - A dict containing: config (full dict), model_name, model_width, model_num_vars, model_PR, log_z_hat, num_trained, duration_seconds, per_bucket_data (list of dicts with label/epochs_trained/hidden_sizes), timestamp (ISO format)
   - Use `pickle.dump(results_dict, f)` with protocol=pickle.HIGHEST_PROTOCOL

9. **Print summary**: Full config, model info, log Z comparison, per-bucket training info, file paths

10. **Discord ping** when complete using `~/.claude/ai-ops/scripts/ping_nick.sh`

Create `notebooks/3-2026/claude_experiments/grid10x10_ukl/analyze_results.py`:

A script that loads the pickle and produces:
1. **Summary table** printed to console: all CSV columns formatted nicely
2. **Bar chart** (matplotlib): log_Z_star vs log_Z_hat side-by-side bars with value labels, saved as `results/logz_comparison.png`
3. **Per-bucket epochs histogram** (matplotlib): histogram of epochs_trained values across all NN buckets, saved as `results/bucket_epochs_histogram.png`
4. **Per-bucket hidden sizes table**: printed to console showing bucket label, hidden sizes, epochs trained
5. **Regenerate CSV** from pickle (proves the pickle has all needed data)
6. The script should take an optional --pickle-path argument (default: results/experiment_results.pkl)

Both scripts must have:
- `sys.path.insert(0, '/home/cohenn1/NCE')` at top
- `#!/usr/bin/env python3` shebang
- Clear docstrings explaining usage
- Absolute paths for NCE imports

IMPORTANT: Follow the Pre-Flight Checklist from CLAUDE.md before running:
- Check CUDA with nvidia-smi
- Check for zombie processes
- Estimate runtime: grid10x10.f10 with iB=10, ecl=1024 -- check how many NN-eligible buckets exist. Width=12, so with iB=10 some buckets will exceed. With ecl=1024=2^10 and binary variables, buckets with scope > 10 vars need NNs. Estimate ~5-15 NN buckets. At 500 epochs each, ~5 min total on CUDA. Ping Discord if > 5 min estimated.
- Run experiment in background if estimated > 5 min

Do NOT invent timeouts. Run without timeout. If the experiment takes long, that is fine.
  </action>
  <verify>
    Both Python files exist and are syntactically valid:
    /home/cohenn1/NCE/venv/bin/python -c "import py_compile; py_compile.compile('notebooks/3-2026/claude_experiments/grid10x10_ukl/run_grid10x10_ukl.py', doraise=True); py_compile.compile('notebooks/3-2026/claude_experiments/grid10x10_ukl/analyze_results.py', doraise=True); print('OK')"
  </verify>
  <done>
    Both scripts created and syntax-valid. run_grid10x10_ukl.py contains full config with all 42 fields, loads grid10x10.f10 from catalog, runs FastGM with UKL loss, outputs CSV and pickle. analyze_results.py loads pickle and produces plots and tables.
  </done>
</task>

<task type="auto">
  <name>Task 2: Execute experiment and verify outputs</name>
  <files>
    notebooks/3-2026/claude_experiments/grid10x10_ukl/results/grid10x10_f10_ukl_results.csv
    notebooks/3-2026/claude_experiments/grid10x10_ukl/results/experiment_results.pkl
    notebooks/3-2026/claude_experiments/grid10x10_ukl/results/logz_comparison.png
    notebooks/3-2026/claude_experiments/grid10x10_ukl/results/bucket_epochs_histogram.png
  </files>
  <action>
BEFORE running:
1. Run nvidia-smi to verify CUDA GPUs are available and have free memory
2. Run `ps aux | grep python | grep -v grep` to check for zombie processes
3. Estimate runtime: need to determine how many NN-eligible buckets grid10x10.f10 has with iB=10, ecl=1024. Can do a quick dry-run count:
   ```
   python -c "
   from nce.benchmark_problems.catalog_utils import get_catalog
   from nce.inference.graphical_model import FastGM
   c = get_catalog()
   m = c['grids/grid10x10.f10']
   config = {'iB': 10, 'ecl': 1024, 'num_epochs': 0, 'device': 'cpu', ...minimal config...}
   # Or just estimate: width=12, iB=10, ecl=1024=2^10, binary vars -> buckets with scope>10 vars need NNs
   "
   ```
4. If estimated runtime > 5 minutes, ping Discord BEFORE launching:
   `~/.claude/ai-ops/scripts/ping_nick.sh "About to run grid10x10.f10 UKL experiment, est ~X min on CUDA"`

EXECUTE:
- Run: `/home/cohenn1/NCE/venv/bin/python notebooks/3-2026/claude_experiments/grid10x10_ukl/run_grid10x10_ukl.py`
- If estimated > 5 min, run in background with no timeout
- Wait for completion

THEN run the analysis script:
- `/home/cohenn1/NCE/venv/bin/python notebooks/3-2026/claude_experiments/grid10x10_ukl/analyze_results.py`

VERIFY outputs:
- CSV exists at results/grid10x10_f10_ukl_results.csv and has correct headers
- Pickle exists at results/experiment_results.pkl
- Plots generated: logz_comparison.png, bucket_epochs_histogram.png
- Print CSV contents to verify all columns populated
  </action>
  <verify>
    ls -la notebooks/3-2026/claude_experiments/grid10x10_ukl/results/ && cat notebooks/3-2026/claude_experiments/grid10x10_ukl/results/grid10x10_f10_ukl_results.csv
  </verify>
  <done>
    Experiment completed successfully on CUDA. CSV contains all 12 columns (problem_name, width, num_vars, log_Z_star, log_Z_hat, err, abs_err, num_trained, time_seconds, num_samples, architecture, loss_fn) with valid values. Pickle file exists. Analysis plots generated. Discord ping sent.
  </done>
</task>

</tasks>

<verification>
1. CSV file has exactly 12 columns matching the requested schema
2. Pickle file is loadable and contains full config, model info, per-bucket data, and log Z values
3. analyze_results.py can regenerate CSV from pickle (round-trip test)
4. All plots exist as PNG files in results/
5. Both scripts have clear docstrings explaining usage for future Claude Code agents
</verification>

<success_criteria>
- grid10x10.f10 ran with UKL loss (unnormalized_kl), bw_ecl=0, ecl=1024, 500 epochs on CUDA
- CSV contains: problem_name, width, num_vars, log_Z_star, log_Z_hat, err, abs_err, num_trained, time_seconds, num_samples, architecture, loss_fn
- Pickle contains complete experiment state for reproduction
- analyze_results.py produces graphs and tables from pickle alone
- Scripts are documented for future Claude Code agents
</success_criteria>

<output>
After completion, create `.planning/quick/18-create-experiment-workflow-run-grid10x10/18-SUMMARY.md`
</output>
