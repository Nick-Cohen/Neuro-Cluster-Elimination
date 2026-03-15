---
estimated_steps: 4
estimated_files: 3
---

# T02: Run neurobe_mode experiments on all 15 problems

**Slice:** S02 — ECL Tuning & Comparison Experiments
**Milestone:** M003

## Description

Run all 15 binary-domain problems through NCE neurobe_mode inference on CUDA. This is the GPU-intensive step that produces the raw results for the comparison table (R038). Each problem trains 1–4 NNs with up to 500 epochs (patience-based early stopping will cut most shorter). Based on NeuroBE runtimes (~1.5 hrs total), NCE should take similar order of magnitude. The experiment script catches per-problem exceptions so one failure doesn't kill the run.

## Steps

1. **Check GPU availability and clean up stale processes** — Run `nvidia-smi` to verify GPU is free. Run `ps aux | grep python` to check for zombie processes from prior runs. Kill any stale workers.

2. **Write `scripts/run_neurobe_experiments.py`** — Script that:
   - Imports `neurobe_binary` benchmark set from T01
   - Iterates all 15 (model, config) pairs
   - For each: creates `FastGM(model=model, nn_config=config, device='cuda')`, calls `run()`, captures `fastgm.logZ` (the log10 partition function estimate), `fastgm.num_trained` (NN count), and elapsed wall-clock time
   - Wraps each problem in try/except — logs failures, continues to next
   - Prints progress per problem: name, NNs trained, log_Z, time
   - Writes results to CSV at `notebooks/March-2025/neurobe_comparison_results.csv` with columns: Problem, NCE_log_Z, NCE_NNs, NCE_time_hrs, Status (success/failed), Error (if any)
   - Prints summary at end: N/15 succeeded, total time

3. **Run experiments in background** — Launch `python scripts/run_neurobe_experiments.py` via `bg_shell` with no timeout. Estimated runtime 1–3 hours. Monitor via `digest`/`highlights`.

4. **Collect and verify results** — When complete, verify CSV has 15 rows, all Status=success. Check that NCE_NNs match expected NeuroBE NN counts (same verification as T01 but from actual training).

## Must-Haves

- [ ] GPU available and free before launch (nvidia-smi check)
- [ ] Experiment script handles per-problem failures gracefully (try/except, continue)
- [ ] All 15 problems complete successfully (Status=success in CSV)
- [ ] Results CSV exists at `notebooks/March-2025/neurobe_comparison_results.csv`
- [ ] Each row has Problem, NCE_log_Z, NCE_NNs, NCE_time_hrs values

## Verification

- `wc -l notebooks/March-2025/neurobe_comparison_results.csv` → 16 (header + 15 data rows)
- `grep -c "success" notebooks/March-2025/neurobe_comparison_results.csv` → 15
- Visual check: NCE_NNs column matches NeuroBE NN counts from CSV

## Observability Impact

- Signals added/changed: Per-problem progress prints (name, NNs, log_Z, time) during inference run
- How a future agent inspects this: Read `notebooks/March-2025/neurobe_comparison_results.csv` for all results; check Status column for failures
- Failure state exposed: Failed problems have Status=failed and Error column populated with exception message

## Inputs

- `nce/benchmark_problems/neurobe_binary.py` — T01's benchmark module with 15 problems and configs
- `scripts/verify_nn_counts.py` passed — confirms NN counts match before GPU run
- GPU availability confirmed via nvidia-smi
- S01's neurobe_mode machinery: DataPreprocessor minmax_01, neurobe_weighted_mse loss, patience early stopping, ReLU activation, neurobe hidden sizes

## Expected Output

- `scripts/run_neurobe_experiments.py` — experiment runner script
- `notebooks/March-2025/neurobe_comparison_results.csv` — raw NCE results (15 rows)
