---
estimated_steps: 5
estimated_files: 2
---

# T03: End-to-end verification script

**Slice:** S02 — Single-Bucket Training Harness with Plots
**Milestone:** M004

## Description

Write `scripts/verify_benchmark_training.py` — the objective stopping condition for S02. This script exercises the full `train_single_bucket()` pipeline end-to-end: loads a .pt file, runs training with a time limit, and validates the output folder contains correct plots and metrics.

If real .pt files from S01 exist in `data/hard_buckets/`, uses the first one. If not (S01 pipeline hasn't completed Phase 2), generates a synthetic .pt file by running a quick exact elimination on smokers_20 (problem 0, the smallest/fastest problem) and saving it in S01's .pt schema. This makes verification self-contained — S02 doesn't block on S01 completion.

## Steps

1. Create `scripts/verify_benchmark_training.py` with argument parsing:
   - `--device` (default: 'cuda')
   - `--time-limit` (default: 30 seconds)
   - `--output-dir` (default: `/tmp/benchmark_training_verify`)

2. Implement `_generate_synthetic_pt(output_path, device)`:
   - Uses `small_problems.problems[0]` (smokers_20) with its default config
   - Creates FastGM, runs `eliminate_variables(up_to=..., exact=True)` for first NN-eligible bucket
   - Computes exact_fw via `bucket.compute_message_exact()`
   - Computes exact_bw via `get_backward_message(fastgm, bucket_label, iB=100, backward_ecl=2**30)`
   - Saves in S01's .pt schema: factors, exact_fw, exact_bw (tensor + labels), bucket_label, scope, domain_sizes, elim_vars, problem_key, auto_ecl
   - Returns path to saved .pt file

3. Implement `_find_pt_file()`:
   - Checks `data/hard_buckets/bucket_list.json` for manifest
   - If found, returns path to first .pt file listed
   - If not found, checks for any .pt file in `data/hard_buckets/`
   - If nothing found, returns None (triggers synthetic generation)

4. Implement main verification flow:
   - Find or generate .pt file
   - Define nn_config: `{loss_fn: 'unnormalized_kl', hidden_sizes: [3, 3], lr: 0.01, num_epochs: 100000, sampling_scheme: 'all', batch_size: 100000, set_size: 100000, num_samples: 100000, seed: 42, device: args.device}` + standard required fields
   - Call `train_single_bucket(pt_path, nn_config, time_limit, output_dir, device)`
   - Validate result dict has expected keys and values make sense

5. Implement output validation assertions:
   - `assert result['epochs_completed'] > 0`, "Training didn't run"
   - `assert len(result['error_tracking_data']) > 0`, "No error tracking data"
   - `assert result['wall_time'] > 0`, "No wall time recorded"
   - Check output folder exists with loss.png, local_error.png, metrics.json
   - Parse metrics.json and assert all required keys present
   - Check PNG files are > 0 bytes
   - Print summary: bucket_id, epochs completed, final loss, final local error, wall time
   - Exit 0 on success, exit 1 on failure with diagnostic message

## Must-Haves

- [ ] Script can run without S01's .pt files (synthetic generation fallback)
- [ ] Validates output folder structure: `{output_dir}/{bucket_id}/loss.png`, `local_error.png`, `metrics.json`
- [ ] Validates metrics.json has all required keys
- [ ] Validates epochs_completed > 0 and error_tracking_data is non-empty
- [ ] Prints clear PASS/FAIL summary with diagnostics on failure
- [ ] Exit code 0 on success, 1 on failure

## Verification

- `python scripts/verify_benchmark_training.py --time-limit 30` exits 0
- Output folder in /tmp contains valid PNG files and parseable metrics.json
- Script works both with and without real .pt files

## Observability Impact

- Signals added/changed: Script prints structured verification results (PASS/FAIL per check)
- How a future agent inspects this: Run the script — exit code tells the story; stdout has details
- Failure state exposed: On failure, prints which specific check failed and the actual vs expected values

## Inputs

- `nce/benchmark/training.py` from T01 — the function under test
- `nce/benchmark/plots.py` from T02 — expected to produce output files
- `data/hard_buckets/*.pt` — real data if available, otherwise synthetic fallback
- `nce/benchmark_problems/small_problems.py` — for synthetic .pt generation

## Expected Output

- `scripts/verify_benchmark_training.py` — ~150 lines, self-contained verification script
- `/tmp/benchmark_training_verify/{bucket_id}/` — output folder with loss.png, local_error.png, metrics.json (generated at runtime)
