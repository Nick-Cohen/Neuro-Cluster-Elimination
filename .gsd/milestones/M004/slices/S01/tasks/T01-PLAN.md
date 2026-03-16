---
estimated_steps: 6
estimated_files: 3
---

# T01: Write Phase 1 worker and Phase 2 precomputation scripts

**Slice:** S01 — Hard Bucket Selection & Precomputation
**Milestone:** M004

## Description

Create the three scripts that form the hard bucket selection pipeline: (1) a worker that trains one problem with error tracking, (2) a coordinator that spawns workers across GPUs, merges results, identifies hard buckets, and runs Phase 2 precomputation, and (3) a verification script that validates the cached output. All three share a data format contract and must be designed together.

## Steps

1. Write `scripts/select_hard_buckets_worker.py`:
   - CLI args: `--problem-index` (int), `--output-path` (str for JSON results file)
   - Load `small_problems.problems[index]` and `small_problems.configs['default'][index]`
   - Override config: `error_tracking=True`, `loss_fn='unnormalized_kl'`, `sampling_scheme='all'`, `num_epochs=10000`, `bw_ecl=config['ecl']` (same as auto_ecl for that problem)
   - Call `prepare_config()` on the config
   - Create `FastGM(model=model, nn_config=config, device='cuda')`
   - Run `fastgm.eliminate_variables(all=True)`
   - Extract `fastgm.error_tracking_data` — list of `(bucket_label, [(epoch, loss, log_Z_err, abs_log_Z_err), ...])`
   - Write results JSON: `{'problem_index': int, 'problem_key': str, 'model_file': str, 'auto_ecl': int, 'buckets': [{'label': int, 'error_data': [...], 'final_abs_log_Z_err': float, 'num_epochs': int}, ...]}`
   - Print summary line: `[Problem {index}] {key} — {n} NN buckets, {m} hard`
   - Handle errors: catch exceptions, write error JSON with traceback, exit 1

2. Write `scripts/select_hard_buckets.py` — Phase 1 coordinator:
   - CLI args: `--threshold` (float, default 0.1), `--gpus` (str, default '0,1,2,3'), `--output-dir` (str, default 'data/hard_buckets'), `--skip-phase1` (flag to skip selection and reuse existing results)
   - Parse GPU list into integers
   - Create temp dir for worker output files
   - Spawn workers via `subprocess.Popen` with `CUDA_VISIBLE_DEVICES=gpu_id`, round-robin across GPUs (6 problems per GPU for 4 GPUs)
   - Wait for all workers with a completion loop: check `proc.poll()`, print progress, handle failures
   - Merge worker JSON results into `selection_results.json`
   - Identify hard buckets: filter for `final_abs_log_Z_err > threshold`
   - Print summary: total NN buckets, hard buckets found, warn if < 3

3. Write `scripts/select_hard_buckets.py` — Phase 2 precomputation (same file, separate function):
   - For each hard bucket (sequentially on cuda:0):
     - Load model from catalog
     - Create fresh `FastGM(model=model, nn_config=exact_config, device='cuda')` where `exact_config` uses `ecl=auto_ecl` but will use `exact=True` elimination
     - Run `fastgm.eliminate_variables(up_to=fastgm.matching_var(bucket_label), exact=True)` to propagate exact upstream messages
     - Get target bucket: `bucket = fastgm.buckets[fastgm.matching_var(bucket_label)]`
     - Compute `exact_fw = bucket.compute_message_exact()`
     - Compute `exact_bw, _ = get_backward_message(fastgm, bucket_label, iB=100, backward_ecl=2**30, return_factor_list=False)`
     - Build save dict with raw tensors (not FastFactor objects): extract `.table` and `.labels` from each factor
     - `torch.save(save_dict, output_path)`
     - Print progress: `[Precompute {i}/{n}] {problem_key} bucket {label} — fw shape {shape}, bw shape {shape}`

4. Write `scripts/select_hard_buckets.py` — manifest generation:
   - Build `bucket_list.json`: `{'threshold': float, 'selection_date': str, 'num_problems': 24, 'total_nn_buckets': int, 'buckets': [{'id': str, 'problem_key': str, 'bucket_label': int, 'selection_error': float, 'auto_ecl': int, 'file': str}, ...]}`
   - Bucket ID format: `{problem_key_sanitized}__{bucket_label}` (e.g. `bn_BN_3__5`)
   - Write to `{output_dir}/bucket_list.json`

5. Write `scripts/verify_hard_buckets.py`:
   - Load `bucket_list.json`, iterate over buckets
   - For each: load `.pt` file, check required keys (`factors`, `exact_fw`, `exact_bw`, `bucket_label`, `scope`, `domain_sizes`, `elim_vars`, `problem_key`, `auto_ecl`, `selection_error`)
   - Verify `exact_fw` and `exact_bw` have `tensor` and `labels` sub-keys
   - Verify tensor shapes are consistent (product of domain sizes for scope matches fw tensor numel)
   - Verify all tensors are finite (no NaN/Inf in exact messages)
   - Verify manifest file count matches actual .pt file count
   - Print per-bucket status line, final PASS/FAIL summary
   - Exit 0 on all pass, exit 1 on any failure

6. Smoke-test all three scripts for syntax validity: `python -c "import ast; ast.parse(open(f).read())"` for each. Create a minimal mock `.pt` file and run verify against it to confirm the verification logic works.

## Must-Haves

- [ ] Worker handles all 24 small_problems indices (0–23) without hardcoded assumptions
- [ ] Worker sets `bw_ecl = config['ecl']` (auto_ecl per problem) for backward info during training
- [ ] Coordinator uses subprocess with `CUDA_VISIBLE_DEVICES` isolation (torch imported only in worker, never in coordinator before fork)
- [ ] Phase 2 uses `exact=True` elimination for upstream messages
- [ ] Phase 2 calls `get_backward_message(fastgm, bucket_label, iB=100, backward_ecl=2**30, return_factor_list=False)` for exact backward
- [ ] `.pt` files store raw tensors + labels, not FastFactor objects
- [ ] `bucket_list.json` manifest is valid JSON with all specified fields
- [ ] `--threshold` default is 0.1 (D043)
- [ ] Verification script checks keys, shapes, finiteness, and manifest consistency

## Verification

- `python -c "import ast; ast.parse(open('scripts/select_hard_buckets_worker.py').read())"` — no syntax errors
- `python -c "import ast; ast.parse(open('scripts/select_hard_buckets.py').read())"` — no syntax errors
- `python -c "import ast; ast.parse(open('scripts/verify_hard_buckets.py').read())"` — no syntax errors
- Quick smoke test: create a mock `.pt` file with expected schema, run `python scripts/verify_hard_buckets.py --dir /tmp/mock_buckets` — exits 0

## Observability Impact

- Signals added/changed: Worker prints structured progress line per problem; coordinator prints per-GPU worker completion; Phase 2 prints per-bucket precomputation progress with tensor shapes
- How a future agent inspects this: Read `selection_results.json` for full Phase 1 data; read `bucket_list.json` for curated hard bucket list; run `verify_hard_buckets.py` for data integrity check
- Failure state exposed: Worker writes error JSON with traceback on failure; coordinator prints failed worker stderr; Phase 2 includes problem_key + bucket_label in exception messages

## Inputs

- `nce/benchmark_problems/small_problems.py` — 24 models with auto_ecl, default configs
- `nce/inference/graphical_model.py` — FastGM, eliminate_variables(up_to=..., exact=True)
- `nce/neural_networks/train.py` — Trainer with error_tracking pattern
- `nce/utils/backward_message.py` — get_backward_message()
- `notebooks/_1-2026/experiment_runner.py` — subprocess spawning pattern (reference, not imported)
- S01-RESEARCH.md — .pt file schema, constraints, common pitfalls

## Expected Output

- `scripts/select_hard_buckets_worker.py` — standalone subprocess entry point, ~100 lines
- `scripts/select_hard_buckets.py` — coordinator + Phase 2 + manifest, ~250 lines
- `scripts/verify_hard_buckets.py` — validation script, ~80 lines
