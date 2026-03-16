# S01: Hard Bucket Selection & Precomputation

**Goal:** Identify hard buckets across all 24 small_problems and cache precomputed factor tensors + exact messages to disk, ready for benchmark training in S02.
**Demo:** `python scripts/select_hard_buckets.py --threshold 0.1 --gpus 0,1,2,3` runs to completion, producing `data/hard_buckets/*.pt` files and `data/hard_buckets/bucket_list.json`. A verification script confirms all cached data loads correctly with expected tensor shapes and metadata.

## Must-Haves

- Phase 1 worker script runs a single problem through FastGM with `error_tracking=True`, UKL loss, `bw_ecl=ecl`, `auto_ecl`, 10000 epochs, and writes per-bucket error data to a temp JSON file
- Phase 1 coordinator distributes 24 problems across N GPUs via subprocess spawning with `CUDA_VISIBLE_DEVICES` isolation, round-robin assignment
- Coordinator merges worker results and identifies hard buckets where `abs_log_Z_err > threshold`
- Phase 2 precomputation: for each hard bucket, runs exact upstream elimination, computes `exact_fw` and `exact_bw`, caches factor tensors + metadata to `.pt` files
- `--threshold` CLI arg with default 0.1 (D043)
- `bucket_list.json` manifest with bucket IDs, problem keys, local errors, paths
- `.pt` file format matches the schema defined in S01-RESEARCH.md
- Verification script loads all cached data and validates keys, tensor shapes, device mapping
- Prints summary of hard buckets found — warns if < 3 (threshold may need lowering)

## Proof Level

- This slice proves: integration
- Real runtime required: yes — real `small_problems` models, real `FastGM.eliminate_variables()`, real `Trainer.train()`, real `get_backward_message()`
- Human/UAT required: no — verification is automated (load + shape check + count)

## Verification

- `python scripts/verify_hard_buckets.py` — loads each `.pt` file from `data/hard_buckets/`, checks required keys, tensor shapes, dtype, and that `bucket_list.json` is consistent with files on disk. Exit 0 on success, exit 1 with diagnostics on failure.
- `ls data/hard_buckets/*.pt | wc -l` — at least 1 file exists (ideally ≥ 3)
- `python -c "import json; d=json.load(open('data/hard_buckets/bucket_list.json')); print(f'{len(d[\"buckets\"])} hard buckets found'); assert len(d['buckets']) >= 1"` — manifest is valid JSON with at least 1 bucket

## Observability / Diagnostics

- Runtime signals: each worker prints `[Problem X/24] {problem_key} — {n} NN buckets trained, {m} hard (abs_log_Z_err > threshold)` on completion; coordinator prints per-GPU progress and final summary
- Inspection surfaces: `data/hard_buckets/selection_results.json` persists full Phase 1 results (all buckets, all errors, not just hard ones) for post-hoc analysis; `bucket_list.json` for the curated subset
- Failure visibility: worker subprocess stderr captured and printed on non-zero exit; Phase 2 errors include problem_key and bucket_label in exception message
- Redaction constraints: none — no secrets involved

## Integration Closure

- Upstream surfaces consumed: `nce/benchmark_problems/small_problems.py` (24 models + auto_ecl), `nce/inference/graphical_model.py` (FastGM, eliminate_variables), `nce/neural_networks/train.py` (Trainer with error_tracking), `nce/utils/backward_message.py` (get_backward_message)
- New wiring introduced in this slice: `scripts/select_hard_buckets.py` (CLI entry point) + `scripts/select_hard_buckets_worker.py` (subprocess entry point) + `scripts/verify_hard_buckets.py` (verification)
- What remains before the milestone is truly usable end-to-end: S02 (training harness that loads cached .pt files and trains), S03 (multi-GPU CLI, JSONL history, comparison charts)

## Tasks

- [x] **T01: Write Phase 1 worker and Phase 2 precomputation scripts** `est:2h`
  - Why: The worker is the subprocess entry point that runs one problem through error-tracked training. The coordinator spawns workers across GPUs, merges results, identifies hard buckets, and runs Phase 2 precomputation. The verification script proves the output is correct. All three scripts are tightly coupled by data format and must be designed together.
  - Files: `scripts/select_hard_buckets_worker.py`, `scripts/select_hard_buckets.py`, `scripts/verify_hard_buckets.py`
  - Do: (1) Worker: takes `--problem-index`, `--output-path` args; loads small_problems model/config at that index; overrides config with `error_tracking=True`, `bw_ecl=ecl`, `loss_fn='unnormalized_kl'`, `sampling_scheme='all'`, `num_epochs=10000`; runs `FastGM.eliminate_variables(all=True)`; writes `error_tracking_data` to JSON output file. (2) Coordinator: parses `--threshold` (default 0.1), `--gpus` (default '0,1,2,3'), `--output-dir` (default 'data/hard_buckets'); spawns workers round-robin across GPUs; waits for completion; merges results; identifies hard buckets; runs Phase 2 precomputation (exact upstream elimination + exact_fw + exact_bw + torch.save); writes `bucket_list.json` manifest and `selection_results.json`. (3) Verification: loads each .pt file, checks keys match schema, tensor shapes are consistent with metadata, loads manifest and cross-checks.
  - Verify: `python scripts/verify_hard_buckets.py` exits 0 after a successful selection run
  - Done when: All three scripts exist, are syntactically valid (`python -c "import ast; ast.parse(open('scripts/select_hard_buckets.py').read())"`), and the verification script can validate a mock .pt file created in a quick smoke test

- [ ] **T02: Run full selection pipeline on 4 GPUs and verify results** `est:6h`
  - Why: The actual execution retires the three key risks (Trainer coupling, selection cost, hard bucket availability) and produces the cached data that S02 depends on. This is the operational proof.
  - Files: `data/hard_buckets/*.pt`, `data/hard_buckets/bucket_list.json`, `data/hard_buckets/selection_results.json`
  - Do: (1) Pre-flight: check `nvidia-smi` for 4 GPUs available, kill stale python processes. (2) Run `python scripts/select_hard_buckets.py --threshold 0.1 --gpus 0,1,2,3` in background (no timeout — estimated 2–6 hours). (3) Monitor via bg_shell digest. (4) On completion: run `python scripts/verify_hard_buckets.py`. (5) If < 3 hard buckets found at 0.1, re-run with `--threshold 0.05` and note in summary. (6) Ping Discord with results summary.
  - Verify: `python scripts/verify_hard_buckets.py` exits 0; `bucket_list.json` has ≥ 1 bucket entry; at least 1 `.pt` file in `data/hard_buckets/`
  - Done when: Cached `.pt` files exist on disk with valid data, verification passes, and Discord pinged with count of hard buckets found and wall-clock time

## Files Likely Touched

- `scripts/select_hard_buckets_worker.py` (new — Phase 1 subprocess entry point)
- `scripts/select_hard_buckets.py` (new — coordinator + Phase 2 precomputation)
- `scripts/verify_hard_buckets.py` (new — verification/validation script)
- `data/hard_buckets/*.pt` (new — cached bucket data, generated by T02)
- `data/hard_buckets/bucket_list.json` (new — manifest, generated by T02)
- `data/hard_buckets/selection_results.json` (new — full Phase 1 results, generated by T02)
