# S01: Hard Bucket Selection & Precomputation — UAT

**Milestone:** M004
**Written:** 2026-03-12

## UAT Type

- UAT mode: artifact-driven
- Why this mode is sufficient: The slice produces cached data files (.pt) and a JSON manifest. Correctness is verified by loading files and checking keys, shapes, finiteness, and manifest consistency — all automated by verify_hard_buckets.py. No UI or runtime behavior to inspect.

## Preconditions

- Pipeline has completed: `ps aux | grep select_hard_buckets.py | grep -v grep` returns no coordinator process
- `data/hard_buckets/` directory exists with .pt files and bucket_list.json
- Python environment activated: `source venv/bin/activate`

## Smoke Test

```bash
python scripts/verify_hard_buckets.py
# Expected: exits 0, prints "N checks passed, 0 failed"
```

## Test Cases

### 1. Verification script passes on real data

1. `python scripts/verify_hard_buckets.py`
2. **Expected:** Exit 0 with per-file PASS results and summary showing 0 failures

### 2. At least 1 hard bucket cached

1. `ls data/hard_buckets/*.pt | wc -l`
2. **Expected:** ≥ 1 file

### 3. Manifest is valid and consistent

1. `python -c "import json; d=json.load(open('data/hard_buckets/bucket_list.json')); print(f'{len(d[\"buckets\"])} hard buckets found'); assert len(d['buckets']) >= 1"`
2. **Expected:** Prints count ≥ 1, no assertion error

### 4. Selection results contain all 24 problems

1. `python -c "import json; d=json.load(open('data/hard_buckets/selection_results.json')); print(f'{len(d)} problems'); assert len(d) == 24"`
2. **Expected:** Prints "24 problems", no assertion error

### 5. .pt files load with correct schema

1. Pick any .pt file: `f=$(ls data/hard_buckets/*.pt | head -1)`
2. `python -c "import torch; d=torch.load('$f', map_location='cpu'); print(list(d.keys())); assert 'exact_fw' in d; assert 'factors' in d; print('OK')"`
3. **Expected:** Prints key list including exact_fw, exact_bw, factors, etc. Prints "OK".

## Edge Cases

### Fewer than 3 hard buckets

1. Check bucket count: `python -c "import json; d=json.load(open('data/hard_buckets/bucket_list.json')); print(len(d['buckets']))"`
2. **Expected:** If < 3, re-run with `--threshold 0.05 --skip-phase1` and repeat verification

### No stale GPU processes after completion

1. `nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l`
2. **Expected:** 0 (no python processes holding GPU memory)

## Failure Signals

- `verify_hard_buckets.py` exits non-zero with per-file FAIL diagnostics
- 0 .pt files in `data/hard_buckets/`
- `bucket_list.json` missing or malformed JSON
- `selection_results.json` has fewer than 24 entries (worker crashes)
- NaN or Inf in cached tensors (verify script checks this)

## Requirements Proved By This UAT

- R039 — Hard bucket selection precomputation: the pipeline identifies hard buckets and saves curated list to disk
- R040 — Precomputed message caching (partial): exact_fw and exact_bw cached per bucket in .pt files; approx_bw at multiple bw_ecl levels is in the schema but validation of all levels deferred to S02

## Not Proven By This UAT

- R040 (full): approx_bw at all specified bw_ecl levels (2^2, 2^3, 2^5, 2^10, 2^15, 2^25) — Phase 2 code writes what's available; S02 will validate training from cached data
- R041–R045: Training harness, multi-GPU benchmark CLI, history tracking, comparison charts — all S02/S03 scope
- Backward message exactness for every problem: proven implicitly by successful Phase 2 precomputation for each bucket, but no explicit per-bucket exactness assertion

## Notes for Tester

- The pipeline takes ~2.5+ hours to run from scratch. Don't re-run unless cached data is missing or corrupt.
- If re-running, use `--skip-phase1` to skip the expensive training phase and only redo Phase 2 precomputation.
- Worker temp files in `/tmp/hard_bucket_selection_*/` can be deleted after verification passes.
- Some problems have 0 NN buckets (all buckets solved exactly) — this is expected and not an error.
