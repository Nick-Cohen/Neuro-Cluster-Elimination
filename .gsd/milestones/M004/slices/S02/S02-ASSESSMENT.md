# S02 Roadmap Assessment

## Verdict: Roadmap is fine — no changes needed.

## What S02 Delivered

- `nce/benchmark/training.py` with `train_single_bucket()` — custom epoch loop with time-limit, checkpoint error tracking, preloaded exact messages
- `nce/benchmark/plots.py` with `plot_loss_curve()` and `plot_local_error_curve()` — standalone matplotlib functions
- `scripts/verify_benchmark_training.py` — end-to-end verification with synthetic .pt fallback, all 8 checks pass on CUDA
- Per-bucket output folder structure: `{output_dir}/{bucket_id}/loss.png`, `local_error.png`, `metrics.json`

All three tasks (T01–T03) completed and verified. The S02→S03 boundary contract is satisfied exactly.

## Success Criteria Coverage

- `select_hard_buckets.py` identifies hard buckets → S01 [x] (done)
- `bucket_benchmark.py config.yaml fast --gpus 0,1,2,3` produces per-bucket output → S03
- JSONL history file with config hash, timing, epochs, local errors → S03
- Comparison chart vs historical best filtered by duration → S03
- Multi-GPU 1-bucket-per-GPU cycling → S03

All criteria have at least one remaining owner. Coverage check passes.

## Risk Retirement

S02 retired its target risk (medium — whether a custom epoch loop could replace Trainer.train() for benchmark training). Answer: yes, Trainer is used for __init__ chain only; the custom loop is ~40 lines and works correctly (D047).

## Boundary Map Accuracy

S02→S03 produces exactly match what was specified:
- ✅ `train_single_bucket()` function with correct return dict
- ✅ Per-bucket output folder structure with loss.png, local_error.png, metrics.json
- ✅ Plot functions exported from `nce/benchmark`

## Operational Note

S01's cached .pt files are not currently on disk (`data/hard_buckets/` is empty). The selection pipeline scripts exist but the data wasn't persisted or was cleaned up. S03 will need to either re-run the selection pipeline or confirm the data exists before proceeding. This is an operational precondition, not a roadmap change — S03's boundary map already specifies consuming S01's .pt files.

## Requirement Coverage

R039–R040 (S01, validated by prior slice). R041 (time-limited single-bucket training) is now functionally complete from S02 — formal validation deferred to S03's end-to-end run. R042–R045 are S03's scope. No requirement ownership changes needed.
