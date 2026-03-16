# GSD State

**Active Milestone:** M004 — Single-Bucket Learning Benchmark
**Active Slice:** S01 — Hard Bucket Selection & Precomputation
**Phase:** executing
**Requirements Status:** 7 active · 17 validated · 4 deferred · 4 out of scope

## Milestone Registry
- ✅ **M001:** Config & Visualization
- ✅ **M002:** Test Suite
- ✅ **M003:** NeuroBE Reproduction Mode
- 🔄 **M004:** Single-Bucket Learning Benchmark

## Recent Decisions
- Fixed Phase 1 coordinator to use worker pool (max 1 per GPU) — original all-at-once spawning caused OOM

## Blockers
- None

## Next Action
Resume T02: Pipeline is running autonomously (coordinator PID 3305375). On resume: check if completed, run verification, finalize must-haves, ping Discord. See T02-SUMMARY.md resume instructions.

## Active Process
- Selection pipeline: `python scripts/select_hard_buckets.py --threshold 0.1 --gpus 0,1,2,3`
- Coordinator PID: 3305375
- Worker temp dir: `/tmp/hard_bucket_selection_hvmlbbvv/`
- Progress at handoff: 10/24 problems complete, 2 hard buckets found (or_chain_10 buckets 88, 154)
