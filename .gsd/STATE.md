# GSD State

**Active Milestone:** M004 — Single-Bucket Learning Benchmark
**Active Slice:** None
**Phase:** planned
**Requirements Status:** 7 active (R039–R045 mapped to M004) · 29 validated · 4 deferred · 4 out of scope

## Milestone Registry
- ✅ **M001:** Config & Visualization
- ✅ **M002:** Test Suite
- ✅ **M003:** NeuroBE Reproduction Mode
- 🔄 **M004:** Single-Bucket Learning Benchmark — roadmap complete, 3 slices planned

## Slice Status (M004)
- [ ] **S01:** Hard Bucket Selection & Precomputation `risk:high`
- [ ] **S02:** Single-Bucket Training Harness with Plots `risk:medium`
- [ ] **S03:** Multi-GPU CLI, History Tracking & Comparison `risk:low`

## Recent Decisions
- D038: Bucket reconstruction via eliminate_variables(up_to=...), not FastGM stub
- D039: Benchmark as standalone scripts + nce/benchmark/ module
- D040: Three-slice risk-ordered structure
- D041: Precomputed data as .pt files with JSON manifest
- D042: Per-worker JSONL + coordinator merge
- D043: Configurable hardness threshold (default 0.1)

## Blockers
- None

## Next Action
Begin S01: Hard Bucket Selection & Precomputation
