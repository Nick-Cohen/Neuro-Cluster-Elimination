# Queue

<!-- Append-only log of queued milestones. -->

## 2026-03-12

### M003: NeuroBE Reproduction Mode
- **Why:** Direct comparison between NCE and NeuroBE C++ on 15 binary-domain problems requires matching algorithm details (normalization, early stopping, loss, batch size)
- **Key deliverables:** neurobe_mode config flag, min-max [0,1] normalization with denormalization, patience-2 early stopping, matched ecl values per problem, normalization round-trip tests, combined comparison results table
- **Requirements:** R033–R038
- **Depends on:** M002 (test infrastructure)

## 2026-03-15

### M004: Single-Bucket Learning Benchmark
- **Why:** No systematic way to evaluate whether config changes improve NN learning quality on hard buckets. Need a reusable benchmark with historical tracking to compare loss functions, hyperparameters, and training strategies over time.
- **Key deliverables:** One-time hard bucket selection across 24 small_problems, precomputed message caching, time-limited single-bucket training (fast=1min, slow=1h), multi-GPU parallel execution, per-bucket output plots, JSONL history with comparison charts, CLI entry point
- **Requirements:** R039–R045
- **Depends on:** None (uses existing infrastructure from M001/M002; independent of M003)
