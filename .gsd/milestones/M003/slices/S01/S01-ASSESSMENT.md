# S01 Roadmap Assessment

## Verdict: Roadmap unchanged

S01 delivered everything in the boundary map. No new risks emerged. S02's scope and dependencies remain correct.

## What S01 Retired

- **Normalization round-trip correctness** — retired. DataPreprocessor minmax_01 mode with 2 passing round-trip tests (known log10 values, degenerate edge case).
- **Early stopping semantics** — retired. Patience-based stopping implemented with `count > stop_iter` matching NeuroBE's counter logic, verified by test.
- **Weighted MSE coupling** — retired. New `neurobe_weighted_mse` loss operates on [0,1]-normalized targets with IS weights, separate from existing logspace loss path.

## Success Criteria Coverage

- `neurobe_mode: true` produces NeuroBE-faithful training → **S01 ✅**
- NN counts match NeuroBE for all 15 problems → **S02**
- Min-max normalization round-trip verified by tests → **S01 ✅**
- Combined comparison table → **S02**
- `pytest tests/` all green → **S01 ✅** (134 pass), S02 must maintain

All criteria have at least one remaining owner. No blocking issues.

## Requirement Coverage

- R033 (minmax normalization): implemented and tested in S01
- R034 (patience early stopping): implemented and tested in S01
- R035 (neurobe_mode preset): implemented and tested in S01
- R037 (round-trip test): implemented and passing in S01
- R036 (ecl tuning): remains for S02, no changes
- R038 (comparison table): remains for S02, no changes

## Boundary Map

S01→S02 boundary is accurate. S02 consumes neurobe_mode config preset and all training machinery from S01, plus NeuroBE results CSV from `Clean-NeuroBE/results/`.

## Risks for S02

- **ecl off-by-one** (from roadmap) — still unretired, S02's primary risk. D031 documents the formula but it needs per-problem verification against NeuroBE NN counts.
- No new risks surfaced from S01.
