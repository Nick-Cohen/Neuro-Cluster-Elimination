# Queue

<!-- Append-only log of queued milestones. -->

## 2026-03-12

### M003: NeuroBE Reproduction Mode
- **Why:** Direct comparison between NCE and NeuroBE C++ on 15 binary-domain problems requires matching algorithm details (normalization, early stopping, loss, batch size)
- **Key deliverables:** neurobe_mode config flag, min-max [0,1] normalization with denormalization, patience-2 early stopping, matched ecl values per problem, normalization round-trip tests, combined comparison results table
- **Requirements:** R033–R038
- **Depends on:** M002 (test infrastructure)
