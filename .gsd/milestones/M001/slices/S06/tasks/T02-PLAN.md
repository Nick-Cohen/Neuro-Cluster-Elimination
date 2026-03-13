---
estimated_steps: 4
estimated_files: 3
---

# T02: Verification script and config docs update

**Slice:** S06 — Logging System
**Milestone:** M001

## Description

Write an end-to-end verification script that runs inference on a small problem with `log_file` configured, then validates the JSONL output satisfies R016. Also update the config reference docs to include the new `log_file` field in the output section.

## Steps

1. **Write `scripts/verify_logging.py`** — Script that:
   - Imports `nbe_sanity_check` benchmark and runs inference on problem[0] with a nested config that sets `log_file` to a temp path.
   - After inference, opens the log file and validates:
     - File exists and is non-empty
     - Every line parses as valid JSON
     - At least one event of each type exists: `bucket_training_start`, `epoch_loss`, `bucket_training_end`
     - Every event has `timestamp`, `event`, `bucket_id` fields
     - `epoch_loss` events have `epoch` and `loss` fields
     - Timestamps are monotonically non-decreasing (chronological order)
   - Runs a second inference without `log_file` set and confirms no log file is created (disabled-by-default check)
   - Prints PASS/FAIL with details on failures
   - Uses a small config (few epochs, small network) to keep runtime under 30 seconds

2. **Update `docs/config_reference.md`** — Add `log_file` row to the output section table with: readable name `log_file`, internal name `log_file`, type `str | None`, default `None`, purpose description explaining JSONL training event logging.

3. **Add doc-sync comment to config_schema.py** — Add inline comment on the `log_file` field entry in NESTED_SECTIONS pointing to the config reference docs (following pattern established by S03).

4. **Run verification** — Execute `python scripts/verify_logging.py` and confirm PASS.

## Must-Haves

- [ ] `scripts/verify_logging.py` exists and is executable
- [ ] Script validates JSONL format (every line is valid JSON)
- [ ] Script validates required event types are present
- [ ] Script validates required fields per event type
- [ ] Script validates chronological ordering
- [ ] Script validates disabled-by-default behavior
- [ ] Script prints clear PASS/FAIL output
- [ ] `docs/config_reference.md` updated with `log_file` field
- [ ] Doc-sync comment added to config_schema.py log_file entry

## Verification

- `python scripts/verify_logging.py` — prints PASS
- `grep 'log_file' docs/config_reference.md` — field is documented
- `grep 'log_file' nce/config_schema.py | grep -i doc` — doc-sync comment exists

## Observability Impact

- Signals added/changed: None (this task adds verification, not runtime signals)
- How a future agent inspects this: Run `python scripts/verify_logging.py` to verify logging system works
- Failure state exposed: Script prints specific failure details (which check failed, expected vs actual)

## Inputs

- `nce/training_logger.py` — T01 output, the logger module being verified
- `nce/config_schema.py` — T01 output, with `log_file` field added
- `nce/benchmark_problems/` — nbe_sanity_check for test problem
- `docs/config_reference.md` — existing output section to extend

## Expected Output

- `scripts/verify_logging.py` — new verification script, PASS on execution
- `docs/config_reference.md` — updated with `log_file` field in output section
- `nce/config_schema.py` — doc-sync comment added to log_file field
