# S06: Logging System

**Goal:** Structured per-bucket training event logging to a configurable JSONL log file during inference.
**Demo:** Set `log_file` path in config, run inference on a small problem, find structured JSONL events (epoch_loss, bucket_training_start/end, early_stopping) in the log file — grep-friendly and machine-parseable.

## Must-Haves

- `log_file` config field in `output` section (default: `None` — disabled)
- New `nce/training_logger.py` module using Python stdlib `logging` with JSONL formatter
- `epoch_loss` event emitted for every training epoch with `bucket_id`, `epoch`, `loss`
- `bucket_training_start` and `bucket_training_end` events with bucket metadata
- `early_stopping` event emitted at every early-exit return point in Trainer.train()
- `val_loss` event emitted when validation loss is recorded
- Logger setup is idempotent (no duplicate handlers on repeated calls)
- No existing print statements replaced — logging is additive
- Logger namespace (`nce.training`) does not interfere with pyGMs logging suppression in stats.py
- Config docs updated with `log_file` field

## Proof Level

- This slice proves: integration
- Real runtime required: yes (must run actual inference to produce log output)
- Human/UAT required: no

## Verification

- `python scripts/verify_logging.py` — runs inference on a small problem with `log_file` set, then validates:
  - Log file exists and is non-empty
  - Every line is valid JSON (JSONL format)
  - At least one `bucket_training_start` event exists
  - At least one `epoch_loss` event exists with `bucket_id`, `epoch`, `loss` fields
  - At least one `bucket_training_end` event exists
  - Events are chronologically ordered by timestamp
  - Running without `log_file` set produces no log file (disabled by default)
- `python -c "from nce.training_logger import setup_training_logger, get_training_logger"` — import check

## Observability / Diagnostics

- Runtime signals: JSONL log file with structured events — this slice IS the observability feature
- Inspection surfaces: The log file itself; `grep bucket_id logfile.jsonl` for per-bucket filtering, `jq` for structured queries
- Failure visibility: If logger setup fails, FastGM.__init__ will raise (no silent swallowing). Missing events are detectable by checking event types in the log file.
- Redaction constraints: None — training metrics contain no secrets or PII

## Integration Closure

- Upstream surfaces consumed: `nce/config_schema.py` (output section for new field), `nce/inference/graphical_model.py` (FastGM.__init__ for logger setup), `nce/inference/bucket.py` (compute_message_nn for bucket events), `nce/neural_networks/train.py` (Trainer.train for epoch/early-stopping events)
- New wiring introduced in this slice: FastGM.__init__ calls `setup_training_logger()` when `log_file` is set; Trainer.train() and bucket.compute_message_nn() emit events via the logger
- What remains before the milestone is truly usable end-to-end: S07 (regression verification)

## Tasks

- [x] **T01: Implement training logger module and wire into inference pipeline** `est:1h`
  - Why: Core implementation — creates the logger, adds the config field, hooks all emission points
  - Files: `nce/training_logger.py`, `nce/config_schema.py`, `nce/inference/graphical_model.py`, `nce/inference/bucket.py`, `nce/neural_networks/train.py`
  - Do: Create `nce/training_logger.py` with `setup_training_logger(log_file_path)` and event emission helpers using JSONL format. Add `log_file` field to output section of config_schema.py. Call setup in FastGM.__init__. Emit `epoch_loss` at train.py line 489, `val_loss` at validation recording sites, `early_stopping` at all `return traced_losses_data` exit points (12 sites), `bucket_training_start`/`bucket_training_end` in bucket.py compute_message_nn. Ensure idempotent handler setup and unbuffered writes.
  - Verify: `python -c "from nce.training_logger import setup_training_logger, get_training_logger"` succeeds; manual smoke test with a small problem confirms log file is created
  - Done when: All 5 event types are emitted from the correct code locations, logger is properly configured via config, no existing behavior changes

- [x] **T02: Verification script and config docs update** `est:30m`
  - Why: Proves R016 is satisfied end-to-end and updates docs for the new field
  - Files: `scripts/verify_logging.py`, `docs/config_reference.md`
  - Do: Write verification script that runs inference on nbe_sanity_check problem[0] with log_file set, then validates JSONL content (valid JSON per line, required event types present, required fields present, chronological ordering). Also run without log_file to confirm disabled-by-default. Update config_reference.md output section with `log_file` row.
  - Verify: `python scripts/verify_logging.py` prints PASS
  - Done when: Verification script passes, config docs include `log_file` field

## Files Likely Touched

- `nce/training_logger.py` (new)
- `nce/config_schema.py`
- `nce/inference/graphical_model.py`
- `nce/inference/bucket.py`
- `nce/neural_networks/train.py`
- `scripts/verify_logging.py` (new)
- `docs/config_reference.md`
