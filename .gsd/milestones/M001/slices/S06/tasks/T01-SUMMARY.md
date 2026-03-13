---
id: T01
parent: S06
milestone: M001
provides:
  - nce/training_logger.py leaf module with JSONL event logger
  - log_file config field in output section
  - Logger wiring in FastGM.__init__, bucket.py, and train.py
key_files:
  - nce/training_logger.py
  - nce/config_schema.py
  - nce/inference/graphical_model.py
  - nce/inference/bucket.py
  - nce/neural_networks/train.py
key_decisions:
  - Used dedicated nce.training logger namespace (not root) to avoid conflicts with stats.py mute_everything
  - Custom _FlushHandler subclass ensures flush-after-every-write for crash safety
  - Skipped emitting early_stopping on the final normal-completion return (9th site) — it's not an early exit
patterns_established:
  - Event emission pattern: guard with `if self.bucket.gm._training_logger:` then call helper
  - Logger accessed via gm._training_logger attribute (FastGM owns the logger, consumers reach through gm reference)
observability_surfaces:
  - JSONL log file with 5 event types: epoch_loss, val_loss, early_stopping, bucket_training_start, bucket_training_end
  - Each event has timestamp (ISO UTC), event type, and bucket_id fields
  - grep/jq friendly: `grep '"event":"epoch_loss"' logfile.jsonl`
duration: 25m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T01: Implement training logger module and wire into inference pipeline

**Created JSONL training logger as leaf module, added log_file config field, and wired emission points into FastGM, bucket, and trainer.**

## What Happened

Created `nce/training_logger.py` with `setup_training_logger()` (idempotent file handler setup), `get_training_logger()`, and 5 event emission helpers. Each helper emits a single JSON line with timestamp, event type, bucket_id, and event-specific fields via the `nce.training` logger namespace.

Added `log_file` field to the `output` section of `config_schema.py` with default `None` (disabled by default).

Wired `FastGM.__init__` to call `setup_training_logger()` when `log_file` is set, storing the logger as `self._training_logger` (or `None` if disabled).

Hooked `bucket_training_start`/`bucket_training_end` in `compute_message_nn()` — start emits before `t.train()`, end emits after per_bucket_training_log append.

Hooked `epoch_loss` at the `self.losses.append()` call in `Trainer.train()`, `val_loss` at the `self.val_losses.append()` call, and `early_stopping` at all 8 active early-exit return points with descriptive reason strings.

## Verification

- `python -c "from nce.training_logger import setup_training_logger, get_training_logger"` — PASS
- `python -c "from nce.config_schema import NESTED_SECTIONS; assert 'log_file' in dict(NESTED_SECTIONS)['output']"` — PASS
- `grep -c "log_early_stopping(" nce/neural_networks/train.py` → 8 (matches 8 active early-exit return points)
- `grep "training_logger" nce/inference/bucket.py nce/inference/graphical_model.py` — confirms wiring in both files
- Functional test: wrote 6 events to temp JSONL file, validated JSON parsing, required fields, and idempotent handler setup — PASS
- Import chain test: FastGM, FastBucket, Trainer all import cleanly — PASS
- Leaf module constraint: only stdlib imports (json, logging, datetime) — PASS
- `_training_logger = None` when no log_file configured — PASS

### Slice-level verification (partial — T02 creates the full script):
- `python -c "from nce.training_logger import ..."` — PASS (import check)
- `python scripts/verify_logging.py` — not yet created (T02 scope)

## Diagnostics

- Inspect log output: `cat <log_file>` for raw JSONL, `grep '"event":"epoch_loss"' <log_file>` for filtering
- Check logger state: `python -c "import logging; l = logging.getLogger('nce.training'); print(len(l.handlers), l.level)"`
- Missing events indicate a hook point was missed; empty log file with log_file configured indicates setup failure
- Logger namespace `nce.training` won't be suppressed by `stats.py`'s `logging.disable(CRITICAL)` because we set `propagate=False` on the logger

## Deviations

- Plan says "12 sites" for early_stopping hooks. Actual count: 9 active `return traced_losses_data` sites (3 are commented out). Of those 9, 8 are actual early exits and 1 is normal completion. Hooked all 8 early exits; skipped the final normal-completion return since emitting "early_stopping" for normal completion would be semantically incorrect.

## Known Issues

None.

## Files Created/Modified

- `nce/training_logger.py` — new leaf module with JSONL logger setup and 5 event emission helpers
- `nce/config_schema.py` — added `log_file` field to output section
- `nce/inference/graphical_model.py` — import + logger setup in `__init__`, `_training_logger` attribute
- `nce/inference/bucket.py` — import + bucket_training_start/end events in compute_message_nn
- `nce/neural_networks/train.py` — import + epoch_loss, val_loss, and early_stopping events at all relevant sites
