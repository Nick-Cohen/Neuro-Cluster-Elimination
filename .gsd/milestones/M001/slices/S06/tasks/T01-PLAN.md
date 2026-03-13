---
estimated_steps: 6
estimated_files: 5
---

# T01: Implement training logger module and wire into inference pipeline

**Slice:** S06 — Logging System
**Milestone:** M001

## Description

Create `nce/training_logger.py` — a leaf module that provides JSONL-formatted file logging for training events. Add the `log_file` config field to the output section of config_schema.py. Wire logger setup into FastGM.__init__. Hook emission points in Trainer.train() (epoch_loss, val_loss, early_stopping events) and bucket.py compute_message_nn() (bucket_training_start/end events).

The logger uses Python's stdlib `logging` module with a dedicated `nce.training` namespace to avoid interfering with the root logger suppression used by stats.py. The file handler uses JSONL format (one JSON object per line) and flushes after each write to prevent data loss on crash.

## Steps

1. **Create `nce/training_logger.py`** — Define:
   - `setup_training_logger(log_file_path: str) -> logging.Logger` — creates/returns the `nce.training` logger with a FileHandler writing JSONL. Must be idempotent (clear existing handlers before adding new one). Set handler to flush after each write.
   - `get_training_logger() -> logging.Logger` — returns the `nce.training` logger (may have no handlers if not set up).
   - Event helper functions: `log_epoch_loss(logger, bucket_id, epoch, loss)`, `log_val_loss(logger, bucket_id, epoch, val_loss)`, `log_early_stopping(logger, bucket_id, epoch, reason, final_loss)`, `log_bucket_training_start(logger, bucket_id, hidden_sizes, num_epochs, loss_fn, num_samples)`, `log_bucket_training_end(logger, bucket_id, epochs_trained, final_loss)`.
   - Each helper emits a single `logger.info()` call with a JSON-serialized dict containing `timestamp` (ISO format), `event` (type string), and event-specific fields.
   - The module must NOT import from `nce.inference` or `nce.neural_networks` (leaf module constraint).

2. **Add `log_file` config field to `nce/config_schema.py`** — Add `'log_file': _field('log_file', default=None)` to the `output` section of NESTED_SECTIONS. This follows the existing pattern for output config fields.

3. **Wire logger setup in `nce/inference/graphical_model.py`** — In FastGM.__init__, after config is processed, check `self.config.get('log_file')`. If set (non-None), call `setup_training_logger(log_file_path)` and store the logger as `self._training_logger`. If not set, set `self._training_logger = None`.

4. **Hook bucket.py compute_message_nn()** — At the start of compute_message_nn (before trainer creation), emit `bucket_training_start` with bucket label, hidden_sizes, num_epochs from config, loss_fn, and sample count. After training completes (after the per_bucket_training_log append), emit `bucket_training_end` with epochs_trained and final loss. Access logger via `self.gm._training_logger`. Guard with `if self.gm._training_logger:`.

5. **Hook Trainer.train() — epoch_loss and val_loss events** — At the `self.losses.append()` call (line ~489), emit `log_epoch_loss` with bucket label, global_epoch_num, and loss value. At the `self.val_losses.append()` call (line ~626), emit `log_val_loss`. Access logger via `self.bucket.gm._training_logger`. Guard with `if self.bucket.gm._training_logger:`.

6. **Hook Trainer.train() — early_stopping events** — At each `return traced_losses_data` exit point (~12 sites), emit `log_early_stopping` with bucket label, current epoch, a descriptive reason string, and the current loss value. Each site has a different reason (e.g., "val_loss_below_0.0001", "no_improvement_1000_epochs", "nbe_3_consecutive_increases", "convex_early_stopping"). Guard with `if self.bucket.gm._training_logger:`.

## Must-Haves

- [ ] `nce/training_logger.py` exists as a leaf module (no imports from nce.inference or nce.neural_networks)
- [ ] Logger uses `nce.training` namespace, not root logger
- [ ] JSONL format: each log line is a valid JSON object with `timestamp`, `event`, `bucket_id` fields
- [ ] File handler flushes after each write (no buffered loss on crash)
- [ ] `setup_training_logger` is idempotent (no duplicate handlers)
- [ ] `log_file` field added to output section of config_schema.py with default None
- [ ] FastGM.__init__ calls setup when log_file is configured
- [ ] `epoch_loss` event emitted at every `self.losses.append()` call
- [ ] `val_loss` event emitted at every `self.val_losses.append()` call
- [ ] `early_stopping` event emitted at every `return traced_losses_data` exit point in Trainer.train()
- [ ] `bucket_training_start` and `bucket_training_end` emitted in compute_message_nn
- [ ] No existing print statements removed or modified

## Verification

- `python -c "from nce.training_logger import setup_training_logger, get_training_logger"` — import succeeds
- `python -c "from nce.config_schema import NESTED_SECTIONS; assert 'log_file' in dict(NESTED_SECTIONS)['output']"` — config field exists
- `grep -c "log_early_stopping" nce/neural_networks/train.py` — returns count matching number of early exit points hooked
- `grep "training_logger" nce/inference/bucket.py nce/inference/graphical_model.py` — confirms wiring exists

## Observability Impact

- Signals added/changed: New JSONL log file with 5 event types (epoch_loss, val_loss, early_stopping, bucket_training_start, bucket_training_end)
- How a future agent inspects this: `cat <log_file>` for raw events, `grep '"event":"epoch_loss"' <log_file>` for filtering, `python -c "import json; ..."` for structured queries
- Failure state exposed: Missing events in log file indicate a hook point was missed; empty log file with log_file configured indicates setup failure

## Inputs

- `nce/config_schema.py` — existing output section pattern to follow for new field
- `nce/neural_networks/train.py` — 12 `return traced_losses_data` sites to hook with early_stopping events
- `nce/inference/bucket.py:315-350` — compute_message_nn flow for bucket start/end events
- `nce/inference/graphical_model.py:40-70` — FastGM.__init__ for logger setup placement
- `nce/utils/stats.py:10-16` — reference for existing logging suppression to avoid conflicting with

## Expected Output

- `nce/training_logger.py` — new module with setup function and 5 event emission helpers
- `nce/config_schema.py` — `log_file` field added to output section
- `nce/inference/graphical_model.py` — logger setup in __init__, `_training_logger` attribute
- `nce/inference/bucket.py` — bucket_training_start/end events in compute_message_nn
- `nce/neural_networks/train.py` — epoch_loss, val_loss, and early_stopping events at all relevant sites
