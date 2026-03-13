# S06: Logging System — Research

**Date:** 2026-03-12

## Summary

S06 delivers R016: structured per-bucket training event logging to a configurable log file. The codebase currently has zero structured logging — all output is via `print()` (~196 calls across graphical_model.py, bucket.py, and train.py). Python's stdlib `logging` module is the natural fit: it supports file handlers, formatters, and can coexist with existing print statements without replacing them.

The implementation surface is small. A new `log_file` config field goes in the `output` section of config_schema.py. A thin logging setup function configures a Python logger with a file handler when the path is set. Hook points in `Trainer.train()` (epoch loss recorded at line 489) and `FastBucket.compute_message_nn()` (bucket training start/end around lines 322-342) emit structured log lines. The key design choice is whether to log every epoch or only at intervals — R016 says "epoch, loss, bucket id" written "line-by-line," which implies every epoch.

The risk is genuinely low. No existing behavior changes (prints stay), no new dependencies, no complex state management. The main pitfall is I/O overhead from logging every epoch for buckets trained for thousands of epochs — but file I/O for a single formatted line is negligible compared to a training epoch.

## Requirement Mapping

| Requirement | What This Slice Must Deliver |
|---|---|
| **R016** (primary) | Log file path in config → structured per-bucket training events (epoch, loss, bucket id) written line-by-line during inference |

## Recommendation

Use Python's stdlib `logging` module with a dedicated logger name (e.g., `nce.training`). Create a new `nce/logging.py` module (or `nce/log.py` to avoid shadowing stdlib `logging`) that:

1. Exports a `setup_training_logger(log_file_path)` function that adds a FileHandler with structured formatting
2. Provides a module-level logger that training code can import and call

Add a `log_file` field to the `output` section of config_schema.py (default: `None` — disabled). FastGM.__init__ calls the setup function when the path is non-None. Trainer and bucket code emit events via the logger.

**Format:** One JSON object per line (JSONL). This is grep-friendly, machine-parseable, and trivially loaded with `json.loads()` per line. Each line includes: `timestamp`, `event` (type string), `bucket_id`, and event-specific fields.

**Why not plain text?** The requirement says "structured" — JSONL is the simplest structured format that doesn't require a parser. A researcher can still `grep bucket_42 logfile.jsonl` or `cat logfile.jsonl | python -c "import json,sys; [print(json.loads(l)['loss']) for l in sys.stdin]"`.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| File logging with handlers/formatters | Python stdlib `logging` | Battle-tested, supports rotation, multiple handlers, zero dependencies |
| Structured line format | JSONL (one `json.dumps()` per line) | Machine-parseable, grep-friendly, no schema overhead |

## Existing Code and Patterns

- `nce/neural_networks/train.py` — **Primary hook point.** Line 489: `self.losses.append((global_epoch_num, loss.item()))` is where every epoch's loss is recorded. This is the natural place to also emit a log event. Early stopping events (~12 distinct exit points with print statements) should also be logged.
- `nce/inference/bucket.py:322-342` — **Bucket-level hooks.** `compute_message_nn()` appends to `per_bucket_training_log` after training completes. Good place for bucket_training_start and bucket_training_end events.
- `nce/inference/graphical_model.py:408-416` — **Elimination-level hooks.** `process_bucket()` prints "Bucket X: training NN/DT" at the start. Good place for elimination-level events.
- `nce/config_schema.py:119-127` — **Output section** of NESTED_SECTIONS. New `log_file` field goes here.
- `nce/utils/stats.py:10-16` — Uses `logging.disable(logging.CRITICAL)` to suppress pyGMs logging during stats computation. Shows the stdlib logging module is already in the dependency tree.
- `nce/inference/graphical_model.py:57` — `per_bucket_training_log` initialization. The logger setup should happen nearby (in `__init__` after config is processed).

## Constraints

- **No existing print statements should be replaced.** R016 adds file logging alongside existing stdout output. Replacing prints is a separate (larger) effort.
- **Internal config stays flat.** The new field's internal name (e.g., `log_file`) must go through `prepare_config` like everything else.
- **Logger must not interfere with pyGMs logging suppression.** The `nce.training` logger namespace avoids collision with the root logger that stats.py disables.
- **No new dependencies.** stdlib `logging` and `json` only.
- **Trainer doesn't hold a reference to the bucket label after construction.** It accesses `self.bucket.label`. This is available throughout training.

## Event Schema

Proposed events (all include `timestamp` and `bucket_id`):

| Event | When | Extra Fields |
|---|---|---|
| `bucket_training_start` | Before `t.train()` in compute_message_nn | `hidden_sizes`, `num_epochs`, `loss_fn`, `num_samples` |
| `epoch_loss` | After each epoch loss is recorded (train.py:489) | `epoch`, `loss` |
| `val_loss` | After validation loss is recorded (train.py:624) | `epoch`, `val_loss` |
| `early_stopping` | At any early stopping exit point | `epoch`, `reason`, `final_loss` |
| `bucket_training_end` | After training completes in compute_message_nn | `epochs_trained`, `final_loss` |

The `epoch_loss` event on every epoch is the core of R016. The others provide context.

## Implementation Plan (High-Level)

1. **Add config field:** `log_file` in output section of config_schema.py, internal name `log_file`, default `None`
2. **Create `nce/training_logger.py`:** Setup function + event emission helpers. Named `training_logger` to avoid shadowing `logging`.
3. **Hook FastGM.__init__:** Call setup when `log_file` is set in config
4. **Hook Trainer.train():** Emit `epoch_loss` at line 489, `val_loss` at line 624, `early_stopping` at exit points
5. **Hook bucket.py compute_message_nn():** Emit `bucket_training_start` before training, `bucket_training_end` after
6. **Update config docs:** Add `log_file` to docs/config_reference.md output section
7. **Verification:** Run inference on a small problem with log_file set, verify JSONL output contains expected events

## Common Pitfalls

- **Shadowing stdlib `logging`** — Don't name the module `nce/logging.py`. Use `nce/training_logger.py` instead.
- **Logger leaking between tests/runs** — The file handler must be added per-run, not globally at import time. Setup only when `log_file` config is set. Clean up handlers if logger already has them (idempotent setup).
- **Buffered I/O losing tail events on crash** — Use `flush=True` on the file handler or set the handler's stream to unbuffered mode, so events are written immediately.
- **Circular imports** — `training_logger.py` must not import from `nce.inference` or `nce.neural_networks`. It's a leaf module that others import from.
- **Excessive log volume** — A bucket trained for 10,000 epochs produces 10,000 `epoch_loss` lines. For a problem with 50 NN buckets, that's 500K lines. This is fine for a log file (a few MB), but worth noting. No sampling/throttling — the user asked for every epoch.

## Open Risks

- **Multiple FastGM instances sharing a log file** — If a user creates two FastGMs with the same log_file path (unlikely in practice), events interleave. Low risk; document that log_file is per-run.
- **Trainer early exit points are numerous (~12 return statements)** — Missing one means some early stopping events won't be logged. Need careful audit of all `return traced_losses_data` sites.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| Python logging | N/A (stdlib, no specialized skill needed) | not applicable |

No specialized skills are relevant for this slice — it's pure Python stdlib work.

## Sources

- Python stdlib `logging` module (standard library documentation — no external lookup needed)
- Codebase audit: `rg` across nce/ for print statements, logging usage, config fields
