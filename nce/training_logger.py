"""JSONL training event logger for NCE.

Leaf module — no imports from nce.inference or nce.neural_networks.

Uses a dedicated ``nce.training`` logger namespace so it does not
interfere with the root-logger suppression in nce.utils.stats.

Public API:
    setup_training_logger(log_file_path) -> logging.Logger
    get_training_logger() -> logging.Logger
    log_epoch_loss(logger, bucket_id, epoch, loss)
    log_val_loss(logger, bucket_id, epoch, val_loss)
    log_early_stopping(logger, bucket_id, epoch, reason, final_loss)
    log_bucket_training_start(logger, bucket_id, hidden_sizes, num_epochs, loss_fn, num_samples)
    log_bucket_training_end(logger, bucket_id, epochs_trained, final_loss)
"""

import json
import logging
from datetime import datetime, timezone

_LOGGER_NAME = "nce.training"


class _FlushHandler(logging.FileHandler):
    """FileHandler that flushes after every emit — no data loss on crash."""

    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_training_logger(log_file_path: str) -> logging.Logger:
    """Create (or reconfigure) the ``nce.training`` file logger.

    Idempotent: clears existing handlers before attaching a new one,
    so repeated calls with different paths are safe.

    Args:
        log_file_path: Path to the JSONL log file.

    Returns:
        The configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    # Prevent propagation to root logger (which may be suppressed)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    # Idempotent: remove all existing handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    handler = _FlushHandler(log_file_path, mode="a")
    # Raw message output — the helpers already emit JSON strings
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def get_training_logger() -> logging.Logger:
    """Return the ``nce.training`` logger (may have no handlers)."""
    return logging.getLogger(_LOGGER_NAME)


# ------------------------------------------------------------------
# Event emission helpers
# ------------------------------------------------------------------

def _emit(logger: logging.Logger, event: str, bucket_id, **fields):
    """Emit a single JSONL line with timestamp, event type, and fields."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "bucket_id": bucket_id,
        **fields,
    }
    logger.info(json.dumps(record))


def log_epoch_loss(logger, bucket_id, epoch, loss):
    _emit(logger, "epoch_loss", bucket_id, epoch=epoch, loss=loss)


def log_val_loss(logger, bucket_id, epoch, val_loss):
    _emit(logger, "val_loss", bucket_id, epoch=epoch, val_loss=val_loss)


def log_early_stopping(logger, bucket_id, epoch, reason, final_loss):
    _emit(logger, "early_stopping", bucket_id, epoch=epoch, reason=reason, final_loss=final_loss)


def log_bucket_training_start(logger, bucket_id, hidden_sizes, num_epochs, loss_fn, num_samples):
    _emit(
        logger, "bucket_training_start", bucket_id,
        hidden_sizes=hidden_sizes,
        num_epochs=num_epochs,
        loss_fn=loss_fn,
        num_samples=num_samples,
    )


def log_bucket_training_end(logger, bucket_id, epochs_trained, final_loss):
    _emit(logger, "bucket_training_end", bucket_id, epochs_trained=epochs_trained, final_loss=final_loss)
