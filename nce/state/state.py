"""Save, load, and inspect post-inference FastGM state.

This module is a thin serialization layer — it does not reconstruct a live FastGM,
just preserves and restores inspection data (loss curves, configs, optional NN weights).
"""

import copy
import pickle
from pathlib import Path
from typing import Union

import torch


def save_state(fastgm, path: Union[str, Path], save_weights: bool = False) -> None:
    """Extract a serializable dict from a post-inference FastGM and pickle it.

    Args:
        fastgm: A FastGM instance after inference has been run.
        path: Filesystem path to write the pickle file.
        save_weights: If False (default), strip nn_state_dict and normalizing_constant
            from training log entries before saving. If True, preserve them.

    Raises:
        TypeError: If an attribute contains an unpicklable object (re-raised with context).
    """
    path = Path(path)

    # Extract the training log — deep copy so stripping doesn't mutate the live object
    training_log = copy.deepcopy(
        getattr(fastgm, "per_bucket_training_log", [])
    )

    if not save_weights:
        for entry in training_log:
            entry.pop("nn_state_dict", None)
            entry.pop("normalizing_constant", None)

    # Extract elim_order as plain list of variable labels (ints)
    raw_order = getattr(fastgm, "elim_order", None)
    if raw_order is not None:
        elim_order = [
            v.label if hasattr(v, "label") else v for v in raw_order
        ]
    else:
        elim_order = None

    state = {
        "per_bucket_training_log": training_log,
        "config": getattr(fastgm, "config", {}),
        "logZ": getattr(fastgm, "logZ", None),
        "elim_order": elim_order,
        "num_trained": getattr(fastgm, "num_trained", 0),
        "bucket_complexities": getattr(fastgm, "bucket_complexities", []),
    }

    try:
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=4)
    except TypeError as exc:
        raise TypeError(
            f"save_state: unpicklable object in FastGM state — {exc}"
        ) from exc


def load_state(path: Union[str, Path]) -> dict:
    """Load a previously saved FastGM state dict from a pickle file.

    Args:
        path: Filesystem path to the pickle file.

    Returns:
        The state dict as saved by save_state().

    Raises:
        FileNotFoundError: If the path does not exist.
        pickle.UnpicklingError: If the file is not a valid pickle.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"load_state: no file at {path}")

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError as exc:
        raise pickle.UnpicklingError(
            f"load_state: failed to unpickle {path} — {exc}"
        ) from exc


def undo_normalization(
    outputs: torch.Tensor, normalizing_constant: float
) -> torch.Tensor:
    """Convert normalized NN outputs back to log10 space.

    Mirrors DataPreprocessor.undo_normalization() but accepts the normalizing
    constant as an explicit argument (as stored in per_bucket_training_log).

    Args:
        outputs: Normalized outputs from a neural network (natural-log space, centered).
        normalizing_constant: The scalar used during training normalization.

    Returns:
        Tensor in log10 space.
    """
    ln10 = torch.log(torch.tensor(10.0)).to(outputs.device)

    # Add back the normalizing constant
    outputs = outputs + normalizing_constant

    # Convert from natural-log to log10
    outputs = outputs / ln10

    return outputs
