"""Single-bucket benchmark training harness.

Core function: train_single_bucket() loads a precomputed .pt file,
reconstructs the live FastGM + bucket, trains the NN with a wall-clock
time limit, and tracks local error at checkpoint epochs using preloaded
exact forward/backward messages.

The custom epoch loop replaces Trainer.train() — Trainer is only used
for its __init__ chain (SampleGenerator, DataPreprocessor, DataLoader)
and _get_loss_fn(). This avoids the 600+ lines of concerns in train()
(early stopping, validation, display, traced losses) that the benchmark
doesn't need.
"""

import copy
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import torch

from nce.benchmark_problems.small_problems import small_problems
from nce.config_schema import prepare_config
from nce.inference.factor import FastFactor
from nce.inference.factor_nn import FactorNN
from nce.inference.graphical_model import FastGM
from nce.neural_networks.net import Net
from nce.neural_networks.train import Trainer, get_error_tracking_epochs, get_scaled_error_tracking_epochs

from nce.benchmark.plots import plot_loss_curve, plot_local_error_curve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_bucket_data(pt_path, device):
    """Load a precomputed .pt file and reconstruct FastFactor objects.

    Moves tensors to the target device. Returns a structured dict with:
      - factors: list of FastFactor objects
      - exact_fw: FastFactor (exact forward message)
      - exact_bw: FastFactor (exact backward message)
      - metadata: dict with bucket_label, scope, domain_sizes, elim_vars,
                   problem_key, auto_ecl
    """
    data = torch.load(pt_path, map_location='cpu', weights_only=False)

    # Reconstruct exact forward/backward as FastFactor objects on device
    exact_fw = FastFactor(
        data['exact_fw']['tensor'].to(device),
        data['exact_fw']['labels'],
    )
    exact_bw = FastFactor(
        data['exact_bw']['tensor'].to(device),
        data['exact_bw']['labels'],
    )

    # Reconstruct bucket factors
    factors = []
    for f_data in data['factors']:
        factors.append(FastFactor(
            f_data['tensor'].to(device),
            f_data['labels'],
        ))

    metadata = {
        'bucket_label': data['bucket_label'],
        'scope': data['scope'],
        'domain_sizes': data['domain_sizes'],
        'elim_vars': data['elim_vars'],
        'problem_key': data['problem_key'],
        'auto_ecl': data['auto_ecl'],
    }

    return {
        'factors': factors,
        'exact_fw': exact_fw,
        'exact_bw': exact_bw,
        'metadata': metadata,
    }


def _find_problem(problem_key):
    """Look up model index in small_problems by matching modelfile.

    Args:
        problem_key: The model's modelfile string (e.g. 'or_chain_10.fg.uai')

    Returns:
        int: Index into small_problems.problems

    Raises:
        ValueError: If problem_key not found in small_problems
    """
    for idx, model in enumerate(small_problems.problems):
        if model.modelfile == problem_key:
            return idx
    raise ValueError(
        f"Problem key '{problem_key}' not found in small_problems. "
        f"Available: {[m.modelfile for m in small_problems.problems]}"
    )


def _json_default(obj):
    """JSON serializer for types not natively serializable."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    return str(obj)


def _write_metrics(result, nn_config, bucket_data, output_path):
    """Write structured metrics JSON for a training run.

    Args:
        result: The result dict from the training loop.
        nn_config: The config dict that was used.
        bucket_data: The loaded bucket data dict (with metadata).
        output_path: File path for the output JSON.

    Returns:
        The output_path that was written.
    """
    metadata = bucket_data['metadata']

    # Config hash for reproducibility tracking
    config_str = json.dumps(
        sorted(nn_config.items(), key=lambda x: str(x[0])),
        default=str,
    )
    config_hash = hashlib.md5(config_str.encode()).hexdigest()

    # Convert error_tracking_data tuples to serializable lists
    error_tracking = [
        list(t) for t in result.get('error_tracking_data', [])
    ]
    losses = [
        list(t) for t in result.get('losses', [])
    ]

    metrics = {
        'epochs_completed': result.get('epochs_completed', 0),
        'final_loss': result.get('final_loss'),
        'final_local_error': result.get('final_local_error'),
        'error_tracking': error_tracking,
        'losses': losses,
        'wall_time': result.get('wall_time'),
        'config_hash': config_hash,
        'bucket_metadata': {
            'problem_key': metadata.get('problem_key'),
            'bucket_label': metadata.get('bucket_label'),
            'auto_ecl': metadata.get('auto_ecl'),
            'selection_error': metadata.get('selection_error'),
        },
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    return output_path


def _reconstruct_bucket(problem_idx, bucket_label, nn_config, device):
    """Reconstruct a live FastGM and bucket via exact elimination.

    Creates the FastGM from the problem model, runs exact elimination
    up to (but not including) the target bucket variable, then returns
    the live bucket with all upstream messages propagated.

    Args:
        problem_idx: Index into small_problems.problems
        bucket_label: Integer label of the target bucket variable
        nn_config: Prepared config dict for FastGM
        device: Target device string ('cuda' or 'cpu')

    Returns:
        Tuple of (FastGM, FastBucket)
    """
    model = small_problems.problems[problem_idx]
    config = copy.deepcopy(nn_config)

    fastgm = FastGM(model=model, nn_config=config, device=device)

    target_var = fastgm.matching_var(bucket_label)
    if target_var is None:
        raise ValueError(
            f"No matching var for bucket_label={bucket_label} in problem "
            f"index {problem_idx} (modelfile={model.modelfile})"
        )

    fastgm.eliminate_variables(up_to=target_var, exact=True)

    bucket = fastgm.buckets[target_var]
    return fastgm, bucket


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_single_bucket(bucket_pt_path, nn_config, time_limit_seconds,
                        output_dir, device):
    """Train a neural network factor for a single precomputed bucket.

    Loads the precomputed .pt file, reconstructs the live FastGM and bucket,
    trains the NN with a wall-clock time limit, and tracks local error at
    checkpoint epochs using preloaded exact forward/backward messages.

    Args:
        bucket_pt_path: Path to the precomputed .pt file from S01
        nn_config: Training config dict (will be merged with per-bucket
                   metadata and passed through prepare_config())
        time_limit_seconds: Wall-clock time limit for training (checked
                           at epoch boundaries)
        output_dir: Directory for output files (plots, metrics)
        device: Target device string ('cuda' or 'cpu')

    Returns:
        dict with keys:
            epochs_completed: int — number of epochs finished
            final_loss: float — loss value at last completed epoch
            final_local_error: float — abs(log_Z_err) at last checkpoint
            error_tracking_data: list of (epoch, loss, log_z_err, abs_log_z_err)
            losses: list of (epoch, loss) tuples
            wall_time: float — total training wall time in seconds
            bucket_id: str — sanitized bucket identifier
            config_used: dict — the prepared config that was actually used
    """
    # -----------------------------------------------------------------------
    # 1. Load precomputed data
    # -----------------------------------------------------------------------
    print(f"[BenchmarkTraining] Loading {bucket_pt_path}...")
    bucket_data = _load_bucket_data(bucket_pt_path, device)
    metadata = bucket_data['metadata']
    exact_fw = bucket_data['exact_fw']
    exact_bw = bucket_data['exact_bw']

    problem_key = metadata['problem_key']
    bucket_label = metadata['bucket_label']
    auto_ecl = metadata['auto_ecl']

    # Build sanitized bucket ID for output paths
    safe_key = problem_key.replace('/', '_').replace('.', '_')
    bucket_id = f"{safe_key}__bucket_{bucket_label}"

    print(f"[BenchmarkTraining] Loaded: problem={problem_key}, "
          f"bucket={bucket_label}, auto_ecl={auto_ecl}")

    # -----------------------------------------------------------------------
    # 2. Find the problem in small_problems
    # -----------------------------------------------------------------------
    problem_idx = _find_problem(problem_key)

    # -----------------------------------------------------------------------
    # 3. Merge config with per-bucket metadata and prepare
    # -----------------------------------------------------------------------
    config = copy.deepcopy(nn_config)

    # Per-bucket overrides from .pt metadata
    config['ecl'] = auto_ecl
    if 'iB' not in config:
        config['iB'] = small_problems.configs['default'][problem_idx].get('iB', 100)

    # Benchmark-enforced settings
    config['error_tracking'] = False       # Benchmark handles error tracking externally
    config['sampling_scheme'] = 'all'      # Required for full-assignment training
    config['device'] = device

    # Pass through prepare_config for validation and alias resolution
    config = prepare_config(config, strict=False)

    num_epochs = config.get('num_epochs', 10000)
    batch_size = config.get('batch_size', 100000)

    # -----------------------------------------------------------------------
    # 4. Reconstruct live FastGM and bucket
    # -----------------------------------------------------------------------
    print(f"[BenchmarkTraining] Reconstructing FastGM and bucket "
          f"(problem_idx={problem_idx}, bucket={bucket_label})...")
    fastgm, bucket = _reconstruct_bucket(problem_idx, bucket_label, config, device)
    message_size = bucket.get_message_size()
    print(f"[BenchmarkTraining] Reconstruction complete. "
          f"message_size={message_size:.0f}")

    # -----------------------------------------------------------------------
    # 5. Create Net and Trainer (for setup only)
    # -----------------------------------------------------------------------
    net = Net(bucket, hidden_sizes=config.get('hidden_sizes', []))
    trainer = Trainer(net, bucket)

    # -----------------------------------------------------------------------
    # 6. Load training data and create batches
    # -----------------------------------------------------------------------
    all_data = trainer.dataloader.load_all()[0]
    x_all, y_all, bw_all = all_data['x'], all_data['y'], all_data['bw']

    batches = []
    for i in range(0, len(x_all), batch_size):
        end_idx = min(i + batch_size, len(x_all))
        batches.append({
            'x': x_all[i:end_idx],
            'y': y_all[i:end_idx],
            'bw': bw_all[i:end_idx] if bw_all is not None else None,
        })

    # -----------------------------------------------------------------------
    # 7. Precompute exact_contribution (constant — computed once)
    # -----------------------------------------------------------------------
    exact_contribution = (exact_fw * exact_bw).sum_all_entries()

    # -----------------------------------------------------------------------
    # 8. Set up checkpoint tracking
    # -----------------------------------------------------------------------
    checkpoint_epochs = set(get_scaled_error_tracking_epochs(num_epochs, batch_size, message_size))
    error_tracking_data = []
    losses = []
    loss_fn = trainer.loss_fn

    print(f"[BenchmarkTraining] Starting training: {num_epochs} max epochs, "
          f"{time_limit_seconds}s time limit, {len(batches)} batches, "
          f"checkpoints at {sorted(checkpoint_epochs)}")

    # -----------------------------------------------------------------------
    # 9. Checkpoint at epoch 0 (before any training)
    # -----------------------------------------------------------------------
    if 0 in checkpoint_epochs:
        with torch.no_grad():
            initial_loss = trainer.compute_epoch_loss(batches, loss_fn).item()
            approx_factor = FactorNN(net, trainer.data_preprocessor)
            approx_exact = approx_factor.to_exact()
            approx_contribution = (approx_exact * exact_bw).sum_all_entries()
            log_z_err = approx_contribution - exact_contribution
            error_tracking_data.append((0, initial_loss, log_z_err, abs(log_z_err)))
            print(f"[BenchmarkTraining] Checkpoint epoch 0: "
                  f"loss={initial_loss:.6e}, log_Z_err={log_z_err:.6f}, "
                  f"|log_Z_err|={abs(log_z_err):.6f}")

    # -----------------------------------------------------------------------
    # 10. Custom epoch loop with time-limit checking
    # -----------------------------------------------------------------------
    start_time = time.time()
    epochs_completed = 0

    for epoch in range(1, num_epochs + 1):
        # Train one epoch
        loss = trainer.train_epoch(batches)
        loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
        losses.append((epoch, loss_val))
        epochs_completed = epoch

        # Step scheduler if trainer has one
        if trainer.use_scheduler and hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
            trainer.scheduler.step()

        # Checkpoint error tracking
        if epoch in checkpoint_epochs:
            with torch.no_grad():
                approx_factor = FactorNN(net, trainer.data_preprocessor)
                approx_exact = approx_factor.to_exact()
                approx_contribution = (approx_exact * exact_bw).sum_all_entries()
                log_z_err = approx_contribution - exact_contribution
                error_tracking_data.append(
                    (epoch, loss_val, log_z_err, abs(log_z_err))
                )
                print(f"[BenchmarkTraining] Checkpoint epoch {epoch}: "
                      f"loss={loss_val:.6e}, log_Z_err={log_z_err:.6f}, "
                      f"|log_Z_err|={abs(log_z_err):.6f}")

        # Wall-clock time check at epoch boundary
        elapsed = time.time() - start_time
        if elapsed >= time_limit_seconds:
            print(f"[BenchmarkTraining] Time limit reached at epoch {epoch} "
                  f"({elapsed:.1f}s >= {time_limit_seconds}s)")
            break

    wall_time = time.time() - start_time

    # -----------------------------------------------------------------------
    # 11. Determine final values
    # -----------------------------------------------------------------------
    final_loss = losses[-1][1] if losses else None
    final_local_error = error_tracking_data[-1][3] if error_tracking_data else None

    print(f"[BenchmarkTraining] Training complete: {epochs_completed} epochs "
          f"in {wall_time:.1f}s, final_loss={final_loss}, "
          f"final_local_error={final_local_error}")

    # -----------------------------------------------------------------------
    # 12. Build result dict
    # -----------------------------------------------------------------------
    result = {
        'epochs_completed': epochs_completed,
        'final_loss': final_loss,
        'final_local_error': final_local_error,
        'error_tracking_data': error_tracking_data,
        'losses': losses,
        'wall_time': wall_time,
        'bucket_id': bucket_id,
        'config_used': config,
    }

    # -----------------------------------------------------------------------
    # 13. Output stage: plots + metrics
    # -----------------------------------------------------------------------
    bucket_output_dir = os.path.join(output_dir, bucket_id)
    os.makedirs(bucket_output_dir, exist_ok=True)
    print(f"[BenchmarkTraining] Saving outputs to {bucket_output_dir}/")

    loss_plot_path = os.path.join(bucket_output_dir, 'loss.png')
    error_plot_path = os.path.join(bucket_output_dir, 'local_error.png')
    metrics_path = os.path.join(bucket_output_dir, 'metrics.json')

    # Loss plot (best-effort — write metrics even if plots fail)
    try:
        if losses:
            plot_loss_curve(
                losses, loss_plot_path,
                title=f"Training Loss — bucket {bucket_label}",
            )
            result['loss_plot_path'] = loss_plot_path
    except Exception as e:
        print(f"[BenchmarkTraining] WARNING: loss plot failed: {e}")

    # Local error plot (best-effort)
    try:
        if error_tracking_data:
            plot_local_error_curve(
                error_tracking_data, error_plot_path,
                title=f"Local Error — bucket {bucket_label}",
            )
            result['error_plot_path'] = error_plot_path
    except Exception as e:
        print(f"[BenchmarkTraining] WARNING: error plot failed: {e}")

    # Metrics JSON (always written)
    try:
        _write_metrics(result, config, bucket_data, metrics_path)
        result['metrics_path'] = metrics_path
    except Exception as e:
        print(f"[BenchmarkTraining] WARNING: metrics write failed: {e}")

    result['output_dir'] = bucket_output_dir
    return result
