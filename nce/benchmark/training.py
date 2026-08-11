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
from pathlib import Path

import torch

from nce.benchmark_problems.small_problems import small_problems
from nce.config_schema import prepare_config
from nce.inference.factor import FastFactor
from nce.inference.factor_nn import FactorNN, MessageGenerator
from nce.inference.graphical_model import FastGM
from nce.neural_networks.net import Net
from nce.neural_networks.train import Trainer, get_error_tracking_epochs, get_scaled_error_tracking_epochs
from nce.utils.backward_message import get_backward_message

from nce.benchmark.plots import plot_loss_curve, plot_local_error_curve, plot_top_assignments
from nce.utils.plots import plot_fastfactor_comparison
from nce.utils.dtype_utils import get_dtype


BW_CACHE_DIR = Path('/home/cohenn1/NCE/data/hard_buckets/bw_cache')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_bucket_data(pt_path, device, dtype=torch.float32):
    """Load a precomputed .pt file and reconstruct FastFactor objects.

    Moves tensors to the target device and dtype. When dtype=float64,
    checks for a cached .f64.pt file first; if not found, loads the
    float32 version, converts, and saves the float64 cache.

    Returns a structured dict with:
      - factors: list of FastFactor objects
      - exact_fw: FastFactor (exact forward message)
      - exact_bw: FastFactor (exact backward message)
      - metadata: dict with bucket_label, scope, domain_sizes, elim_vars,
                   problem_key, auto_ecl
    """
    pt_path = Path(pt_path)

    # Float64 caching: use a .f64.pt file alongside the original
    if dtype == torch.float64:
        f64_path = pt_path.with_suffix('.f64.pt')
        if f64_path.exists():
            print(f"[BenchmarkTraining] Loading cached float64 data: {f64_path}")
            data = torch.load(f64_path, map_location='cpu', weights_only=False)
        else:
            print(f"[BenchmarkTraining] No float64 cache, converting from {pt_path}")
            data = torch.load(pt_path, map_location='cpu', weights_only=False)
            # Convert tensors to float64 and save cache
            data['exact_fw']['tensor'] = data['exact_fw']['tensor'].to(torch.float64)
            data['exact_bw']['tensor'] = data['exact_bw']['tensor'].to(torch.float64)
            for f_data in data['factors']:
                f_data['tensor'] = f_data['tensor'].to(torch.float64)
            torch.save(data, f64_path)
            print(f"[BenchmarkTraining] Saved float64 cache: {f64_path}")
    else:
        data = torch.load(pt_path, map_location='cpu', weights_only=False)

    # Reconstruct exact forward/backward as FastFactor objects on device
    exact_fw = FastFactor(
        data['exact_fw']['tensor'].to(device=device, dtype=dtype),
        data['exact_fw']['labels'],
    )
    exact_bw = FastFactor(
        data['exact_bw']['tensor'].to(device=device, dtype=dtype),
        data['exact_bw']['labels'],
    )

    # Reconstruct bucket factors
    factors = []
    for f_data in data['factors']:
        factors.append(FastFactor(
            f_data['tensor'].to(device=device, dtype=dtype),
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


def _load_or_compute_bw_message(fastgm, bucket_label, bucket_id, bw_ib2, device,
                                dtype=torch.float32):
    """Load a cached backward message or compute it on the fly.

    Checks BW_CACHE_DIR/bw_ib2_{bw_ib2}/{bucket_id}.pt for a cached version.
    If not found, computes via get_backward_message and caches for future use.
    When dtype=float64, uses a separate cache directory (bw_ib2_{N}_f64/).

    Returns:
        FastFactor: The backward message
    """
    bw_iB = bw_ib2
    bw_ecl = (2 ** bw_ib2) - 1

    # Separate cache directories for float32 vs float64
    if dtype == torch.float64:
        cache_dir = BW_CACHE_DIR / f'bw_ib2_{bw_ib2}_f64'
    else:
        cache_dir = BW_CACHE_DIR / f'bw_ib2_{bw_ib2}'
    cache_path = cache_dir / f'{bucket_id}.pt'

    if cache_path.exists():
        print(f"[BenchmarkTraining] Loading cached bw message: {cache_path}")
        data = torch.load(cache_path, map_location='cpu', weights_only=False)
        bw_msg = FastFactor(data['tensor'].to(device=device, dtype=dtype), data['labels'])
        return bw_msg

    # For float64, try loading float32 cache and converting
    if dtype == torch.float64:
        f32_cache_path = BW_CACHE_DIR / f'bw_ib2_{bw_ib2}' / f'{bucket_id}.pt'
        if f32_cache_path.exists():
            print(f"[BenchmarkTraining] Converting float32 bw cache to float64: {f32_cache_path}")
            data = torch.load(f32_cache_path, map_location='cpu', weights_only=False)
            data['tensor'] = data['tensor'].to(torch.float64)
            # Save float64 cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, cache_path)
            print(f"[BenchmarkTraining] Saved float64 bw cache: {cache_path}")
            bw_msg = FastFactor(data['tensor'].to(device=device, dtype=dtype), data['labels'])
            return bw_msg

    # Not cached — compute on the fly
    print(f"[BenchmarkTraining] No cached bw message for bw_iB2={bw_ib2}, computing...")
    bw_msg, _ = get_backward_message(
        fastgm,
        bucket_label,
        backward_factors=None,
        iB=bw_iB,
        backward_ecl=bw_ecl,
        approximation_method='wmb',
        return_factor_list=False,
    )

    # Cache for future use
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'tensor': bw_msg.tensor.detach().cpu().to(dtype),
        'labels': list(bw_msg.labels),
        'bw_ib2': bw_ib2,
        'bw_iB': bw_iB,
        'bw_ecl': bw_ecl,
        'bucket_id': bucket_id,
        'bucket_label': bucket_label,
    }, cache_path)
    print(f"[BenchmarkTraining] Cached bw message to {cache_path}")

    # Ensure returned message is on correct device and dtype
    bw_msg = FastFactor(bw_msg.tensor.to(device=device, dtype=dtype), bw_msg.labels)
    return bw_msg


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


def _write_metrics(result, nn_config, bucket_data, output_path,
                   scope_width=None, message_size=None):
    """Write structured metrics JSON for a training run.

    Args:
        result: The result dict from the training loop.
        nn_config: The config dict that was used.
        bucket_data: The loaded bucket data dict (with metadata).
        output_path: File path for the output JSON.
        scope_width: Number of variables in the message scope.
        message_size: Product of domain sizes of scope variables.

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
            'scope_width': scope_width,
            'message_size': message_size,
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

    When proposal_sampling is enabled in the config, also populates
    approximate upstream/downstream factors for the proposal tree
    (requires bw_ecl or bw_ib2 to be set).

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

    # If proposal sampling is requested, ensure backward factor population
    # is enabled so approximate_upstream/downstream_factors get populated
    if config.get('proposal_sampling', False):
        config['populate_bw_factors'] = True

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
                        output_dir, device, noise_seed=None,
                        multiply_messages=None):
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
        noise_seed: If not None, add Gaussian noise to exact_fw with this
                    seed.  Variance = var(exact_fw.tensor) / 10.
        multiply_messages: If not None, a pair [fw_mult, bw_mult].
                    Forward message is multiplied by fw_mult (after noise),
                    backward messages (exact and approx) by bw_mult.
                    Exact contribution is recomputed from the scaled messages.

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
    # Resolve precision dtype early (before loading data)
    dtype = get_dtype(nn_config)

    print(f"[BenchmarkTraining] Loading {bucket_pt_path}...")
    bucket_data = _load_bucket_data(bucket_pt_path, device, dtype=dtype)
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
    # 1b. Optionally add Gaussian noise to exact forward message
    # -----------------------------------------------------------------------
    if noise_seed is not None:
        fw_var = exact_fw.tensor.var().item()
        noise_std = (fw_var / 10) ** 0.5
        rng = torch.Generator(device=device)
        rng.manual_seed(noise_seed)
        noise = torch.randn(exact_fw.tensor.shape, generator=rng,
                            device=device, dtype=exact_fw.tensor.dtype) * noise_std
        exact_fw.tensor = exact_fw.tensor + noise
        print(f"[BenchmarkTraining] Added noise: seed={noise_seed}, "
              f"fw_var={fw_var:.6f}, noise_std={noise_std:.6f}")

    # -----------------------------------------------------------------------
    # 1c. Optionally scale forward and backward messages
    # -----------------------------------------------------------------------
    if multiply_messages is not None:
        fw_mult, bw_mult = multiply_messages
        exact_fw.tensor = exact_fw.tensor * fw_mult
        exact_bw.tensor = exact_bw.tensor * bw_mult
        print(f"[BenchmarkTraining] Scaled messages: fw*{fw_mult}, bw*{bw_mult}")

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
    if not config.get('proposal_sampling', False) and 'sampling_scheme' not in config:
        config['sampling_scheme'] = 'all'  # Default: full-assignment training
    config['device'] = device

    # Pass through prepare_config for validation and alias resolution
    config = prepare_config(config, strict=False)

    # Force AMP off for float64 mode
    if dtype == torch.float64:
        config['use_amp'] = False
        print(f"[BenchmarkTraining] Float64 mode enabled (AMP disabled)")

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
    # Resolve string-format hidden_sizes (e.g. 'neurobe,3' or 'nbe,2')
    hidden_sizes = config.get('hidden_sizes', [])
    if isinstance(hidden_sizes, str) and hidden_sizes.startswith('neurobe'):
        b = int(hidden_sizes.split(',')[1]) if ',' in hidden_sizes else 1
        scope_size = len(bucket.get_message_scope())
        h = scope_size * b
        hidden_sizes = [h, h]
    elif isinstance(hidden_sizes, str) and hidden_sizes.startswith('nbe'):
        import math
        b = int(hidden_sizes.split(',')[1]) if ',' in hidden_sizes else 1
        h = b * math.ceil(math.log2(message_size)) if message_size > 1 else b
        hidden_sizes = [h, h]
    print(f"[BenchmarkTraining] hidden_sizes={hidden_sizes}")

    net = Net(bucket, hidden_sizes=hidden_sizes)
    if dtype == torch.float64:
        net = net.to(dtype=dtype)
    trainer = Trainer(net, bucket)

    # -----------------------------------------------------------------------
    # 5b. Set up backward message if use_bw_approx is enabled
    # -----------------------------------------------------------------------
    if config.get('use_bw_approx', False):
        bw_ib2 = config.get('bw_iB', config.get('iB', 10))
        print(f"[BenchmarkTraining] use_bw_approx=True, bw_iB={bw_ib2}")
        bw_msg = _load_or_compute_bw_message(
            fastgm, bucket_label, bucket_id, bw_ib2, device, dtype=dtype
        )
        # Scale approximate backward message if multiply_messages is active
        if multiply_messages is not None:
            _, bw_mult = multiply_messages
            bw_msg.tensor = bw_msg.tensor * bw_mult
            print(f"[BenchmarkTraining] Scaled approx bw message by {bw_mult}")
        # Set on dataloader so load_all() picks up bw values
        trainer.dataloader.bw_modifier = bw_msg
        trainer.data_preprocessor.use_bw_approx = True
        print(f"[BenchmarkTraining] Backward message set, shape={list(bw_msg.tensor.shape)}")

    # -----------------------------------------------------------------------
    # 6. Load training data and create batches
    # -----------------------------------------------------------------------
    proposal_log_probs_all = None

    if config.get('proposal_sampling', False):
        # --- Proposal sampling path ---
        from nce.sampling.proposal_sampler import build_proposal_for_bucket
        import nce.sampling.no_replacement_sampler_v3  # attach sample_no_replacement_v3
        import math as _math

        proposal_ecl = config.get('bw_ecl', 0)
        num_proposal_samples = config.get('num_samples', 10000)
        proposal_mix = config.get('proposal_mix', 'full')  # 'full', 'half', 'no_replacement'
        proposal_temperature = float(config.get('proposal_temperature', 1.0))

        msg_scope = trainer.sample_generator.message_scope
        domain_sizes = [fastgm.matching_var(v).states for v in msg_scope]

        # Build the proposal tree unless we're pure uniform (which shouldn't hit this branch)
        print(f"[BenchmarkTraining] Building proposal tree "
              f"(ecl={proposal_ecl}, T={proposal_temperature})...")
        proposal_tree = build_proposal_for_bucket(
            bucket, fastgm, ecl=proposal_ecl, temperature=proposal_temperature
        )
        print(f"[BenchmarkTraining] Proposal tree: {len(proposal_tree.levels)} levels, "
              f"vars={proposal_tree.variables}")

        if proposal_mix == 'no_replacement':
            # Vectorized v3: faithful re-implementation of the OR-tree no-replacement
            # algorithm (iterative threshold reduction + phase-2 decimal resolution).
            # Returns effective_log_probs tailored for the IS wrapper:
            #   phase-1: eff = -log10(nws)  → weight = nws
            #   phase-2: eff = log10(q)     → weight = 1/q
            print(f"[BenchmarkTraining] No-replacement sampling (v3): "
                  f"N={num_proposal_samples}")
            import torch as _torch
            rng = _torch.Generator(device=device)
            seed = config.get('seed', 42)
            rng.manual_seed(int(seed))
            nr_samples, nr_log_probs_log10, nr_eff_log_probs_log10 = \
                proposal_tree.sample_no_replacement_v3_recursive(
                    num_proposal_samples, M=1, rng=rng, mode='save')
            assignments = torch.stack(
                [nr_samples[v] for v in msg_scope], dim=1
            ).to(device)
            # Existing IS wrapper consumes natural-log "effective q"
            proposal_log_probs_nat = nr_eff_log_probs_log10.to(
                device=device, dtype=torch.float32
            ) * _math.log(10)
            print(f"[BenchmarkTraining] Got {assignments.shape[0]} unique samples; "
                  f"eff_log_prob range=[{nr_eff_log_probs_log10.min():.3f}, "
                  f"{nr_eff_log_probs_log10.max():.3f}] (log10)")
        elif proposal_mix == 'half':
            # 50/50 stratified: N/2 from uniform, N/2 from WMB proposal.
            # Each sample keeps the log-prob under the distribution it was drawn from,
            # so 1/q_source acts as the IS weight per sample.
            n_half = num_proposal_samples // 2
            n_wmb = num_proposal_samples - n_half  # give the remainder to proposal
            print(f"[BenchmarkTraining] Mixed proposal: {n_half} uniform + {n_wmb} WMB samples")

            # Uniform samples: independent draw per variable
            uniform_cols = [
                torch.randint(0, d, (n_half,), device=device, dtype=torch.long)
                for d in domain_sizes
            ]
            uniform_assignments = torch.stack(uniform_cols, dim=1)
            # log q_u(x) = -sum log(d_v), same for every uniform sample (natural log)
            log_uniform_density = -sum(_math.log(d) for d in domain_sizes)
            uniform_log_probs = torch.full(
                (n_half,), log_uniform_density, device=device, dtype=torch.float32
            )

            # WMB samples from the proposal tree (returns log-probs in log10)
            wmb_samples_dict, wmb_log_probs_log10 = proposal_tree.sample(n_wmb)
            wmb_assignments = torch.stack(
                [wmb_samples_dict[v] for v in msg_scope], dim=1
            ).to(device)
            wmb_log_probs = wmb_log_probs_log10.to(device=device, dtype=torch.float32) * _math.log(10)

            # Concatenate
            assignments = torch.cat([uniform_assignments, wmb_assignments], dim=0)
            proposal_log_probs_nat = torch.cat([uniform_log_probs, wmb_log_probs], dim=0)
        elif proposal_mix == 'half_nr':
            # 50/50 stratified: N/2 from uniform, N/2 from no-replacement WMB.
            n_half = num_proposal_samples // 2
            n_nr = num_proposal_samples - n_half
            print(f"[BenchmarkTraining] Mixed proposal: {n_half} uniform + {n_nr} NR samples")

            # Uniform half — log q_u(x) = -sum log(d_v) per sample
            uniform_cols = [
                torch.randint(0, d, (n_half,), device=device, dtype=torch.long)
                for d in domain_sizes
            ]
            uniform_assignments = torch.stack(uniform_cols, dim=1)
            log_uniform_density = -sum(_math.log(d) for d in domain_sizes)
            uniform_log_probs = torch.full(
                (n_half,), log_uniform_density, device=device, dtype=torch.float32
            )

            # No-replacement half — uses the IS-wrapper's "effective q" convention
            import torch as _torch
            rng = _torch.Generator(device=device)
            seed = config.get('seed', 42)
            rng.manual_seed(int(seed))
            nr_samples, _, nr_eff_log_probs_log10 = \
                proposal_tree.sample_no_replacement_v3_recursive(
                    n_nr, M=1, rng=rng, mode='save')
            nr_assignments = torch.stack(
                [nr_samples[v] for v in msg_scope], dim=1
            ).to(device)
            nr_log_probs = nr_eff_log_probs_log10.to(
                device=device, dtype=torch.float32
            ) * _math.log(10)
            print(f"[BenchmarkTraining] Got {nr_assignments.shape[0]} unique NR samples")

            assignments = torch.cat([uniform_assignments, nr_assignments], dim=0)
            proposal_log_probs_nat = torch.cat([uniform_log_probs, nr_log_probs], dim=0)
        else:
            # Pure proposal sampling (original behavior)
            print(f"[BenchmarkTraining] Sampling {num_proposal_samples} proposal samples...")
            samples_dict, proposal_log_probs = proposal_tree.sample(num_proposal_samples)
            assignments = torch.stack(
                [samples_dict[v] for v in msg_scope], dim=1
            ).to(device)
            proposal_log_probs_nat = proposal_log_probs.to(
                device=device, dtype=torch.float32
            ) * _math.log(10)

        # Compute forward message values at the (mixed) samples
        y_log10 = trainer.sample_generator.compute_message_values(assignments)

        # Compute backward message values if applicable
        bw_log10 = None
        if hasattr(trainer.dataloader, 'bw_modifier') and trainer.dataloader.bw_modifier is not None:
            bw_log10 = trainer.sample_generator.compute_backward_values(
                assignments, backward_factors=[trainer.dataloader.bw_modifier]
            )

        # Normalize through the preprocessor
        trainer.data_preprocessor._initialize_normalizing_constant(y_log10, bw_log10)
        y_all, bw_all = trainer.data_preprocessor.normalize(y_log10, bw_log10)

        # One-hot encode assignments
        x_all = trainer.data_preprocessor.one_hot_encode(bucket, assignments)

        # Keep proposal log probs for importance weighting (natural log)
        proposal_log_probs_all = proposal_log_probs_nat.to(dtype=dtype)

        print(f"[BenchmarkTraining] Proposal sampling complete: "
              f"x={list(x_all.shape)}, y={list(y_all.shape)}, "
              f"proposal_log_probs range=[{proposal_log_probs_nat.min():.3f}, "
              f"{proposal_log_probs_nat.max():.3f}] (nat log)")
    else:
        # --- Standard full-enumeration path ---
        all_data = trainer.dataloader.load_all()[0]
        x_all, y_all, bw_all = all_data['x'], all_data['y'], all_data['bw']

    # Cast training data to target precision
    if dtype == torch.float64:
        x_all = x_all.to(dtype=dtype)
        y_all = y_all.to(dtype=dtype)
        if bw_all is not None:
            bw_all = bw_all.to(dtype=dtype)

    # Replace training targets with noised exact_fw values and reinitialize normalization
    if noise_seed is not None:
        noised_log10 = exact_fw.tensor.flatten().to(dtype=y_all.dtype)
        # Undo normalization on bw to get raw log10 bw values for reinit
        bw_raw = None
        if bw_all is not None:
            ln10 = torch.log(torch.tensor(10.0, device=bw_all.device))
            bw_raw = bw_all / ln10  # bw was converted to natural log but not centered
        # Reinitialize normalizing constant from noised data
        trainer.data_preprocessor._initialize_normalizing_constant(noised_log10, bw_raw)
        y_all, bw_all = trainer.data_preprocessor.normalize(noised_log10, bw_raw)
        print(f"[BenchmarkTraining] Replaced training targets with noised exact_fw")

    # Replace -inf targets with -10 for neurobe_weighted_mse (zero-probability assignments)
    if config.get('loss_fn', '') == 'neurobe_weighted_mse':
        num_inf = torch.isinf(y_all).sum().item()
        if num_inf > 0:
            y_all = torch.clamp(y_all, min=-10.0)
            print(f"[BenchmarkTraining] Clamped {num_inf} -inf target values to -10")

    # Compute global_max_targets for UKL numerical stability (mirrors Trainer.train())
    # Must be computed ONCE from all training data and used for ALL batches
    if config.get('loss_fn', '') == 'unnormalized_kl':
        with torch.no_grad():
            if bw_all is not None and trainer.data_preprocessor.bw_normalizing_constant is not None:
                targets_for_max = y_all + bw_all - trainer.data_preprocessor.bw_normalizing_constant
            else:
                targets_for_max = y_all
            trainer.data_preprocessor.global_max_targets = targets_for_max.max().item()
            print(f"[BenchmarkTraining] global_max_targets: {trainer.data_preprocessor.global_max_targets:.4f}")

    batches = []
    for i in range(0, len(x_all), batch_size):
        end_idx = min(i + batch_size, len(x_all))
        batch = {
            'x': x_all[i:end_idx],
            'y': y_all[i:end_idx],
            'bw': bw_all[i:end_idx] if bw_all is not None else None,
        }
        if proposal_log_probs_all is not None:
            batch['proposal_log_probs'] = proposal_log_probs_all[i:end_idx]
        batches.append(batch)

    # -----------------------------------------------------------------------
    # 7. Precompute exact_contribution (constant — computed once)
    # -----------------------------------------------------------------------
    exact_contribution = (exact_fw * exact_bw).sum_all_entries()

    # -----------------------------------------------------------------------
    # 8. Set up checkpoint tracking
    # -----------------------------------------------------------------------
    error_track_every = config.get('error_track_every')
    if error_track_every is not None:
        step = int(error_track_every)
        checkpoint_epochs = set(range(0, num_epochs + 1, step))
        checkpoint_epochs.add(num_epochs)
    else:
        checkpoint_epochs = set(get_scaled_error_tracking_epochs(num_epochs, batch_size, message_size))
    error_tracking_data = []
    losses = []
    loss_fn = trainer.loss_fn

    # If proposal sampling, wrap the loss function to apply importance weights
    if proposal_log_probs_all is not None:
        _base_loss_fn = trainer.loss_fn
        _loss_name = config.get('loss_fn', '')

        def _compute_is_weights(proposal_lp):
            """Self-normalized IS weights from proposal log-probs (natural log)."""
            log_w = -proposal_lp
            log_w_shifted = log_w - log_w.max()
            w = torch.exp(log_w_shifted)
            w = w / w.sum() * len(w)  # Normalize so weights sum to N
            return w.detach()

        def _importance_weighted_loss(outputs, targets, bw_hat=None,
                                      _proposal_lp=None, **kwargs):
            """Wrap base loss to apply IS weights from proposal sampling.

            Per-sample loss is weighted by 1/proposal_prob, self-normalized.
            Supports UKL and neurobe_weighted_mse.
            """
            if _proposal_lp is None:
                return _base_loss_fn(outputs, targets, bw_hat, **kwargs)

            w = _compute_is_weights(_proposal_lp)

            if _loss_name == 'neurobe_weighted_mse':
                # Per-sample: target_i * (ln_range/sum_ln) * (outputs_i - targets_i)^2
                epsilon = 1e-10
                ln_range = trainer.data_preprocessor.ln_max - trainer.data_preprocessor.ln_min
                sum_ln = trainer.data_preprocessor.sum_ln
                safe_sum_ln = sum_ln if abs(sum_ln) > epsilon else epsilon
                w_target = targets * ln_range / safe_sum_ln
                per_sample = w_target * (outputs - targets) ** 2
                return torch.mean(w * per_sample)

            # Default: UKL inline
            bw_normalizing_constant = trainer.data_preprocessor.bw_normalizing_constant
            max_val = trainer.data_preprocessor.global_max_targets

            if bw_hat is not None:
                bw_hat_d = bw_hat.detach()
                if bw_normalizing_constant is not None:
                    bw_hat_d = bw_hat_d - bw_normalizing_constant
                outputs_adj = outputs + bw_hat_d
                targets_adj = targets + bw_hat_d
            else:
                outputs_adj = outputs
                targets_adj = targets

            if max_val is None:
                mv = torch.max(torch.max(targets_adj), torch.max(outputs_adj.detach()))
            else:
                mv = max(max_val, outputs_adj.detach().max().item())
                mv = torch.tensor(mv, device=outputs.device)
            mv = mv.detach()

            log_p = torch.clamp(targets_adj - mv, min=-100)
            log_q = torch.clamp(outputs_adj - mv, min=-100)
            p = torch.exp(log_p)
            q = torch.exp(log_q)
            per_sample = p * (log_p - log_q) - p + q

            return torch.sum(w * per_sample)

        trainer.loss_fn = _importance_weighted_loss
        loss_fn = _importance_weighted_loss

        # Also need to pass proposal_log_probs through train_batch.
        # Override train_epoch to handle the extra batch field.
        _original_train_epoch = trainer.train_epoch

        def _proposal_train_epoch(batches, plot=False, epoch=None):
            batch_losses = []
            for batch in batches:
                x_b = batch['x'].to(config['device'])
                y_b = batch['y'].to(config['device'])
                bw_b = batch.get('bw')
                if bw_b is not None:
                    bw_b = bw_b.to(config['device'])
                plp = batch.get('proposal_log_probs')
                if plp is not None:
                    plp = plp.to(config['device'])

                trainer.net.train()
                if isinstance(trainer.optimizer, list):
                    for opt in trainer.optimizer:
                        opt.zero_grad()
                else:
                    trainer.optimizer.zero_grad()

                outputs = trainer.net(x_b)
                loss = loss_fn(outputs.reshape(-1), y_b, bw_b, _proposal_lp=plp)
                loss.backward()

                grad_clip_norm = config.get('grad_clip_norm', None)
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), grad_clip_norm)
                trainer.optimizer.step()

                batch_losses.append(loss.cpu().item())
                del x_b, y_b, bw_b, plp, loss
                torch.cuda.empty_cache()

            return torch.tensor(sum(batch_losses) / len(batch_losses))

        trainer.train_epoch = _proposal_train_epoch

    # Optional max_steps budget — override num_epochs so total gradient
    # updates ≈ max_steps. Lets callers specify compute budget invariant
    # to per-bucket batch count.
    max_steps = config.get('max_steps')
    if max_steps is not None:
        steps_per_epoch = max(1, len(batches))
        new_num_epochs = max(1, int(max_steps) // steps_per_epoch)
        print(f"[BenchmarkTraining] max_steps={max_steps}: overriding "
              f"num_epochs {num_epochs} → {new_num_epochs} "
              f"(steps_per_epoch={steps_per_epoch})")
        num_epochs = new_num_epochs
        # Recompute checkpoint epochs to match new num_epochs
        if error_track_every is not None:
            step = int(error_track_every)
            checkpoint_epochs = set(range(0, num_epochs + 1, step))
            checkpoint_epochs.add(num_epochs)
        else:
            checkpoint_epochs = set(get_scaled_error_tracking_epochs(num_epochs, batch_size, message_size))

    sorted_checkpoints = sorted(checkpoint_epochs)
    if len(sorted_checkpoints) <= 20:
        ckpt_str = str(sorted_checkpoints)
    else:
        ckpt_str = (f"{len(sorted_checkpoints)} epochs "
                    f"(first: {sorted_checkpoints[:3]}, last: {sorted_checkpoints[-3:]})")
    print(f"[BenchmarkTraining] Starting training: {num_epochs} max epochs, "
          f"{time_limit_seconds}s time limit, {len(batches)} batches, "
          f"checkpoints at {ckpt_str}")

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

        # DEBUG: break into debugger right before NaN propagates
        import math
        if math.isnan(loss_val):
            print(f"[DEBUG] NaN loss detected at epoch {epoch}. Dropping into debugger.")
            print(f"  Useful things to inspect:")
            print(f"    trainer.net    — the NN model (check params with .named_parameters())")
            print(f"    batches[0]     — first batch: 'x', 'y', 'bw' keys")
            print(f"    trainer.loss_fn — the loss function")
            print(f"  Try: outputs = trainer.net(batches[0]['x'])")
            print(f"       torch.isnan(outputs).any(), outputs.min(), outputs.max()")
            # breakpoint()
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
    # 10b. Final checkpoint (always measure error at last epoch)
    # -----------------------------------------------------------------------
    if epochs_completed > 0 and (not error_tracking_data or error_tracking_data[-1][0] != epochs_completed):
        with torch.no_grad():
            final_loss_val = losses[-1][1]
            approx_factor = FactorNN(net, trainer.data_preprocessor)
            approx_exact = approx_factor.to_exact()
            approx_contribution = (approx_exact * exact_bw).sum_all_entries()
            log_z_err = approx_contribution - exact_contribution
            error_tracking_data.append(
                (epochs_completed, final_loss_val, log_z_err, abs(log_z_err))
            )
            print(f"[BenchmarkTraining] Final checkpoint epoch {epochs_completed}: "
                  f"loss={final_loss_val:.6e}, log_Z_err={log_z_err:.6f}, "
                  f"|log_Z_err|={abs(log_z_err):.6f}")

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
        'noise_seed': noise_seed,
        'multiply_messages': multiply_messages,
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

    # Build title suffix with scope width info
    import math
    scope_width = len(bucket.get_message_scope())
    log2_size = math.log2(message_size) if message_size > 0 else 0
    width_suffix = f", Width {scope_width} ({log2_size:.1f})"

    # Loss plot (best-effort — write metrics even if plots fail)
    try:
        if losses:
            plot_loss_curve(
                losses, loss_plot_path,
                title=f"Training Loss — bucket {bucket_label}{width_suffix}",
            )
            result['loss_plot_path'] = loss_plot_path
    except Exception as e:
        print(f"[BenchmarkTraining] WARNING: loss plot failed: {e}")

    # Local error plot (best-effort)
    try:
        if error_tracking_data:
            plot_local_error_curve(
                error_tracking_data, error_plot_path,
                title=f"Local Error — bucket {bucket_label}{width_suffix}",
            )
            result['error_plot_path'] = error_plot_path
    except Exception as e:
        print(f"[BenchmarkTraining] WARNING: error plot failed: {e}")

    # Approximation plot (exact vs approx factor comparison)
    try:
        approx_plot_path = os.path.join(bucket_output_dir, 'approximation.png')
        approx_factor = FactorNN(net, trainer.data_preprocessor)
        p = plot_fastfactor_comparison(
            exact_fw, approx_factor,
            title=f"Exact vs Approx — bucket {bucket_label}{width_suffix}",
            show=False, show_loss_curve=False,
        )
        p.savefig(approx_plot_path, bbox_inches="tight")
        p.close('all')
        result['approx_plot_path'] = approx_plot_path
    except Exception as e:
        print(f"[BenchmarkTraining] WARNING: approximation plot failed: {e}")

    # Top-assignments comparison plot (importance-weighted)
    try:
        top_assign_path = os.path.join(bucket_output_dir, 'top_assignments.png')
        approx_factor = FactorNN(net, trainer.data_preprocessor)
        approx_exact = approx_factor.to_exact()
        plot_top_assignments(
            exact_fw, approx_exact, exact_bw, top_assign_path,
            title_prefix=f"Bucket {bucket_label}{width_suffix}",
        )
        result['top_assignments_plot_path'] = top_assign_path
    except Exception as e:
        print(f"[BenchmarkTraining] WARNING: top_assignments plot failed: {e}")

    # Metrics JSON (always written)
    try:
        _write_metrics(result, config, bucket_data, metrics_path,
                       scope_width=scope_width, message_size=message_size)
        result['metrics_path'] = metrics_path
    except Exception as e:
        print(f"[BenchmarkTraining] WARNING: metrics write failed: {e}")

    # NN weights (raw state dict)
    try:
        weights_path = os.path.join(bucket_output_dir, 'nn_weights.pt')
        torch.save(net.state_dict(), weights_path)
        result['nn_weights_path'] = weights_path
    except Exception as e:
        print(f"[BenchmarkTraining] WARNING: nn_weights save failed: {e}")

    # MessageGenerator (picklable object that can regenerate the full table message)
    try:
        gen_path = os.path.join(bucket_output_dir, 'generate_nn.pkl')
        scope = bucket.get_message_scope()
        domain_sizes = [fastgm.matching_var(v).states for v in scope]
        generator = MessageGenerator(
            net, trainer.data_preprocessor, scope, domain_sizes, fastgm.lower_dim,
        )
        torch.save(generator, gen_path)
        result['generate_nn_path'] = gen_path
        print(f"[BenchmarkTraining] Saved MessageGenerator to {gen_path}")
    except Exception as e:
        print(f"[BenchmarkTraining] WARNING: generate_nn save failed: {e}")

    result['output_dir'] = bucket_output_dir
    return result
