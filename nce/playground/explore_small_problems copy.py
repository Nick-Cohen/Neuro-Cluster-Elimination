"""
Playground: Explore and train on small_problems benchmark set.

Run cells with #%% in VS Code / PyCharm / Spyder for interactive use.
Or run the whole file: python nce/playground/explore_small_problems.py
"""

# %% ── Imports ──────────────────────────────────────────────────────
import torch
import numpy as np
import copy
from nce.inference.graphical_model import FastGM
from nce.inference.factor import FastFactor
from nce.benchmark_problems import small_problems
from nce.config_schema import prepare_config

# %% ── List all problems ───────────────────────────────────────────
print("=" * 70)
print("small_problems benchmark set")
print("=" * 70)
for i, p in enumerate(small_problems.problems):
    name = p.file.split('/')[-1].replace('.uai', '')
    print(f"  [{i:2d}] {name:35s}  vars={p.num_vars:3d}  width={p.width}")

# %% ── SELECT A PROBLEM ────────────────────────────────────────────
# Change this index to pick a different problem
PROBLEM_IDX = 1  # BN_3 — small, fast, good for testing

problem = small_problems.problems[PROBLEM_IDX]
problem_name = problem.file.split('/')[-1].replace('.uai', '')
print(f"\nSelected: [{PROBLEM_IDX}] {problem_name}")
print(f"  vars={problem.num_vars}, factors={problem.num_factors}, width={problem.width}")

# %% ── CONFIG ──────────────────────────────────────────────────────
# Start from the benchmark default, then override what you want.
# Edit these values to experiment.

config = copy.deepcopy(small_problems.configs['default'][PROBLEM_IDX])

# ── Overrides (edit these) ──
config['device'] = 'cuda'           # 'cuda' or 'cpu'
config['num_epochs'] = 500          # quick iteration; bump to 5000-10000 for real runs
config['hidden_sizes'] = [32, 32]   # network architecture
config['loss_fn'] = 'logspace_mse_fdb'  # try: 'unnormalized_kl', 'linspace_mse_fdb', 'neurobe_weighted_mse'
config['lr'] = 0.001
config['batch_size'] = 4096
config['num_samples'] = 50000
config['sampling_scheme'] = 'uniform'  # 'uniform', 'all', 'mg'
config['track_errors'] = False       # set True to compute per-bucket NN error (slower)

print(f"\nConfig:")
for k in ['ecl', 'iB', 'num_epochs', 'hidden_sizes', 'loss_fn', 'lr', 'batch_size',
          'num_samples', 'sampling_scheme', 'device']:
    print(f"  {k}: {config[k]}")

# %% ── BUILD THE GRAPHICAL MODEL ──────────────────────────────────
print(f"\nBuilding FastGM for {problem_name}...")
fastgm = FastGM(model=problem, nn_config=config, device=config['device'])

# Show bucket structure
n_buckets = len(fastgm.buckets)
large_buckets = fastgm.get_large_message_buckets(iB=config['iB'], ecl=config['ecl'])
adjusted_width, n_exceeding = fastgm.get_adjusted_width(ecl=config['ecl'])

print(f"  Total buckets: {n_buckets}")
print(f"  Buckets needing NN: {len(large_buckets)}")
print(f"  Adjusted width: {adjusted_width:.1f}")
print(f"  Message scopes exceeding ecl: {n_exceeding}")

# %% ── SHOW MESSAGE SCOPES ────────────────────────────────────────
# Which buckets are exact vs NN?
print(f"\nBucket breakdown (ecl={config['ecl']}, iB={config['iB']}):")
nn_buckets = []
for var in fastgm.elim_order:
    scope = fastgm.message_scopes.get(var.label, [])
    msg_size = int(np.prod([fastgm.matching_var(v).states for v in scope])) if scope else 1
    is_nn = var.label in large_buckets
    tag = "NN" if is_nn else "exact"
    if is_nn:
        nn_buckets.append((var.label, len(scope), msg_size))
    # Only print NN buckets to keep output clean
    if is_nn:
        print(f"  Bucket {var.label:3d}: width={len(scope):2d}  msg_size={msg_size:>10,}  [{tag}]")

if not nn_buckets:
    print("  (all buckets are exact — lower ecl or iB to force NN training)")

# %% ── RUN INFERENCE (TRAIN NNs) ──────────────────────────────────
print(f"\nRunning inference on {problem_name}...")
print(f"  Training {len(large_buckets)} NN buckets, {config['num_epochs']} epochs each")
print("-" * 50)

fastgm.eliminate_variables(all=True)

print("-" * 50)
print(f"  Log partition function (approx): {fastgm.log_partition_function:.6f}")
if problem.logSS is not None:
    exact_logz = float(problem.logSS)
    error = fastgm.log_partition_function - exact_logz
    print(f"  Log partition function (exact):  {exact_logz:.6f}")
    print(f"  Error (log10):                   {error:.6f}")

# %% ── TRAINING LOG ───────────────────────────────────────────────
# Per-bucket training details (loss curves, epochs, hidden sizes)
print(f"\nPer-bucket training log ({len(fastgm.per_bucket_training_log)} NN buckets):")
for entry in fastgm.per_bucket_training_log:
    label = entry.get('label', '?')
    epochs = entry.get('epochs_trained', '?')
    hsizes = entry.get('hidden_sizes', '?')
    losses = entry.get('losses', [])
    final_loss = losses[-1] if losses else '?'
    print(f"  Bucket {label}: {epochs} epochs, hidden={hsizes}, final_loss={final_loss:.6f}" if isinstance(final_loss, float) else f"  Bucket {label}: {epochs} epochs, hidden={hsizes}")

# %% ── PLOT LOSS CURVES ───────────────────────────────────────────
# Uncomment to plot (requires display or saves to file)
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt

    if fastgm.per_bucket_training_log:
        fig, axes = plt.subplots(1, min(len(fastgm.per_bucket_training_log), 4),
                                  figsize=(4 * min(len(fastgm.per_bucket_training_log), 4), 3),
                                  squeeze=False)
        for i, entry in enumerate(fastgm.per_bucket_training_log[:4]):
            ax = axes[0][i]
            losses = entry.get('losses', [])
            if losses:
                ax.plot(losses)
                ax.set_title(f"Bucket {entry.get('label', '?')}")
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_yscale('symlog', linthresh=1e-3)
        plt.suptitle(f'{problem_name} — Loss Curves', y=1.02)
        plt.tight_layout()
        outpath = f'nce/playground/{problem_name}_loss_curves.png'
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f"\nLoss curves saved to: {outpath}")
        plt.close()
    else:
        print("\nNo NN buckets trained — nothing to plot.")
except Exception as e:
    print(f"\nPlotting skipped: {e}")

# %% ── COMPARE LOSS FUNCTIONS ─────────────────────────────────────
# Quick A/B test: run the same problem with two loss functions.
# Uncomment and edit to use.

# loss_fns_to_compare = ['logspace_mse_fdb', 'unnormalized_kl']
# results = {}
# for lf in loss_fns_to_compare:
#     cfg = copy.deepcopy(config)
#     cfg['loss_fn'] = lf
#     cfg['num_epochs'] = 200  # quick comparison
#     gm = FastGM(model=problem, nn_config=cfg, device=cfg['device'])
#     gm.eliminate_variables(all=True)
#     logz = gm.log_partition_function
#     exact = float(problem.logSS) if problem.logSS else None
#     err = logz - exact if exact else None
#     results[lf] = {'logz': logz, 'error': err}
#     print(f"  {lf:30s}  logZ={logz:.4f}  err={err:.4f}" if err else f"  {lf:30s}  logZ={logz:.4f}")

# %% ── COMPARE ACROSS PROBLEMS ────────────────────────────────────
# Sweep multiple problems with the same config.
# Uncomment and edit to use.

# problem_indices = [1, 2, 3, 4]  # BN_3, BN_5, BN_7, BN_10
# for idx in problem_indices:
#     p = small_problems.problems[idx]
#     name = p.file.split('/')[-1].replace('.uai', '')
#     cfg = copy.deepcopy(config)
#     cfg['num_epochs'] = 200
#     gm = FastGM(model=p, nn_config=cfg, device=cfg['device'])
#     gm.eliminate_variables(all=True)
#     exact = float(p.logSS) if p.logSS else None
#     err = gm.log_partition_function - exact if exact else None
#     print(f"  [{idx:2d}] {name:30s}  logZ={gm.log_partition_function:.4f}  err={err:.4f}" if err else f"  [{idx:2d}] {name:30s}  logZ={gm.log_partition_function:.4f}")

# %% ── INSPECT A SINGLE BUCKET ────────────────────────────────────
# Rebuild and inspect a specific bucket before elimination.
# Uncomment and edit to use.

# fastgm2 = FastGM(model=problem, nn_config=config, device=config['device'])
# bucket_label = large_buckets[0] if large_buckets else list(fastgm2.message_scopes.keys())[0]
# bucket = fastgm2.get_bucket(bucket_label)
# print(f"\nBucket {bucket_label}:")
# print(f"  Width: {bucket.get_width()}")
# print(f"  EC: {bucket.get_ec():,}")
# print(f"  Message scope: {bucket.get_message_scope()}")
# print(f"  Factors: {len(bucket.factors)}")
# for i, f in enumerate(bucket.factors):
#     print(f"    [{i}] scope={f.labels}, shape={f.tensor.shape}")
