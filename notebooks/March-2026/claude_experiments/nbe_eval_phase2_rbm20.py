# %% [markdown]
# # Phase 2: rbm_20 Deep Dive with 5 Ablation Variants
#
# Run 5 ablation variants on rbm_20 (index 3, 40 vars, width 20) to measure
# the contribution of each NBE component.
#
# rbm_20 has the highest width (20) among easily runnable sanity check problems,
# so NBE's adaptive sampling matters most here.
#
# Variants:
# | Variant            | num_samples | hidden_sizes | loss_fn                |
# |--------------------|-------------|--------------|------------------------|
# | NBE-full           | nbe,0.1     | nbe,3        | weighted_logspace_mse  |
# | NBE-fixed-samples  | 50000       | nbe,3        | weighted_logspace_mse  |
# | NBE-fixed-arch     | nbe,0.1     | [3,3]        | weighted_logspace_mse  |
# | NBE-fixed-loss     | nbe,0.1     | nbe,3        | unnormalized_kl        |
# | Baseline           | 50000       | [3,3]        | unnormalized_kl        |

# %%
import time
import traceback
from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM

# Model index 3 = rbm_20 (40 vars, width 20)
MODEL_IDX = 3

print("=" * 70)
print("Phase 2: rbm_20 Deep Dive - 5 Ablation Variants")
print("=" * 70)
model = nbe_sanity_check.problems[MODEL_IDX]
print(f"Problem: {model.modelfile}")
print(f"Num vars: {model.num_vars}")
print(f"Width: {model.width}")
print()

# %%
# Define the 5 ablation variants
ablation_variants = [
    {
        'name': 'NBE-full',
        'description': 'Full NBE config (adaptive everything)',
        'overrides': {
            'num_samples': 'nbe,0.1',
            'hidden_sizes': 'nbe,3',
            'loss_fn': 'weighted_logspace_mse',
        }
    },
    {
        'name': 'NBE-fixed-samples',
        'description': 'Remove adaptive sampling (fixed 50k)',
        'overrides': {
            'num_samples': 50000,
            'hidden_sizes': 'nbe,3',
            'loss_fn': 'weighted_logspace_mse',
        }
    },
    {
        'name': 'NBE-fixed-arch',
        'description': 'Remove adaptive architecture ([3,3] fixed)',
        'overrides': {
            'num_samples': 'nbe,0.1',
            'hidden_sizes': [3, 3],
            'loss_fn': 'weighted_logspace_mse',
        }
    },
    {
        'name': 'NBE-fixed-loss',
        'description': 'Remove weighted loss (unnormalized_kl)',
        'overrides': {
            'num_samples': 'nbe,0.1',
            'hidden_sizes': 'nbe,3',
            'loss_fn': 'unnormalized_kl',
        }
    },
    {
        'name': 'Baseline',
        'description': 'Standard config (fixed samples + arch + kl loss)',
        'overrides': {
            'num_samples': 50000,
            'hidden_sizes': [3, 3],
            'loss_fn': 'unnormalized_kl',
        }
    },
]

# %%
# Results storage
results = []

# %%
# Run each variant
for i, variant in enumerate(ablation_variants):
    print(f"\n{'=' * 70}")
    print(f"Variant {i+1}/{len(ablation_variants)}: {variant['name']}")
    print(f"Description: {variant['description']}")
    print(f"Overrides: {variant['overrides']}")
    print('=' * 70)

    try:
        # Create a fresh copy of the benchmark config
        config = dict(nbe_sanity_check.configs['nbe'][MODEL_IDX])

        # Apply variant overrides
        for k, v in variant['overrides'].items():
            config[k] = v

        # Set device to cuda and enable error tracking
        config['device'] = 'cuda'
        config['track_errors'] = True

        # Create FastGM and run inference
        t0 = time.time()
        fastgm = FastGM(model=model, nn_config=config, device='cuda')
        t_create = time.time() - t0
        print(f"FastGM created in {t_create:.2f}s")

        t0 = time.time()
        log_z = fastgm.get_log_partition_function()
        t_infer = time.time() - t0

        result = {
            'name': variant['name'],
            'log_z': float(log_z),
            'num_trained': fastgm.num_trained,
            'time': t_infer,
            'status': 'OK',
            'error': None,
        }
        print(f"\nResult: log Z = {log_z:.6f}")
        print(f"Num trained: {fastgm.num_trained}")
        print(f"Time: {t_infer:.2f}s")

    except Exception as e:
        print(f"\nERROR in variant {variant['name']}: {e}")
        traceback.print_exc()
        result = {
            'name': variant['name'],
            'log_z': None,
            'num_trained': None,
            'time': None,
            'status': 'ERROR',
            'error': str(e),
        }

    results.append(result)

# %%
# Print summary table
print()
print("=" * 70)
print("SUMMARY TABLE - Phase 2: rbm_20 Ablation")
print("=" * 70)
print(f"{'Variant':<22} {'Log Z':>12} {'Num Trained':>12} {'Time (s)':>10} {'Status':>8}")
print("-" * 70)
for r in results:
    log_z_str = f"{r['log_z']:.4f}" if r['log_z'] is not None else "ERROR"
    num_trained_str = str(r['num_trained']) if r['num_trained'] is not None else "N/A"
    time_str = f"{r['time']:.1f}" if r['time'] is not None else "N/A"
    print(f"{r['name']:<22} {log_z_str:>12} {num_trained_str:>12} {time_str:>10} {r['status']:>8}")
print()

# Compute differences relative to NBE-full
full_result = next((r for r in results if r['name'] == 'NBE-full'), None)
if full_result and full_result['log_z'] is not None:
    print(f"Log Z differences vs NBE-full ({full_result['log_z']:.4f}):")
    for r in results:
        if r['name'] != 'NBE-full' and r['log_z'] is not None:
            diff = r['log_z'] - full_result['log_z']
            print(f"  {r['name']:<22}: {diff:+.4f}")
print()
print("Phase 2 complete.")
