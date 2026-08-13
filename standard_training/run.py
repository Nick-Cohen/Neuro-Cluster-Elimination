#!/usr/bin/env python3
"""Standard Training Example - Full Epochs

This is a complete working example showing standard training without early stopping.

USAGE:
    python examples/standard_training/run.py

OUTPUT:
    - Console output showing training progress  
    - results.json with log_Z estimate and metadata
    - config.json showing the exact configuration used
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, '/home/cohenn1/NCE')

from nce.inference.graphical_model import FastGM
from nce.benchmark_problems.small_problems import small_problems

# Output directory
OUTPUT_DIR = Path(__file__).parent
PLOTS_DIR = OUTPUT_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# The config - demonstrates error tracking
config = {
    'inference': {
        'device': 'cuda',
        'ib2': 19,
        'approximation_method': 'nn',
    },
    'nn': {
        'hidden_sizes': [32, 32],
    },
    'training': {
        'num_epochs': 20,  # Reduced for faster demonstration
        'loss_fn': 'logspace_mse_fdb',
        'batch_size': 512,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'skip_early_stopping': True,
        'seed': 42,
    },
    'sampling': {
        'sampling_scheme': 'all',
        'num_samples': 100000,
        'set_size': 100000,
        'val_set': False,
        'lower_dim': False,
    },
    'output': {
        'debug': False,
        'error_tracking': False,  # Disabled for faster execution
        'traced_losses': [],
    },
}

print("=" * 70)
print("Standard Training Example")
print("=" * 70)
print(f"Test problem: smokers_5.uai")
print(f"Config: 20 epochs, batch_size=512")
print(f"Output directory: {OUTPUT_DIR}")
print()

# Save config
with open(OUTPUT_DIR / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)
print(f"✓ Config saved to: {OUTPUT_DIR / 'config.json'}")

# Run inference
print("\nRunning inference...")
start = time.time()

test_problem = small_problems.problems[0]  # smokers_5.uai
fastgm = FastGM(model=test_problem, nn_config=config, device='cuda')
log_Z = fastgm.get_log_partition_function()

duration = time.time() - start

# Get error tracking data (if available)
error_tracking = []
# Note: error_tracking disabled in this example for faster execution
# To enable: set config['output']['error_tracking'] = True
    
# Save results
results = {
    'log_Z': float(log_Z),
    'duration_seconds': duration,
    'problem': 'smokers_5.uai',
    'error_tracking': error_tracking,
    'config': config,
}

with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"log_Z estimate: {log_Z:.6f}")
print(f"Duration: {duration:.2f}s")

print(f"\n✓ Results saved to: {OUTPUT_DIR / 'results.json'}")
print("\nTo reproduce these exact results:")
print(f"  python {Path(__file__).relative_to(Path.cwd())}")
