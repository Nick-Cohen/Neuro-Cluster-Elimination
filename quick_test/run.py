#!/usr/bin/env python3
"""Quick Test Example - NeuroBE Mode

This is a complete working example you can run with one command.
Everything is pre-configured. Just run it.

USAGE:
    python examples/quick_test/run.py

OUTPUT:
    - Console output showing training progress
    - results.json with log_Z estimate and metadata
    - If plots enabled: convergence graphs in examples/quick_test/plots/
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
OUTPUT_DIR.mkdir(exist_ok=True)

# The config - exactly as shown in documentation
config = {
    'inference': {
        'device': 'cuda',
        'ib2': 19,
        'approximation_method': 'nn',
        'neurobe_mode': True,
    },
    'nn': {
        'hidden_sizes': [32, 32],
        'activation': 'relu',
    },
    'training': {
        'num_epochs': 50,
        'loss_fn': 'logspace_mse_fdb',
        'batch_size': 2048,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'skip_early_stopping': True,
        'seed': 42,
    },
    'sampling': {
        'sampling_scheme': 'all',
        'num_samples': 100000,
        'set_size': 100000,
        'val_set': True,
        'lower_dim': False,
    },
    'output': {
        'debug': False,
        'error_tracking': False,
        'traced_losses': [],
    },
}

print("=" * 70)
print("Quick Test Example - NeuroBE Mode")
print("=" * 70)
print(f"Test problem: smokers_5.uai (smallest benchmark problem)")
print(f"Config: NeuroBE mode, batch_size=2048, 50 epochs max")
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

# Save results
results = {
    'log_Z': float(log_Z),
    'duration_seconds': duration,
    'problem': 'smokers_5.uai',
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
