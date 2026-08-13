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
        'error_tracking': True,  # Enable to track convergence
        'traced_losses': [],
    },
}

print("=" * 70)
print("Standard Training Example")
print("=" * 70)
print(f"Test problem: smokers_5.uai")
print(f"Config: 20 epochs, batch_size=512, error tracking enabled")
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

# Get error tracking data
error_tracking = []
if hasattr(fastgm, 'error_tracking_data') and fastgm.error_tracking_data:
    # Parse the error tracking data
    # Format: [(bucket_id, [(epoch, loss, log_Z_err, abs_log_Z_err), ...]), ...]
    for entry in fastgm.error_tracking_data:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            bucket_id, checkpoint_data = entry[0], entry[1]
            
            if isinstance(checkpoint_data, (list, tuple)):
                for checkpoint in checkpoint_data:
                    if isinstance(checkpoint, (list, tuple)) and len(checkpoint) >= 4:
                        epoch, loss, log_Z_err, abs_log_Z_err = checkpoint[:4]
                        error_tracking.append({
                            'epoch': int(epoch),
                            'loss': float(loss),
                            'log_Z_err': float(log_Z_err),
                            'abs_log_Z_err': float(abs_log_Z_err),
                        })
    
    # Create convergence plots
    if error_tracking:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            epochs = [e['epoch'] for e in error_tracking]
            abs_errors = [e['abs_log_Z_err'] for e in error_tracking]
            losses = [e['loss'] for e in error_tracking]
            
            # Plot 1: Log Z error convergence
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1.plot(epochs, abs_errors, 'b-o', linewidth=2, markersize=8)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('|log Z error|', fontsize=12)
            ax1.set_title('Convergence: Absolute log Z Error', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # Plot 2: Training loss
            ax2.plot(epochs, losses, 'r-o', linewidth=2, markersize=8)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Training Loss', fontsize=12)
            ax2.set_title('Training Loss over Epochs', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            
            plt.tight_layout()
            plot_path = PLOTS_DIR / 'convergence.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Convergence plots saved to: {plot_path}")
            
        except Exception as e:
            print(f"⚠ Could not create plots: {e}")
            import traceback
            traceback.print_exc()
    
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

if error_tracking:
    print(f"\nError tracking ({len(error_tracking)} checkpoints):")
    print(f"  Initial |log_Z_err|: {error_tracking[0]['abs_log_Z_err']:.6f} (epoch {error_tracking[0]['epoch']})")
    print(f"  Final |log_Z_err|: {error_tracking[-1]['abs_log_Z_err']:.6f} (epoch {error_tracking[-1]['epoch']})")
    improvement = error_tracking[0]['abs_log_Z_err'] / max(error_tracking[-1]['abs_log_Z_err'], 1e-10)
    print(f"  Improvement: {improvement:.1f}x")
    print(f"  Plot: {PLOTS_DIR / 'convergence.png'}")

print(f"\n✓ Results saved to: {OUTPUT_DIR / 'results.json'}")
print("\nTo reproduce these exact results:")
print(f"  python {Path(__file__).relative_to(Path.cwd())}")
