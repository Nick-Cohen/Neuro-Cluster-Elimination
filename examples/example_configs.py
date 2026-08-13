"""Example NCE Configs - Working Examples

Two real configs that you can copy and run:
1. Quick 1-minute test (batch_size 2048, NeuroBE mode, no early stopping)
2. Standard training (bw_ib2=10, unnormalized KL loss)
"""

# Example 1: Quick 1-Minute Test
# - Fast training with large batches
# - NeuroBE mode enabled
# - No early stopping (trains all epochs)
example_1_quick_test = {
    'inference': {
        'device': 'cuda',
        'ib2': 19,  # Binary width 19 (sets both iB=19 and ecl=2^19-1=524287)
        'approximation_method': 'nn',
        'neurobe_mode': True,  # NeuroBE-faithful: ReLU, min-max normalization, patience stopping
    },
    'nn': {
        'hidden_sizes': [32, 32],  # Two hidden layers with 32 units each
        'activation': 'relu',  # ReLU activation (NeuroBE default)
    },
    'training': {
        'num_epochs': 50,  # Short run for 1-minute test
        'loss_fn': 'logspace_mse_fdb',  # Standard loss function
        'batch_size': 2048,  # Large batch = faster training
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'skip_early_stopping': True,  # No early stopping - train all 50 epochs
        'seed': 42,
    },
    'sampling': {
        'sampling_scheme': 'all',  # Enumerate all assignments
        'num_samples': 100000,
        'set_size': 100000,
        'val_set': True,
        'lower_dim': False,  # Standard full-dimensional sampling
    },
    'output': {
        'debug': False,
        'error_tracking': False,
        'traced_losses': [],  # No additional loss functions to track
    },
}

# Example 2: Standard Training (Different Settings)
# - Larger batch size for comparison
# - Tracks error over epochs
# - Shows typical settings
example_2_standard = {
    'inference': {
        'device': 'cuda',
        'ib2': 19,
        'approximation_method': 'nn',
    },
    'nn': {
        'hidden_sizes': [32, 32],
    },
    'training': {
        'num_epochs': 10,  # Short for demonstration
        'loss_fn': 'logspace_mse_fdb',
        'batch_size': 512,  # Medium batch size
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'skip_early_stopping': True,  # Train all epochs (no early stopping)
        'seed': 42,
    },
    'sampling': {
        'sampling_scheme': 'all',
        'num_samples': 100000,
        'set_size': 100000,
        'val_set': False,  # No validation set (faster)
        'lower_dim': False,
    },
    'output': {
        'debug': False,
        'error_tracking': False,  # Disabled for faster execution
        'traced_losses': [],
    },
}


if __name__ == '__main__':
    """Run both examples on a small test problem."""
    import sys
    sys.path.insert(0, '/home/cohenn1/NCE')
    
    from nce.inference.graphical_model import FastGM
    from nce.benchmark_problems.small_problems import small_problems
    import time
    
    # Use smokers_5.uai - smallest problem in small_problems
    test_problem = small_problems.problems[0]
    print(f"Test problem: smokers_5.uai (smallest problem)")
    print()
    
    # Example 1: Quick 1-minute test
    print("=" * 70)
    print("EXAMPLE 1: Quick 1-Minute Test (NeuroBE mode, batch_size 2048)")
    print("=" * 70)
    print("Config:")
    print(f"  - num_epochs: {example_1_quick_test['training']['num_epochs']}")
    print(f"  - batch_size: {example_1_quick_test['training']['batch_size']}")
    print(f"  - loss_fn: {example_1_quick_test['training']['loss_fn']}")
    print(f"  - neurobe_mode: {example_1_quick_test['inference']['neurobe_mode']}")
    print(f"  - skip_early_stopping: {example_1_quick_test['training']['skip_early_stopping']}")
    print()
    
    start = time.time()
    fastgm_1 = FastGM(model=test_problem, nn_config=example_1_quick_test, device='cuda')
    log_Z_1 = fastgm_1.get_log_partition_function()
    duration_1 = time.time() - start
    
    print(f"Results:")
    print(f"  - log_Z: {log_Z_1:.6f}")
    print(f"  - Duration: {duration_1:.2f}s")
    print()
    
    # Example 2: Standard training
    print("=" * 70)
    print("EXAMPLE 2: Standard Training (no early stopping, error tracking)")
    print("=" * 70)
    print("Config:")
    print(f"  - num_epochs: {example_2_standard['training']['num_epochs']}")
    print(f"  - batch_size: {example_2_standard['training']['batch_size']}")
    print(f"  - loss_fn: {example_2_standard['training']['loss_fn']}")
    print(f"  - skip_early_stopping: {example_2_standard['training']['skip_early_stopping']}")
    print()
    
    start = time.time()
    fastgm_2 = FastGM(model=test_problem, nn_config=example_2_standard, device='cuda')
    log_Z_2 = fastgm_2.get_log_partition_function()
    duration_2 = time.time() - start
    
    print(f"Results:")
    print(f"  - log_Z: {log_Z_2:.6f}")
    print(f"  - Duration: {duration_2:.2f}s")
    
    print()
    print("=" * 70)
    print("Both examples completed successfully!")
    print("=" * 70)
