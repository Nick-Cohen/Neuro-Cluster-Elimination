"""Example NCE neural network configs.

Both flat (legacy) and nested (recommended) formats are valid.
``prepare_config()`` auto-detects the format, validates, resolves aliases,
and returns a flat dict with internal key names that consumer code reads.

See docs/config_reference.md for the full field reference.
"""

# ── Flat config (legacy) ─────────────────────────────────────────────
# Uses internal key names directly. Still fully supported.

flat_config = {
    'ecl': 14,
    'iB': 10,
    'approximation_method': 'nn',
    'device': 'cuda',
    'hidden_sizes': [32, 32],
    'loss_fn': 'logspace_mse_fdb',
    'num_epochs': 500,
    'lr': 0.001,
    'batch_size': 256,
    'optimizer': 'adam',
    'num_samples': 5000,
    'sampling_scheme': 'uniform',
    'val_set': True,
    'debug': False,
}

# ── Nested config (recommended) ─────────────────────────────────────
# Organized by section. Uses readable names (aliases resolve automatically).
# Same parameter values as flat_config above.

nested_config = {
    'inference': {
        'exact_computation_limit': 14,
        'i_bound': 10,
        'approximation_method': 'nn',
        'device': 'cuda',
    },
    'nn': {
        'hidden_sizes': [32, 32],
    },
    'training': {
        'loss_fn': 'logspace_mse_fdb',
        'num_epochs': 500,
        'learning_rate': 0.001,
        'batch_size': 256,
        'optimizer': 'adam',
    },
    'sampling': {
        'num_samples': 5000,
        'sampling_scheme': 'uniform',
        'val_set': True,
    },
    'output': {
        'debug': False,
    },
}
