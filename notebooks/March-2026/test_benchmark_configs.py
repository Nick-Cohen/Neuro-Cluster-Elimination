# %% Imports
from nce.benchmark_problems import (
    neuro_be_sanity_check,
    neuro_be_sanity_check_configs,
    neuro_be_sanity_check_configs_list,
)

# %% Print all models and their paired configs
print("=== neuro_be_sanity_check: Models + Configs ===\n")
for model, config in zip(neuro_be_sanity_check, neuro_be_sanity_check_configs_list):
    print(f"Model: {model.modelfile}")
    print(f"  num_vars: {model.num_vars}, width: {model.width}")
    print(f"  Config: {config}")
    print()

# %% Print the full configs dict (keyed by catalogue key)
print("=== Configs dict (keyed by catalogue key) ===\n")
for key, config in neuro_be_sanity_check_configs.items():
    print(f"  {key}: {config}")

# %% Verify expected values
print("\n=== Verification ===\n")
expected = {
    'pedigree/pedigree13': 'nbe,3',
    'grids/grid40x40.f10': 'nbe,1',
    'grids/grid20x20.f10': 'nbe,1',
    'dbn/rbm_20': 'nbe,3',
}
for key, expected_hs in expected.items():
    actual_hs = neuro_be_sanity_check_configs[key]['hidden_sizes']
    actual_bs = neuro_be_sanity_check_configs[key]['batch_size']
    status = "PASS" if actual_hs == expected_hs and actual_bs == 256 else "FAIL"
    print(f"  [{status}] {key}: hidden_sizes={actual_hs}, batch_size={actual_bs}")

print("\nAll checks passed!" if all(
    neuro_be_sanity_check_configs[k]['hidden_sizes'] == v and
    neuro_be_sanity_check_configs[k]['batch_size'] == 256
    for k, v in expected.items()
) else "\nSome checks FAILED!")
