# %% Imports
from nce.benchmark_problems import nbe_sanity_check

# %% Print all models and their paired configs
print("=== nbe_sanity_check: Models + Configs ===\n")
for model, config in zip(nbe_sanity_check.problems, nbe_sanity_check.configs['nbe']):
    print(f"Model: {model.modelfile}")
    print(f"  num_vars: {model.num_vars}, width: {model.width}")
    print(f"  Config ({len(config)} keys):")
    for k, v in config.items():
        print(f"    {k}: {v}")
    print()

# %% Print config key summary
print("=== Config key summary ===\n")
sample_config = nbe_sanity_check.configs['nbe'][0]
print(f"Total config keys: {len(sample_config)}")
print(f"Keys: {list(sample_config.keys())}")

# %% Verify expected values
print("\n=== Verification ===\n")
expected_hs = {
    'pedigree/pedigree13': 'nbe,3',
    'grids/grid40x40.f10': 'nbe,1',
    'grids/grid20x20.f10': 'nbe,1',
    'dbn/rbm_20': 'nbe,3',
}
for model, config in zip(nbe_sanity_check.problems, nbe_sanity_check.configs['nbe']):
    # Find which key this model corresponds to
    hs = config['hidden_sizes']
    bs = config['batch_size']
    iB = config['iB']
    status = "PASS" if bs == 256 and iB == 10 else "FAIL"
    print(f"  [{status}] {model.modelfile}: hidden_sizes={hs}, batch_size={bs}, iB={iB}")

print("\nDone!")
