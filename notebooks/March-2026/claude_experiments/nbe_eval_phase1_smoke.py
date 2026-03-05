# %% [markdown]
# # Phase 1 Full: 500-Epoch Smoke Test on grid10x10.f5.wrap
#
# Run the full NBE pipeline on grid10x10.f5.wrap with the benchmark num_epochs=500.
# With ecl=2^10=1024 and approximation_method='nn', buckets with width > 10
# will train neural networks.
#
# Phase 0a exact baseline: log Z = 169.40834045410156

# %%
import time
from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM

EXACT_LOG_Z = 169.40834045410156  # From Phase 0a

# Load grid10x10.f5.wrap (index 4)
model = nbe_sanity_check.problems[4]
config = dict(nbe_sanity_check.configs['nbe'][4])

# Use cuda for training
config['device'] = 'cuda'

print("=" * 60)
print("Phase 1 Full: 500-Epoch Smoke Test")
print("=" * 60)
print(f"Problem: {model.modelfile}")
print(f"Num vars: {model.num_vars}")
print(f"Config:")
print(f"  num_epochs={config['num_epochs']}")
print(f"  ecl={config['ecl']} (2^{config['iB']})")
print(f"  iB={config['iB']}")
print(f"  loss_fn={config['loss_fn']}")
print(f"  hidden_sizes={config['hidden_sizes']}")
print(f"  num_samples={config['num_samples']}")
print(f"  approximation_method={config['approximation_method']}")
print(f"  dope_factors={config['dope_factors']}")
print(f"  device={config['device']}")
print()
print(f"Phase 0a exact baseline: {EXACT_LOG_Z}")
print()

# %%
# Create FastGM
print("Creating FastGM...")
t0 = time.time()
fastgm = FastGM(model=model, nn_config=config, device=config['device'])
t_create = time.time() - t0
print(f"FastGM created in {t_create:.2f}s")
print()

# %%
# Run full inference with 500 epochs
print(f"Running get_log_partition_function() with num_epochs={config['num_epochs']}...")
t0 = time.time()
log_z = fastgm.get_log_partition_function()
t_total = time.time() - t0

# %%
# Print results
print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Problem: {model.modelfile}")
print(f"Num vars: {model.num_vars}")
print(f"Log Z estimate (NBE, {config['num_epochs']} epochs): {log_z}")
print(f"Exact baseline (Phase 0a):                   {EXACT_LOG_Z}")
abs_err = abs(float(log_z) - EXACT_LOG_Z)
print(f"Absolute error: {abs_err:.6f}")
print(f"Num trained (NN buckets): {fastgm.num_trained}")
print(f"Total time: {t_total:.2f}s")
print()
print("Phase 1 full complete.")
