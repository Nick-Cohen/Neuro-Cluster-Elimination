# %% [markdown]
# # Phase 1 Full: 500-Epoch Smoke Test on grid10x10.f5.wrap
#
# Run the full NBE pipeline on grid10x10.f5.wrap with the benchmark num_epochs=500.
# This is the "full" version of the Phase 1 practice run.
#
# Expected: num_trained=0 for grid10x10 (max width=21, ecl=2^22=4M > 2^21=2M).
# All buckets run exactly, so log Z = exact baseline from Phase 0a.
#
# Phase 0a exact baseline: log Z = 169.40834045410156

# %%
import time
import os
import sys
from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM

EXACT_LOG_Z = 169.40834045410156  # From Phase 0a

# Load grid10x10.f5.wrap (index 4)
model = nbe_sanity_check.problems[4]
config = dict(nbe_sanity_check.configs['nbe'][4])

# Use benchmark defaults (num_epochs=500, ecl=2^22, iB=10)
# Override only device - use cuda for training (even though no NN expected here)
config['device'] = 'cuda'

print("=" * 60)
print("Phase 1 Full: 500-Epoch Smoke Test")
print("=" * 60)
print(f"Problem: {model.modelfile}")
print(f"Num vars: {model.num_vars}")
print(f"Config:")
print(f"  num_epochs={config['num_epochs']} (benchmark default)")
print(f"  ecl={config['ecl']} = 2^{config['ecl'].bit_length()-1}")
print(f"  iB={config['iB']} (benchmark default)")
print(f"  loss_fn={config['loss_fn']}")
print(f"  hidden_sizes={config['hidden_sizes']}")
print(f"  num_samples={config['num_samples']}")
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
print("Running get_log_partition_function() with num_epochs=500...")
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
print(f"Log Z estimate (NBE, 500 epochs): {log_z}")
print(f"Exact baseline (Phase 0a):        {EXACT_LOG_Z}")
abs_err = abs(float(log_z) - EXACT_LOG_Z)
print(f"Absolute error: {abs_err:.6f}")
print(f"Num trained (NN buckets): {fastgm.num_trained}")
print(f"Total time: {t_total:.2f}s")
print()

if fastgm.num_trained == 0:
    print("Note: num_trained=0 because all bucket message sizes < ecl=2^22.")
    print("grid10x10 max width=21, so max message size=2^21=2M < 4M=2^22.")
    print("All buckets run exactly, producing the same result as Phase 0a.")
    print("This is expected behavior - the benchmark config is designed for larger models.")
    print()
    print("To test NBE training on grid10x10, lower ecl to 2^15 or 2^16.")
else:
    print(f"NBE trained {fastgm.num_trained} buckets with NNs.")

print()
print("Phase 1 full (500-epoch) complete.")
