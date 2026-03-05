# %% [markdown]
# # Phase 0a: Exact Computation Baseline
#
# Verify that exact bucket elimination works on grid10x10.f5.wrap
# by setting ecl=2**30 and iB=30 (forcing all buckets to compute exactly).
#
# This tests the core elimination pipeline without any NN training.

# %%
import time
from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM

# Load grid10x10.f5.wrap (index 4 in nbe_sanity_check)
model = nbe_sanity_check.problems[4]
config = dict(nbe_sanity_check.configs['nbe'][4])

# Override to force exact computation everywhere
config['ecl'] = 2**30       # very high exact computation limit -> no NN training
config['iB'] = 30           # very high i-bound -> no mini-bucket splitting
config['device'] = 'cpu'    # no GPU needed for exact computation on small problem

print(f"Model: {model.modelfile}")
print(f"Num vars: {model.num_vars}")
print(f"Config overrides: ecl=2^30, iB=30, device=cpu")
print(f"dope_factors: {config['dope_factors']}")
print()

# %%
# Create FastGM
# Note: dope_factors is called automatically by the constructor since config['dope_factors']=True
print("Creating FastGM...")
t0 = time.time()
fastgm = FastGM(model=model, nn_config=config, device='cpu')
t_create = time.time() - t0
print(f"FastGM created in {t_create:.2f}s")
print()

# %%
# Compute log partition function (exact elimination)
print("Computing log partition function (exact mode)...")
t0 = time.time()
log_z = fastgm.get_log_partition_function()
t_elim = time.time() - t0

print(f"Log Z estimate (exact): {log_z}")
print(f"Elimination time: {t_elim:.2f}s")
print(f"Num trained (should be 0 for exact): {fastgm.num_trained}")
print()

# %%
# Compare to known exact value if available
if hasattr(model, 'ln_z') and model.ln_z is not None:
    print(f"True log Z (model.ln_z): {model.ln_z}")
    abs_err = abs(float(log_z) - float(model.ln_z))
    print(f"Absolute error: {abs_err:.6f}")
elif hasattr(model, 'logZ') and model.logZ is not None:
    print(f"True log Z (model.logZ): {model.logZ}")
    abs_err = abs(float(log_z) - float(model.logZ))
    print(f"Absolute error: {abs_err:.6f}")
else:
    print("No exact log Z available for comparison.")
    print("Save this value as the exact baseline for later phases.")

print()
print("Phase 0a complete.")
