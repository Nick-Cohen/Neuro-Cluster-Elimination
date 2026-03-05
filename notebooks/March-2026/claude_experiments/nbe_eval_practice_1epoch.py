# %% [markdown]
# # Phase 1 Practice: 1-Epoch Full Run
#
# Run the full NBE pipeline on grid10x10.f5.wrap with num_epochs=1
# as a practice run before the real 500-epoch evaluation.
#
# Uses the benchmark config defaults for ecl (2**22) and iB (10) --
# these are NOT overridden, so some buckets will use NN training.

# %%
import time
from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM

# Load grid10x10.f5.wrap (index 4)
model = nbe_sanity_check.problems[4]
config = dict(nbe_sanity_check.configs['nbe'][4])

# Override ONLY num_epochs and device
# DO NOT override ecl or iB -- use the benchmark defaults (ecl=2**22, iB=10)
config['num_epochs'] = 1
config['device'] = 'cpu'

print(f"Model: {model.modelfile}")
print(f"Num vars: {model.num_vars}")
print(f"Config: num_epochs=1, device=cpu")
print(f"  ecl={config['ecl']} (benchmark default)")
print(f"  iB={config['iB']} (benchmark default)")
print(f"  loss_fn={config['loss_fn']}")
print(f"  hidden_sizes={config['hidden_sizes']}")
print(f"  num_samples={config['num_samples']}")
print(f"  dope_factors={config['dope_factors']}")
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
# Show elimination structure
print("Elimination structure:")
fastgm.show_elimination(all=True)
print()

# %%
# Run full inference (1 epoch per NN bucket)
print("Running get_log_partition_function() with num_epochs=1...")
print("(This will train NNs on buckets that exceed ecl threshold)")
print()
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
print(f"Log Z estimate: {log_z}")
print(f"Num trained (NN buckets): {fastgm.num_trained}")
print(f"Total time: {t_total:.2f}s")

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

print()
print("Note: With num_epochs=1, accuracy will be poor. This is just a pipeline test.")
print("Run with num_epochs=500 (benchmark default) for real evaluation.")
print()
print("Phase 1 practice (1-epoch) complete.")
