# %% [markdown]
# # Phase 0b: Single Bucket NN Test
#
# Test NN training on ONE large bucket in isolation using pedigree13.
# This verifies that the training pipeline (sampling, loss function, optimizer)
# works for a single bucket before running full inference.
#
# pedigree13 has 1077 vars and width 32, so it should have large buckets.
#
# Note: pedigree13 on CPU may be slow for the exact elimination steps.
# If available, consider using device='cuda'.

# %%
import time
from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM

# Load pedigree13 (index 0)
model = nbe_sanity_check.problems[0]
config = dict(nbe_sanity_check.configs['nbe'][0])

# Override config for single-bucket test
config['ecl'] = 2**30       # exact mode for all preceding buckets
config['iB'] = 30           # no mini-bucket splitting
config['num_epochs'] = 1    # just 1 epoch to test the pipeline
config['device'] = 'cpu'

print(f"Model: {model.modelfile}")
print(f"Num vars: {len(model.X)}")
print(f"Config: ecl=2^30, iB=30, num_epochs=1, device=cpu")
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
# Find large buckets
# CRITICAL: get_large_message_buckets returns INTEGER labels (var.label values),
# NOT Var objects. We must convert with matching_var() before passing to eliminate_variables.
print("Finding large buckets (iB=15)...")
large_labels = fastgm.get_large_message_buckets(iB=15, debug=True)
print(f"\nFound {len(large_labels)} large buckets with iB=15")

# If no results with iB=15, try iB=10
if len(large_labels) == 0:
    print("No large buckets found with iB=15, trying iB=10...")
    large_labels = fastgm.get_large_message_buckets(iB=10, debug=True)
    print(f"Found {len(large_labels)} large buckets with iB=10")

if len(large_labels) == 0:
    print("ERROR: No large buckets found. Cannot proceed with single-bucket test.")
    print("Try a different model or lower iB threshold.")
    raise SystemExit(1)

# Pick the first large bucket
target_label = large_labels[0]  # This is an INTEGER
print(f"\nTarget bucket label (integer): {target_label}")
print(f"Type: {type(target_label)}")
print()

# %%
# Show elimination structure
print("Elimination structure:")
fastgm.show_elimination(all=True)
print()

# %%
# Eliminate all variables up to the target bucket
# CRITICAL: eliminate_variables(up_to=...) expects a Var object, not an int.
# Use matching_var() to convert the integer label to a Var.
target_var = fastgm.matching_var(target_label)
print(f"Target var (converted from int {target_label}): {target_var}")
print(f"Target var type: {type(target_var)}")
print()

print(f"Eliminating variables up to bucket {target_label}...")
t0 = time.time()
fastgm.eliminate_variables(up_to=target_var)
t_elim = time.time() - t0
print(f"Elimination took {t_elim:.2f}s")
print()

# %%
# Get the target bucket and print info
# get_bucket() accepts either int or Var
bucket = fastgm.get_bucket(target_label)
print(f"Bucket label: {bucket.label}")
print(f"Message scope: {bucket.get_message_scope()}")
print(f"Message dimension: {bucket.get_message_dimension()}")
print(f"Message size: {bucket.get_message_size()}")
print(f"Num factors: {len(bucket.factors)}")
print()

# %%
# Train NN on this single bucket
print(f"Training NN on bucket {target_label} (1 epoch)...")
try:
    t0 = time.time()
    message = bucket.compute_message_nn()
    t_train = time.time() - t0
    print(f"Training completed in {t_train:.2f}s")
    print(f"Message tensor shape: {message.tensor.shape}")
    print(f"Message labels: {message.labels}")
    print()
    print("Phase 0b complete: single bucket NN training succeeded.")
except Exception as e:
    print(f"ERROR during NN training: {e}")
    print()
    print("Troubleshooting tips:")
    print("  - Check that weighted_logspace_mse is registered in losses.py")
    print("  - Check that nbe num_samples string resolves correctly")
    print("  - Check that nbe hidden_sizes string resolves correctly")
    print("  - Try running on cuda if CPU is too slow")
    raise
