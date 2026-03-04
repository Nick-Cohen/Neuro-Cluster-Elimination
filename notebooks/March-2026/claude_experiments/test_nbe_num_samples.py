"""Test NeuroBE num_samples function on benchmark models.

Loads the benchmark set, finds the bucket with the largest width,
eliminates up to it, then calls the nbe num_samples function on that bucket.

Note: No benchmark model in this set has a bucket with width exactly 10.
The grid10x10.f5.wrap model has max bucket width 4 (optimized elimination order).
The rbm_20 model has buckets with width 20, which is used for the main test.
"""
import sys
sys.path.insert(0, '/home/cohenn1/NCE')

from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM
from nce.inference.bucket import FastBucket

# --- Static function test ---
print("=== Static function test ===")
# Test case: w=20, l=3, eps=0.1
result = FastBucket.compute_nbe_num_samples(20, 3, 0.1)
print(f"w=20, l=3, eps=0.1: {result}")
assert result['total'] == 48997, f"Expected 48997, got {result['total']}"

# Additional test case
result2 = FastBucket.compute_nbe_num_samples(20, 3, 0.35)
print(f"w=20, l=3, eps=0.35: {result2}")
assert result2['total'] == 13999, f"Expected 13999, got {result2['total']}"

# Verify train/val split
assert result['n_train'] == 39197
assert result['n_val'] == 9800
assert result['n_train'] + result['n_val'] == result['total']
print("Static tests passed!\n")

# --- Load rbm_20 (has width-20 buckets) ---
print("=== Loading rbm_20 (model index 3) ===")
model = nbe_sanity_check.problems[3]  # dbn/rbm_20
config = dict(nbe_sanity_check.configs['nbe'][3])
config['ecl'] = 2**30  # exact
config['iB'] = 30      # exact
config['device'] = 'cpu'

fastgm = FastGM(model=model, nn_config=config, device='cpu')

# Show width distribution
print("\n=== Width distribution ===")
width_counts = {}
for var in fastgm.elim_order:
    bucket = fastgm.buckets[var]
    w = len(bucket.get_message_scope())
    width_counts[w] = width_counts.get(w, 0) + 1
for w in sorted(width_counts.keys()):
    print(f"  Width {w}: {width_counts[w]} buckets")

# Find the first bucket with the largest width (should be 20 for rbm_20)
target_var = None
target_w = 0
for var in fastgm.elim_order:
    bucket = fastgm.buckets[var]
    w = len(bucket.get_message_scope())
    if w > target_w:
        target_w = w
        target_var = var

print(f"\nUsing bucket with largest width: var {target_var.label} (width {target_w})")

# Eliminate up to (but not including) the target bucket
print(f"\n=== Eliminating up to bucket {target_var.label} ===")
fastgm.eliminate_variables(up_to=target_var)

# Get the bucket and compute nbe num_samples
bucket = fastgm.buckets[target_var]
w = len(bucket.get_message_scope())
dims = bucket.get_message_dimension()
l = max(dims) if dims else 2
epsilon = 0.1  # dbn type

print(f"\n=== NBE num_samples for bucket {target_var.label} ===")
print(f"  Width (w): {w}")
print(f"  Max domain size (l): {l}")
print(f"  Domain sizes: {dims}")
print(f"  Epsilon: {epsilon}")

# Call instance method
result = bucket.get_nbe_num_samples(epsilon)
print(f"  Result: {result}")
print(f"  Total samples: {result['total']}")
print(f"  Training (80%): {result['n_train']}")
print(f"  Validation (20%): {result['n_val']}")

# Also verify static call matches
result_static = FastBucket.compute_nbe_num_samples(w, l, epsilon)
assert result == result_static, "Instance and static methods should match"
print("  Instance/static consistency: PASSED")

# --- Also test on grid10x10.f5.wrap (max width 4) ---
print("\n=== Loading grid10x10.f5.wrap (model index 4) ===")
model2 = nbe_sanity_check.problems[4]
config2 = dict(nbe_sanity_check.configs['nbe'][4])
config2['ecl'] = 2**30
config2['iB'] = 30
config2['device'] = 'cpu'

fastgm2 = FastGM(model=model2, nn_config=config2, device='cpu')

# Find the max-width bucket
target_var2 = None
target_w2 = 0
for var in fastgm2.elim_order:
    bucket = fastgm2.buckets[var]
    w = len(bucket.get_message_scope())
    if w > target_w2:
        target_w2 = w
        target_var2 = var

print(f"Max width bucket: var {target_var2.label} (width {target_w2})")
fastgm2.eliminate_variables(up_to=target_var2)
bucket2 = fastgm2.buckets[target_var2]
result3 = bucket2.get_nbe_num_samples(0.35)
print(f"  NBE num_samples (eps=0.35): {result3}")

print("\nAll tests passed!")
