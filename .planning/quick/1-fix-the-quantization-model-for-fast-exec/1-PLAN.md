---
phase: quick
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - nce/neural_networks/quantization.py
  - notebooks/March-2026/claude_experiments/test_quantization.py
autonomous: false
requirements: [QUANT-REWRITE]

must_haves:
  truths:
    - "Quantization solver uses recursive binary splitting instead of DP with D&C"
    - "For K=1 base case, value is computed as logsumexp(f[m:n] + b[m:n]) - logsumexp(b[m:n])"
    - "For K=2, every possible split of sorted message is tried, picking the one minimizing total UKL loss"
    - "For K>2, recursion splits each piece (e.g. K=4 splits into 2 halves, each half into 2)"
    - "Quantized message on grid10x10 bucket visually matches exact message shape"
  artifacts:
    - path: "nce/neural_networks/quantization.py"
      provides: "Recursive binary splitting quantization solver"
      contains: "class QuantizationSolver"
    - path: "notebooks/March-2026/claude_experiments/test_quantization.py"
      provides: "Test script producing comparison plot"
  key_links:
    - from: "nce/neural_networks/quantization.py"
      to: "nce/inference/bucket.py"
      via: "quantize_message() called from compute_message_quantization()"
      pattern: "from nce.neural_networks.quantization import quantize_message"
---

<objective>
Rewrite the quantization solver in quantization.py to use a simpler recursive binary splitting
algorithm, replacing the current DP with divide-and-conquer approach. Then validate it works
correctly on a real bucket from grid10x10.f5.wrap.

Purpose: The current DP+D&C solver is complex and may not be correct for UKL. The recursive
binary splitting is simpler, easier to verify, and naturally handles the UKL objective.

Output: Rewritten quantization.py + a test script with saved comparison plot.
</objective>

<execution_context>
@/home/cohenn1/.claude/get-shit-done/workflows/execute-plan.md
@/home/cohenn1/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@nce/neural_networks/quantization.py
@nce/inference/bucket.py (lines 387-472, compute_message_quantization method)
@nce/utils/plots.py (plot_fastfactor_comparison function)
@nce/inference/graphical_model.py (FastGM class, lines 390-420 for quantization dispatch)
@nce/neural_networks/losses.py (unnormalized_kl function, lines 35-105)

<interfaces>
<!-- Key types and contracts the executor needs -->

From nce/inference/factor.py:
```python
class FastFactor:
    tensor: torch.Tensor   # values in log-space
    labels: list           # variable labels
    def __init__(self, tensor, labels): ...
```

From nce/neural_networks/quantization.py (current API to preserve):
```python
def quantize_message(
    exact_message: FastFactor,
    backward_message: Optional[FastFactor],
    num_states: int,
    loss_type: str = 'ukl',
    device: str = 'cuda'
) -> Tuple[FastFactor, dict]:
    """Returns (quantized_message, info_dict)"""
```

From nce/neural_networks/losses.py:
```python
# UKL formula (for reference, the quantization uses a segment-level version):
# unsummed = p_tilde * (log_p_tilde - log_q_tilde) - p_tilde + q_tilde
# where p_tilde = exp(targets - max_val), q_tilde = exp(outputs - max_val)
# result = sum(unsummed)
```

From nce/utils/plots.py:
```python
def plot_fastfactor_comparison(exact_factor, approx_factor, message_gradient=None,
    title="FastFactor Comparison", show=True, sort_indices=True,
    show_loss_curve=True, n_biggest=None, pred_alpha=None):
    """Returns plt object. Note: no built-in savefig - call plt.savefig() before show."""
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Rewrite QuantizationSolver with recursive binary splitting</name>
  <files>nce/neural_networks/quantization.py</files>
  <action>
Rewrite the QuantizationSolver class to use recursive binary splitting instead of DP with D&C.
Keep the same public API (quantize_message function signature and return type unchanged).

**New algorithm for QuantizationSolver:**

1. **Constructor:** Store f_sorted, b_sorted, num_states (K), loss_type, device. No prefix sum
   precomputation needed (compute on-the-fly with logsumexp which is numerically stable).

2. **segment_value(l, r):** Compute the optimal constant value for segment f_sorted[l:r] (exclusive r).
   - For UKL: `q = logsumexp(f[l:r] + b[l:r]) - logsumexp(b[l:r])` using torch.logsumexp
   - For MSE: `q = mean(f[l:r])`

3. **segment_ukl_loss(l, r, q):** Compute the UKL loss for assigning constant value q to segment [l, r).
   This is the unnormalized KL divergence between the exact values f[l:r] and constant q,
   weighted by the backward message b[l:r].

   Use the same UKL formula from losses.py adapted for a segment:
   ```python
   # targets = f[l:r] + b[l:r]  (exact joint)
   # outputs = q + b[l:r]        (quantized joint, q is constant)
   # max_val = max(targets.max(), outputs.max())
   # p_tilde = exp(targets - max_val)
   # q_tilde = exp(outputs - max_val)
   # loss = sum(p_tilde * (log_p_tilde - log_q_tilde) - p_tilde + q_tilde)
   ```

   For MSE: `loss = sum((f[l:r] - q)^2)`

4. **solve() method:** Implement recursive binary splitting.

   `_split(l, r, k)` returns `(boundaries, values)` for segment [l, r) with k quanta:
   - **Base case (k=1):** Compute q = segment_value(l, r). Return ([l, r], [q])
   - **k=2:** Try every split point s in [l+1, r). For each s:
     - Compute q_left = segment_value(l, s), loss_left = segment_ukl_loss(l, s, q_left)
     - Compute q_right = segment_value(s, r), loss_right = segment_ukl_loss(s, r, q_right)
     - Total loss = loss_left + loss_right
     Pick s minimizing total loss. Return ([l, s_best, r], [q_left_best, q_right_best])
   - **k>2:** Split into k_left = k//2 and k_right = k - k_left. Use k=2 logic to find
     the best binary split point. Then recursively call _split(l, s_best, k_left) and
     _split(s_best, r, k_right). Merge the boundary/value lists.

   The top-level solve() calls `_split(0, N, K)` and returns (boundaries, values, total_cost).

5. **Keep quantize_message() function:** Same signature, same return type. Just use the new solver.
   Remove the verify_monotonicity function (no longer relevant for recursive splitting).

**Important implementation notes:**
- Use torch.logsumexp for numerical stability in segment_value and segment_ukl_loss.
- The split search in k=2 is O(N) per level, and recursion depth is O(log K), so total is O(N K log K) in the worst case but typically much faster since segments shrink.
- For k>2, the key insight: first find the best binary split as if k=2, then recursively subdivide each half with its allocated quanta. This is a greedy heuristic (not globally optimal like DP), but much simpler and faster.
- Preserve the info dict structure in the return value of quantize_message.
  </action>
  <verify>
    python -c "
import torch; import sys; sys.path.insert(0, '/home/cohenn1/NCE')
from nce.neural_networks.quantization import QuantizationSolver, quantize_message
from nce.inference.factor import FastFactor
# Test basic K=2 splitting
f = torch.linspace(-5, 5, 100)
b = torch.zeros(100)
solver = QuantizationSolver(f, b, 2, 'ukl', 'cpu')
boundaries, values, cost = solver.solve()
print(f'K=2: boundaries={boundaries}, values={[round(v,2) for v in values]}, cost={cost:.4f}')
assert len(boundaries) == 3, f'Expected 3 boundaries, got {len(boundaries)}'
assert len(values) == 2, f'Expected 2 values, got {len(values)}'
# Test K=4
solver4 = QuantizationSolver(f, b, 4, 'ukl', 'cpu')
b4, v4, c4 = solver4.solve()
print(f'K=4: {len(b4)-1} segments, cost={c4:.4f}')
assert len(b4) == 5
# Test quantize_message wrapper
ff = FastFactor(f.reshape(10,10), [0,1])
bf = FastFactor(b.reshape(10,10), [0,1])
qm, info = quantize_message(ff, bf, 4, 'ukl', 'cpu')
print(f'quantize_message: shape={qm.tensor.shape}, unique_vals={len(torch.unique(qm.tensor))}')
assert qm.tensor.shape == (10,10)
print('All tests passed')
"
  </verify>
  <done>
    QuantizationSolver uses recursive binary splitting. segment_value computes logsumexp formula.
    K=1,2,4 all produce correct number of segments. quantize_message returns correct shape.
  </done>
</task>

<task type="auto">
  <name>Task 2: Test on grid10x10 bucket and save comparison plot</name>
  <files>notebooks/March-2026/claude_experiments/test_quantization.py</files>
  <action>
Create a test script that loads grid10x10.f5.wrap.uai, runs the quantization on a single bucket,
and saves a comparison plot.

```python
"""Test quantization on grid10x10.f5.wrap with K=8."""
import sys
sys.path.insert(0, '/home/cohenn1/NCE')

import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt

from nce.inference.graphical_model import FastGM
from nce.utils.plots import plot_fastfactor_comparison

# Config
config = {
    'approximation_method': 'quantization',
    'quantization_states': 8,
    'loss_fn': 'unnormalized_kl',
    'use_bw_approx': True,
    'bw_ecl': 1024,
    'ecl': 2**15,
    'iB': 10,
    'plot_messages': False,
}

uai_file = '/home/cohenn1/NCE/nce/problems/8-5-benchmarks/grid10x10.f5.wrap.uai'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
gm = FastGM(uai_file=uai_file, device=device, nn_config=config)

# Find a bucket that needs approximation (complexity > ecl)
target_bucket = None
for bucket in gm.buckets:
    complexity = bucket.compute_complexity()
    if complexity > config['ecl']:
        target_bucket = bucket
        print(f"Selected bucket {bucket.label} with complexity {complexity}")
        break

if target_bucket is None:
    print("No bucket exceeds ecl, using first bucket with complexity > 100")
    for bucket in gm.buckets:
        complexity = bucket.compute_complexity()
        if complexity > 100:
            target_bucket = bucket
            print(f"Selected bucket {bucket.label} with complexity {complexity}")
            break

assert target_bucket is not None, "No suitable bucket found"

# Compute exact message for comparison
exact_message = target_bucket.compute_message_exact()
print(f"Exact message shape: {exact_message.tensor.shape}, numel: {exact_message.tensor.numel()}")

# Compute quantized message
quantized_message = target_bucket.compute_message_quantization(
    num_states=config['quantization_states'],
    loss_fn=config['loss_fn']
)
print(f"Quantized message shape: {quantized_message.tensor.shape}")
print(f"Unique values: {len(torch.unique(quantized_message.tensor))}")

# Plot comparison and save
plt_obj = plot_fastfactor_comparison(
    exact_message, quantized_message,
    title=f"Bucket {target_bucket.label}: Quantized (K={config['quantization_states']}) vs Exact",
    show=False,
    show_loss_curve=False
)

output_path = '/home/cohenn1/NCE/notebooks/March-2026/claude_experiments/quantization_grid10x10_K8.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to {output_path}")
plt.close('all')
```

Run this script after creating it. The script should:
1. Load grid10x10.f5.wrap.uai
2. Find a bucket exceeding ecl (complexity > 2^15 = 32768)
3. Compute exact and quantized messages
4. Save comparison plot to notebooks/March-2026/claude_experiments/quantization_grid10x10_K8.png
  </action>
  <verify>
    cd /home/cohenn1/NCE && mkdir -p notebooks/March-2026/claude_experiments && /home/cohenn1/NCE/venv/bin/python notebooks/March-2026/claude_experiments/test_quantization.py
  </verify>
  <done>
    Script runs without error. Plot file exists at notebooks/March-2026/claude_experiments/quantization_grid10x10_K8.png.
    Output shows bucket selection, exact message shape, quantized message shape, and unique value count equal to K=8.
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <what-built>Recursive binary splitting quantization solver and test on grid10x10 bucket with K=8</what-built>
  <how-to-verify>
    1. Open the saved plot at notebooks/March-2026/claude_experiments/quantization_grid10x10_K8.png
    2. Verify the quantized message (orange) follows the general shape of the exact message (blue) with 8 distinct step levels
    3. Check that the step transitions are placed at reasonable points (where the exact message value changes significantly)
    4. Confirm no NaN or obviously wrong values (e.g., all zeros, all same value)
  </how-to-verify>
  <resume-signal>Type "approved" if the plot looks reasonable, or describe issues</resume-signal>
</task>

</tasks>

<verification>
1. quantization.py contains QuantizationSolver with recursive binary splitting (no DP, no D&C)
2. segment_value uses logsumexp formula for UKL
3. quantize_message API unchanged (same signature and return type)
4. Test script runs on grid10x10 and produces comparison plot
5. Plot visually confirms quantization quality
</verification>

<success_criteria>
- QuantizationSolver.solve() uses recursive _split() method, not DP tables
- K=1,2,4,8 all produce correct number of segments
- grid10x10 bucket produces quantized message with exactly K unique values
- Comparison plot saved successfully and shows reasonable approximation
</success_criteria>

<output>
After completion, create `.planning/quick/1-fix-the-quantization-model-for-fast-exec/1-SUMMARY.md`
</output>
