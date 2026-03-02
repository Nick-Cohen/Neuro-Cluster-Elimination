"""
Quantization-based message approximation.

Implements optimal K-segment quantization of messages using dynamic programming
with divide-and-conquer optimization for O(K N log N) complexity.

For UKL loss:
    val = logsumexp(f[m:n] + b[m:n]) - logsumexp(b[m:n])

For MSE loss:
    val = mean(f[m:n])
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from ..inference.factor import FastFactor


class QuantizationSolver:
    """
    Optimal K-segment quantization solver using DP with D&C optimization.

    Given a sorted message f and backward message b (sorted by f's values),
    finds optimal partition into K contiguous segments minimizing total loss.
    """

    def __init__(
        self,
        f_sorted: torch.Tensor,
        b_sorted: torch.Tensor,
        num_states: int,
        loss_type: str = 'ukl',
        device: str = 'cuda'
    ):
        """
        Initialize the quantization solver.

        Args:
            f_sorted: Sorted exact message values (log space), shape (N,)
            b_sorted: Backward message sorted by f's ordering (log space), shape (N,)
            num_states: Number of quantization levels (K)
            loss_type: 'ukl' or 'mse'
            device: Device for computation
        """
        self.f = f_sorted.to(device)
        self.b = b_sorted.to(device)
        self.K = num_states
        self.N = len(f_sorted)
        self.loss_type = loss_type
        self.device = device

        # Precompute arrays for O(1) interval cost
        self._precompute_prefix_sums()

    def _precompute_prefix_sums(self):
        """Precompute prefix sums for O(1) interval cost evaluation."""

        if self.loss_type == 'ukl':
            # For UKL: need u = f + b, t = b
            u = self.f + self.b
            t = self.b

            # Global max for numerical stability
            self.su = u.max().item()
            self.st = t.max().item()

            # Scaled weights
            wu = torch.exp(u - self.su)
            wt = torch.exp(t - self.st)

            # Prefix sums (0-indexed, W[i] = sum of wu[0:i])
            # We prepend 0 for easier interval computation
            self.W = torch.zeros(self.N + 1, device=self.device)
            self.P = torch.zeros(self.N + 1, device=self.device)
            self.V = torch.zeros(self.N + 1, device=self.device)

            self.W[1:] = torch.cumsum(wu, dim=0)
            self.P[1:] = torch.cumsum(wu * self.f, dim=0)
            self.V[1:] = torch.cumsum(wt, dim=0)

        else:  # MSE
            # For MSE: need prefix sums of f and f^2
            self.S = torch.zeros(self.N + 1, device=self.device)  # sum of f
            self.S2 = torch.zeros(self.N + 1, device=self.device)  # sum of f^2

            self.S[1:] = torch.cumsum(self.f, dim=0)
            self.S2[1:] = torch.cumsum(self.f ** 2, dim=0)

    def _cost_ukl(self, l: int, r: int) -> float:
        """
        Compute UKL cost for interval [l, r] (0-indexed, inclusive).

        Cost = P - W * (su + log(W)) + W * (st + log(V))

        where W, P, V are computed from prefix sums over [l, r].
        """
        # Convert to 1-indexed for prefix sum access
        l1 = l + 1
        r1 = r + 1

        Wv = self.W[r1] - self.W[l1 - 1]
        Pv = self.P[r1] - self.P[l1 - 1]
        Vv = self.V[r1] - self.V[l1 - 1]

        # Handle edge cases
        if Wv <= 0 or Vv <= 0:
            return float('inf')

        cost = Pv - Wv * (self.su + torch.log(Wv)) + Wv * (self.st + torch.log(Vv))
        return cost.item()

    def _cost_mse(self, l: int, r: int) -> float:
        """
        Compute MSE cost for interval [l, r] (0-indexed, inclusive).

        For MSE with constant val = mean(f[l:r+1]):
        Cost = sum((f[i] - val)^2) = sum(f^2) - n * val^2
             = S2[r+1] - S2[l] - (S[r+1] - S[l])^2 / n
        """
        l1 = l + 1
        r1 = r + 1

        n = r - l + 1
        if n <= 0:
            return float('inf')

        sum_f = self.S[r1] - self.S[l1 - 1]
        sum_f2 = self.S2[r1] - self.S2[l1 - 1]

        cost = sum_f2 - (sum_f ** 2) / n
        return cost.item()

    def _cost(self, l: int, r: int) -> float:
        """Compute cost for interval [l, r] based on loss type."""
        if self.loss_type == 'ukl':
            return self._cost_ukl(l, r)
        else:
            return self._cost_mse(l, r)

    def _compute_value_ukl(self, l: int, r: int) -> float:
        """
        Compute optimal quantization value for interval [l, r] under UKL.

        val = log(Z) - log(B) = log(sum(exp(u))) - log(sum(exp(t)))
            = (su + log(W)) - (st + log(V))
        """
        l1 = l + 1
        r1 = r + 1

        Wv = self.W[r1] - self.W[l1 - 1]
        Vv = self.V[r1] - self.V[l1 - 1]

        if Wv <= 0 or Vv <= 0:
            # Fallback to mean of f
            return self.f[l:r+1].mean().item()

        val = (self.su + torch.log(Wv)) - (self.st + torch.log(Vv))
        return val.item()

    def _compute_value_mse(self, l: int, r: int) -> float:
        """Compute optimal quantization value for interval [l, r] under MSE."""
        l1 = l + 1
        r1 = r + 1

        n = r - l + 1
        if n <= 0:
            return 0.0

        sum_f = self.S[r1] - self.S[l1 - 1]
        return (sum_f / n).item()

    def _compute_value(self, l: int, r: int) -> float:
        """Compute optimal quantization value for interval based on loss type."""
        if self.loss_type == 'ukl':
            return self._compute_value_ukl(l, r)
        else:
            return self._compute_value_mse(l, r)

    def solve(self) -> Tuple[List[int], List[float], float]:
        """
        Find optimal K-segment partition using DP with D&C optimization.

        Returns:
            boundaries: List of K+1 boundary indices [0, b1, b2, ..., N]
            values: List of K quantization values for each segment
            total_cost: Total loss of the optimal partition
        """
        N = self.N
        K = self.K

        # Handle edge cases
        if K >= N:
            # Each element gets its own segment
            boundaries = list(range(N + 1))
            values = [self.f[i].item() for i in range(N)]
            return boundaries, values, 0.0

        if K == 1:
            # Single segment
            boundaries = [0, N]
            values = [self._compute_value(0, N - 1)]
            total_cost = self._cost(0, N - 1)
            return boundaries, values, total_cost

        # DP with divide and conquer
        # dp[j] = min cost for covering prefix [0, j) with current number of buckets
        # prev[j] = dp values from previous k

        INF = float('inf')

        # Base case: k=1 (one bucket covering [0, j))
        prev = np.full(N + 1, INF)
        prev[0] = 0.0
        for j in range(1, N + 1):
            prev[j] = self._cost(0, j - 1)

        # Store optimal split points for backtracking
        # opt_history[k][j] = optimal split point for dp[k][j]
        opt_history = []

        # DP for k = 2 to K
        for k in range(2, K + 1):
            cur = np.full(N + 1, INF)
            opt = np.full(N + 1, -1, dtype=int)

            # Use divide and conquer optimization
            self._solve_dc(k, 0, N, 0, N - 1, prev, cur, opt)

            opt_history.append(opt.copy())
            prev = cur.copy()

        # Backtrack to find boundaries
        boundaries = [N]
        j = N
        for k in range(K - 1, 0, -1):
            opt_k = opt_history[k - 1]
            i = opt_k[j]
            boundaries.append(i)
            j = i
        boundaries.append(0)
        boundaries.reverse()

        # Compute values for each segment
        values = []
        for seg_idx in range(K):
            l = boundaries[seg_idx]
            r = boundaries[seg_idx + 1] - 1
            values.append(self._compute_value(l, r))

        total_cost = prev[N]

        return boundaries, values, total_cost

    def _solve_dc(
        self,
        k: int,
        lo: int,
        hi: int,
        opt_lo: int,
        opt_hi: int,
        prev: np.ndarray,
        cur: np.ndarray,
        opt: np.ndarray
    ):
        """
        Divide and conquer DP solver.

        Solves dp[k][lo:hi+1] given that optimal split points are in [opt_lo, opt_hi].

        The monotonicity property ensures opt[j] <= opt[j+1], enabling D&C.
        """
        if lo > hi:
            return

        mid = (lo + hi) // 2

        # Find best split point for dp[k][mid]
        best_cost = float('inf')
        best_i = opt_lo

        # Search range: [opt_lo, min(mid - 1, opt_hi)]
        # We need at least one element in the last segment, so i < mid
        search_hi = min(mid - 1, opt_hi)

        for i in range(opt_lo, search_hi + 1):
            if i < k - 1:
                # Need at least k-1 elements for k-1 buckets
                continue

            cost = prev[i] + self._cost(i, mid - 1)
            if cost < best_cost:
                best_cost = cost
                best_i = i

        cur[mid] = best_cost
        opt[mid] = best_i

        # Recurse
        self._solve_dc(k, lo, mid - 1, opt_lo, best_i, prev, cur, opt)
        self._solve_dc(k, mid + 1, hi, best_i, opt_hi, prev, cur, opt)


def quantize_message(
    exact_message: FastFactor,
    backward_message: Optional[FastFactor],
    num_states: int,
    loss_type: str = 'ukl',
    device: str = 'cuda'
) -> Tuple[FastFactor, dict]:
    """
    Quantize a message into K levels optimally under the given loss.

    Args:
        exact_message: The exact message to quantize (FastFactor)
        backward_message: The backward message for UKL loss (FastFactor or None)
        num_states: Number of quantization levels K
        loss_type: 'ukl' or 'mse'
        device: Device for computation

    Returns:
        quantized_message: FastFactor with quantized values
        info: Dict with boundaries, values, total_cost, sorted_indices
    """
    # Flatten tensors
    f = exact_message.tensor.flatten().to(device)
    N = len(f)

    # Get backward message (or zeros if not provided)
    if backward_message is not None:
        b = backward_message.tensor.flatten().to(device)
        if len(b) != N:
            raise ValueError(f"Backward message size {len(b)} != exact message size {N}")
    else:
        b = torch.zeros(N, device=device)

    # Sort by f values
    sorted_indices = torch.argsort(f)
    f_sorted = f[sorted_indices]
    b_sorted = b[sorted_indices]

    # Solve quantization
    solver = QuantizationSolver(f_sorted, b_sorted, num_states, loss_type, device)
    boundaries, values, total_cost = solver.solve()

    # Create quantized message (unsort back to original order)
    quantized_sorted = torch.zeros(N, device=device)
    for seg_idx in range(num_states):
        l = boundaries[seg_idx]
        r = boundaries[seg_idx + 1]
        quantized_sorted[l:r] = values[seg_idx]

    # Unsort
    inverse_indices = torch.argsort(sorted_indices)
    quantized_flat = quantized_sorted[inverse_indices]

    # Reshape to original shape
    quantized_tensor = quantized_flat.reshape(exact_message.tensor.shape)

    # Create FastFactor
    quantized_message = FastFactor(quantized_tensor, exact_message.labels.copy())

    info = {
        'boundaries': boundaries,
        'values': values,
        'total_cost': total_cost,
        'sorted_indices': sorted_indices.cpu().numpy(),
        'num_states': num_states,
        'loss_type': loss_type,
        'N': N,
    }

    return quantized_message, info


def verify_monotonicity(
    f_sorted: torch.Tensor,
    b_sorted: torch.Tensor,
    loss_type: str = 'ukl',
    num_samples: int = 1000,
    device: str = 'cuda'
) -> Tuple[bool, List[int]]:
    """
    Verify that optimal split points are monotonic (required for D&C optimization).

    Tests on a subset of the data to check if opt[j] <= opt[j+1].

    Args:
        f_sorted: Sorted message values
        b_sorted: Backward message (same sort order)
        loss_type: 'ukl' or 'mse'
        num_samples: Number of samples to test
        device: Device for computation

    Returns:
        is_monotonic: True if monotonicity holds
        opt: Optimal split points for k=2
    """
    # Subsample if needed
    N = len(f_sorted)
    if N > num_samples:
        step = N // num_samples
        indices = torch.arange(0, N, step, device=device)[:num_samples]
        f_sub = f_sorted[indices]
        b_sub = b_sorted[indices]
    else:
        f_sub = f_sorted
        b_sub = b_sorted

    N_sub = len(f_sub)

    # Create solver
    solver = QuantizationSolver(f_sub, b_sub, 2, loss_type, device)

    # Compute dp for k=1
    prev = np.full(N_sub + 1, float('inf'))
    prev[0] = 0.0
    for j in range(1, N_sub + 1):
        prev[j] = solver._cost(0, j - 1)

    # Compute optimal split points for k=2
    opt = []
    for j in range(2, N_sub + 1):
        best_i = 0
        best_cost = float('inf')
        for i in range(1, j):
            cost = prev[i] + solver._cost(i, j - 1)
            if cost < best_cost:
                best_cost = cost
                best_i = i
        opt.append(best_i)

    # Check monotonicity
    is_monotonic = all(opt[i] <= opt[i + 1] for i in range(len(opt) - 1))

    return is_monotonic, opt
