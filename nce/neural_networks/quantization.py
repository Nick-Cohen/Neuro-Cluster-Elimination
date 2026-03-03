"""
Quantization-based message approximation.

Implements K-segment quantization of messages using recursive binary splitting.
The algorithm works by recursively splitting the sorted message into halves,
each half subdivided with its allocated quanta. This is a greedy heuristic
(not globally optimal like DP), but simpler and faster.

For UKL loss (optimal segment value):
    val = logsumexp(f[l:r] + b[l:r]) - logsumexp(b[l:r])

For MSE loss (optimal segment value):
    val = mean(f[l:r])
"""

import torch
from typing import List, Tuple, Optional
from ..inference.factor import FastFactor


class QuantizationSolver:
    """
    K-segment quantization solver using recursive binary splitting.

    Given a sorted message f and backward message b (sorted by f's values),
    finds a partition into K contiguous segments minimizing total loss
    via a greedy recursive binary split heuristic.
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

    def segment_value(self, l: int, r: int) -> float:
        """
        Compute the optimal constant value for segment f[l:r] (exclusive r).

        For UKL: q = logsumexp(f[l:r] + b[l:r]) - logsumexp(b[l:r])
        For MSE: q = mean(f[l:r])

        Args:
            l: Start index (inclusive)
            r: End index (exclusive)

        Returns:
            Optimal constant value for this segment
        """
        if r <= l:
            return 0.0

        f_seg = self.f[l:r]
        b_seg = self.b[l:r]

        if self.loss_type == 'ukl':
            # q = logsumexp(f + b) - logsumexp(b)
            joint = f_seg + b_seg
            log_numerator = torch.logsumexp(joint, dim=0)
            log_denominator = torch.logsumexp(b_seg, dim=0)
            return (log_numerator - log_denominator).item()
        else:
            # MSE: mean of f
            return f_seg.mean().item()

    def segment_ukl_loss(self, l: int, r: int, q: float) -> float:
        """
        Compute the UKL loss for assigning constant value q to segment [l, r).

        Uses the unnormalized KL divergence formula:
            targets = f[l:r] + b[l:r]   (exact joint in log space)
            outputs = q + b[l:r]          (quantized joint in log space)
            max_val = max(targets.max(), outputs.max())
            p_tilde = exp(targets - max_val)
            q_tilde = exp(outputs - max_val)
            loss = sum(p_tilde * (log_p_tilde - log_q_tilde) - p_tilde + q_tilde)

        For MSE: loss = sum((f[l:r] - q)^2)

        Args:
            l: Start index (inclusive)
            r: End index (exclusive)
            q: Constant value assigned to this segment

        Returns:
            Loss value for this segment
        """
        if r <= l:
            return 0.0

        f_seg = self.f[l:r]
        b_seg = self.b[l:r]

        if self.loss_type == 'ukl':
            targets = f_seg + b_seg
            outputs = q + b_seg
            max_val = max(targets.max().item(), outputs.max().item())
            p_tilde = torch.exp(targets - max_val)
            q_tilde = torch.exp(outputs - max_val)
            # Clamp to avoid log(0)
            log_p = targets - max_val  # = log(p_tilde), no clamp needed as it's just subtraction
            log_q = outputs - max_val  # = log(q_tilde), same
            loss = (p_tilde * (log_p - log_q) - p_tilde + q_tilde).sum()
            return loss.item()
        else:
            # MSE
            return ((f_seg - q) ** 2).sum().item()

    def _split(self, l: int, r: int, k: int) -> Tuple[List[int], List[float]]:
        """
        Recursive binary splitting for segment [l, r) with k quanta.

        Args:
            l: Start index (inclusive)
            r: End index (exclusive)
            k: Number of quanta to use for this segment

        Returns:
            boundaries: List of boundary indices (length k+1, starts with l, ends with r)
            values: List of k quantization values
        """
        # Guard: if segment has fewer elements than quanta, assign one per element
        n = r - l
        if n <= 0:
            return [l], []
        if k >= n:
            # Each element gets its own value
            boundaries = list(range(l, r + 1))
            values = [self.f[i].item() for i in range(l, r)]
            return boundaries, values

        # Base case: k=1
        if k == 1:
            q = self.segment_value(l, r)
            return [l, r], [q]

        # k=2: try every split point
        if k == 2:
            best_s = l + 1
            best_loss = float('inf')
            best_q_left = 0.0
            best_q_right = 0.0

            for s in range(l + 1, r):
                q_left = self.segment_value(l, s)
                q_right = self.segment_value(s, r)
                loss_left = self.segment_ukl_loss(l, s, q_left)
                loss_right = self.segment_ukl_loss(s, r, q_right)
                total = loss_left + loss_right

                if total < best_loss:
                    best_loss = total
                    best_s = s
                    best_q_left = q_left
                    best_q_right = q_right

            return [l, best_s, r], [best_q_left, best_q_right]

        # k>2: find best binary split as if k=2, then recurse
        k_left = k // 2
        k_right = k - k_left

        # Find best binary split point
        best_s = l + k_left  # must leave room for k_left and k_right elements
        best_loss = float('inf')

        for s in range(l + k_left, r - k_right + 1):
            q_left = self.segment_value(l, s)
            q_right = self.segment_value(s, r)
            loss_left = self.segment_ukl_loss(l, s, q_left)
            loss_right = self.segment_ukl_loss(s, r, q_right)
            total = loss_left + loss_right

            if total < best_loss:
                best_loss = total
                best_s = s

        # Recursively split each half
        left_boundaries, left_values = self._split(l, best_s, k_left)
        right_boundaries, right_values = self._split(best_s, r, k_right)

        # Merge: left_boundaries ends with best_s, right_boundaries starts with best_s
        # so we drop the duplicate best_s from the boundary join
        merged_boundaries = left_boundaries + right_boundaries[1:]
        merged_values = left_values + right_values

        return merged_boundaries, merged_values

    def solve(self) -> Tuple[List[int], List[float], float]:
        """
        Find K-segment partition using recursive binary splitting.

        Returns:
            boundaries: List of K+1 boundary indices [0, b1, b2, ..., N]
            values: List of K quantization values for each segment
            total_cost: Total loss of the partition
        """
        N = self.N
        K = self.K

        # Handle edge case: K >= N (each element gets its own segment)
        if K >= N:
            boundaries = list(range(N + 1))
            values = [self.f[i].item() for i in range(N)]
            return boundaries, values, 0.0

        # Recursive splitting
        boundaries, values = self._split(0, N, K)

        # Compute total cost
        total_cost = 0.0
        for seg_idx in range(K):
            l = boundaries[seg_idx]
            r = boundaries[seg_idx + 1]
            q = values[seg_idx]
            total_cost += self.segment_ukl_loss(l, r, q)

        return boundaries, values, total_cost


def quantize_message(
    exact_message: FastFactor,
    backward_message: Optional[FastFactor],
    num_states: int,
    loss_type: str = 'ukl',
    device: str = 'cuda'
) -> Tuple[FastFactor, dict]:
    """
    Quantize a message into K levels using recursive binary splitting.

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
