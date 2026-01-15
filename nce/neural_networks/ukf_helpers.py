"""
UKF (Unscented Kalman Filter) helper functions for sequential Gaussian loss approximation.

Based on the GaussUKF Testing.ipynb notebook approach.

## Performance Characteristics

The UKF sequential loss function processes training batches sequentially through ukf_accumulate(),
which cannot be parallelized due to the iterative nature of the UKF update equations.

**Time complexity**: O(batch_size) where each iteration takes ~1-1.5ms on GPU

**Recommended batch sizes**:
- Small messages (<1k states): batch_size = 100-500
- Medium messages (1k-10k states): batch_size = 100-1000
- Large messages (10k-100k states): batch_size = 100-500
- Very large messages (>100k states): Use different loss function (elp, logspace_mse_fdb)

**Example timings** (per training batch):
- batch_size = 100:   ~150ms
- batch_size = 1000:  ~1.5s
- batch_size = 10000: ~15s (slow but usable)

For a 2M state message with batch_size=1000:
- 2000 batches per epoch
- ~1.5s per batch
- ~50 minutes per epoch

This is by design - the UKF sequential loss provides theoretically optimal gradients
at the cost of computational efficiency. For large problems, consider:
1. Using smaller batch sizes (100-500)
2. Using elp_least_squares loss instead (5-10x faster)
3. Using logspace_mse_fdb loss instead (10-20x faster)
"""

import torch
import math
import time

def lse1(X):
    """Helper function to perform log-sum-exp across a single array"""
    return torch.log(torch.sum(torch.exp(X - torch.max(X)))) + torch.max(X)


def lse2(a, b):
    """Helper function to perform log-sum-exp on a pair of arrays"""
    return torch.logsumexp(torch.stack((a, b), dim=0), dim=0)


def estimate_gaussian(f, fhat, mb, sb, n_samp=500, batch_size=None, memory_fraction=0.5):
    """
    Estimate the Gaussian distribution of [lse(f+b), lse(fhat+b)]
    where b ~ N(mb, sb²)

    Args:
        f: Forward message values (torch tensor), shape: (message_size,)
        fhat: Approximate forward message values (torch tensor), shape: (message_size,)
        mb: Mean of backward message, shape: (message_size,)
        sb: Standard deviation of backward message (scalar or tensor)
        n_samp: Number of samples to use for estimation (default: 500, uses antithetic sampling for 1000 effective)
        batch_size: Number of backward message samples to process at once (default: auto-detect based on available GPU memory)
        memory_fraction: Fraction of available GPU memory to use for batching (default: 0.5)

    Returns:
        mu: Mean vector [E[lse(f+b)], E[lse(fhat+b)]]
        Sig: 2x2 covariance matrix
    """
    # Ensure f is 1D
    f = f.flatten()
    fhat = fhat.flatten()
    mb = mb.flatten() if hasattr(mb, 'flatten') else mb

    message_size = f.shape[0]

    # Auto-detect batch size based on available GPU memory
    if batch_size is None:
        if f.device.type == 'cuda':
            # Get available GPU memory
            torch.cuda.empty_cache()
            available_memory = torch.cuda.mem_get_info(f.device)[0]  # bytes available

            # Estimate memory needed per backward message sample (with antithetic pair)
            bytes_per_float = 4  # float32
            memory_per_sample_pair = 4 * message_size * bytes_per_float

            # Use specified fraction of available memory
            usable_memory = available_memory * memory_fraction

            # Calculate how many sample pairs we can fit
            max_sample_pairs = max(1, int(usable_memory / memory_per_sample_pair))

            # batch_size is the number of base samples (before antithetic pairing)
            batch_size = min(max_sample_pairs, n_samp)
        else:
            # CPU: use reasonable default
            batch_size = min(100, n_samp)

    # Collect all phi and phi_hat values by processing backward samples in batches
    phi_all = []
    phi_hat_all = []

    # Process backward message samples in batches
    num_batches = (n_samp + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samp)
        batch_n_samp = end_idx - start_idx

        # Sample this batch of backward messages
        # Shape: (batch_n_samp, message_size)
        b_ = torch.randn(batch_n_samp, message_size, device=f.device, dtype=f.dtype) * sb + mb

        # Create antithetic pairs: [b_1, -b_1, b_2, -b_2, ...]
        # Shape: (2*batch_n_samp, message_size)
        b_samples = torch.stack([b_, -b_], dim=1).reshape(-1, message_size)

        # Compute log-sum-exp with exact forward message
        phi_batch = torch.logsumexp(f.unsqueeze(0) + b_samples, dim=1)

        # Compute log-sum-exp with approximate forward message
        phi_hat_batch = torch.logsumexp(fhat.unsqueeze(0) + b_samples, dim=1)

        # Store results
        phi_all.append(phi_batch)
        phi_hat_all.append(phi_hat_batch)

        # Free memory
        del b_, b_samples, phi_batch, phi_hat_batch
        if f.device.type == 'cuda':
            torch.cuda.empty_cache()

    # Concatenate all batches
    phi = torch.cat(phi_all, dim=0)  # Shape: (2*n_samp,)
    phi_hat = torch.cat(phi_hat_all, dim=0)  # Shape: (2*n_samp,)

    # Stack into (2, 2*n_samp) for covariance computation
    x = torch.stack([phi, phi_hat])  # Shape: (2, 2*n_samp)

    # Compute mean and covariance
    mu, Sig = torch.mean(x, 1), torch.cov(x)

    return mu, Sig


def ukf_accumulate(f, fhat, mb, sb, alpha=1., beta=0., k=0.):
    """
    UKF-based approximation of the Gaussian defined by [lse(f+b), lse(fhat+b)] with b ~ N(mb, sb²)

    Accumulates UKF estimates sequentially for each state in the batch.

    PERFORMANCE NOTE: This function processes states sequentially (cannot be parallelized).
    Time complexity: O(batch_size) where each iteration takes ~1-1.5ms.

    For large messages (>10k states), use small batch sizes (100-1000) to keep
    training time reasonable. The sequential nature means:
    - 100 states: ~150ms per batch (reasonable)
    - 1000 states: ~1.5s per batch (acceptable)
    - 10000 states: ~15s per batch (slow but usable)
    - 100000+ states: impractical, use different loss function

    Args:
        f: Input message values (torch vector), shape: (batch_size,)
        fhat: Approximate message values (torch vector), shape: (batch_size,)
        mb: Backward message distribution's mean, shape: (batch_size,)
        sb: Backward message distribution's standard deviation (scalar or shape: (batch_size,))
        alpha, beta, k: Parameters of the UKF approximation (default values work well)

    Returns:
        mu: Mean vector [E[Φ], E[Φ̂]], shape: (1, 2)
        Sig: 2x2 covariance matrix
    """
    # ============================================================================
    # PERFORMANCE FLAG: Set to True to run on CPU (may be faster for small ops)
    # ============================================================================
    FORCE_CPU = True  # Set to True to run this function on CPU
    # ============================================================================

    # Store original device and move to CPU if requested
    original_device = f.device
    if FORCE_CPU and f.device.type == 'cuda':
        f = f.cpu()
        fhat = fhat.cpu()
        mb = mb.cpu()
        if isinstance(sb, torch.Tensor):
            sb = sb.cpu()

    n = 3  # 2 dimensional Gaussians + 1D for noise
    lam = alpha**2 * (n + k) - n
    Wm0, Wc0 = lam / (n + lam), lam / (n + lam) + (1 - alpha**2 + beta)
    Wmi, Wci = 1. / (2 * n + 2 * lam), 1. / (2 * n + 2 * lam)
    delta = math.sqrt(n + lam)

    mu = torch.zeros(1, 2, device=f.device) - 1000.
    Sig = torch.zeros(2, 2, device=f.device)

    # Profiling accumulators
    time_cholesky = 0.0
    time_ffh_prep = 0.0
    time_xmu = 0.0
    time_sigma_points = 0.0
    time_mu_update = 0.0
    time_sig_update = 0.0

    tic = time.time()
    for j in range(len(f)):
        # 1. Cholesky decomposition
        t0 = time.time()
        Chol, info = torch.linalg.cholesky_ex(Sig)
        if info:
            # If not full rank, do manually
            Chol = torch.tensor([[1., 1.], [0, 0.]], device=f.device).T * torch.sqrt(Sig)
        time_cholesky += time.time() - t0

        # 2. Prepare ffh
        t0 = time.time()
        ffh = torch.stack((f[j], fhat[j]))[None, :] + mb[j]
        time_ffh_prep += time.time() - t0

        # 3. Compute Xmu
        t0 = time.time()
        Xmu = lse2(mu, ffh)
        time_xmu += time.time() - t0

        # 4. Generate sigma points
        t0 = time.time()
        # Precompute all perturbations (more efficient than generator expression)
        # Points from covariance structure (4 points for n-1=2)
        sigma_list = []
        for i in range(n - 1):
            sigma_list.append(lse2(mu + delta * Chol[:, i], ffh))
            sigma_list.append(lse2(mu - delta * Chol[:, i], ffh))
        # Points from backward message uncertainty (2 points)
        sigma_list.append(lse2(mu, ffh + delta * sb))
        sigma_list.append(lse2(mu, ffh - delta * sb))
        t_vstack = time.time()
        Xsigmas = torch.vstack(sigma_list)
        time_sigma_points += time.time() - t0

        # Track vstack overhead separately
        if j == 0:
            time_vstack_total = 0.0
        time_vstack_total += time.time() - t_vstack

        # 5. Compute mean update
        t0 = time.time()
        mu = Wm0 * Xmu + Wmi * Xsigmas.sum(0, keepdim=True)
        time_mu_update += time.time() - t0

        # 6. Compute covariance update
        t0 = time.time()
        Sig = (Xmu - mu).T * Wc0 * (Xmu - mu) + (Xsigmas - mu).T @ (Wci * (Xsigmas - mu))
        time_sig_update += time.time() - t0

    total_time = time.time() - tic

    # Print profiling breakdown
    device_str = "CPU" if FORCE_CPU else f.device.type.upper()
    print(f"\n{'='*70}")
    print(f"UKF_ACCUMULATE PROFILING - {len(f)} iterations on {device_str}")
    print(f"{'='*70}")
    print(f"{'Operation':<30} {'Total (s)':<12} {'Avg (ms)':<12} {'% of Total':<12}")
    print(f"{'-'*70}")
    print(f"{'Cholesky decomposition':<30} {time_cholesky:<12.4f} {time_cholesky/len(f)*1000:<12.4f} {time_cholesky/total_time*100:<12.1f}")
    print(f"{'FFH preparation':<30} {time_ffh_prep:<12.4f} {time_ffh_prep/len(f)*1000:<12.4f} {time_ffh_prep/total_time*100:<12.1f}")
    print(f"{'Xmu computation (1 lse2)':<30} {time_xmu:<12.4f} {time_xmu/len(f)*1000:<12.4f} {time_xmu/total_time*100:<12.1f}")
    print(f"{'Sigma points (6 lse2)':<30} {time_sigma_points:<12.4f} {time_sigma_points/len(f)*1000:<12.4f} {time_sigma_points/total_time*100:<12.1f}")
    print(f"{'  └─ vstack overhead':<30} {time_vstack_total:<12.4f} {time_vstack_total/len(f)*1000:<12.4f} {time_vstack_total/total_time*100:<12.1f}")
    print(f"{'Mu update':<30} {time_mu_update:<12.4f} {time_mu_update/len(f)*1000:<12.4f} {time_mu_update/total_time*100:<12.1f}")
    print(f"{'Sig update':<30} {time_sig_update:<12.4f} {time_sig_update/len(f)*1000:<12.4f} {time_sig_update/total_time*100:<12.1f}")
    print(f"{'-'*70}")
    print(f"{'TOTAL':<30} {total_time:<12.4f} {total_time/len(f)*1000:<12.4f} {'100.0':<12}")
    print(f"{'='*70}\n")

    # Move results back to original device if we forced CPU
    if FORCE_CPU and original_device.type == 'cuda':
        mu = mu.to(original_device)
        Sig = Sig.to(original_device)

    return mu, Sig


def ukf_combine(muA, sigA, muB, sigB, alpha=1., beta=0., k=0.):
    """
    LSE-combine two quantities described by Gaussians N(muA, sigA) and N(muB, sigB)

    Args:
        muA, muB: Mean vectors (shape: (1, 2) or (2,))
        sigA, sigB: 2x2 covariance matrices
        alpha, beta, k: UKF parameters

    Returns:
        mu: Combined mean vector (shape: (2,))
        Sig: Combined 2x2 covariance matrix
    """
    n = 2  # 2 dimensional Gaussians
    lam = alpha**2 * (n + k) - n
    Wm0, Wc0 = lam / (n + lam), lam / (n + lam) + (1 - alpha**2 + beta)
    Wmi, Wci = 1. / (2 * n + 2 * lam), 1. / (2 * n + 2 * lam)
    delta = math.sqrt(n + lam)

    # Ensure muA and muB are the right shape
    if muA.dim() == 2:
        muA = muA.squeeze(0)
    if muB.dim() == 2:
        muB = muB.squeeze(0)

    CholA, info = torch.linalg.cholesky_ex(sigA)
    if info:
        CholA = torch.tensor([[1., 1.], [0, 0.]], device=muA.device).T * torch.sqrt(sigA)
    CholB, info = torch.linalg.cholesky_ex(sigB)
    if info:
        CholB = torch.tensor([[1., 1.], [0, 0.]], device=muB.device).T * torch.sqrt(sigB)

    Wm = torch.tensor([Wm0] + [Wmi] * 2 * n, device=muA.device)
    Wc = torch.tensor([Wc0] + [Wci] * 2 * n, device=muA.device)

    ptsA = torch.vstack(
        (muA,) + sum(((muA + delta * CholA[:, i], muA - delta * CholA[:, i])
                      for i in range(n)), tuple())
    )
    ptsB = torch.vstack(
        (muB,) + sum(((muB + delta * CholB[:, i], muB - delta * CholB[:, i])
                      for i in range(n)), tuple())
    )

    ptsX = lse2(ptsA.T[:, :, None].repeat(1, 1, 2 * n + 1),
                ptsB.T[:, None, :].repeat(1, 2 * n + 1, 1))

    mu = ((Wm[None, :, None] * Wm[None, None, :]) * ptsX).sum((1, 2))
    Sig = ((Wc[None, None, :, None] * (ptsX[:, None, :, :] - mu[:, None, None, None])) *
           (Wc[None, None, None, :] * (ptsX[None, :, :, :] - mu[None, :, None, None]))).sum((2, 3))

    return mu, Sig


def loss_seq(f, fhat, mb, sb, Lmu, Lsig, alpha=1., beta=0., k=0.):
    """
    UKF-based approximation of the expected L2 log-partition loss

    Args:
        f: Target message values (batch), shape: (batch_size,)
        fhat: Predicted message values (batch), shape: (batch_size,)
        mb: Backward message mean (batch), shape: (batch_size,)
        sb: Backward message std dev (scalar or batch)
        Lmu: Pre-computed mean of p(Φ, Φ̂) from estimate_gaussian, shape: (2,) or (1, 2)
        Lsig: Pre-computed covariance of p(Φ, Φ̂), shape: (2, 2)
        alpha, beta, k: UKF parameters

    Returns:
        loss: Scalar loss value
    """
    # Estimate p(lse(f), lse(fhat)) for this batch
    muF, SigF = ukf_accumulate(f, fhat, mb, sb, alpha=alpha, beta=beta, k=k)

    # Combine with pre-computed p(Φ, Φ̂)
    if Lsig.dim() == 3:
        Lsig = Lsig.squeeze(0)
    mu, Sig = ukf_combine(Lmu, Lsig, muF, SigF, alpha=alpha, beta=beta, k=k)

    # Closed form Var(X0 - X1) from joint Gaussian
    loss = (mu[0] - mu[1])**2 + (Sig[0, 0] + Sig[1, 1] - 2 * Sig[0, 1])

    return loss
