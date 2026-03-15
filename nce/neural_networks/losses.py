import torch
import torch.nn as nn
import math

def neurobe_weighted_mse(outputs, targets, ln_min, ln_max, sum_ln):
    """NeuroBE weighted MSE loss operating on [0,1]-normalized targets.

    Weights are importance sampling weights derived from the normalization stats:
        w = targets * (ln_max - ln_min) / sum_ln
    Loss:
        mean(w * (outputs - targets)^2)

    Args:
        outputs: NN predictions, [0,1]-normalized
        targets: True values, [0,1]-normalized
        ln_min: Minimum value in natural log space (from DataPreprocessor)
        ln_max: Maximum value in natural log space (from DataPreprocessor)
        sum_ln: Sum of (y_ln - ln_min) over training data (from DataPreprocessor)

    Returns:
        Scalar loss tensor
    """
    epsilon = 1e-10
    ln_range = ln_max - ln_min
    safe_sum_ln = sum_ln if abs(sum_ln) > epsilon else epsilon
    w = targets * ln_range / safe_sum_ln
    loss = (w * (outputs - targets) ** 2).mean()
    return loss


def expected_softmax_kl(outputs, targets, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100, seed=None):
    # set random seed
    batch_size = outputs.numel()
    if seed is not None:
        torch.manual_seed(seed)
    if sigma_f > 0:
        mg_samples = (sigma_g * (1-rho**2)**0.5 * torch.randn(num_bw_samples, batch_size, device=outputs.device) +
                  rho * sigma_g / sigma_f * targets.unsqueeze(0))
    elif sigma_f == 0:
        mg_samples = sigma_g * torch.randn(num_bw_samples, batch_size, device=outputs.device)
    
    # Reshape outputs and targets for broadcasting
    outputs_expanded = outputs.unsqueeze(0)  # (1, len(outputs))
    targets_expanded = targets.unsqueeze(0)  # (1, len(outputs))
    
    # Compute the sampled Z for outputs and targets across all samples
    # Shape: (num_bw_samples, batch_size) + (1, batch_size) -> (num_bw_samples, batch_size)
    unnormalized_P = targets_expanded + mg_samples
    unnormalized_Q = outputs_expanded + mg_samples
    sampled_Z_targets = torch.logsumexp(unnormalized_P, dim=1)  # (num_bw_samples,)
    sampled_Z_outputs = torch.logsumexp(unnormalized_Q, dim=1)  # (num_bw_samples,)
    log_P = unnormalized_P - sampled_Z_targets.unsqueeze(1)
    log_Q = unnormalized_Q - sampled_Z_outputs.unsqueeze(1)
    P = torch.exp(log_P)
    # compute the average KL divergence over samples
    kl_per_sample = nn.functional.kl_div(input=log_Q, target=P, reduction='none', log_target=False).sum(dim=1)
    
    loss = kl_per_sample.mean()
    return loss

def unnormalized_kl(outputs, targets, bw_hat=None, sigma_f=None, sigma_g=None, bw_normalizing_constant=None, max_val=None):
    """
    Unnormalized KL divergence loss.

    Parameters:
    -----------
    outputs : torch.Tensor
        Predicted message values (in log space, normalized)
    targets : torch.Tensor
        True message values (in log space, normalized)
    bw_hat : torch.Tensor, optional
        Backward message values to add to outputs and targets before computing KL.
        When use_bw_approx=True, this contains the approximate backward message.
        Treated as constant (gradients detached).
    sigma_f : float, optional
        Standard deviation of forward messages (for scaling) - unused
    sigma_g : float, optional
        Standard deviation of backward messages (for scaling) - unused
    bw_normalizing_constant : torch.Tensor, optional
        The bw value at argmax(y + bw) from training data.
        When provided, this is subtracted from bw_hat before adding to outputs/targets.
        This ensures numerical stability by making the bw contribution at the max entry = 0.
    max_val : float, optional
        Global maximum value for numerical stability. When provided, this is used instead
        of computing max from the current batch. IMPORTANT: For batched training, this
        should be computed ONCE from the full dataset and passed to every batch. Using
        per-batch max causes gradient inconsistency and training divergence.

    Returns:
    --------
    torch.Tensor
        Scalar loss value
    """

    # If bw_hat is provided, add it to both outputs and targets
    # This implements backward message weighting for use_bw_approx mode
    # bw_hat is treated as a constant (gradients not tracked through it)
    if bw_hat is not None:
        bw_hat_detached = bw_hat.detach()

        # If bw_normalizing_constant is provided, subtract it from bw_hat
        # This ensures that at argmax(y + bw), the bw contribution is 0
        if bw_normalizing_constant is not None:
            bw_hat_detached = bw_hat_detached - bw_normalizing_constant

        outputs = outputs + bw_hat_detached
        targets = targets + bw_hat_detached

    # for non-log-valued, equation is
    # sum ~p(x) [ log [~p(x)/~q(x)] - ~p(x) + ~q(x) ]

    # Compute max_val for numerical stability (prevents exp overflow)
    # CRITICAL: For batched training, max_val MUST be computed from full dataset
    # and passed in. Per-batch max_val causes gradient inconsistency.
    if max_val is None:
        # Fallback: compute from current batch (works for full-batch training)
        max_val = torch.max(torch.max(targets), torch.max(outputs.detach()))
    else:
        # Use provided global max_val, but also check if outputs exceed it
        max_val = max(max_val, outputs.detach().max().item())
        max_val = torch.tensor(max_val, device=outputs.device)
    max_val = max_val.detach()

    log_p_tilde = targets - max_val
    log_q_tilde = outputs - max_val
    p_tilde = torch.exp(log_p_tilde)
    q_tilde = torch.exp(log_q_tilde)
    unsummed = p_tilde * (log_p_tilde - log_q_tilde) - p_tilde + q_tilde
    result = torch.sum(unsummed, dim=0)

    return result

def unnormalized_kl_old(outputs, targets, bw_hat=None, sigma_f=None, sigma_g=None):
    """
    Unnormalized KL divergence loss.

    If sigma_f and sigma_g are provided, applies scaling factor to log-space values:
    scaling_factor = sigma_f^2 / (sigma_f^2 + sigma_g^2)

    Parameters:
    -----------
    outputs : torch.Tensor
        Predicted message values (in log space)
    targets : torch.Tensor
        True message values (in log space)
    bw_hat : torch.Tensor, optional
        Backward message values to add to outputs and targets before computing KL.
        When use_bw_approx=True, this contains the approximate backward message.
        Treated as constant (gradients detached).
    sigma_f : float, optional
        Standard deviation of forward messages (for scaling)
    sigma_g : float, optional
        Standard deviation of backward messages (for scaling)

    Returns:
    --------
    torch.Tensor
        Scalar loss value
    """

    # If bw_hat is provided, add it to both outputs and targets
    # This implements backward message weighting for use_bw_approx mode
    # bw_hat is treated as a constant (gradients not tracked through it)
    if bw_hat is not None:
        bw_hat_detached = bw_hat.detach()
        outputs = outputs + bw_hat_detached
        targets = targets + bw_hat_detached

    # Apply scaling if sigmas provided
    if sigma_f is not None and sigma_g is not None:
        scaling_factor = (sigma_f ** 2 + 1e-12) / (sigma_f ** 2 + sigma_g ** 2 + 1e-12)
        outputs = outputs * scaling_factor
        targets = targets * scaling_factor

    # for non-log-valued, equation is
    # sum ~p(x) [ log [~p(x)/~q(x)] - ~p(x) + ~q(x) ]
    # normalize by max of BOTH outputs and targets to prevent overflow
    # This is critical: if targets are all negative (batch without global max)
    # but outputs are near 0, using max(targets) alone causes exp(outputs - max_targets) to overflow
    max_val = torch.max(torch.max(targets), torch.max(outputs.detach()))
    max_val = max_val.detach()
    log_p_tilde = targets - max_val
    log_q_tilde = outputs - max_val
    p_tilde = torch.exp(log_p_tilde)
    q_tilde = torch.exp(log_q_tilde)
    unsummed = p_tilde * (log_p_tilde - log_q_tilde) - p_tilde + q_tilde
    result = torch.sum(unsummed, dim=0)

    return result

def power_exponential(outputs, targets, bw_hat=None, alpha=0.1):
    max_elt = max(torch.max(outputs-4), torch.max(targets))
    normalizing_factor = max_elt - torch.log(torch.tensor(outputs.numel(), device=outputs.device))
    adjusted_outputs = alpha * (outputs - normalizing_factor)
    adjusted_targets = alpha * (targets - normalizing_factor)
    sqr_difs = (torch.exp(adjusted_outputs) - torch.exp(adjusted_targets)) ** 2
    return 1/(alpha**2) * torch.mean(sqr_difs)

def linspace_mse_fdb(outputs, targets, bw_hat=None, detach_norm=True):
    logZ_t = torch.logsumexp(targets, dim=0, keepdim=True)
    logZ_o = torch.logsumexp(outputs, dim=0, keepdim=True)
    if detach_norm:
        logZ_o = logZ_o.detach()  # stop-gradient through normalizer

    p_hat = torch.exp(outputs - logZ_o)
    p     = torch.exp(targets - logZ_t)
    squared_difs = (p_hat - p)**2
    return torch.mean(squared_difs)

def scaled_mse(outputs, targets, bw_hat=None, sigma_f=0, sigma_g=0, rho=0, detach_norm=True):
    """
    Scaled MSE loss function.

    Same as linspace_mse_fdb, but before exponentiating, multiplies inputs/outputs
    by the scaling factor: (sigma_f**2) / (sigma_f**2 + sigma_g**2)

    This scaling factor weights the contribution based on forward vs backward message variance.

    Parameters:
    -----------
    outputs : torch.Tensor
        Predicted message values (in log space)
    targets : torch.Tensor
        True message values (in log space)
    bw_hat : torch.Tensor, optional
        Message gradient values (unused)
    sigma_f : float
        Standard deviation of forward messages
    sigma_g : float
        Standard deviation of backward messages
    rho : float
        Correlation coefficient (unused in this loss)
    detach_norm : bool
        If True, stop gradient through normalizer (default: True)

    Returns:
    --------
    torch.Tensor
        Scalar loss value
    """
    # Compute scaling factor
    scaling_factor = (sigma_f ** 2) / (sigma_f ** 2 + sigma_g ** 2 + 1e-12)

    # Scale outputs and targets BEFORE computing logsumexp
    scaled_outputs = outputs * scaling_factor
    scaled_targets = targets * scaling_factor
    
    # adjust outputs for convergence
    adj_max_elt = max(torch.max(scaled_outputs-4), torch.max(scaled_targets))
    normalizing_factor = adj_max_elt

    adjusted_outputs = scaled_outputs - normalizing_factor
    adjusted_targets = scaled_targets - normalizing_factor

    difs = torch.exp(adjusted_outputs) - torch.exp(adjusted_targets)
    sqr_difs = difs ** 2
    avg_sqr_difs = torch.mean(sqr_difs)
    return avg_sqr_difs

def linspace_mse_fdb0(outputs, targets, bw_hat=None):
    adj_max_elt = max(torch.max(outputs-4), torch.max(targets))
    normalizing_factor = adj_max_elt - torch.log(torch.tensor(outputs.numel(), device=outputs.device))
    adjusted_outputs = outputs - normalizing_factor
    adjusted_targets = targets - normalizing_factor

    difs = torch.exp(adjusted_outputs) - torch.exp(adjusted_targets)
    sqr_difs = difs ** 2
    avg_sqr_difs = torch.mean(sqr_difs)
    return avg_sqr_difs

def logspace_mse_fdb(outputs, targets, bw_hat=None):
    difs = outputs - targets
    sqr_difs = difs ** 2
    avg_sqr_difs = torch.mean(sqr_difs)
    return avg_sqr_difs
    
def elp(outputs, targets, bw_hat=None, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100, seed=None):
    """
    Computes the loss using sampled message gradient guesses.
    Outputs and targets are expected to be in log space.
    """
    # Get dimensions
    batch_size = outputs.numel()
    
    # Sample mg with shape (num_bw_samples, batch_size)
    # This generates all samples at once for efficiency
    # mg_samples = (sigma_g * (1-rho**2)**0.5 * torch.randn(num_bw_samples, batch_size, device=outputs.device).detach() + 
    #               rho * sigma_g / sigma_f * targets.unsqueeze(0))  # targets broadcasted to (1, batch_size)
    
    # set random seed
    if seed is not None:
        torch.manual_seed(seed)
    if sigma_f > 0:
        mg_samples = (sigma_g * (1-rho**2)**0.5 * torch.randn(num_bw_samples, batch_size, device=outputs.device).detach() +
                  rho * sigma_g / sigma_f * targets.unsqueeze(0))
    elif sigma_f == 0:
        mg_samples = (sigma_g * torch.randn(num_bw_samples, batch_size, device=outputs.device).detach())
    
    # Reshape outputs and targets for broadcasting
    outputs_expanded = outputs.unsqueeze(0)  # (1, batch_size)
    targets_expanded = targets.unsqueeze(0)  # (1, batch_size)
    
    # Compute the sampled Z for outputs and targets across all samples
    # Shape: (num_bw_samples, batch_size) + (1, batch_size) -> (num_bw_samples, batch_size)
    sampled_Z_outputs = torch.logsumexp(outputs_expanded + mg_samples, dim=1)  # (num_bw_samples,)
    sampled_Z_targets = torch.logsumexp(targets_expanded + mg_samples, dim=1)  # (num_bw_samples,)
    
    # Compute squared differences for all samples
    squared_diffs = (sampled_Z_outputs - sampled_Z_targets)**2  # (num_bw_samples,)
    
    # Return the average over all samples
    return torch.mean(squared_diffs)

def elp_cancellation(outputs, targets, bw_hat=None, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100, seed=None):
    """
    Computes the loss using sampled message gradient guesses.
    Outputs and targets are expected to be in log space.
    """
    # Get dimensions
    batch_size = outputs.numel()
    
    # Sample mg with shape (num_bw_samples, batch_size)
    # This generates all samples at once for efficiency
    # mg_samples = (sigma_g * (1-rho**2)**0.5 * torch.randn(num_bw_samples, batch_size, device=outputs.device).detach() + 
    #               rho * sigma_g / sigma_f * targets.unsqueeze(0))  # targets broadcasted to (1, batch_size)
    
    # set random seed
    if seed is not None:
        torch.manual_seed(seed)
    if sigma_f > 0:
        mg_samples = (sigma_g * (1-rho**2)**0.5 * torch.randn(num_bw_samples, batch_size, device=outputs.device).detach() +
                  rho * sigma_g / sigma_f * targets.unsqueeze(0))
    elif sigma_f == 0:
        mg_samples = (sigma_g * torch.randn(num_bw_samples, batch_size, device=outputs.device).detach())
    
    # Reshape outputs and targets for broadcasting
    outputs_expanded = outputs.unsqueeze(0)  # (1, batch_size)
    targets_expanded = targets.unsqueeze(0)  # (1, batch_size)
    
    # Compute the sampled Z for outputs and targets across all samples
    # Shape: (num_bw_samples, batch_size) + (1, batch_size) -> (num_bw_samples, batch_size)
    sampled_Z_outputs = torch.logsumexp(outputs_expanded + mg_samples, dim=1)  # (num_bw_samples,)
    sampled_Z_targets = torch.logsumexp(targets_expanded + mg_samples, dim=1)  # (num_bw_samples,)
    
    # Compute squared differences for all samples
    difs = (sampled_Z_outputs - sampled_Z_targets)  # (num_bw_samples,)
    
    # Return the average over all samples
    return torch.mean(difs)**2


def mg_sampled_loss_loo_fdb(outputs, targets, bw_hat=None, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100, max_memory_gb=10):
    """
    Vectorized leave-one-out sampled loss with memory check and fallback
    """
    return mg_sampled_loss_loo_fdb_vectorized(outputs, targets, bw_hat, sigma_f, sigma_g, rho, num_bw_samples)

    batch_size = outputs.numel()
    
    # Estimate memory usage for vectorized version
    # Main tensors: loo_messages (batch_size^2), mg_samples (num_bw_samples * batch_size), 
    # log_Z_hat_loo (batch_size * num_bw_samples), squared_diffs (batch_size * num_bw_samples)
    bytes_per_float = 4  # float32
    
    estimated_memory_bytes = (
        batch_size**2 * num_bw_samples +  # loo_messages
        num_bw_samples * batch_size +  # mg_samples
        batch_size * num_bw_samples * 2  # log_Z_hat_loo + squared_diffs
    ) * bytes_per_float
    
    estimated_memory_gb = estimated_memory_bytes / (1024**3)
    
    if estimated_memory_gb > max_memory_gb:
        return mg_sampled_loss_loo_fdb_loop(outputs, targets, bw_hat, sigma_f, sigma_g, rho, num_bw_samples)
    else:
        return mg_sampled_loss_loo_fdb_vectorized(outputs, targets, bw_hat, sigma_f, sigma_g, rho, num_bw_samples)

# def mg_sampled_loss_loo_fdb_vectorized(outputs, targets, bw_hat=None, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100):
#     """
#     Your existing vectorized version
#     """
#     batch_size = outputs.numel()
   
#     # Sample backward messages: (num_bw_samples, batch_size)
#     mg_samples = (sigma_g * (1-rho**2)**0.5 * torch.randn(num_bw_samples, batch_size, device=outputs.device).detach() +
#                   rho * sigma_g / sigma_f * targets.unsqueeze(0))

#     # Compute log Z once: (num_bw_samples,)
#     log_Z = torch.logsumexp(targets.unsqueeze(0) + mg_samples, dim=1)
   
#     # Create leave-one-out messages for ALL states at once
#     loo_messages = targets.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, batch_size)
#     loo_messages[torch.arange(batch_size), torch.arange(batch_size)] = outputs  # Diagonal = learned values
   
#     # Expand for broadcasting with mg_samples
#     loo_messages_expanded = loo_messages.unsqueeze(1)  # (batch_size, 1, batch_size)
#     mg_samples_expanded = mg_samples.unsqueeze(0)      # (1, num_bw_samples, batch_size)
   
#     # Compute log Z_hat^(-i) for all i simultaneously: (batch_size, num_bw_samples)
#     log_Z_hat_loo = torch.logsumexp(loo_messages_expanded + mg_samples_expanded, dim=2)
   
#     # Compute losses for all states: (batch_size, num_bw_samples)
#     squared_diffs = (log_Z.unsqueeze(0) - log_Z_hat_loo)**2
   
#     return torch.sum(squared_diffs)

def mg_sampled_loss_loo_fdb_vectorized(outputs, targets, bw_hat=None, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100):
    def looZ(nZs, nerrs_linspace):
        return torch.log(torch.exp(nZs) + nerrs_linspace)
    def smooth_repel(x, nZ_value=4, repel_to=-54.5980, epsilon=1e-6, sharpness=1e6):
        """
        Maps x to almost exactly x, except when very close to critical_value,
        then smoothly pushes toward repel_to.
        
        Args:
            x: input tensor
            critical_value: the value to avoid
            repel_to: where to push values that are too close
            epsilon: how close is "very close"
            sharpness: how sharp the transition is (higher = more sudden)
        """
        critical_value = -torch.exp(torch.tensor(nZ_value, device=x.device))  # Convert nZ_value to log space
        distance = x - critical_value
        
        # Smooth transition function: 0 when far, 1 when very close
        transition = torch.exp(-sharpness * distance**2 / epsilon**2)
        
        # Repulsion amount
        repulsion = repel_to - critical_value
        
        return x + transition * repulsion
    batch_size = outputs.numel()
   
    # Sample backward messages: (num_bw_samples, batch_size)
    if sigma_f > 0:
        mg_samples = (sigma_g * (1-rho**2)**0.5 * torch.randn(num_bw_samples, batch_size, device=outputs.device).detach() +
                  rho * sigma_g / sigma_f * targets.unsqueeze(0))
    elif sigma_f == 0:
        mg_samples = (sigma_g * torch.randn(num_bw_samples, batch_size, device=outputs.device).detach())

    # Compute log Z once: (num_bw_samples,)
    logZs = torch.logsumexp(targets.unsqueeze(0) + mg_samples, dim=1).reshape(1,-1)
   
    nlogZs = 4 * torch.ones_like(logZs, device=logZs.device).reshape(-1,1)  # Initialize with a constant value for numerical stability
    # Create leave-one-out messages for ALL states at once
    normalizers = (logZs - 4).reshape(-1,1)
    
    # compute what we add to nlogZ
    nlin_difs = torch.exp(outputs.unsqueeze(0) + mg_samples - normalizers) - torch.exp(targets.unsqueeze(0) + mg_samples - normalizers)
    nlin_difs = smooth_repel(nlin_difs)
    nlogZ_hats = looZ(nlogZs, nlin_difs) # shape: (num_bw_samples, batch_size)

    # return torch.sum(torch.abs(nlogZs - nlogZ_hats)) # trying no square...
    return torch.sum((nlogZs - nlogZ_hats)**2)

def mg_sampled_loss_loo_fdb_loop(outputs, targets, bw_hat=None, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100):
    """
    Memory-efficient for-loop version for large batch sizes
    """
    batch_size = outputs.numel()
    
    # Sample backward messages: (num_bw_samples, batch_size)
    mg_samples = (sigma_g * (1-rho**2)**0.5 * torch.randn(num_bw_samples, batch_size, device=outputs.device).detach() +
                  rho * sigma_g / sigma_f * targets.unsqueeze(0))
    
    # Compute log Z_true once: (num_bw_samples,)
    log_Z_true = torch.logsumexp(targets.unsqueeze(0) + mg_samples, dim=1)
    
    total_loss = 0.0
    
    for i in range(batch_size):
        # Create leave-one-out message for state i
        loo_message = targets.clone()  # Start with all true values
        loo_message[i] = outputs[i]    # Replace position i with learned value
        
        # Compute log Z_hat^(-i) for this state: (num_bw_samples,)
        log_Z_hat_loo_i = torch.logsumexp(loo_message.unsqueeze(0) + mg_samples, dim=1)
        
        # Compute loss for state i: (num_bw_samples,)
        squared_diffs_i = (log_Z_true - log_Z_hat_loo_i)**2
        
        # Accumulate
        total_loss += torch.sum(squared_diffs_i)
    
    return total_loss

def gil1_linear_space(outputs, targets, bw_hat):
    # compute difference of outputs and targets
    diff = torch.abs(outputs - targets)
    # weight by the grad
    weighted_diff = diff * bw_hat
    # sum the loss
    return torch.sum(weighted_diff) / len(outputs)

def l1c(outputs, targets, bw_hat = None): # gil1c with IS
    # log10 = torch.log(torch.tensor(10.0)).to(outputs.device)
    max_elt = max(torch.max(outputs), torch.max(targets))
    # max_elt.detach_()
    

    target_sampled_Z = torch.logsumexp((targets - max_elt).flatten(), dim=0)
    output_sampled_Z = torch.logsumexp((outputs - max_elt).flatten(), dim=0)
    
    return torch.abs(target_sampled_Z - output_sampled_Z)

def huber_gil1c(outputs, targets, bw_hat, delta = 1):
    # get the exponentiated difference of the outputs and targets
    # convert to linear space to take the difference of the products
    # adding mgh in first for numerical stability to prevent really tiny differences being swallowed even when bw_hat is very large
    
    # bw_hat = 0 # debug
    s1 = outputs + bw_hat
    s2 = targets + bw_hat
    
    max_s = max(torch.max(s1),torch.max(s2))
    max_s.detach_()
    
    target_sampled_Z = torch.logsumexp((s2 - max_s).flatten(), dim=0)
    output_sampled_Z = torch.logsumexp((s1 - max_s).flatten(), dim=0)
    
    # abs_y_minus_y_hat
    dif = torch.abs(target_sampled_Z - output_sampled_Z)
    if dif <= delta:
        print('under delta')
        return 0.5 * dif ** 2
    else:
        print('over delta')
        return delta * dif - 0.5 * delta ** 2

def l1(outputs, targets, bw_hat = None):
    """
    Converts to linear space, takes l1 and converts back to logspace_e
    """
    max_elt = max(torch.max(outputs), torch.max(targets))
    
    difs = torch.abs(torch.exp(outputs - max_elt) - torch.exp(targets - max_elt))
    sum_difs = torch.sum(difs)
    out = torch.log(sum_difs) + max_elt
    return out
    
def from_logspace_l1(outputs, targets, bw_hat = None):
    # log10 = torch.log(torch.tensor(10.0)).to(outputs.device)
    max_elt = max(torch.max(outputs), torch.max(targets))
    
    difs = torch.abs(torch.pow(10.0, outputs - max_elt) - torch.pow(10.0, targets - max_elt))
    sum_difs = torch.sum(difs)
    out = torch.log10(sum_difs) + max_elt
    return out

def from_logspace_mse(outputs, targets, bw_hat = None):
    max_elt = max(torch.max(outputs), torch.max(targets)).detach()
    
    sq_difs = (torch.exp(outputs - max_elt) - torch.exp(targets - max_elt)) ** 2
    #sq_difs = (torch.pow(10.0, outputs - max_elt) - torch.pow(10.0, targets - max_elt)) ** 2
    sum_difs = torch.sum(sq_difs)
    out = sum_difs #/ len(outputs)
    out = torch.log(out) + 2 * max_elt
    #out = torch.log10(out) + 2 * max_elt
    return out

def from_logspace_gil2(outputs, targets, bw_hat):
    max_elt = max(torch.max(outputs+bw_hat), torch.max(targets+bw_hat))
    max_elt.detach_()
    
    sq_difs = (torch.exp(outputs + bw_hat - max_elt) - torch.exp(targets + bw_hat - max_elt)) ** 2
    sum_difs = torch.sum(sq_difs)
    out = sum_difs #/ len(outputs)
    out = torch.log10(out) + 2 * max_elt
    return out

def gil1(outputs, targets, bw_hat):
    # get the exponentiated difference of the outputs and targets
    # convert to linear space to take the difference of the products
    # adding mgh in first for numerical stability to prevent really tiny differences being swallowed even when bw_hat is very large
    
    # bw_hat = 0 # debug
    s1 = outputs + bw_hat
    s2 = targets + bw_hat

    max_s = max(torch.max(s1),torch.max(s2))
    max_s.detach_()

    delta = torch.exp(s1 - max_s) - torch.exp(s2 - max_s)
    abs_delta = torch.abs(delta)
    
    # add the entries
    sum_difs = torch.sum(abs_delta)
    
    # take log and add back in the max
    out = torch.log(sum_difs) + max_s
    
    return out

def gil1c(outputs, targets, bw_hat):
    # get the exponentiated difference of the outputs and targets
    # convert to linear space to take the difference of the products
    # adding mgh in first for numerical stability to prevent really tiny differences being swallowed even when bw_hat is very large
    
    # bw_hat = 0 # debug

    s1 = outputs + bw_hat
    s2 = targets + bw_hat

    max_s = max(torch.max(s1),torch.max(s2))
    max_s.detach_()

    target_sampled_Z = torch.logsumexp((s2 - max_s).flatten(), dim=0)
    output_sampled_Z = torch.logsumexp((s1 - max_s).flatten(), dim=0)
    
    # return output_sampled_Z - target_sampled_Z
    return torch.abs(target_sampled_Z - output_sampled_Z)

def gil1c_linear(outputs, targets, bw_hat, normalizer):
    # get the exponentiated difference of the outputs and targets
    # convert to linear space to take the difference of the products
    # adding mgh in first for numerical stability to prevent really tiny differences being swallowed even when bw_hat is very large
    
    # bw_hat = 0 # debug

    s1 = outputs + bw_hat
    s2 = targets + bw_hat

    # max_s = max(torch.max(s1),torch.max(s2))
    max_s2 = torch.max(s2)
    # max_s.detach_()
    
    # adjust normalizer if new max is seen
    def change_val(old_val, new_val):
        old_val.add_(-old_val)
        old_val.add_(new_val)
    if normalizer < max_s2 or normalizer == 0:
        change_val(normalizer, max_s2)
    
    
    linspace_target_sampled_Z = torch.sum(torch.exp((s2 - normalizer).flatten()), dim=0)
    #
    a=linspace_target_sampled_Z
    linspace_output_sampled_Z = torch.sum(torch.exp((s1 - normalizer).flatten()), dim=0)
    #debug
    b=linspace_output_sampled_Z
    
    # return output_sampled_Z - target_sampled_Z
    output = torch.abs(linspace_target_sampled_Z - linspace_output_sampled_Z)
    return output
    # return (linspace_target_sampled_Z - linspace_output_sampled_Z)**2

def gil1c_linear2(outputs, targets, bw_hat, normalizer):
    s1 = outputs + bw_hat
    s2 = targets + bw_hat
    
    max_s1 = torch.max(s1)
    max_s2 = torch.max(s2)
    max_s = max(max_s1, max_s2)
    
    normalized_output_linspace = torch.logsumexp((s1 - max_s1).flatten(), dim=0)
    normalized_target_linspace = torch.logsumexp((s2 - max_s2).flatten(), dim=0)   

def w_gil1c(outputs, targets, bw_hat, normalizer):
    s1 = outputs + bw_hat
    s2 = targets + bw_hat
    
    max_s = max(torch.max(s1),torch.max(s2))
    max_s.detach_()
    
    target_sampled_Z = torch.logsumexp((s2 - max_s).flatten(), dim=0)
    output_sampled_Z = torch.logsumexp((s1 - max_s).flatten(), dim=0)
    
    # return output_sampled_Z - target_sampled_Z
    log_err_ratio = torch.abs(target_sampled_Z - output_sampled_Z)
    # compute log first
    # logw = (max(target_sampled_Z, output_sampled_Z) + max_s) - normalizer
    logw = (target_sampled_Z + max_s) - normalizer
    # logw = (max(target_sampled_Z, output_sampled_Z) + max_s) - log_Z_hat
    # keep track of samples Z from the batch
    # log_bZ_hat = target_sampled_Z + max_s
    return torch.exp(logw) * log_err_ratio

def z_err(outputs, targets, bw_hat):
    # bw_hat = 0 # debug
    s1 = outputs + bw_hat
    s2 = targets + bw_hat
    
    max_s = max(torch.max(s1),torch.max(s2))
    max_s.detach_()
    
    target_sampled_Z = torch.logsumexp((s2 - max_s).flatten(), dim=0)
    output_sampled_Z = torch.logsumexp((s1 - max_s).flatten(), dim=0)
    
    return output_sampled_Z - target_sampled_Z

def gil2(outputs, targets, bw_hat, normalizer = torch.tensor([9.0])):
    # get the exponentiated difference of the outputs and targets
    # convert to linear space to take the difference of the products
    # adding mgh in first for numerical stability to prevent really tiny differences being swallowed even when bw_hat is very large
    
    # bw_hat = 0 # debug
    # s1 = outputs + bw_hat/2
    # s2 = targets + bw_hat/2
    
    # max_s = max(torch.max(s1),torch.max(s2))
    # max_s.detach_()

    # delta = torch.exp(s1 - max_s) - torch.exp(s2 - max_s)
    delta = torch.exp(outputs) - torch.exp(targets)
    sqr_deltas = delta ** 2
    
    # add weights
    w_sqr_deltas = sqr_deltas * torch.exp(bw_hat)
    
    # add the entries
    mean_w_sqr_difs = torch.mean(w_sqr_deltas)
    
    # add back in the max
    # out = sum_sqr_difs * torch.exp(2 * max_s)
    
    return mean_w_sqr_difs

# just the square of the gil1c err, has different grads
def gil2c(outputs, targets, bw_hat):
    # get the exponentiated difference of the outputs and targets
    # convert to linear space to take the difference of the products
    # adding mgh in first for numerical stability to prevent really tiny differences being swallowed even when bw_hat is very large
    
    # bw_hat = 0 # debug
    s1 = outputs + bw_hat
    s2 = targets + bw_hat
    
    max_s = max(torch.max(s1),torch.max(s2))
    max_s.detach_()
    
    target_sampled_Z = torch.logsumexp((s2 - max_s).flatten(), dim=0)
    output_sampled_Z = torch.logsumexp((s1 - max_s).flatten(), dim=0)
    
    # return output_sampled_Z - target_sampled_Z
    return (target_sampled_Z - output_sampled_Z) ** 2

def from_logspace_gil1c_old(outputs, targets, bw_hat):
    # get the exponentiated difference of the outputs and targets
    # convert to linear space to take the difference of the products
    # adding mgh in first for numerical stability to prevent really tiny differences being swallowed even when bw_hat is very large
    
    # bw_hat = 0 # debug
    s1 = outputs + bw_hat
    s2 = targets + bw_hat
    
    max_s = max(torch.max(s1),torch.max(s2))
    max_s.detach_()

    delta = torch.exp(s1 - max_s) - torch.exp(s2 - max_s)
    sum_difs = torch.sum(delta)
    
    # take abs
    abs_sum = torch.abs(sum_difs)
    
    # convert to samples log Z err
    
    
    # take log and add back in the max
    out = torch.log(abs_sum) + max_s
    
    return out

# DBE
def logspace_mse(outputs, targets, bw_hat = None, IS_weights = None):
    return nn.MSELoss()(outputs, targets)
    if IS_weights is None:
        return nn.MSELoss()(outputs, targets)
    else:
        # compute difference of outputs and targets
        diffs = outputs - targets
        diffs_sq = diffs ** 2
        weighted_diffs_sq = diffs_sq / torch.exp(IS_weights)
        # sum the loss
        return torch.sum(weighted_diffs_sq) / len(outputs)

# NeuroBE
def weighted_logspace_mse(outputs, targets, bw_hat = None):
    epsilon = 1e-6
    ln_max = torch.max(targets)
    ln_min = torch.min(targets)
    normalized_targets = (targets - ln_min) / (epsilon + ln_max - ln_min)
    weights = len(targets) * normalized_targets / (epsilon + torch.sum(normalized_targets))
    # check if weights is ever negative
    if torch.any(weights < 0):
        print('Negative weight values in NeuroBE loss')
    unsummed = weights * (outputs - targets).pow(2)
    out = torch.mean(unsummed)
    return out

def weighted_logspace_mse_pedigree(outputs, targets, bw_hat = None):
    den = torch.logsumexp(targets.flatten(), dim=0)
    weights = torch.exp(targets - den)
    unsummed = 1 * (outputs - targets) ** 2
    # unsummed = weights * (outputs - targets) ** 2
    # check if unsummed is ever negative
    return torch.mean(unsummed) * 10**3

def logspace_mse2(outputs, targets, bw_hat = None):
    # compute difference of outputs and targets
    diff = outputs - targets
    # sum the loss
    return torch.sum(diff**2) / len(outputs)

def logspace_mse_IS(outputs, targets, mh_hat = None, weights = 1):
    """
    message gradient weighted importance sampling
    p weights is numerator of p/q, in our case 1/message_size
    """
    debug = False
    
    if weights == 1:
        print('No weights used.')
    weights.detatch()
    sqr_diff = (outputs - targets)**2
    if debug:
        print('sqr difs are ', sqr_diff[:10])
        print('weights[:10] is ', weights[:10])
        print('exp weigbts[:10] is ', weights[:10])
    weighted_sqr_diff = sqr_diff * weights # apply importance weights by dividing by factor proportional to sampling probability
    if debug:
        print('weighted sqr difs are ', weighted_sqr_diff[:10])
        exit(1)
    return torch.sum(weighted_sqr_diff) / len(outputs)
    
def logspace_mse_pathIS(outputs, targets, bw_hat): # path cost weighted importance sampling
    bw_hat.detach()
    sqr_diff = (outputs - targets)**2
    weighted_sqr_diff = sqr_diff / torch.exp(bw_hat + targets) # apply importance weights by dividing by factor proportional to sampling probability
    return torch.sum(weighted_sqr_diff) / len(outputs)

def logspace_l1(outputs, targets, bw_hat = None):
    return nn.L1Loss(outputs, targets)

def combined_gil1_ls_mse(outputs, targets, bw_hat):
    # compute difference of outputs and targets
    return 100 * from_logspace_gil1(outputs, targets, bw_hat) + logspace_mse(outputs, targets)

def elp_least_squares(outputs, targets, bw_hat=None, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100, seed=None):
    """
    Expected Log Partition function least squares loss (Newton-style weighted regression).

    This implements the same objective that the decision tree optimizer uses:
    minimizing E[(log Z(f + b) - log Z(s + b))²]

    But instead of weighted least squares on the original targets, we compute:
    1. First derivative: dL/ds = -2 * E[DP * psb]
    2. Second derivative: d²L/ds² = 2 * E[psb² - DP*(psb - psb²)]
    3. Newton update: adjusted_targets = outputs - dL/d²L
    4. Weighted MSE: minimize Σ w_i * (outputs - adjusted_targets)²

    where:
    - psb = exp(s + b - log Z(s + b)) is the softmax probability at assignment
    - DP = log Z(f + b) - log Z(s + b) is the partition function difference
    - b = sampled backward messages
    - w_i = d²L/ds² (second derivatives as weights)

    This matches the decision tree optimization in decision_tree.py lines 119-138.

    Parameters:
    -----------
    outputs : torch.Tensor
        Predicted message values (in log space), shape (batch_size,)
    targets : torch.Tensor
        True message values (in log space), shape (batch_size,)
    bw_hat : torch.Tensor, optional
        Message gradient values (unused)
    sigma_f : float
        Standard deviation of forward messages
    sigma_g : float
        Standard deviation of backward messages
    rho : float
        Correlation between forward and backward messages
    num_bw_samples : int
        Number of backward samples for computing derivatives (default: 100)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    torch.Tensor
        Scalar loss value
    """
    batch_size = outputs.numel()

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Handle None values by using defaults or empirical estimates
    if sigma_f is None or sigma_g is None or rho is None:
        # If statistics not available, estimate from targets
        sigma_f = float(torch.std(targets).item()) if sigma_f is None else sigma_f
        sigma_g = sigma_f * 0.5 if sigma_g is None else sigma_g  # Assume sb ~ 0.5 * sf as default
        rho = 0.5 if rho is None else rho  # Assume moderate correlation

    # Compute backward message sampling parameters (same as decision tree)
    sf = sigma_f + 1e-12
    sb = sigma_g + 1e-12

    # Correlation-adjusted parameters
    # alpha = 1 + Sx/sf² where Sx = rho * sigma_f * sigma_g
    alpha = 1.0 + (rho * sigma_f * sigma_g) / (sf ** 2)
    # sb' = sb * sqrt(1 - rho²)
    sb_prime = torch.sqrt(torch.tensor((1.0 - rho**2), device=outputs.device)) * sb

    # Center targets for correlation structure
    f_centered = targets - targets.mean()

    # Sample backward messages: (num_bw_samples, batch_size)
    noise = torch.randn(num_bw_samples, batch_size, device=outputs.device)
    bw_samples = sb_prime * noise + (alpha - 1.0) * f_centered.unsqueeze(0)

    # Expand for broadcasting
    outputs_expanded = outputs.unsqueeze(0)  # (1, batch_size)
    targets_expanded = targets.unsqueeze(0)  # (1, batch_size)

    # Compute partition functions
    # log Z(f + b) and log Z(s + b) for each sample: (num_bw_samples,)
    log_Z_targets = torch.logsumexp(targets_expanded + bw_samples, dim=1)
    log_Z_outputs = torch.logsumexp(outputs_expanded + bw_samples, dim=1)

    # Partition function difference: (num_bw_samples, 1)
    DP = (log_Z_targets - log_Z_outputs).unsqueeze(1)

    # Compute softmax probabilities: psb = exp(s + b - log Z(s + b))
    # Shape: (num_bw_samples, batch_size)
    psb = torch.exp(outputs_expanded + bw_samples - log_Z_outputs.unsqueeze(1))

    # Compute FIRST derivative (gradient): dL/ds = -2 * E[DP * psb]
    # Shape: (batch_size,)
    dL = -2.0 * torch.mean(DP * psb, dim=0)

    # Compute SECOND derivative (Hessian diagonal): d²L/ds² = 2 * E[psb² - DP*(psb - psb²)]
    # Shape: (batch_size,)
    d2L = 2.0 * torch.mean(psb**2 - DP * (psb - psb**2), dim=0)

    # Force positive definiteness (as in decision tree code line 129)
    d2L = torch.clamp(d2L, min=1.0 / (batch_size ** 2))

    # The actual loss we're minimizing is the expected squared partition function error:
    # E[(log Z(f+b) - log Z(s+b))²]
    #
    # The decision tree minimizes this by:
    # 1. Computing Newton-adjusted targets
    # 2. Doing weighted regression with d²L as sample weights
    #
    # For neural networks, we directly minimize the partition function error
    # Let gradients flow through everything - NO detach!
    partition_errors_squared = DP.squeeze() ** 2  # (num_bw_samples,)
    loss = torch.mean(partition_errors_squared)

    return loss


def elp_least_squares_v2(outputs, targets, bw_hat=None, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100, seed=None, expansion_point=None):
    """
    Expected Log Partition function least squares loss with moving average expansion point.

    This version implements the decision tree's moving average approach:
    1. Track an expansion point `s` (like decision tree line 104, 144)
    2. Use moving average for derivatives (like decision tree line 131)
    3. Update expansion point incrementally (like decision tree line 144)

    The key difference from elp_least_squares is that we compute derivatives
    around a **smoothly-evolving expansion point** rather than the current outputs.

    Parameters:
    -----------
    outputs : torch.Tensor
        Predicted message values (in log space), shape (batch_size,)
    targets : torch.Tensor
        True message values (in log space), shape (batch_size,)
    bw_hat : torch.Tensor, optional
        Message gradient values (unused)
    sigma_f : float
        Standard deviation of forward messages
    sigma_g : float
        Standard deviation of backward messages
    rho : float
        Correlation between forward and backward messages
    num_bw_samples : int
        Number of backward samples for computing derivatives (default: 100)
    seed : int, optional
        Random seed for reproducibility
    expansion_point : torch.Tensor, optional
        Previous expansion point for Taylor expansion. If None, initialized to outputs.
        Shape: (batch_size,)

    Returns:
    --------
    tuple: (loss, updated_expansion_point, dL_ma, d2L_ma)
        loss : torch.Tensor - Scalar loss value
        updated_expansion_point : torch.Tensor - Updated expansion point for next iteration
        dL_ma : torch.Tensor - Moving average of first derivatives
        d2L_ma : torch.Tensor - Moving average of second derivatives
    """
    batch_size = outputs.numel()

    # Initialize expansion point if not provided
    if expansion_point is None:
        s = outputs.detach().clone()  # Start at current outputs
    else:
        s = expansion_point.detach().clone()

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Handle None values by using defaults or empirical estimates
    if sigma_f is None or sigma_g is None or rho is None:
        sigma_f = float(torch.std(targets).item()) if sigma_f is None else sigma_f
        sigma_g = sigma_f * 0.5 if sigma_g is None else sigma_g
        rho = 0.5 if rho is None else rho

    # Compute backward message sampling parameters (same as decision tree)
    sf = sigma_f + 1e-12
    sb = sigma_g + 1e-12

    # Correlation-adjusted parameters (decision tree lines 106-107)
    alpha = 1.0 + (rho * sigma_f * sigma_g) / (sf ** 2)
    sb_prime = torch.sqrt(torch.tensor((1.0 - rho**2), device=outputs.device)) * sb

    # Center targets for correlation structure (decision tree line 119)
    f_centered = targets - targets.mean()

    # Sample backward messages: (num_bw_samples, batch_size)
    noise = torch.randn(num_bw_samples, batch_size, device=outputs.device)
    bw_samples = sb_prime * noise + (alpha - 1.0) * f_centered.unsqueeze(0)

    # Expand for broadcasting
    s_expanded = s.unsqueeze(0)  # Use expansion point, not outputs!
    targets_expanded = targets.unsqueeze(0)

    # Compute partition functions AROUND THE EXPANSION POINT
    # This is the key difference - we compute derivatives around `s`, not `outputs`
    log_Z_targets = torch.logsumexp(targets_expanded + bw_samples, dim=1)
    log_Z_s = torch.logsumexp(s_expanded + bw_samples, dim=1)

    # Partition function difference
    DP = (log_Z_targets - log_Z_s).unsqueeze(1)

    # Compute softmax probabilities around expansion point
    psb = torch.exp(s_expanded + bw_samples - log_Z_s.unsqueeze(1))

    # Compute FIRST derivative (gradient): dL/ds = -2 * E[DP * psb]
    dL_current = -2.0 * torch.mean(DP * psb, dim=0)

    # Compute SECOND derivative (Hessian diagonal): d²L/ds² = 2 * E[psb² - DP*(psb - psb²)]
    d2L_current = 2.0 * torch.mean(psb**2 - DP * (psb - psb**2), dim=0)

    # Force positive definiteness
    d2L_current = torch.clamp(d2L_current, min=1.0 / (batch_size ** 2))

    # NOTE: In the decision tree, derivatives are smoothed with moving average (line 131)
    # However, for the loss function, we'll return the current derivatives
    # The Trainer class should handle the moving average across batches/epochs

    # Compute Newton-adjusted targets (decision tree line 134)
    # Clip the Newton step to prevent large jumps
    rngf = torch.std(targets) + 1e-12
    newton_step = torch.clamp(dL_current / d2L_current, -rngf, rngf)
    adjusted_targets = s - newton_step

    # Compute weighted MSE loss
    # Weights are the second derivatives (confidence in each adjustment)
    weights = d2L_current / d2L_current.sum()  # Normalized weights

    # The loss is weighted MSE between outputs and adjusted targets
    # This encourages the network to match the Newton-adjusted targets
    weighted_errors = weights * (outputs - adjusted_targets.detach()) ** 2
    loss = torch.sum(weighted_errors)

    # Update expansion point for next iteration (decision tree line 144)
    # s = s + (c-s)*.5  where c is the new prediction
    # This means: move 50% toward the new outputs
    s_updated = s + 0.5 * (outputs.detach() - s)

    return loss, s_updated, dL_current, d2L_current

def ukf_sequential(outputs, targets, bw_hat=None, Lmu=None, Lsig=None, sigma_f=0, sigma_g=0, rho=0):
    """
    UKF-based sequential loss function.
    
    This loss requires pre-computed Gaussian statistics (Lmu, Lsig) which represent
    the distribution p(Φ, Φ̂) where Φ = lse(f+b) and Φ̂ = lse(fhat+b).
    
    The training loop should periodically recompute these statistics using estimate_gaussian().
    
    Args:
        outputs : torch.Tensor
            Predicted message values (batch), in log space
        targets : torch.Tensor
            True message values (batch), in log space  
        bw_hat : torch.Tensor, optional
            Message gradient values (unused, for interface compatibility)
        Lmu : torch.Tensor
            Pre-computed mean [E[Φ], E[Φ̂]], shape (2,) or (1, 2)
        Lsig : torch.Tensor
            Pre-computed covariance matrix, shape (2, 2)
        sigma_f : float
            Standard deviation of forward messages
        sigma_g : float
            Standard deviation of backward messages
        rho : float
            Correlation coefficient between forward and backward messages
            
    Returns:
        torch.Tensor
            Scalar loss value
    """
    from .ukf_helpers import loss_seq
    
    if Lmu is None or Lsig is None:
        raise ValueError("ukf_sequential requires pre-computed Lmu and Lsig. "
                        "These should be computed periodically in the training loop.")
    
    # Compute adjusted backward message parameters (from notebook)
    sf = sigma_f + 1e-12
    sb = sigma_g + 1e-12
    Sx = rho * sigma_f * sigma_g  # Cross-covariance
    
    # Alpha parameter: 1 + Sx/sf²
    al_ = 1.0 + Sx / (sf ** 2)
    
    # Adjusted std dev: sqrt((1 - rho²) * sb²)
    sb_ = torch.sqrt(torch.tensor((1.0 - Sx**2 / (sb**2 * sf**2)) * sb**2, device=outputs.device))
    
    # Backward message mean: (alpha - 1) * (targets - targets.mean())
    mb = (al_ - 1.0) * (targets - targets.mean())
    
    # Compute loss using UKF sequential approximation
    loss = loss_seq(targets, outputs, mb, sb_, Lmu, Lsig)
    
    return loss
