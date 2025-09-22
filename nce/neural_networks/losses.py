import torch
import torch.nn as nn
import math

def unnormalized_kl(outputs, targets, mg_hat=None):
    # for non-log-valued, equation is
    # sum ~p(x) [ log [~p(x)/~q(x)] - ~p(x) + ~q(x) ]
    # normalize
    max_targets = torch.max(targets)
    max_targets = max_targets.detach()
    log_p_tilde = targets - max_targets
    log_q_tilde = outputs - max_targets
    p_tilde = torch.exp(log_p_tilde)
    q_tilde = torch.exp(log_q_tilde)
    unsummed = p_tilde * (log_p_tilde - log_q_tilde) - p_tilde + q_tilde
    return torch.sum(unsummed, dim=0)

def power_exponential(outputs, targets, mg_hat=None, alpha=0.1):
    max_elt = max(torch.max(outputs-4), torch.max(targets))
    normalizing_factor = max_elt - torch.log(torch.tensor(outputs.numel(), device=outputs.device))
    adjusted_outputs = alpha * (outputs - normalizing_factor)
    adjusted_targets = alpha * (targets - normalizing_factor)
    sqr_difs = (torch.exp(adjusted_outputs) - torch.exp(adjusted_targets)) ** 2
    return 1/(alpha**2) * torch.mean(sqr_difs)

def linspace_mse_fdb(outputs, targets, mg_hat=None, detach_norm=True):
    logZ_t = torch.logsumexp(targets, dim=0, keepdim=True)
    logZ_o = torch.logsumexp(outputs, dim=0, keepdim=True)
    if detach_norm:
        logZ_o = logZ_o.detach()  # stop-gradient through normalizer

    p_hat = torch.exp(outputs - logZ_o)
    p     = torch.exp(targets - logZ_t)
    squared_difs = (p_hat - p)**2
    return torch.mean(squared_difs)

def linspace_mse_fdb0(outputs, targets, mg_hat=None):
    adj_max_elt = max(torch.max(outputs-4), torch.max(targets))
    normalizing_factor = adj_max_elt - torch.log(torch.tensor(outputs.numel(), device=outputs.device))
    adjusted_outputs = outputs - normalizing_factor
    adjusted_targets = targets - normalizing_factor

    difs = torch.exp(adjusted_outputs) - torch.exp(adjusted_targets)
    sqr_difs = difs ** 2
    avg_sqr_difs = torch.mean(sqr_difs)
    return avg_sqr_difs

def logspace_mse_fdb(outputs, targets, mg_hat=None):
    difs = outputs - targets
    sqr_difs = difs ** 2
    avg_sqr_difs = torch.mean(sqr_difs)
    return avg_sqr_difs
    
def mg_sampled_loss_fdb(outputs, targets, mg_hat=None, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100, seed=None):
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

def mg_sampled_loss_fdb_cancellation(outputs, targets, mg_hat=None, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100, seed=None):
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


def mg_sampled_loss_loo_fdb(outputs, targets, mg_hat=None, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100, max_memory_gb=10):
    """
    Vectorized leave-one-out sampled loss with memory check and fallback
    """
    return mg_sampled_loss_loo_fdb_vectorized(outputs, targets, mg_hat, sigma_f, sigma_g, rho, num_bw_samples)

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
        # print(f"Estimated memory usage: {estimated_memory_gb:.2f} GB > {max_memory_gb} GB. Using for-loop version.")
        return mg_sampled_loss_loo_fdb_loop(outputs, targets, mg_hat, sigma_f, sigma_g, rho, num_bw_samples)
    else:
        # print(f"Estimated memory usage: {estimated_memory_gb:.2f} GB <= {max_memory_gb} GB. Using vectorized version.")
        return mg_sampled_loss_loo_fdb_vectorized(outputs, targets, mg_hat, sigma_f, sigma_g, rho, num_bw_samples)

# def mg_sampled_loss_loo_fdb_vectorized(outputs, targets, mg_hat=None, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100):
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

def mg_sampled_loss_loo_fdb_vectorized(outputs, targets, mg_hat=None, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100):
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

def mg_sampled_loss_loo_fdb_loop(outputs, targets, mg_hat=None, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100):
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


# def mg_sampled_loss_loo_fdb(outputs, targets, mg_hat=None, sigma_f=0, sigma_g=0, rho=0, num_bw_samples=100):
#     """
#     Vectorized leave-one-out sampled loss - no loops!
#     Built for full data batches
#     """
#     batch_size = outputs.numel()
    
#     # Sample backward messages: (num_bw_samples, batch_size)
#     mg_samples = (sigma_g * (1-rho**2)**0.5 * torch.randn(num_bw_samples, batch_size, device=outputs.device).detach() +
#                   rho * sigma_g / sigma_f * targets.unsqueeze(0))
    
#     # Compute log Z_true once: (num_bw_samples,)
#     log_Z_true = torch.logsumexp(targets.unsqueeze(0) + mg_samples, dim=1)
    
#     # Create leave-one-out messages for ALL states at once
#     # Start with targets repeated for each "leave-one-out" scenario
#     # Shape: (batch_size, batch_size) where row i has outputs[i] in position i, targets elsewhere
#     loo_messages = targets.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, batch_size)
#     loo_messages[torch.arange(batch_size), torch.arange(batch_size)] = outputs  # Diagonal = learned values
    
#     # Expand for broadcasting with mg_samples
#     # loo_messages: (batch_size, 1, batch_size), mg_samples: (1, num_bw_samples, batch_size)
#     loo_messages_expanded = loo_messages.unsqueeze(1)  # (batch_size, 1, batch_size)
#     mg_samples_expanded = mg_samples.unsqueeze(0)      # (1, num_bw_samples, batch_size)
    
#     # Compute log Z_hat^(-i) for all i simultaneously: (batch_size, num_bw_samples)
#     log_Z_hat_loo = torch.logsumexp(loo_messages_expanded + mg_samples_expanded, dim=2)
    
#     # Compute losses for all states: (batch_size, num_bw_samples)
#     squared_diffs = (log_Z_true.unsqueeze(0) - log_Z_hat_loo)**2
    
#     # Average over samples and states
#     return torch.sum(squared_diffs)


def gil1_linear_space(outputs, targets, mg_hat):
    # compute difference of outputs and targets
    diff = torch.abs(outputs - targets)
    # weight by the grad
    weighted_diff = diff * mg_hat
    # sum the loss
    return torch.sum(weighted_diff) / len(outputs)

def l1c(outputs, targets, mg_hat = None): # gil1c with IS
    # log10 = torch.log(torch.tensor(10.0)).to(outputs.device)
    max_elt = max(torch.max(outputs), torch.max(targets))
    # max_elt.detach_()
    

    target_sampled_Z = torch.logsumexp((targets - max_elt).flatten(), dim=0)
    output_sampled_Z = torch.logsumexp((outputs - max_elt).flatten(), dim=0)
    
    return torch.abs(target_sampled_Z - output_sampled_Z)

def huber_gil1c(outputs, targets, mg_hat, delta = 1):
    # get the exponentiated difference of the outputs and targets
    # convert to linear space to take the difference of the products
    # adding mgh in first for numerical stability to prevent really tiny differences being swallowed even when mg_hat is very large
    
    # mg_hat = 0 # debug
    s1 = outputs + mg_hat
    s2 = targets + mg_hat
    
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

def l1(outputs, targets, mg_hat = None):
    """
    Converts to linear space, takes l1 and converts back to logspace_e
    """
    max_elt = max(torch.max(outputs), torch.max(targets))
    
    difs = torch.abs(torch.exp(outputs - max_elt) - torch.exp(targets - max_elt))
    sum_difs = torch.sum(difs)
    out = torch.log(sum_difs) + max_elt
    return out
    
def from_logspace_l1(outputs, targets, mg_hat = None):
    # log10 = torch.log(torch.tensor(10.0)).to(outputs.device)
    max_elt = max(torch.max(outputs), torch.max(targets))
    
    difs = torch.abs(torch.pow(10.0, outputs - max_elt) - torch.pow(10.0, targets - max_elt))
    sum_difs = torch.sum(difs)
    out = torch.log10(sum_difs) + max_elt
    return out

def from_logspace_mse(outputs, targets, mg_hat = None):
    max_elt = max(torch.max(outputs), torch.max(targets)).detach()
    
    sq_difs = (torch.exp(outputs - max_elt) - torch.exp(targets - max_elt)) ** 2
    #sq_difs = (torch.pow(10.0, outputs - max_elt) - torch.pow(10.0, targets - max_elt)) ** 2
    sum_difs = torch.sum(sq_difs)
    out = sum_difs #/ len(outputs)
    out = torch.log(out) + 2 * max_elt
    #out = torch.log10(out) + 2 * max_elt
    return out

def from_logspace_gil2(outputs, targets, mg_hat):
    max_elt = max(torch.max(outputs+mg_hat), torch.max(targets+mg_hat))
    max_elt.detach_()
    
    sq_difs = (torch.exp(outputs + mg_hat - max_elt) - torch.exp(targets + mg_hat - max_elt)) ** 2
    sum_difs = torch.sum(sq_difs)
    out = sum_difs #/ len(outputs)
    out = torch.log10(out) + 2 * max_elt
    return out

def gil1(outputs, targets, mg_hat):
    # get the exponentiated difference of the outputs and targets
    # convert to linear space to take the difference of the products
    # adding mgh in first for numerical stability to prevent really tiny differences being swallowed even when mg_hat is very large
    
    # mg_hat = 0 # debug
    # print('outputs: ', outputs[:5])
    # print('targets: ', targets[:5])
    s1 = outputs + mg_hat
    s2 = targets + mg_hat
    
    max_s = max(torch.max(s1),torch.max(s2))
    max_s.detach_()

    delta = torch.exp(s1 - max_s) - torch.exp(s2 - max_s)
    abs_delta = torch.abs(delta)
    
    # add the entries
    sum_difs = torch.sum(abs_delta)
    
    # take log and add back in the max
    out = torch.log(sum_difs) + max_s
    
    return out

def gil1c(outputs, targets, mg_hat):
    # get the exponentiated difference of the outputs and targets
    # convert to linear space to take the difference of the products
    # adding mgh in first for numerical stability to prevent really tiny differences being swallowed even when mg_hat is very large
    
    # mg_hat = 0 # debug
    
    # print(mg_hat[:10])
    # exit(1)
    
    s1 = outputs + mg_hat
    s2 = targets + mg_hat
    
    max_s = max(torch.max(s1),torch.max(s2))
    max_s.detach_()
    
    target_sampled_Z = torch.logsumexp((s2 - max_s).flatten(), dim=0)
    output_sampled_Z = torch.logsumexp((s1 - max_s).flatten(), dim=0)
    
    # return output_sampled_Z - target_sampled_Z
    return torch.abs(target_sampled_Z - output_sampled_Z)

def gil1c_linear(outputs, targets, mg_hat, normalizer):
    # get the exponentiated difference of the outputs and targets
    # convert to linear space to take the difference of the products
    # adding mgh in first for numerical stability to prevent really tiny differences being swallowed even when mg_hat is very large
    
    # mg_hat = 0 # debug
    
    # print(mg_hat[:10])
    # exit(1)
    
    s1 = outputs + mg_hat
    s2 = targets + mg_hat
    
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

def gil1c_linear2(outputs, targets, mg_hat, normalizer):
    s1 = outputs + mg_hat
    s2 = targets + mg_hat
    
    max_s1 = torch.max(s1)
    max_s2 = torch.max(s2)
    max_s = max(max_s1, max_s2)
    
    normalized_output_linspace = torch.logsumexp((s1 - max_s1).flatten(), dim=0)
    normalized_target_linspace = torch.logsumexp((s2 - max_s2).flatten(), dim=0)   

def w_gil1c(outputs, targets, mg_hat, normalizer):
    s1 = outputs + mg_hat
    s2 = targets + mg_hat
    
    max_s = max(torch.max(s1),torch.max(s2))
    max_s.detach_()
    
    target_sampled_Z = torch.logsumexp((s2 - max_s).flatten(), dim=0)
    output_sampled_Z = torch.logsumexp((s1 - max_s).flatten(), dim=0)
    
    # return output_sampled_Z - target_sampled_Z
    log_err_ratio = torch.abs(target_sampled_Z - output_sampled_Z)
    # compute batch's Z divided by sampled Z bZ_hat/Z_hat
    print('got here')
    # compute log first
    # logw = (max(target_sampled_Z, output_sampled_Z) + max_s) - normalizer
    logw = (target_sampled_Z + max_s) - normalizer
    # logw = (max(target_sampled_Z, output_sampled_Z) + max_s) - log_Z_hat
    # keep track of samples Z from the batch
    # log_bZ_hat = target_sampled_Z + max_s
    return torch.exp(logw) * log_err_ratio

def z_err(outputs, targets, mg_hat):
    # mg_hat = 0 # debug
    s1 = outputs + mg_hat
    s2 = targets + mg_hat
    
    max_s = max(torch.max(s1),torch.max(s2))
    max_s.detach_()
    
    target_sampled_Z = torch.logsumexp((s2 - max_s).flatten(), dim=0)
    output_sampled_Z = torch.logsumexp((s1 - max_s).flatten(), dim=0)
    
    return output_sampled_Z - target_sampled_Z

def gil2(outputs, targets, mg_hat, normalizer = torch.tensor([9.0])):
    # get the exponentiated difference of the outputs and targets
    # convert to linear space to take the difference of the products
    # adding mgh in first for numerical stability to prevent really tiny differences being swallowed even when mg_hat is very large
    
    # mg_hat = 0 # debug
    # print('outputs: ', outputs[:5])
    # print('targets: ', targets[:5])
    # s1 = outputs + mg_hat/2
    # s2 = targets + mg_hat/2
    
    # max_s = max(torch.max(s1),torch.max(s2))
    # max_s.detach_()

    # delta = torch.exp(s1 - max_s) - torch.exp(s2 - max_s)
    delta = torch.exp(outputs) - torch.exp(targets)
    sqr_deltas = delta ** 2
    
    # add weights
    w_sqr_deltas = sqr_deltas * torch.exp(mg_hat)
    
    # add the entries
    mean_w_sqr_difs = torch.mean(w_sqr_deltas)
    
    # add back in the max
    # out = sum_sqr_difs * torch.exp(2 * max_s)
    
    return mean_w_sqr_difs

# just the square of the gil1c err, has different grads
def gil2c(outputs, targets, mg_hat):
    # get the exponentiated difference of the outputs and targets
    # convert to linear space to take the difference of the products
    # adding mgh in first for numerical stability to prevent really tiny differences being swallowed even when mg_hat is very large
    
    # mg_hat = 0 # debug
    s1 = outputs + mg_hat
    s2 = targets + mg_hat
    
    max_s = max(torch.max(s1),torch.max(s2))
    max_s.detach_()
    
    target_sampled_Z = torch.logsumexp((s2 - max_s).flatten(), dim=0)
    output_sampled_Z = torch.logsumexp((s1 - max_s).flatten(), dim=0)
    
    # return output_sampled_Z - target_sampled_Z
    return (target_sampled_Z - output_sampled_Z) ** 2

def from_logspace_gil1c_old(outputs, targets, mg_hat):
    # get the exponentiated difference of the outputs and targets
    # convert to linear space to take the difference of the products
    # adding mgh in first for numerical stability to prevent really tiny differences being swallowed even when mg_hat is very large
    
    # mg_hat = 0 # debug
    s1 = outputs + mg_hat
    s2 = targets + mg_hat
    
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
def logspace_mse(outputs, targets, mg_hat = None, IS_weights = None):
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
def weighted_logspace_mse(outputs, targets, mg_hat = None):
    ln_max = torch.max(targets)
    ln_min = torch.min(targets)
    normalized_targets = (targets - ln_min) / (ln_max - ln_min)
    weights = len(targets) * normalized_targets / (torch.sum(normalized_targets))
    # check if weights is ever negative
    if torch.any(weights < 0):
        print('Negative weight values in NeuroBE loss')
    unsummed = weights * (outputs - targets).pow(2)
    out = torch.mean(unsummed)
    return out

def weighted_logspace_mse_pedigree(outputs, targets, mg_hat = None):
    den = torch.logsumexp(targets.flatten(), dim=0)
    weights = torch.exp(targets - den)
    unsummed = 1 * (outputs - targets) ** 2
    # unsummed = weights * (outputs - targets) ** 2
    # check if unsummed is ever negative
    return torch.mean(unsummed) * 10**3

def logspace_mse2(outputs, targets, mg_hat = None):
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
    
def logspace_mse_pathIS(outputs, targets, mg_hat): # path cost weighted importance sampling
    mg_hat.detach()
    sqr_diff = (outputs - targets)**2
    weighted_sqr_diff = sqr_diff / torch.exp(mg_hat + targets) # apply importance weights by dividing by factor proportional to sampling probability
    return torch.sum(weighted_sqr_diff) / len(outputs)

def logspace_l1(outputs, targets, mg_hat = None):
    return nn.L1Loss(outputs, targets)

def combined_gil1_ls_mse(outputs, targets, mg_hat):
    # compute difference of outputs and targets
    return 100 * from_logspace_gil1(outputs, targets, mg_hat) + logspace_mse(outputs, targets)