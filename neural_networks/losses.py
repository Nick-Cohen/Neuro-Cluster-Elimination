import torch
import torch.nn as nn
import math
    

def gil1(outputs, targets, mg_hat):
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

def from_logspace_gil1(outputs, targets, mg_hat):
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
    s1 = outputs + mg_hat
    s2 = targets + mg_hat
    
    max_s = max(torch.max(s1),torch.max(s2))
    max_s.detach_()
    
    target_sampled_Z = torch.logsumexp((s2 - max_s).flatten(), dim=0)
    output_sampled_Z = torch.logsumexp((s1 - max_s).flatten(), dim=0)
    
    # return output_sampled_Z - target_sampled_Z
    return torch.abs(target_sampled_Z - output_sampled_Z)

def z_err(outputs, targets, mg_hat):
    # mg_hat = 0 # debug
    s1 = outputs + mg_hat
    s2 = targets + mg_hat
    
    max_s = max(torch.max(s1),torch.max(s2))
    max_s.detach_()
    
    target_sampled_Z = torch.logsumexp((s2 - max_s).flatten(), dim=0)
    output_sampled_Z = torch.logsumexp((s1 - max_s).flatten(), dim=0)
    
    # return output_sampled_Z - target_sampled_Z without abs
    return target_sampled_Z - output_sampled_Z

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