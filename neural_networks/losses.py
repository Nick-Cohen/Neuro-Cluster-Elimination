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
    max_elt.detach_()
    
    target_sampled_Z = torch.logsumexp((targets - max_elt).flatten(), dim=0)
    output_sampled_Z = torch.logsumexp((outputs - max_elt).flatten(), dim=0)
    
    return torch.abs(target_sampled_Z - output_sampled_Z)

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
    print('outputs: ', outputs[:5])
    print('targets: ', targets[:5])
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

def from_logspace_gil1c(outputs, targets, mg_hat):
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
    
    return torch.abs(target_sampled_Z - output_sampled_Z)

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

def gil1c(outputs, targets, mg_hat):
    # compute difference of outputs and targets
    diff = outputs - targets
    # weight by the grad
    diff = diff * mg_hat
    # sum the loss
    return torch.abs(torch.sum(diff)) / len(outputs)

def gil2(outputs, targets, mg_hat):
    # compute difference of outputs and targets
    diffsq = (outputs - targets)**2
    # weight by the grad
    weighted_diff = diffsq * mg_hat
    # sum the loss
    return torch.sum(weighted_diff) / len(outputs)

def logspace_mse(outputs, targets, mg_hat = None, IS_weights = None):
    if IS_weights is None:
        return nn.MSELoss()(outputs, targets)
    else:
        # compute difference of outputs and targets
        diffs = outputs - targets
        diffs_sq = diffs ** 2
        weighted_diffs_sq = diffs_sq / torch.exp(IS_weights)
        # sum the loss
        return torch.sum(weighted_diffs_sq) / len(outputs)

def logspace_mse2(outputs, targets, mg_hat = None):
    # compute difference of outputs and targets
    diff = outputs - targets
    # sum the loss
    return torch.sum(diff**2) / len(outputs)

def logspace_mse_mgIS(outputs, targets, mg_hat): # message gradient weighted importance sampling
    debug = False
    mg_hat.detach()
    sqr_diff = (outputs - targets)**2
    if debug:
        print('sqr difs are ', sqr_diff[:10])
        print('mg_hat is ', mg_hat[:10])
        print('exp mg_hat is ', torch.exp(mg_hat)[:10])
    weighted_sqr_diff = sqr_diff / torch.exp(mg_hat) # apply importance weights by dividing by factor proportional to sampling probability
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