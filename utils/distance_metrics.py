from fastElim import *
import torch
import copy

# grad informed metrics take the distance metric between each of the entries weighted by the gradient at the corresponding entry
# This weighting is outside of logspace, so to weight the difference of an entry it will be distance_metric(exact_entry, approximate_entry) * 10**(log_valued_gradient)
# All log-valued terms are base 10
# Messages and approximate messages are given in log space. The logspace functions do not exponentiate the messages before computing the distance metric, but if logspace is not specified, the messages are exponentiated before the distance metric is computed.
# All gradient arguments are given in log space
# l1 distances with cancellation do not take abs between each terms but only take abs at the very end after adding all the weighted or unweighted differences together



def match_FastFactor(matchee, matcher):
    # A helper function that automatically permutes two tensors with the same but permuted labels to match their label order is needed to help compute many of these distance measures
    # function should change the matcher tensor and labels by permuting the labels and tensor of the matcher together to match the labels of the matchee
    
    # Find the permutation of labels in matcher that matches the matchee
    perm = [matcher.labels.index(label) for label in matchee.labels]
    # Permute the matcher tensor in-place
    matcher.tensor = matcher.tensor.permute(*perm)
    # Update matcher labels to match the matchee
    matcher.labels = matchee.labels
    return matcher  # returning matcher as per your instructions


# def distance(message, message_hat, distance_metric, grad_informed=False, grad=None, cancellation=False, logspace=False):
#     # Check if labels match, if not, use match_FastFactor to align them
#     if message.labels != message_hat.labels:
#         message_hat = match_FastFactor(message, message_hat)

#     # Determine the distance metric to use (l1 or mse)
#     if distance_metric.lower() in ["l1", "1"]:
#         if cancellation:
#             return logspace_l1_with_cancellation(message, message_hat) if logspace else l1_with_cancellation(message, message_hat)
#         elif grad_informed:
#             return logspace_grad_informed_l1(message, message_hat, grad) if logspace else grad_informed_l1(message, message_hat, grad)
#         else:
#             return logspace_l1(message, message_hat) if logspace else l1(message, message_hat)
    
#     elif distance_metric.lower() in ["l2", "2", "mse", "MSE"]:
#         if grad_informed:
#             return logspace_grad_informed_mse(message, message_hat, grad) if logspace else grad_informed_mse(message, message_hat, grad)
#         else:
#             return logspace_mse(message, message_hat) if logspace else mse(message, message_hat)

#     else:
#         raise ValueError(f"Unknown distance metric: {distance_metric}")

def sampled_distance(distance_metric, num_samples, mess, mess_hat, mg=None):
    # create copies of mess and mess_hat
    sampled_mess = copy.deepcopy(mess)
    sampled_mess_hat = copy.deepcopy(mess_hat)
   
    # get num_samples random indices from mess.tensor
    flat_indices = torch.randperm(mess.tensor.numel())[:num_samples]
    
    # Convert flat indices to multi-dimensional indices
    indices = np.unravel_index(flat_indices.numpy(), mess.tensor.shape)
   
    # Create mask tensors
    mask = torch.zeros_like(mess.tensor, dtype=torch.bool)
    mask[indices] = True
   
    # Apply mask to tensors
    sampled_mess.tensor = torch.where(mask, mess.tensor, torch.tensor(float('-inf')))
    sampled_mess_hat.tensor = torch.where(mask, mess_hat.tensor, torch.tensor(float('-inf')))
   
    # compute the sampled distance
    if mg is None:
        return distance_metric(sampled_mess, sampled_mess_hat)
    else:
        sampled_mg = copy.deepcopy(mg)
        sampled_mg.tensor = torch.where(mask, mg.tensor, torch.tensor(float('-inf')))
        return distance_metric(sampled_mess, sampled_mess_hat, sampled_mg)
    
def l1(mess, mess_hat):
    if mess.labels != mess_hat.labels:
        mess_hat = match_FastFactor(mess, mess_hat)

    # Calculate the max over both tensors
    max_value = max(torch.max(mess.tensor), torch.max(mess_hat.tensor))

    # Subtract the max value from both tensors for numerical stability
    exp_mess = torch.pow(10, mess.tensor - max_value)
    exp_mess_hat = torch.pow(10, mess_hat.tensor - max_value)

    # Compute the absolute differences
    abs_diff = torch.abs(exp_mess - exp_mess_hat)

    # Sum the absolute differences
    sum_abs_diff = torch.sum(abs_diff)

    # Take the log10 of the sum of absolute differences and add the max back
    log_distance = torch.log10(sum_abs_diff) + max_value

    return log_distance

def grad_informed_l1(mess, mess_hat, mg):
    if mess.labels != mess_hat.labels:
        mess_hat = match_FastFactor(mess, mess_hat)

    # Calculate the max over both tensors and the gradient
    max_value = max(torch.max(mess.tensor), torch.max(mess_hat.tensor))

    # Subtract the max value from both tensors for numerical stability
    exp_mess = torch.pow(10, mess.tensor - max_value)
    exp_mess_hat = torch.pow(10, mess_hat.tensor - max_value)
    dif_tensor = torch.abs(exp_mess - exp_mess_hat)
    log_dif_tensor = torch.log10(dif_tensor) + max_value
    dif_FastFactor = FastFactor(log_dif_tensor, mess.labels)
    
    return (dif_FastFactor * mg).sum_all_entries()


# What I used to call the nn-error
def l1_with_cancellation(mess, mess_hat):
    if mess.labels != mess_hat.labels:
        mess_hat = match_FastFactor(mess, mess_hat)

    # Calculate the max over both tensors for numerical stability
    max_value = max(torch.max(mess.tensor), torch.max(mess_hat.tensor))

    # Subtract the max value from both tensors for stability
    exp_mess = torch.pow(10, mess.tensor - max_value)
    exp_mess_hat = torch.pow(10, mess_hat.tensor - max_value)

    # Perform the sum of the differences before taking the absolute value (for cancellation)
    diff_sum = torch.sum(exp_mess - exp_mess_hat)

    # Take the absolute value of the sum, and add back the max value
    log_distance = torch.log10(torch.abs(diff_sum)) + max_value

    return log_distance

# grad informed nn-error
def grad_informed_l1_with_cancellation(mess, mess_hat, mg):
    
    # mg is approximate, hence the double hats
    z_hat = (mess * mg).sum_all_entries()
    z_hat_hat = (mess_hat * mg).sum_all_entries()
    return abs(z_hat-z_hat_hat)
    
    
    if mess.labels != mess_hat.labels:
        mess_hat = match_FastFactor(mess, mess_hat)

    # Calculate the max over both tensors for numerical stability
    max_value = max(torch.max(mess.tensor), torch.max(mess_hat.tensor))
    
    # Subtract the max value from both tensors for numerical stability
    exp_mess = torch.pow(10, mess.tensor - max_value)
    exp_mess_hat = torch.pow(10, mess_hat.tensor - max_value)
    
    # Compute the difference
    dif_tensor = exp_mess - exp_mess_hat
    
    # Keep track of negative values
    negative_mask = dif_tensor < 0
    
    # Take absolute value and convert back to log space
    log_abs_dif_tensor = torch.log10(torch.abs(dif_tensor)) + max_value
    
    dif_FastFactor = FastFactor(log_abs_dif_tensor, mess.labels)
   
    # Multiply with message gradient
    product = dif_FastFactor * mg
    
    if product.labels != mess.labels:
        product = match_FastFactor(mess, product)
    
    # Extract the tensor from the product
    product_tensor = product.tensor
    
    # Find the max of the product for numerical stability
    product_max = torch.max(product_tensor)
    
    # Subtract max, exponentiate, and reapply signs
    exp_product = torch.exp((product_tensor - product_max) * math.log(10))
    exp_product[negative_mask] = -exp_product[negative_mask]
    
    # Sum all entries
    sum_exp_product = torch.sum(exp_product)
    
    # Take log10 and add max back
    result = torch.log10(abs(sum_exp_product)) + product_max
    
    # # Reapply sign to the final result
    # if sum_exp_product < 0:
    #     result = -result

    return resul

def logspace_l1(mess, mess_hat):
    if mess.labels != mess_hat.labels:
        mess_hat = match_FastFactor(mess, mess_hat)
    return torch.sum(torch.abs(mess.tensor - mess_hat.tensor))

def logspace_grad_informed_l1(mess, mess_hat, mg):
    if mess.labels != mess_hat.labels:
        mess_hat = match_FastFactor(mess, mess_hat)

    # Compute the absolute difference directly in log space
    abs_diff = torch.abs(mess.tensor - mess_hat.tensor)
    
    # Create a FastFactor with the absolute difference
    abs_diff_factor = FastFactor(abs_diff, mess.labels)
    
    # Multiply (add in log space) with the message gradient and sum all entries
    return (abs_diff_factor * mg).sum_all_entries()

def mse(mess, mess_hat):
    if mess.labels != mess_hat.labels:
        mess_hat = match_FastFactor(mess, mess_hat)

    # Calculate the max over both tensors for numerical stability
    max_value = max(torch.max(mess.tensor), torch.max(mess_hat.tensor))

    # Subtract the max value from both tensors for stability
    exp_mess = torch.pow(10, mess.tensor - max_value)
    exp_mess_hat = torch.pow(10, mess_hat.tensor - max_value)

    # Compute the squared difference
    squared_error = (exp_mess - exp_mess_hat) ** 2

    # Sum the squared differences and convert back to log10 space
    sum_squared_error = torch.sum(squared_error)
    log_distance = torch.log10(sum_squared_error) + 2 * max_value

    return log_distance

def grad_informed_mse(mess, mess_hat, mg):
    if mess.labels != mess_hat.labels:
        mess_hat = match_FastFactor(mess, mess_hat)
    
    # Calculate the max over both tensors and the gradient
    max_value = max(torch.max(mess.tensor), torch.max(mess_hat.tensor))
    
    # Subtract the max value from both tensors for numerical stability
    exp_mess = torch.pow(10, mess.tensor - max_value)
    exp_mess_hat = torch.pow(10, mess_hat.tensor - max_value)
    
    # Compute the squared difference
    diff_tensor = (exp_mess - exp_mess_hat) ** 2
    
    # Convert back to log space
    log_diff_tensor = torch.log10(diff_tensor) + 2 * max_value  # Add 2 * max_value because we squared the difference
    
    diff_FastFactor = FastFactor(log_diff_tensor, mess.labels)
   
    return (diff_FastFactor * mg).sum_all_entries()

def logspace_mse(mess, mess_hat):
    if mess.labels != mess_hat.labels:
        mess_hat = match_FastFactor(mess, mess_hat)

    # Compute the squared difference in log space
    squared_error = (mess.tensor - mess_hat.tensor) ** 2

    # Sum the squared differences
    return torch.sum(squared_error)

def logspace_grad_informed_mse(mess, mess_hat, mg):
    if mess.labels != mess_hat.labels:
        mess_hat = match_FastFactor(mess, mess_hat)

    # Compute the difference directly in log space
    diff = mess.tensor - mess_hat.tensor
    
    # Square the difference
    squared_diff = diff ** 2
    log_squared_diff = torch.log10(squared_diff)
    
    # Create a FastFactor with the squared difference
    squared_diff_factor = FastFactor(squared_diff, mess.labels)
   
    # Multiply with message gradient (addition in log space) and sum all entries
    return (squared_diff_factor * mg).sum_all_entries()

def true_err(mess, mess_hat, mg, verbose=False):
    Z_hat = (mess_hat * mg).sum_all_entries()
    Z = (mess * mg).sum_all_entries()
    if verbose:
        print("Z_hat is", Z_hat)
        print("Z is", Z)
    return Z_hat - Z


    