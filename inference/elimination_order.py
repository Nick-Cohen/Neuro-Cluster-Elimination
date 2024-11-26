import numpy as np
from .factor import FastFactor
from .bucket import FastBucket

def wtminfill_order(factors_or_buckets, variables_not_eliminated=None):
    """
    Find an elimination order using the weighted min-fill heuristic for a subset of variables.
    
    Args:
    factors_or_buckets (list): Either a list of FastFactor objects or a list of FastBucket objects.
    variables_not_eliminated (list): List of variable labels not eliminated. If None, all variables are considered.
    
    Returns:
    list: The elimination order for the specified variables (first eliminated first).
    """
    # Extract factors from buckets if necessary
    if isinstance(factors_or_buckets[0], FastBucket):
        factors = [factor for bucket in factors_or_buckets for factor in bucket.factors]
    else:
        factors = factors_or_buckets

    # Get all unique variables
    all_variables = set()
    for factor in factors:
        all_variables.update(factor.labels)
    all_variables = list(all_variables)

    # If variables_to_eliminate is not specified, use all variables
    if variables_not_eliminated is None:
        variables_to_eliminate = all_variables
    else:
        variables_to_eliminate = [var for var in all_variables if var not in variables_not_eliminated]

    # Create adjacency matrix
    n = len(all_variables)
    adj_matrix = np.zeros((n, n), dtype=int)
    var_to_index = {var: i for i, var in enumerate(all_variables)}

    for factor in factors:
        for i, var1 in enumerate(factor.labels):
            for var2 in factor.labels[i+1:]:
                idx1, idx2 = var_to_index[var1], var_to_index[var2]
                adj_matrix[idx1, idx2] = adj_matrix[idx2, idx1] = 1

    # Initialize priority queue for variables to eliminate
    pq = [(_compute_fill_weight(adj_matrix, var_to_index[var]), var) 
            for var in variables_to_eliminate]
    pq.sort(reverse=True)  # Higher weight = higher priority to eliminate

    elimination_order = []
    while pq:
        _, var = pq.pop()
        elimination_order.append(var)

        # Update adjacency matrix
        var_idx = var_to_index[var]
        neighbors = np.where(adj_matrix[var_idx] == 1)[0]
        for i in neighbors:
            for j in neighbors:
                if i != j:
                    adj_matrix[i, j] = adj_matrix[j, i] = 1
        adj_matrix[var_idx, :] = adj_matrix[:, var_idx] = 0

        # Update priorities for remaining variables to eliminate
        pq = [(_compute_fill_weight(adj_matrix, var_to_index[v]), v) 
                for _, v in pq if v in variables_to_eliminate]
        pq.sort(reverse=True)

    # Add variables not eliminated to the end of the elimination order
    elimination_order.extend(variables_not_eliminated)
    
    return elimination_order

def _compute_fill_weight(adj_matrix, var_idx):
    """
    Compute the weighted min-fill score for a variable.
    """
    neighbors = np.where(adj_matrix[var_idx] == 1)[0]
    fill_edges = 0
    for i in neighbors:
        for j in neighbors:
            if i < j and adj_matrix[i, j] == 0:
                fill_edges += 1
    return fill_edges * len(neighbors)  # weight by cluster size
   