import torch, copy


def _get_backward_factors(gm, bucket_var, backward_factors=None):
    """
    Extract backward factors (downstream factors) for a bucket.

    If backward_factors is not provided, this function eliminates variables up to
    (but not including) bucket_var and collects all factors from downstream buckets.

    Args:
        gm: The graphical model
        bucket_var: The bucket variable
        backward_factors: Optional list of downstream factors. If None, will be computed.

    Returns:
        List of FastFactor objects representing the backward factors
    """
    if backward_factors is None:
        gm.eliminate_variables(up_to=bucket_var, exact=True)
        backward_factors = []
        for var in gm.elim_order[gm.elim_order.index(gm.matching_var(bucket_var))+1:]:
            bucket = gm.buckets[var]
            bucket_factors = bucket.factors
            for factor in bucket_factors:
                backward_factors.append(factor.to_exact())
    return backward_factors


def get_backward_message(gm, bucket_var, backward_factors=None, iB = 100, backward_ecl=None, approximation_method=None, return_factor_list=False):
    """
    Compute the backward message for a bucket.

    Args:
        gm: The graphical model
        bucket_var: The bucket variable
        backward_factors: Optional list of downstream factors
        iB: Mini-bucket i-bound
        backward_ecl: Backward message exact complexity limit
        approximation_method: Method for approximating backward message ('wmb', 'nn', etc.)
        return_factor_list: If True, returns list of factors instead of multiplied product
                           (used for batched learning to avoid materializing full product)

    Returns:
        If return_factor_list=False: (bw_msg, message) where bw_msg is a single FastFactor (product of all factors)
        If return_factor_list=True: (factor_list, message) where factor_list is a list of FastFactors
    """
    from nce.inference.graphical_model import FastGM
    from nce.inference.factor import FastFactor
    from nce.inference.elimination_order import wtminfill_order

    # Get backward factors (downstream factors)
    backward_factors = _get_backward_factors(gm, bucket_var, backward_factors)

    # Get bucket and scope
    bucket = gm.get_bucket(bucket_var)
    bucket_scope = bucket.get_message_scope()

    # Only compute exact forward message if complexity is low (< 2^20)
    # Otherwise, message can be too large and cause OOM
    bucket_ec = bucket.get_ec()
    if bucket_ec < 2**20:
        message = bucket.compute_message_exact()
    else:
        message = None
        # print(f"  Forward message complexity {bucket_ec} >= 2^20, skipping exact computation")

    # Handle edge cases - use 0-dim tensor for scalar factors (empty labels)
    if bucket_scope == []:
        return FastFactor(torch.tensor(0.0, device=gm.device, requires_grad=False), []), message
    if backward_factors == []:
        return FastFactor(torch.tensor(0.0, device=gm.device, requires_grad=False), []), message

    # Compute elimination order for downstream factors
    downstream_elim_order = wtminfill_order(backward_factors, variables_not_eliminated=bucket_scope)
    # print("deo is ", downstream_elim_order)
    # device_copy=str(gm.device)

    # Create a copy of the config for the downstream GM
    downstream_config = copy.deepcopy(gm.config)
    downstream_config['populate_bw_factors'] = False

    # Override approximation method if specified
    if approximation_method is not None:
        downstream_config['approximation_method'] = approximation_method
        print(f"Using approximation method '{approximation_method}' for downstream GM")

    # Override ecl in config BEFORE creating FastGM (critical fix!)
    if backward_ecl is not None:
        # Use explicitly specified backward_ecl
        downstream_config['ecl'] = backward_ecl
    else:
        # Default to forward ecl (so forward ecl automatically applies to backward)
        downstream_config['ecl'] = gm.config['ecl']

    # Compute effective iB from ecl (for binary variables: ecl = 2^iB)
    # This ensures WMB actually partitions when ecl is exceeded
    import math
    effective_iB_from_ecl = int(math.log2(downstream_config['ecl'])) if downstream_config['ecl'] > 0 else 0
    # Use min of provided iB and effective iB from ecl
    effective_iB = min(iB, effective_iB_from_ecl)
    print(f"  Backward ecl={downstream_config['ecl']}, effective iB from ecl: {effective_iB_from_ecl}, using min(iB={iB}, effective={effective_iB_from_ecl}) = {effective_iB}")

    # Override iB in config with the effective value
    downstream_config['iB'] = effective_iB

    downstream_gm = FastGM(factors=backward_factors, elim_order=downstream_elim_order, reference_fastgm=gm, device=gm.device, nn_config=downstream_config)
    downstream_gm.is_primary = False

    # Print backward computation info
    backward_induced_width = downstream_gm.get_max_width()

    # Calculate max table size (ec) across all buckets
    import numpy as np
    max_ec = 0
    for key in downstream_gm.message_scopes:
        mess_vars = downstream_gm.message_scopes[key]
        mess_size = int(np.prod([downstream_gm.matching_var(var).states for var in mess_vars]))
        if mess_size > max_ec:
            max_ec = mess_size

    print(f"  Backward computation: iB={effective_iB}, ecl={downstream_gm.ecl}, induced_width={backward_induced_width}, max_table_size={max_ec}")

    # Check if approximation will be used (need BOTH width <= iB AND ec <= ecl for exact)
    will_approximate = backward_induced_width > effective_iB or max_ec > downstream_gm.ecl
    if will_approximate:
        reasons = []
        if backward_induced_width > effective_iB:
            reasons.append(f"width {backward_induced_width} > iB {effective_iB}")
        if max_ec > downstream_gm.ecl:
            reasons.append(f"max_table_size {max_ec} > ecl {downstream_gm.ecl}")
        print(f"  -> Will use approximation ({', '.join(reasons)})")
    else:
        print(f"  -> Exact computation (width {backward_induced_width} <= iB {effective_iB} and max_table_size {max_ec} <= ecl {downstream_gm.ecl})")

    downstream_gm.eliminate_variables(all_but=bucket_scope)

    if return_factor_list:
        # Return list of factors for batched learning (avoids materializing full product)
        factor_list = downstream_gm.get_all_factors()
        if not factor_list:
            # Empty list case: return list with scalar factor (0-dim tensor)
            factor_list = [FastFactor(torch.tensor(0.0, device=gm.device, requires_grad=False), [])]
        # print(f"  Backward message: returning list of {len(factor_list)} factors (batched mode)")
        return factor_list, message
    else:
        # Return multiplied product (original behavior for backward compatibility)
        bw_msg = downstream_gm.get_joint_distribution()
        if bw_msg is None:
            bw_msg = FastFactor(torch.tensor(0.0, device=gm.device, requires_grad=False), [])

        # Debug: print first 10 values of flattened approximate backward message
        flat_bw = bw_msg.tensor.flatten()
        # print(f"  Backward message first 10 values: {flat_bw[:min(10, len(flat_bw))].tolist()}")

        return bw_msg, message


# Alias for backward compatibility
def get_message_gradient(gm, bucket_var, backward_factors=None, iB=100, backward_ecl=None, approximation_method=None, return_factor_list=False):
    """
    Alias for get_backward_message() for backward compatibility.

    This function name is kept for legacy code that uses 'message gradient' terminology.
    New code should use get_backward_message() instead.

    Note: The parameter 'backward_factors' was previously called 'gradient_factors'.
    """
    return get_backward_message(gm, bucket_var, backward_factors, iB, backward_ecl, approximation_method, return_factor_list)
