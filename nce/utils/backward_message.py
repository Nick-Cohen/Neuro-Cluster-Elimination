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


def get_backward_message(gm, bucket_var, backward_factors=None, iB = 100, backward_ecl=None, approximation_method=None, return_factor_list=False, return_partitions=False):
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
        return_partitions: If True, also returns the number of WMB partitions used

    Returns:
        If return_factor_list=False: (bw_msg, message) or (bw_msg, message, bw_partitions)
        If return_factor_list=True: (factor_list, message) or (factor_list, message, bw_partitions)
        The third element (bw_partitions) is only included if return_partitions=True
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

    # Handle edge cases - use 0-dim tensor for scalar factors (empty labels)
    if bucket_scope == []:
        result = (FastFactor(torch.tensor(0.0, device=gm.device, requires_grad=False), []), message)
        return result + (0,) if return_partitions else result
    if backward_factors == []:
        result = (FastFactor(torch.tensor(0.0, device=gm.device, requires_grad=False), []), message)
        return result + (0,) if return_partitions else result

    # Separate scalar factors (empty labels) from non-scalar factors.
    # Scalar factors are constants in logspace - we need to accumulate them
    # and add back to the final result.
    scalar_factors = [f for f in backward_factors if not f.labels]
    backward_factors_filtered = [f for f in backward_factors if f.labels]

    # Accumulate scalar constants (sum in logspace = product in linear space)
    scalar_constant = torch.tensor(0.0, device=gm.device, requires_grad=False)
    for sf in scalar_factors:
        scalar_constant = scalar_constant + sf.tensor.squeeze()

    # If all factors were scalar, return the accumulated constant as the backward message
    if not backward_factors_filtered:
        result = (FastFactor(scalar_constant, []), message)
        return result + (0,) if return_partitions else result

    # Compute elimination order for downstream factors
    downstream_elim_order = wtminfill_order(backward_factors_filtered, variables_not_eliminated=bucket_scope)

    # Create a copy of the config for the downstream GM
    downstream_config = copy.deepcopy(gm.config)
    downstream_config['populate_bw_factors'] = False

    # Override approximation method if specified
    if approximation_method is not None:
        downstream_config['approximation_method'] = approximation_method

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

    # Override iB in config with the effective value
    downstream_config['iB'] = effective_iB

    downstream_gm = FastGM(factors=backward_factors_filtered, elim_order=downstream_elim_order, reference_fastgm=gm, device=gm.device, nn_config=downstream_config)
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

    # Check if approximation will be used (need BOTH width <= iB AND ec <= ecl for exact)
    will_approximate = backward_induced_width > effective_iB or max_ec > downstream_gm.ecl
    if will_approximate:
        reasons = []
        if backward_induced_width > effective_iB:
            reasons.append(f"width {backward_induced_width} > iB {effective_iB}")
        if max_ec > downstream_gm.ecl:
            reasons.append(f"max_table_size {max_ec} > ecl {downstream_gm.ecl}")

    elim_result = downstream_gm.eliminate_variables(all_but=bucket_scope)

    # Capture any scalar constants accumulated during elimination (from root_bucket)
    # These are scalars produced when WMB mini-buckets eliminate to scalars
    elim_scalar = torch.tensor(0.0, device=gm.device, requires_grad=False)
    if elim_result is not None and not elim_result.labels:
        # elim_result is a scalar factor - add its value to our accumulated constant
        elim_scalar = elim_result.tensor.squeeze()

    # Total scalar constant = input scalars + elimination scalars
    total_scalar_constant = scalar_constant + elim_scalar

    # Track backward pass partitions
    bw_partitions = downstream_gm.wmb_fw_partitions  # Partitions during backward message computation

    if return_factor_list:
        # Return list of factors for batched learning (avoids materializing full product)
        factor_list = downstream_gm.get_all_factors()
        if not factor_list:
            # Empty list case: return list with scalar factor containing the accumulated constant
            factor_list = [FastFactor(total_scalar_constant, [])]
        else:
            # Add scalar constant to the first factor in the list (logspace addition)
            if total_scalar_constant != 0.0:
                first_factor = factor_list[0]
                factor_list[0] = FastFactor(first_factor.tensor + total_scalar_constant, first_factor.labels)
        result = (factor_list, message)
        return result + (bw_partitions,) if return_partitions else result
    else:
        # Return multiplied product (original behavior for backward compatibility)
        bw_msg = downstream_gm.get_joint_distribution()
        if bw_msg is None:
            bw_msg = FastFactor(total_scalar_constant, [])
        else:
            # Add scalar constant to the backward message (logspace addition)
            if total_scalar_constant != 0.0:
                bw_msg = FastFactor(bw_msg.tensor + total_scalar_constant, bw_msg.labels)

        result = (bw_msg, message)
        return result + (bw_partitions,) if return_partitions else result


# Alias for backward compatibility
def get_message_gradient(gm, bucket_var, backward_factors=None, iB=100, backward_ecl=None, approximation_method=None, return_factor_list=False, return_partitions=False):
    """
    Alias for get_backward_message() for backward compatibility.

    This function name is kept for legacy code that uses 'message gradient' terminology.
    New code should use get_backward_message() instead.

    Note: The parameter 'backward_factors' was previously called 'gradient_factors'.
    """
    return get_backward_message(gm, bucket_var, backward_factors, iB, backward_ecl, approximation_method, return_factor_list, return_partitions)
