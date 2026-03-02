"""
Utility function to compute backward message sensitivity for a graphical model.

Sensitivity = lse(f+b_wmb) - lse(b_wmb) - lse(f+b) + lse(b)

Where:
- f = exact forward message
- b = exact backward message
- b_wmb = WMB approximate backward message at given bw_ecl

IMPORTANT: Uses populate_bw_factors=True and bucket.approximate_downstream_factors
"""

import torch
import numpy as np
from nce.inference.graphical_model import FastGM
from nce.inference.factor import FastFactor
from nce.utils.backward_message import get_backward_message


def compute_bucket_sensitivity(forward_message, backward_wmb, backward_exact):
    """
    Compute sensitivity for a single bucket.

    Args:
        forward_message: Exact forward message (FastFactor)
        backward_wmb: WMB approximate backward message (FastFactor)
        backward_exact: Exact backward message (FastFactor)

    Returns:
        float: Sensitivity value = lse(f+b_wmb) - lse(b_wmb) - lse(f+b) + lse(b)
    """
    f_bw_wmb = forward_message * backward_wmb
    f_bw_exact = forward_message * backward_exact

    term1 = f_bw_wmb.sum_all_entries()
    term2 = backward_wmb.sum_all_entries()
    term3 = f_bw_exact.sum_all_entries()
    term4 = backward_exact.sum_all_entries()

    sensitivity = term1 - term2 - term3 + term4
    return float(sensitivity)


def compute_sensitivity_sweep(
    model,
    bw_ecl_values=None,
    exact_ecl=2**35,
    device='cuda',
    zero_threshold=1e-4,
    plateau_threshold=1e-5,
    plateau_count=3,
    verbose=False
):
    """
    Compute backward sensitivity for all buckets across multiple bw_ecl values.

    Uses populate_bw_factors=True and bucket.approximate_downstream_factors.

    Args:
        model: Model object with order, evidence, etc.
        bw_ecl_values: List of bw_ecl values to test (default: 0, 2^1, ..., 2^30)
        exact_ecl: ECL for exact computation (default: 2^35)
        device: Torch device
        zero_threshold: Consider values below this as zero
        plateau_threshold: Consider values converged if change is below this
        plateau_count: Number of consecutive plateau values to trigger early exit
        verbose: Print progress

    Returns:
        dict: {bucket_label: {bw_ecl: sensitivity}}
    """
    if bw_ecl_values is None:
        bw_ecl_values = [0] + [2**i for i in range(1, 31)]

    results = {}

    for bw_ecl in bw_ecl_values:
        bw_iB = 0 if bw_ecl == 0 else int(np.log2(bw_ecl))

        if verbose:
            print(f"\n  Testing bw_ecl={bw_ecl} (bw_iB={bw_iB})...")

        # Config with this bw_ecl and populate_bw_factors=True
        config = {
            'device': device,
            'ecl': exact_ecl,
            'iB': 100,
            'approximation_method': 'wmb',
            'populate_bw_factors': True,  # This populates approximate_downstream_factors
            'loss_fn': 'mse',
            'bw_ecl': bw_ecl,
        }

        # Create GM with this config
        gm = FastGM(model=model, nn_config=config)
        elim_order = list(gm.elim_order)

        # Process buckets in elimination order
        for bucket_idx, var in enumerate(elim_order):
            bucket_label = var

            # Initialize results dict for this bucket
            if bucket_label not in results:
                results[bucket_label] = {}

            # Skip if already converged for this bucket
            if results[bucket_label].get('_converged', False):
                # Fill with converged value
                results[bucket_label][bw_ecl] = results[bucket_label].get('_final_value', 0.0)
                continue

            try:
                # Eliminate up to this bucket
                gm.eliminate_variables(up_to=bucket_label, exact=True)
                bucket = gm.get_bucket(bucket_label)

                # Compute exact forward message
                exact_forward = bucket.compute_message_exact()

                # Check if scalar bucket
                msg_scope = bucket.get_message_scope()
                if not msg_scope:
                    continue

                # Compute exact backward message (high ecl)
                exact_backward, _ = get_backward_message(
                    gm, bucket_label,
                    backward_factors=None,  # None = compute fresh
                    iB=2**35,
                    backward_ecl=exact_ecl,
                    approximation_method='wmb',
                    return_factor_list=False
                )

                # Compute WMB backward message using pre-computed factors
                if bw_ecl == 0:
                    shape = [gm.matching_var(v).states for v in msg_scope]
                    backward_wmb = FastFactor(torch.zeros(shape, device=device), msg_scope)
                else:
                    backward_wmb, _ = get_backward_message(
                        gm, bucket_label,
                        backward_factors=bucket.approximate_downstream_factors,  # Use pre-computed
                        iB=2**35,
                        backward_ecl=bw_ecl,
                        approximation_method='wmb',
                        return_factor_list=False
                    )

                # Compute sensitivity
                sensitivity = compute_bucket_sensitivity(exact_forward, backward_wmb, exact_backward)
                results[bucket_label][bw_ecl] = sensitivity

                # Check for convergence
                if abs(sensitivity) < zero_threshold:
                    results[bucket_label]['_converged'] = True
                    results[bucket_label]['_final_value'] = 0.0
                    if verbose:
                        print(f"    Bucket {bucket_label}: converged at bw_iB={bw_iB}")

            except Exception as e:
                if verbose:
                    print(f"    Bucket {bucket_label} ERROR: {e}")
                results[bucket_label][bw_ecl] = None

    # Clean up internal tracking keys
    for bucket_label in results:
        results[bucket_label].pop('_converged', None)
        results[bucket_label].pop('_final_value', None)

    return results


def compute_sensitivity_sweep_efficient(
    model,
    bw_ecl_values=None,
    exact_ecl=2**35,
    device='cuda',
    zero_threshold=1e-4,
    plateau_threshold=1e-5,
    plateau_count=3,
    verbose=False
):
    """
    Compute backward sensitivity efficiently - process each bucket once through all bw_ecl values.

    Args:
        model: Model object
        bw_ecl_values: List of bw_ecl values to test
        exact_ecl: ECL for exact computation
        device: Torch device
        zero_threshold: Consider values below this as zero
        plateau_threshold: Consider converged if change below this
        plateau_count: Consecutive plateau values to trigger early exit
        verbose: Print progress

    Returns:
        dict: {bucket_label: {bw_ecl: sensitivity}}
    """
    if bw_ecl_values is None:
        bw_ecl_values = [0] + [2**i for i in range(1, 31)]

    # Get elimination order from a reference GM
    config_ref = {
        'device': device,
        'ecl': exact_ecl,
        'iB': 100,
        'approximation_method': 'wmb',
        'populate_bw_factors': True,
        'loss_fn': 'mse',
        'bw_ecl': exact_ecl,
    }
    gm_ref = FastGM(model=model, nn_config=config_ref)
    elim_order = list(gm_ref.elim_order)
    n_buckets = len(elim_order)

    results = {}

    for bucket_idx, var in enumerate(elim_order):
        bucket_label = var

        if verbose:
            print(f"  [{bucket_idx+1}/{n_buckets}] Processing bucket {bucket_label}...")

        results[bucket_label] = {}

        # Test each bw_ecl value with early exit
        found_zero = False
        converged_at = None
        plateau_count_local = 0
        prev_sensitivity = None
        final_plateau_value = None

        for bw_ecl in bw_ecl_values:
            if found_zero:
                results[bucket_label][bw_ecl] = final_plateau_value if final_plateau_value is not None else 0.0
                continue

            bw_iB = 0 if bw_ecl == 0 else int(np.log2(bw_ecl))

            try:
                # Create GM with this bw_ecl
                config = {
                    'device': device,
                    'ecl': exact_ecl,
                    'iB': 100,
                    'approximation_method': 'wmb',
                    'populate_bw_factors': True,
                    'loss_fn': 'mse',
                    'bw_ecl': bw_ecl,
                }
                gm = FastGM(model=model, nn_config=config)
                gm.eliminate_variables(up_to=bucket_label, exact=True)
                bucket = gm.get_bucket(bucket_label)

                # Compute exact forward message
                exact_forward = bucket.compute_message_exact()

                # Check if scalar bucket
                msg_scope = bucket.get_message_scope()
                if not msg_scope:
                    if verbose:
                        print(f"    Scalar bucket, skipping")
                    break  # Skip all bw_ecl values for this bucket

                # Compute exact backward message
                exact_backward, _ = get_backward_message(
                    gm, bucket_label,
                    backward_factors=None,
                    iB=2**35,
                    backward_ecl=exact_ecl,
                    approximation_method='wmb',
                    return_factor_list=False
                )

                # Compute WMB backward message
                if bw_ecl == 0:
                    shape = [gm.matching_var(v).states for v in msg_scope]
                    backward_wmb = FastFactor(torch.zeros(shape, device=device), msg_scope)
                else:
                    backward_wmb, _ = get_backward_message(
                        gm, bucket_label,
                        backward_factors=bucket.approximate_downstream_factors,
                        iB=2**35,
                        backward_ecl=bw_ecl,
                        approximation_method='wmb',
                        return_factor_list=False
                    )

                sensitivity = compute_bucket_sensitivity(exact_forward, backward_wmb, exact_backward)

                # Check for convergence
                if abs(sensitivity) < zero_threshold:
                    sensitivity = 0.0
                    found_zero = True
                    converged_at = bw_iB
                    final_plateau_value = 0.0
                elif prev_sensitivity is not None and abs(sensitivity - prev_sensitivity) < plateau_threshold:
                    plateau_count_local += 1
                    if plateau_count_local >= plateau_count:
                        found_zero = True
                        converged_at = bw_iB - plateau_count
                        final_plateau_value = sensitivity
                else:
                    plateau_count_local = 0

                results[bucket_label][bw_ecl] = sensitivity
                prev_sensitivity = sensitivity

            except Exception as e:
                if verbose:
                    print(f"    ERROR at bw_ecl={bw_ecl}: {e}")
                results[bucket_label][bw_ecl] = None

        if verbose:
            if converged_at is not None:
                print(f"    Converged at bw_iB={converged_at}")
            elif prev_sensitivity is not None:
                print(f"    Did not converge (last={prev_sensitivity:.6f})")

    return results


def compute_total_sensitivity(model, bw_ecl, exact_ecl=2**35, device='cuda', verbose=False):
    """
    Compute total backward sensitivity (sum across all buckets) at a given bw_ecl.
    """
    results = compute_sensitivity_sweep_efficient(
        model,
        bw_ecl_values=[bw_ecl],
        exact_ecl=exact_ecl,
        device=device,
        verbose=verbose
    )

    total = 0.0
    for bucket_label, bucket_sens in results.items():
        sens = bucket_sens.get(bw_ecl)
        if sens is not None:
            total += abs(sens)

    return total
