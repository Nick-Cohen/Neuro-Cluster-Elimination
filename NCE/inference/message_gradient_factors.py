from .factor import FastFactor
from .bucket import FastBucket
from .graphical_model import FastGM
from .elimination_order import wtminfill_order
import pyGMs as gm
from pyGMs import wmb
from pyGMs.neuro import *
from pyGMs.graphmodel import eliminationOrder
from pyGMs import Var
import torch
from typing import List
import copy

"""
Input: graphical model, iB
Output: None

The function will very closely mirror how get_wmb_message_gradient works but with some subtle differences.

The function will populate all of the original graphical model's buckets approximate_mg_factors datafields.

It will proceed as follows:

It will make a copy of the base graphical model (it is given before any elimination is done by assumption).

The function will process through the buckets in elimination order.

For each bucket, as it is processed, first, all downstream FastFactors (every fast factor from every bucket in the copied FastGM) are written to the original FastGM's (not the copy) bucket's datafield of approximate downstream factors. Then, the bucket is examined for factors which do not have the elimination variable in it. These are set aside into a list called independent_factors.

The message for the bucket is computed from the remaining ,non-set-aside, factors. It is computed exactly if the scope is below or equal to the iB and with wmb if it is above the iB. Now, all minibucket messages as well as independent, set-aside factors are moved together to the next bucket in the elimination order whose elimination variable is in any of the independent factors or minibucket messages.

The function will proceed to the end of the elimination order.
"""

def populate_gradient_factors_old(base_gm: FastGM, iB: int) -> None:
    # Make a copy of the base graphical model
    copied_gm = copy.deepcopy(base_gm)

    # Process buckets in elimination order
    for i,current_var in enumerate(base_gm.elim_order):
        current_bucket = base_gm.get_bucket(current_var)
        copied_bucket = copied_gm.get_bucket(current_var)

        # Collect all downstream factors from the copied GM that exist in buckets after the current bucket in the elimination order
        all_downstream_factors = []
        for j in range(i+1, len(base_gm.elim_order)):
            downstream_bucket = copied_gm.get_bucket(base_gm.elim_order[j])
            for factor in downstream_bucket.factors:
                all_downstream_factors.append(factor)

        # Set the approximate_downstream_factors for the current bucket in the original GM
        current_bucket.approximate_downstream_factors = all_downstream_factors

        # Separate factors that don't include the elimination variable
        independent_factors = []
        elimination_factors = []
        for factor in copied_bucket.factors:
            if current_var not in factor.labels:
                independent_factors.append(factor)
            else:
                elimination_factors.append(factor)

        # Compute the message
        if copied_bucket.get_width() <= iB:
            # Compute exact message if scope is below or equal to iB
            messages =[copied_gm.process_bucket(copied_bucket)]
        else:
            # Use WMB if scope is above iB
            messages = copied_bucket.compute_wmb_message(iB)
            
        # Combine independent factors and message(s)
        factors_to_move = independent_factors + messages

        # Move factors to the next relevant bucket
        for factor in factors_to_move:
            next_var = copied_gm.find_next_bucket(factor.labels, current_var)
            if next_var:
                next_bucket = copied_gm.get_bucket(next_var)
                next_bucket.factors.append(factor)

        # Remove processed factors from the current bucket
        copied_bucket.factors.clear()

def populate_gradient_factors(base_gm: FastGM, iB: int) -> None:
    # Make a copy of the base graphical model
    copied_gm = copy.deepcopy(base_gm)

    scheme_info = copied_gm.get_senders_receivers()
    scheme = {info['var']: info['sends_to'] for info in scheme_info}
    # Process buckets in elimination order
    # for each bucket:
    for i,current_var in enumerate(base_gm.elim_order):
        # gather downstream factors and populate the base_gms's bucket's approximate_downstream_factors
        current_bucket = base_gm.get_bucket(current_var)
        approximate_bucket = copied_gm.get_bucket(current_var)
        current_bucket.approximate_downstream_factors = get_downstream_factors(copied_gm, current_var)
        # separate factors into has_summation_var and does_not_have_summation_var sets
        has_summation_var = [factor for factor in approximate_bucket.factors if current_var in factor.labels]
        does_not_have_summation_var = [factor for factor in approximate_bucket.factors if current_var not in factor.labels]
        # send wmb message based on has_summation_var set factors to the bucket indicated by the scheme
        # first redefine factors to make the width-checking and wmb function work
        approximate_bucket.factors = has_summation_var
        # Compute the message
        if approximate_bucket.get_width() <= iB:
            # Compute exact message if scope is below or equal to iB
            # messages = [copied_gm.process_bucket(approximate_bucket)]
            messages = [approximate_bucket.compute_message_exact()]
        else:
            # Use WMB if scope is above iB
            messages = approximate_bucket.compute_wmb_message(iB)
        for message in messages:
            assert message.tensor is not None, f"{current_var}"
        # also accumulate each factor in the does_not_have_summation_var set to the bucket indicated by the scheme
        messages += does_not_have_summation_var
        # send to the correct bucket in copied gm
        if scheme[current_var] is not None:
            next_bucket = copied_gm.get_bucket(scheme[current_var])
            next_bucket.factors += messages

def get_downstream_factors(gm: FastGM, bucket_var: Var) -> List[FastFactor]:
    downstream_factors = []
    for i in range(gm.elim_order.index(bucket_var)+1, len(gm.elim_order)):
        downstream_bucket = gm.get_bucket(gm.elim_order[i])
        for factor in downstream_bucket.factors:
            downstream_factors.append(factor)
    return downstream_factors

def get_wmb_message_gradient_factors(factors: List[FastFactor], message_scope, config) -> List[FastFactor]:
    # determine the elimination order
    # i) Put scope vars at end of elim order, all other = elim vars
    # ii) Use min fill heuristic to order elim vars
    if not factors:
        return []
    elim_order = []
    vars_in_grad_factors = set()
    for f in factors:
        for label in f.labels:
            vars_in_grad_factors.add(label)
    variables_not_eliminated = [var for var in message_scope if var in vars_in_grad_factors]
    elim_order = wtminfill_order(factors, variables_not_eliminated=variables_not_eliminated)

        
        
    # Create a graphical model with the factors
    gm = FastGM(factors=factors, nn_config=config, elim_order=elim_order)
    # Check if factors is empty
    # if there are no gradient factors return a scalar factor
    if len(factors) == 0:
        return FastFactor(tensor=torch.tensor([0.0], device=config['device']), labels=[])
    # use wmb elimination to eliminate all the lim vars
    mg_hat_factors = FastGM._wmb_eliminate(gm, target_scope=message_scope, i_bound=config['iB'], weights='max', combine_factors=False)
    # return all remaining factors
    for factor in mg_hat_factors:
        for label in factor.labels:
            assert label in message_scope
    return mg_hat_factors