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

def populate_gradient_factors(base_gm: FastGM, iB: int) -> None:
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
        if len(elimination_factors[0].labels) <= iB:
            # Compute exact message if scope is below or equal to iB
            message = copied_gm.process_bucket(copied_bucket)
        else:
            # Use WMB if scope is above iB
            message = copied_gm.copied_bucket.compute_wmb_message(iB)

        # Combine independent factors and message(s)
        factors_to_move = independent_factors + [message]

        # Move factors to the next relevant bucket
        for factor in factors_to_move:
            next_var = copied_gm.find_next_bucket(factor.labels, current_var)
            if next_var:
                next_bucket = copied_gm.get_bucket(next_var)
                next_bucket.factors.append(factor)

        # Remove processed factors from the current bucket
        copied_bucket.factors.clear()
