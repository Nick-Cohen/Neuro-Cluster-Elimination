from .factor import FastFactor
from .bucket import FastBucket
from .elimination_order import wtminfill_order
from nce.utils.stats import get_message_stats
import pyGMs as gm
from pyGMs import wmb
from pyGMs.neuro import *
from pyGMs.graphmodel import eliminationOrder
from pyGMs.filetypes import readEvidence14
from pyGMs import Var
import torch
import numpy as np
import math
import os
from typing import List
from tqdm.notebook import tqdm


class FastGM:
    def __init__(self, elim_order=None, buckets=None, factors=None, uai_file=None, device="cuda", reference_fastgm=None, nn_config=None, stats=None):
        self.iB = nn_config['iB']
        self.ecl = nn_config['ecl']
        self.num_trained = 0
        self.uai_file = uai_file
        self.device = device
        self.vars = []
        self.elim_order = None
        self.config = nn_config
        self.traced_losses_data = []
        self.message_stats = []
        self.stats=stats # cheater stats with variances
        self.is_primary = True
        if nn_config is not None:
            self.sampling_scheme = nn_config['sampling_scheme']
            self.traced_losses = nn_config['traced_losses']
            self.lower_dim = nn_config['lower_dim']
            self.hidden_sizes = nn_config['hidden_sizes']
            self.loss_fn = nn_config['loss_fn']
            self.optimizer = nn_config['optimizer']
            self.lr = nn_config['lr']
            self.lr_decay = nn_config['lr_decay']
            self.patience = nn_config['patience']
            self.min_lr = nn_config['min_lr']
            self.num_samples = nn_config['num_samples']
            self.num_epochs = nn_config['num_epochs']
            self.batch_size = nn_config['batch_size']
            self.set_size = nn_config['set_size']
            self.seed = nn_config['seed']
            self.gather_message_stats = nn_config['gather_message_stats']
            

        if uai_file is not None:
            self._load_from_uai(uai_file)
        elif buckets is not None:
            self.buckets = buckets
            self.elim_order = elim_order
        elif factors is not None:
            self._load_vars_from_factors(factors)
            if elim_order is not None:
                self.load_elim_order(elim_order, reference_fastgm)
                # print(self.elim_order)
            else:
                print("Computing elim order")
                self.load_elim_order(wtminfill_order(factors), reference_fastgm)
            self.buckets = self._create_buckets_from_factors(factors)
        else:
            raise ValueError("Either buckets, factors, or a UAI file must be provided")
        
        self.message_scopes = {}
        self.calculate_message_scopes()
        if self.config.get('dope_factors'):
            self.dope_factors()
        if self.config.get('sigma_g_global') is None and ('approx_smg' in self.loss_fn or 'approx_smg2' in self.config.get('loss_fn2', '')):
            self.config['sigma_g_global'] = -1
            self.populate_global_stats()

    def __repr__(self):
        return f"FastGM(elim_order={self.elim_order}, buckets={self.buckets})"

    def __str__(self):
        output = []
        for var in self.elim_order:
            bucket = self.buckets[var]
            factor_strs = []
            for factor in bucket.factors:
                factor_strs.append(f"f({', '.join(map(str, factor.labels))})")
            output.append(f"Bucket {var}: {' '.join(factor_strs)}")
        return "\n".join(output)
    
    def _load_from_uai(self, uai_file):
        import os
        # Load the UAI file
        ord_file = uai_file + ".vo"
        evid_file = uai_file + ".evid"
        #check if evid file exists
        gm_model = uai_to_GM(uai_file=uai_file,     order_file=ord_file)
        if os.path.exists(evid_file):
            evid = readEvidence14(evid_file)
            gm_model.condition(evid)
 
        
        self.vars = gm_model.vars
        
        # Convert PyGM factors to FastFactors
        fast_factors = []
        for factor in gm_model.factors:
            tensor = torch.tensor(factor.table, dtype=torch.float32).to(self.device)
            labels = [var.label for var in factor.vars]
            fast_factors.append(FastFactor(torch.log10(tensor), labels))
        
        # Set elimination order
        self.elim_order = gm_model.elim_order
        
        # Create buckets from FastFactors
        self.buckets = self._create_buckets_from_factors(fast_factors)

    def _load_vars_from_factors(self, factors):
        var_domains = {}
        for factor in factors:
            # print(factor.labels)
            if factor.labels:
                for i, label in enumerate(factor.labels):
                    if label not in var_domains:
                        var_domains[label] = factor.tensor.shape[i]
        for label, domain_size in var_domains.items():
            self.vars.append(Var(label, domain_size))

    def _create_buckets_from_factors(self, factors):
        if self.elim_order is None:
            raise ValueError("Elimination order must be set before creating buckets")
        
        # print(self.elim_order)
        buckets = {var: FastBucket(self, var.label, [], self.device, [var]) for var in self.elim_order}
        unplaced_factors = set(factors)

        for var in self.elim_order:
            factors_to_place = []
            for factor in unplaced_factors:
                if var in factor.labels:
                    factors_to_place.append(factor)
            
            for factor in factors_to_place:
                buckets[var].factors.append(factor)
                unplaced_factors.remove(factor)

        if unplaced_factors:
            raise ValueError(f"Some factors could not be placed in buckets: {unplaced_factors}")

        return buckets 

    def get_factors(self):
        factors = []
        for bucket in list(self.buckets.values()):
            for factor in bucket.factors:
                factors.append(factor)
        return factors

    def get_bucket(self, bucket):
        if type(bucket) != int:
            bucket_id = bucket.label
        else:
            bucket_id = bucket
        return self.buckets[self.matching_var(bucket_id)]

    def matching_var(self, var_index):
        for var in self.vars:
            if var_index == var.label:
                return var
    
    def load_elim_order(self, elim_order, reference_fastgm=None): 
        if reference_fastgm is not None:
            self.elim_order = [reference_fastgm.matching_var(var_index) for var_index in elim_order]
        elif isinstance(elim_order[0], Var):
            self.elim_order = elim_order
        elif isinstance(elim_order[0], int):
            self.elim_order = []
            for var_index in elim_order:
                if self.matching_var(var_index):
                    self.elim_order.append(self.matching_var(var_index))
                else:
                    raise ValueError("No matching var found for idx ", var_index)
                    
            # self.elim_order = [self.matching_var(var_index) for var_index in elim_order]
        else:
            raise ValueError("Elimination order must be a list of Var objects or integers")
        # print(self.elim_order)
            
    def eliminate_variables(self, elim_vars=None, up_to=None, through=None, all=False, all_but=None, exact=False):
        if sum(map(bool, [elim_vars, up_to is not None, through is not None, all, all_but])) != 1:
            raise ValueError("Exactly one of elim_vars, up_to, through, all_but, or all must be specified")

        if all:
            vars_to_eliminate = self.elim_order
        elif elim_vars:
            vars_to_eliminate = elim_vars
        elif up_to is not None:
            vars_to_eliminate = self.elim_order[:self.elim_order.index(up_to)]
        elif through is not None:
            vars_to_eliminate = self.elim_order[:self.elim_order.index(through) + 1]
        elif all_but:
            vars_to_keep = set(all_but)
            vars_to_eliminate = [var for var in self.elim_order if var not in vars_to_keep]
            
        # remove variables already eliminated from vars_to_eliminate
        vars_to_eliminate = [var for var in vars_to_eliminate if var in self.buckets]
        
        # Create a dummy root bucket to collect the final result
        root_bucket = FastBucket(self, 'root', [], self.device, [], isRoot=True)
        
        max_width = 0
        
        for key in self.message_scopes:
            mess_vars = self.message_scopes[key]
            mess_size = int(np.prod([self.matching_var(var).states for var in mess_vars]))
            if key in vars_to_eliminate and (len(mess_vars) > self.iB or mess_size > self.ecl):
                self.num_trained += 1
        with tqdm(total=self.num_trained, desc="Num NNs to train") as pbar:
        # if True:
            for var in vars_to_eliminate:
                # if type(var) != int:
                #     var = var.label
                current_bucket = self.buckets[var]
                
                # debug test
                # print(var)
                # fs = self.get_bucket(59).factors
                # fails_test = False
                # for f in fs:
                #     if f.tensor is None:
                #         fails_test = True
                #         print("fails")
                
                # print(current_bucket.label)
                message = self.process_bucket(current_bucket, exact=exact)
                if not message.is_nn:
                    assert message.tensor is not None
                else:
                    # pass
                    pbar.update(1)
                if message.labels:  # If the message is not a scalar
                    # Find the next appropriate bucket
                    if len(message.labels) > max_width:
                        max_width = len(message.labels)
                    next_bucket = self.find_next_bucket(message.labels, var)
                    if next_bucket:
                        next_bucket.receive_message(message)
                    else:
                        # If no appropriate bucket found, send to root
                        root_bucket.receive_message(message)
                else:
                    # If the message is a scalar, send to root
                    root_bucket.receive_message(message)
                
                # Remove the eliminated variable's bucket
                del self.buckets[var]

        # Process the root bucket
        if root_bucket.factors:
            result = root_bucket.factors[0]
            for factor in root_bucket.factors[1:]:
                result = result * factor
        else:
            # If no factors were sent to the root, return an identity factor
            result = FastFactor(torch.tensor(0.0, device=self.device).reshape(1), [])

        # If we've eliminated all variables, save the partition function
        if all:
            assert len(result.labels) == 0, "Not all variables were eliminated"
            self.log_partition_function = result.tensor.item()
        return result

    def eliminate_max(self, elim_labels):
        if elim_labels == 'all':
            elim_indices = list(range(len(self.labels)))
            new_labels = []
        else:
            elim_indices = [self.labels.index(label) for label in elim_labels]
            new_labels = [label for label in self.labels if label not in elim_labels]
        
        result_tensor, _ = torch.max(self.tensor, dim=elim_indices[0])
        for dim in elim_indices[1:]:
            result_tensor, _ = torch.max(result_tensor, dim=dim-len(elim_indices)+1)
        
        if type(result_tensor) == float:
            result_tensor = torch.Tensor([result_tensor])
        
        # If we've eliminated all variables, we need to ensure the result is a scalar
        if not new_labels:
            result_tensor = result_tensor.view(1)
        
        return FastFactor(result_tensor, new_labels)
    
    def process_bucket(self, bucket, exact=False):
        """
        Process bucket with support for linear solver options.
        """
        print("Processing bucket: ", bucket.label)
        
        # Check if we should use exact computation based on width
        if exact or (bucket.get_width() <= self.iB and bucket.get_ec() <= self.ecl):
            output_message = bucket.compute_message_exact()
            if self.gather_message_stats:
                get_message_stats(self, bucket, output_message)
            # begin debug
            if output_message.tensor.isnan().any():
                raise ValueError(f"Output message for bucket {bucket.label} contains NaN values")
            return output_message
        else:
            # Check if we should use linear solver
            use_linear_solver = bucket.config.get('use_linear_solver', False)
            if self.config.get('approximation_method') == 'nn':
                print(f"Training NN for bucket: {bucket.label}")
                output_message = bucket.compute_message_nn()
            elif self.config.get('approximation_method') == 'dt':
                print(f"Using decision tree for bucket: {bucket.label}")
                output_message = bucket.compute_message_dt()
            return output_message
    
    def calculate_message_scopes(self):
        """Calculate and save the message scope for each bucket."""
        elimination_scheme = []

        for var in self.elim_order:
            bucket_factors = self.buckets[var].factors
            incoming_messages = []
            outgoing_message_vars = set()

            # Determine incoming messages from previous buckets
            for prev_bucket in elimination_scheme:
                if prev_bucket['sends_to'] is not None and prev_bucket['sends_to'].label == var.label:
                    incoming_messages.append(prev_bucket['outgoing_message'])
                    outgoing_message_vars.update(prev_bucket['outgoing_message'])

            # Add variables from the bucket's own factors
            for factor in bucket_factors:
                outgoing_message_vars.update(factor.labels)

            # Remove the bucket's own variable
            outgoing_message_vars.discard(var.label)

            # Save the scope for this bucket
            self.message_scopes[var.label] = sorted(list(outgoing_message_vars))

            # Find the next bucket to send the message to
            next_var = self.find_next_bucket(list(outgoing_message_vars), var)

            bucket_info = {
                'var': var,
                'sends_to': next_var,
                'outgoing_message': sorted(list(outgoing_message_vars))
            }

            elimination_scheme.append(bucket_info)

    def get_max_width(self):
        m = 0
        for key in self.message_scopes.keys():
            if len(self.message_scopes[key]) > m:
                m = len(self.message_scopes[key])
        return m

    def show_message_scopes(self):
        """Display the calculated bucket scopes."""
        print("Bucket Scopes:")
        for var in self.elim_order:
            print(f"Bucket {var}: {self.message_scopes[var.label]}")
    
    def show_elimination(self, elim_vars=None, up_to=None, through=None, all=False):
        if sum(map(bool, [elim_vars, up_to, through, all])) != 1:
            raise ValueError("Exactly one of elim_vars, up_to, through, or all must be specified")

        if all:
            vars_to_eliminate = self.elim_order
        elif elim_vars:
            vars_to_eliminate = elim_vars
        elif up_to:
            vars_to_eliminate = self.elim_order[:self.elim_order.index(up_to)]
        elif through:
            vars_to_eliminate = self.elim_order[:self.elim_order.index(through) + 1]

        elimination_scheme = []
        max_width = 0

        for var in self.elim_order:
            if var not in vars_to_eliminate:
                continue
            
            bucket_factors = self.buckets[var].factors
            incoming_messages = []
            outgoing_message_vars = set()

            # Determine incoming messages from previous buckets
            for prev_bucket in elimination_scheme:
                if prev_bucket['sends_to'] is not None and prev_bucket['sends_to'].label == var.label:
                    incoming_messages.append(prev_bucket['outgoing_message'])
                    outgoing_message_vars.update(prev_bucket['outgoing_message'])

            # Determine variables in the outgoing message
            for factor in bucket_factors:
                outgoing_message_vars.update(factor.labels)
            outgoing_message_vars.discard(var)

            # Find the next bucket to send the message to
            next_var = self.find_next_bucket(list(outgoing_message_vars), var)

            width = len(list(outgoing_message_vars))
            max_width = max(max_width, width)

            bucket_info = {
                'var': var,
                'factors': [f"f{i}({', '.join(map(str, factor.labels))})" for i, factor in enumerate(bucket_factors)],
                'receives': [f"mess_to_{var}({', '.join(map(str, sorted(msg)))})" for msg in incoming_messages],
                'sends_to': next_var,
                'outgoing_message': sorted(list(outgoing_message_vars)),
                'width': width
            }

            elimination_scheme.append(bucket_info)

        # Print the elimination scheme
        print("Elimination Scheme:")
        for bucket in elimination_scheme:
            print(f"Bucket {bucket['var']}:")
            print(f"  Factors: {', '.join(bucket['factors'])}")
            print(f"  Receives: {', '.join(bucket['receives']) if bucket['receives'] else 'None'}")
            if bucket['sends_to'] is not None:
                print(f"  Sends: mess_{bucket['var']}_to_{bucket['sends_to'].label}({', '.join(map(str, bucket['outgoing_message']))}) to bucket {bucket['sends_to'].label}")
            else:
                print(f"  Sends: mess_{bucket['var']}_to_root({', '.join(map(str, bucket['outgoing_message']))}) to root")
            print(f"  Width: {bucket['width']}")
            print()

        print(f"Maximum width: {max_width}")

    def get_senders_receivers(self):
        vars_to_eliminate = self.elim_order

        elimination_scheme = []
        max_width = 0

        for var in self.elim_order:
            if var not in vars_to_eliminate:
                continue
            
            bucket_factors = self.buckets[var].factors
            incoming_messages = []
            outgoing_message_vars = set()

            # Determine incoming messages from previous buckets
            for prev_bucket in elimination_scheme:
                if prev_bucket['sends_to'] is not None and prev_bucket['sends_to'] == var.label:
                    incoming_messages.append(prev_bucket['outgoing_message'])
                    outgoing_message_vars.update(prev_bucket['outgoing_message'])

            # Determine variables in the outgoing message
            for factor in bucket_factors:
                outgoing_message_vars.update(factor.labels)
            outgoing_message_vars.discard(var)

            # Find the next bucket to send the message to
            next_var = self.find_next_bucket(list(outgoing_message_vars), var)

            width = len(list(outgoing_message_vars))
            max_width = max(max_width, width)

            bucket_info = {
                'var': var,
                'factors': [f"f{i}({', '.join(map(str, factor.labels))})" for i, factor in enumerate(bucket_factors)],
                'receives': [f"mess_to_{var}({', '.join(map(str, sorted(msg)))})" for msg in incoming_messages],
                'sends_to': next_var.label if next_var is not None else None,
                'outgoing_message': sorted(list(outgoing_message_vars)),
                'width': width
            }

            elimination_scheme.append(bucket_info)

        return elimination_scheme

    def get_large_message_buckets(self, iB, debug = False):
        for key in self.message_scopes:
            if len(self.message_scopes[key]) > iB:
                if debug:
                    print('------------------------------------')
                    print('Key is ', key)
                    print('Scope is ', self.message_scopes[key])
                prod = 1
                for idx in self.message_scopes[key]:
                    if debug:
                        print('(Var: ', idx, ', ', end= '')
                    if idx != key:
                        var = self.matching_var(idx)
                        prod *= var.states
                        if debug:
                            print('states: ', var.states, end=') ')
                if debug:
                    print()
                adj_num_vars = math.log2(prod)
                print('bucket: ', key, ', num vars: ', len(self.message_scopes[key]), ', adj num vars: ', adj_num_vars)
    
    def find_next_bucket(self, labels, current_var):
        """Find the next bucket that shares any variable with the given labels."""
        current_index = self.elim_order.index(current_var)
        for var in self.elim_order[current_index:]:  # Look at earlier variables
            if var in self.buckets and any(label in labels for label in self.buckets[var].elim_vars):
                return self.buckets[var]
        return None  # If no appropriate bucket found, this will send the message to root

    def get_joint_distribution(self) -> FastFactor:
        # used for message gradient
        joint = None
        for bucket in self.buckets.values():
            for factor in bucket.factors:
                if joint is None:
                    joint = factor
                else:
                    joint *= factor
        return joint

    def get_log_partition_function(self):
        """
        Returns the log partition function if it has been computed, otherwise computes it.
        """
        if hasattr(self, 'log_partition_function'):
            return self.log_partition_function
        else:
            result = self.eliminate_variables(all=True)
            return self.log_partition_function

    def get_gradient_factors(self, bucket_idx: int) -> List[FastFactor]:
        # this function assumes elimination up to variable bucket_idx has been done
        message_scope = self.get_bucket(bucket_idx).get_message_scope()
        gradient_factors = []
        
        for var in self.elim_order[self.elim_order.index(self.matching_var(bucket_idx))+1:]:
            bucket = self.buckets[var]
            bucket_factors = bucket.factors
            for factor in bucket_factors:
                if set(factor.labels).intersection(message_scope):
                    gradient_factors.append(factor)
        return gradient_factors
    
    # def get_message_gradient(self, bucket_var, gradient_factors=None, iB = 100):
    #     # function should do elimination up to, but not including bucket_var.
    #     # function should gather all factors from all buckets that come after bucket_var and create a new fastGM from them.
    #     # function should eliminate all bucket the variables in the scope of bucket_var's bucket's scope
    #     if gradient_factors is None:
    #         self.eliminate_variables(up_to=bucket_var, exact=True)
    #         gradient_factors = []
    #         for var in self.elim_order[self.elim_order.index(self.matching_var(bucket_var))+1:]:
    #             bucket = self.buckets[var]
    #             bucket_factors = bucket.factors
    #             for factor in bucket_factors:
    #                 gradient_factors.append(factor)
    #     bucket = self.get_bucket(bucket_var)
    #     message = bucket.compute_message_exact()
    #     bucket_scope = bucket.get_message_scope()
    #     # if no downstream function
    #     if gradient_factors == []:
    #         return FastFactor(torch.tensor([0.0], device=self.device), [], requires_grad=False), message
    #     downstream_elim_order = wtminfill_order(gradient_factors, variables_not_eliminated=bucket_scope)
    #     # print("deo is ", downstream_elim_order)
    #     # device_copy=str(self.device)
    #     downstream_gm = FastGM(factors=gradient_factors, elim_order=downstream_elim_order, reference_fastgm=self, device=self.device, nn_config=self.config)
    #     print("Upstream width is ", downstream_gm.get_max_width())
    #     downstream_gm.iB = iB
    #     downstream_gm.eliminate_variables(all_but=bucket_scope)
    #     return downstream_gm.get_joint_distribution(), message

    def removeFactors(self, factors_to_remove):
        for var in self.buckets:
            self.buckets[var].factors = [f for f in self.buckets[var].factors if f not in factors_to_remove]

    def addFactors(self, factors_to_add):
        for factor in factors_to_add:
            if not factor.labels:  # If the factor has no labels (scalar factor)
                # Add to a special 'constant' bucket or handle as needed
                if 'constant' not in self.buckets:
                    self.buckets['constant'] = FastBucket(self, 'constant', [], self.device, [])
                self.buckets['constant'].factors.append(factor)
            else:
                # find earliest bucket in elim order that contains any of the labels
                indices = [self.elim_order.index(self.matching_var(v)) for v in factor.labels]
                # find index of earliest bucket
                earliest_var = factor.labels[indices.index(min(indices))]
                # move to that bucket
                try:
                    self.buckets[self.matching_var(earliest_var)].factors.append(factor)
                except:
                    print('got here')
                    raise(ValueError("No matching var found for idx ", earliest_var))
                
                # self.buckets[var].factors.append(factor)

    # def get_message_stats(self, bucket, output_message, get_Z_estimands=True):
    #     # adds (label, width, fw_var, bw_var, const_pred_linspace_mse_Z_err, ls_one)
    #     from nce.utils.stats import get_fw_bw_correlation
    #     import os, contextlib
    #     stats = dict()
    #     stats['label'] = bucket.label
    #     stats['width'] = len(output_message.labels)
    #     if not bucket.approximate_downstream_factors or len(output_message.labels) == 0:
    #         empty = True
    #     else:
    #         empty = False
    #         with open(os.devnull, "w") as devnull, \
    #             contextlib.redirect_stdout(devnull), \
    #             contextlib.redirect_stderr(devnull):
    #             mg = self.get_message_gradient(bucket.label, bucket.approximate_downstream_factors)[0]
        
    #     stats['mg_var'] = 0 if empty or mg is None else mg.get_variance()
    #     if not get_Z_estimands:
    #         Z_errs = () # linspace_err,logspace_err
    #     else:
    #         Z_linspace_mse_fw_component = torch.logsumexp(output_message.tensor.reshape(-1), dim=0) - math.log(output_message.tensor.numel())
    #         Z_logspace_mse_bw_component = torch.mean(output_message.tensor.reshape(-1))
    #         if len(output_message.labels) == 0:
    #             stats['linspace_err'], stats['logspace_err'] = 0, 0
    #         elif not bucket.approximate_downstream_factors:
    #             stats['linspace_err'], stats['logspace_err'] = 0, Z_linspace_mse_fw_component.item() - Z_logspace_mse_bw_component.item()
    #         else:
    #             if mg is not None:
    #                 Z = (output_message * mg).sum_all_entries()
    #                 Z_component_from_bw_message = torch.logsumexp(mg.tensor.reshape(-1), dim=0)
    #             else:
    #                 Z = output_message.sum_all_entries()
                    
    #             fw_star = Z - Z_component_from_bw_message
    #             stats['linspace_err'], stats['logspace_err'] = \
    #                 Z_linspace_mse_fw_component.item() - fw_star.item(), \
    #                 Z_logspace_mse_bw_component.item() - fw_star.item()
                
    #     stats['output_message_var'] = output_message.get_variance()
    #     stats['correlation'] = get_fw_bw_correlation(output_message, mg) if not empty else 0
    #     stats['output_message_std'] = stats['output_message_var']** 0.5 if stats['output_message_var'] > 0 else 0
    #     stats['mg_std'] = stats['mg_var'] ** 0.5 if stats['mg_var'] > 0 else 0
    #     stats['correlation_correction_factor'] = 2 * stats['correlation'] * stats['output_message_std'] * stats['mg_std']
    #     stats['corrected_err'] = stats['linspace_err'] + stats['correlation_correction_factor']


    #     self.message_stats.append(stats)
    
 
    # def graph_message_stats(
    #     self,
    #     min_width: int = 0,
    #     prob_name: str | None = None,
    #     save_path: str | None = None,
    #     show: bool = True,
    # ) -> None:
    #     """
    #     Figures produced
    #     ----------------
    #     1. Scatter: forward/backward variance vs. # vars eliminated first.
    #     2. Scatter: (variance_f − variance_b) vs. lin/log-space MSE + best-fit lines.
    #     3. Scatter: bucket width vs. forward/backward variance (small dots).

    #     All figures are saved to *save_path* (directory) if provided.
    #     """
    #     import os
    #     import numpy as np
    #     import matplotlib.pyplot as plt

    #     # -------- collect data -------------------------------------------------
    #     idx, widths = [], []
    #     vf, vb = [], []
    #     mse_lin, mse_log = [], []
    #     corrs = []  # log correlation between forward and backward messages

    #     for (_, width, var_f, var_b, ml, mg, corr) in self.message_stats:
    #         if width < min_width:
    #             continue
    #         idx.append(len(idx))
    #         widths.append(width)
    #         vf.append(var_f)
    #         vb.append(var_b)
    #         mse_lin.append(ml)
    #         mse_log.append(mg)
    #         corrs.append(corr)

    #     if not idx:
    #         raise ValueError("No message_stats entries satisfy min_width.")

    #     def _fname(stem: str) -> str:
    #         return f"{stem}{'_'+prob_name if prob_name else ''}.png"

    #     # -------- Figure 1 -----------------------------------------------------
    #     plt.figure()
    #     plt.scatter(idx, vf, marker="o", label="Forward Variance")
    #     plt.scatter(idx, vb, marker="x", label="Backward Variance")
    #     plt.xlabel("# vars eliminated first")
    #     plt.ylabel("Variance of log‐message")
    #     t = "Variance vs. Elimination Order"
    #     if prob_name:
    #         t += f" for {prob_name}"
    #     plt.title(t)
    #     plt.legend()
    #     if save_path:
    #         os.makedirs(save_path, exist_ok=True)
    #         plt.savefig(os.path.join(save_path, _fname("variance_vs_elim")),
    #                     dpi=300, bbox_inches="tight")
    #     if show: plt.show()
    #     else:    plt.close()

    #     # -------- Figure 2 -----------------------------------------------------
    #     ratio = np.array(vf) - np.array(vb)           # *** keep your subtraction ***
    #     abs_lin = np.abs(mse_lin)
    #     abs_log = np.abs(mse_log)

    #     # best-fit line in log-space (fit log10(y) on x, then exponentiate for plotting)
    #     def best_fit(x, y):
    #         mask = np.isfinite(x) & (y > 0)
    #         if mask.sum() < 2:      # degenerate case
    #             return None, None
    #         coeff = np.polyfit(x[mask], np.log10(y[mask]), 1)   # slope, intercept
    #         x_fit = np.linspace(x[mask].min(), x[mask].max(), 200)
    #         y_fit = 10 ** (coeff[1] + coeff[0] * x_fit)
    #         return x_fit, y_fit

    #     x_fit_lin, y_fit_lin = best_fit(ratio, abs_lin)
    #     x_fit_log, y_fit_log = best_fit(ratio, abs_log)

    #     plt.figure()
    #     plt.scatter(ratio, abs_lin, marker="o", label="Lin-space MSE (Z_err)")
    #     plt.scatter(ratio, abs_log, marker="x", label="Log-space MSE (Z_err)")

    #     if x_fit_lin is not None:
    #         plt.plot(x_fit_lin, y_fit_lin, linestyle="--", linewidth=1,
    #                 label="Best fit (lin-space)")
    #     if x_fit_log is not None:
    #         plt.plot(x_fit_log, y_fit_log, linestyle="--", linewidth=1,
    #                 label="Best fit (log-space)")

    #     plt.xlabel("Log-variance ratio  (forward – backward)")
    #     plt.ylabel("Z_err for constant-message prediction")
    #     t = "Variance-Ratio vs. Z_err (lin/log)"
    #     if prob_name:
    #         t += f" for {prob_name}"
    #     plt.title(t)
    #     # plt.yscale("log")
    #     plt.legend()

    #     if save_path:
    #         plt.savefig(os.path.join(save_path, _fname("var_ratio_vs_mse")),
    #                     dpi=300, bbox_inches="tight")
    #     if show: plt.show()
    #     else:    plt.close()

    #     # -------- Figure 3 -----------------------------------------------------
    #     plt.figure()
    #     plt.scatter(widths, vf, s=15, marker="o", label="Forward Variance")  # tiny dots
    #     plt.scatter(widths, vb, s=15, marker="x", label="Backward Variance")
    #     plt.xlabel("Bucket width")
    #     plt.ylabel("Variance of log‐message")
    #     t = "Variance vs. Bucket Width"
    #     if prob_name:
    #         t += f" for {prob_name}"
    #     plt.title(t)
    #     plt.legend()

    #     if save_path:
    #         plt.savefig(os.path.join(save_path, _fname("variance_vs_width")),
    #                     dpi=300, bbox_inches="tight")
    #     if show: plt.show()
    #     else:    plt.close()
  
    # def graph_message_stats_old(
    #     self,
    #     min_width: int = 0,
    #     prob_name: str | None = None,
    #     save_path: str | None = None,
    #     show: bool = True,
    # ) -> None:
    #     """
    #     Figures produced
    #     ----------------
    #     1. Scatter: forward/backward variance vs. # vars eliminated first.
    #     2. Scatter: (variance_f − variance_b) vs. lin/log-space MSE + best-fit lines.
    #     3. Scatter: bucket width vs. forward/backward variance (small dots).
    #     4. Scatter: log forward-backward correlation vs. lin/log-space MSE Z_err.

    #     All figures are saved to *save_path* (directory) if provided.
    #     """
    #     import os
    #     import numpy as np
    #     import matplotlib.pyplot as plt

    #     # -------- collect data -------------------------------------------------
    #     idx, widths = [], []
    #     vf, vb = [], []
    #     mse_lin, mse_log = [], []
    #     corrs = []  # log correlation between forward and backward messages

    #     for stat_dict in self.message_stats:
    #         width = stat_dict['width']
    #         idx = stat_dict['label']
    #         var_f = stat_dict['output_message_var']
    #         var_b = stat_dict['mg_var']
    #         ml = stat_dict['linspace_err']
    #         mg = stat_dict['logspace_err']
    #         corr = stat_dict['correlation']
            
    #         if width < min_width:
    #             continue
    #         idx.append(len(idx))
    #         widths.append(width)
    #         vf.append(var_f)
    #         vb.append(var_b)
    #         mse_lin.append(ml)
    #         mse_log.append(mg)
    #         corrs.append(corr)

    #     if not idx:
    #         raise ValueError("No message_stats entries satisfy min_width.")

    #     def _fname(stem: str) -> str:
    #         return f"{stem}{'_'+prob_name if prob_name else ''}.png"

    #     # -------- Figure 1 -----------------------------------------------------
    #     plt.figure()
    #     plt.scatter(idx, vf, marker="o", label="Forward Variance")
    #     plt.scatter(idx, vb, marker="x", label="Backward Variance")
    #     plt.xlabel("# vars eliminated first")
    #     plt.ylabel("Variance of log‐message")
    #     t = "Variance vs. Elimination Order"
    #     if prob_name:
    #         t += f" for {prob_name}"
    #     plt.title(t)
    #     plt.legend()
    #     if save_path:
    #         os.makedirs(save_path, exist_ok=True)
    #         plt.savefig(os.path.join(save_path, _fname("variance_vs_elim")),
    #                     dpi=300, bbox_inches="tight")
    #     if show: plt.show()
    #     else:    plt.close()

    #     # -------- Figure 2 -----------------------------------------------------
    #     ratio = np.array(vf) - np.array(vb)           # *** keep your subtraction ***
    #     abs_lin = np.abs(mse_lin)
    #     abs_log = np.abs(mse_log)

    #     # best-fit line in log-space (fit log10(y) on x, then exponentiate for plotting)
    #     def best_fit(x, y):
    #         mask = np.isfinite(x) & (y > 0)
    #         if mask.sum() < 2:      # degenerate case
    #             return None, None
    #         coeff = np.polyfit(x[mask], np.log10(y[mask]), 1)   # slope, intercept
    #         x_fit = np.linspace(x[mask].min(), x[mask].max(), 200)
    #         y_fit = 10 ** (coeff[1] + coeff[0] * x_fit)
    #         return x_fit, y_fit

    #     x_fit_lin, y_fit_lin = best_fit(ratio, abs_lin)
    #     x_fit_log, y_fit_log = best_fit(ratio, abs_log)

    #     plt.figure()
    #     plt.scatter(ratio, abs_lin, marker="o", label="Lin-space MSE (Z_err)")
    #     plt.scatter(ratio, abs_log, marker="x", label="Log-space MSE (Z_err)")

    #     if x_fit_lin is not None:
    #         plt.plot(x_fit_lin, y_fit_lin, linestyle="--", linewidth=1,
    #                 label="Best fit (lin-space)")
    #     if x_fit_log is not None:
    #         plt.plot(x_fit_log, y_fit_log, linestyle="--", linewidth=1,
    #                 label="Best fit (log-space)")

    #     plt.xlabel("Log-variance ratio  (forward – backward)")
    #     plt.ylabel("Z_err for constant-message prediction")
    #     t = "Variance-Ratio vs. Z_err (lin/log)"
    #     if prob_name:
    #         t += f" for {prob_name}"
    #     plt.title(t)
    #     # plt.yscale("log")
    #     plt.legend()

    #     if save_path:
    #         plt.savefig(os.path.join(save_path, _fname("var_ratio_vs_mse")),
    #                     dpi=300, bbox_inches="tight")
    #     if show: plt.show()
    #     else:    plt.close()

    #     # -------- Figure 3 -----------------------------------------------------
    #     plt.figure()
    #     plt.scatter(widths, vf, s=15, marker="o", label="Forward Variance")  # tiny dots
    #     plt.scatter(widths, vb, s=15, marker="x", label="Backward Variance")
    #     plt.xlabel("Bucket width")
    #     plt.ylabel("Variance of log‐message")
    #     t = "Variance vs. Bucket Width"
    #     if prob_name:
    #         t += f" for {prob_name}"
    #     plt.title(t)
    #     plt.legend()

    #     if save_path:
    #         plt.savefig(os.path.join(save_path, _fname("variance_vs_width")),
    #                     dpi=300, bbox_inches="tight")
    #     if show: plt.show()
    #     else:    plt.close()

    #     # -------- Figure 4 -----------------------------------------------------
    #     plt.figure()
    #     plt.scatter(corrs, abs_lin, marker="o", label="Lin-space MSE (Z_err)")
    #     plt.scatter(corrs, abs_log, marker="x", label="Log-space MSE (Z_err)")
    #     plt.xlabel("Log forward-backward correlation")
    #     plt.ylabel("Z_err for constant-message prediction")
    #     t = "Log FW-BW Correlation vs. Z_err (lin/log)"
    #     if prob_name:
    #         t += f" for {prob_name}"
    #     plt.title(t)
    #     plt.legend()

    #     if save_path:
    #         plt.savefig(os.path.join(save_path, _fname("correlation_vs_mse")),
    #                     dpi=300, bbox_inches="tight")
    #     if show: plt.show()
    #     else:    plt.close()
  
    def graph_message_stats(
            self,
            min_width: int = 0,
            prob_name: str | None = None,
            save_path: str | None = None,
            show: bool = True,
        ) -> None:
            """
            Figures produced
            ----------------
            1. Scatter: forward/backward variance vs. # vars eliminated first.
            2. Scatter: (variance_f − variance_b) vs. lin/log-space MSE + best-fit lines.
            3. Scatter: bucket width vs. forward/backward variance (small dots).
            4. Scatter: log forward-backward correlation vs. lin/log-space MSE Z_err.

            All figures are saved to *save_path* (directory) if provided.
            CSV data is also saved to the same directory.
            """
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            import pandas as pd

            # -------- collect data -------------------------------------------------
            idx, widths = [], []
            vf, vb = [], []
            mse_lin, mse_log, corrected_mse_lin = [], [], []
            corrs = []  # log correlation between forward and backward messages
            corrections = []
            labels = []  # bucket labels

            for stat_dict in self.message_stats:
                label = stat_dict['label']
                width = stat_dict['width']
                idx = stat_dict['label']
                var_f = stat_dict['output_message_var']
                var_b = stat_dict['mg_var']
                mlin = stat_dict['linspace_err']
                mlog = stat_dict['logspace_err']
                mc = stat_dict['corrected_err']
                corr = stat_dict['correlation']
                correction = stat_dict['correlation_correction_factor']
                if width < min_width:
                    continue
                labels.append(label)
                idx.append(len(idx))
                widths.append(width)
                vf.append(var_f)
                vb.append(var_b)
                mse_lin.append(mlin)
                mse_log.append(mlog)
                corrected_mse_lin.append(mc)
                corrs.append(corr)
                corrections.append(correction)

            if not idx:
                raise ValueError("No message_stats entries satisfy min_width.")

            def _fname(stem: str) -> str:
                return f"{stem}{'_'+prob_name if prob_name else ''}"

            # -------- Save CSV data ------------------------------------------------
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                
                # Create DataFrame with all the data
                df = pd.DataFrame({
                    'bucket_label': labels,
                    'elimination_order': idx,
                    'bucket_width': widths,
                    'forward_variance': vf,
                    'backward_variance': vb,
                    'variance_ratio': np.array(vf) - np.array(vb),
                    'lin_space_mse': mse_lin,
                    'log_space_mse': mse_log,
                    'abs_lin_space_mse': np.abs(mse_lin),
                    'abs_log_space_mse': np.abs(mse_log),
                    'fw_bw_correlation': corrs,
                    'correction_factor': corrections,
                    'corrected_mse': corrected_mse_lin
                })
                
                csv_filename = _fname("message_stats") + ".csv"
                csv_path = os.path.join(save_path, csv_filename)
                df.to_csv(csv_path, index=False)
                print(f"Message stats data saved to: {csv_path}")

            # -------- Figure 1 -----------------------------------------------------
            plt.figure()
            plt.scatter(idx, vf, marker="o", label="Forward Variance")
            plt.scatter(idx, vb, marker="x", label="Backward Variance")
            plt.xlabel("# vars eliminated first")
            plt.ylabel("Variance of log‐message")
            t = "Variance vs. Elimination Order"
            if prob_name:
                t += f" for {prob_name}"
            plt.title(t)
            plt.legend()
            if save_path:
                plt.savefig(os.path.join(save_path, _fname("variance_vs_elim") + ".png"),
                            dpi=300, bbox_inches="tight")
            if show: plt.show()
            else:    plt.close()

            # -------- Figure 2 -----------------------------------------------------
            ratio = np.array(vf) / np.array(vb)           # *** keep your subtraction ***
            abs_lin = np.abs(mse_lin)
            abs_log = np.abs(mse_log)
            abs_corrected = np.abs(corrected_mse_lin)

            # best-fit line in log-space (fit log10(y) on x, then exponentiate for plotting)
            def best_fit(x, y):
                mask = np.isfinite(x) & (y > 0)
                if mask.sum() < 2:      # degenerate case
                    return None, None
                coeff = np.polyfit(x[mask], np.log10(y[mask]), 1)   # slope, intercept
                x_fit = np.linspace(x[mask].min(), x[mask].max(), 200)
                y_fit = 10 ** (coeff[1] + coeff[0] * x_fit)
                return x_fit, y_fit

            # x_fit_lin, y_fit_lin = best_fit(ratio, abs_lin)
            # x_fit_log, y_fit_log = best_fit(ratio, abs_log)

            plt.figure()
            plt.scatter(ratio, abs_lin, marker="o", label="Lin-space MSE (Z_err)")
            plt.scatter(ratio, abs_log, marker="x", label="Log-space MSE (Z_err)")
            plt.scatter(ratio, abs_corrected, marker="^", label="Corrected Lin-space MSE (Z_err)")
            

            # if x_fit_lin is not None:
            #     plt.plot(x_fit_lin, y_fit_lin, linestyle="--", linewidth=1,
            #             label="Best fit (lin-space)")
            # if x_fit_log is not None:
            #     plt.plot(x_fit_log, y_fit_log, linestyle="--", linewidth=1,
            #             label="Best fit (log-space)")

            plt.xlabel("Log-variance ratio  (forward – backward)")
            plt.ylabel("Z_err for constant-message prediction")
            t = "Variance-Ratio vs. Z_err (lin/log)"
            if prob_name:
                t += f" for {prob_name}"
            plt.title(t)
            # plt.yscale("log")
            plt.legend()

            if save_path:
                plt.savefig(os.path.join(save_path, _fname("var_ratio_vs_mse") + ".png"),
                            dpi=300, bbox_inches="tight")
            if show: plt.show()
            else:    plt.close()

            # -------- Figure 3 -----------------------------------------------------
            plt.figure()
            plt.scatter(widths, vf, s=15, marker="o", label="Forward Variance")  # tiny dots
            plt.scatter(widths, vb, s=15, marker="x", label="Backward Variance")
            plt.xlabel("Bucket width")
            plt.ylabel("Variance of log‐message")
            t = "Variance vs. Bucket Width"
            if prob_name:
                t += f" for {prob_name}"
            plt.title(t)
            plt.legend()

            if save_path:
                plt.savefig(os.path.join(save_path, _fname("variance_vs_width") + ".png"),
                            dpi=300, bbox_inches="tight")
            if show: plt.show()
            else:    plt.close()

            # -------- Figure 4 -----------------------------------------------------
            plt.figure()
            plt.scatter(corrs, abs_lin, marker="o", label="Lin-space MSE (Z_err)")
            plt.scatter(corrs, abs_log, marker="x", label="Log-space MSE (Z_err)")
            plt.xlabel("Log forward-backward correlation")
            plt.ylabel("Z_err for constant-message prediction")
            t = "Log FW-BW Correlation vs. Z_err (lin/log)"
            if prob_name:
                t += f" for {prob_name}"
            plt.title(t)
            plt.legend()

            if save_path:
                plt.savefig(os.path.join(save_path, _fname("correlation_vs_mse") + ".png"),
                            dpi=300, bbox_inches="tight")
            if show: plt.show()
            else:    plt.close()
    
    # make minimum value of initial factors something higher than 0
    def dope_factors(self, new_min=-5):
        for bucket in self.buckets.values():
            for factor in bucket.factors:
                factor.tensor[factor.tensor==float('-inf')] = new_min
    
    def get_wmb_message_gradient(self, bucket_var, i_bound, weights='max'):
        self.eliminate_variables(up_to=bucket_var)

        gradient_factors = []
        
        
        for var in self.elim_order[self.elim_order.index(self.matching_var(bucket_var))+1:]:
            bucket = self.buckets[var]
            gradient_factors.extend(bucket.factors)

        # if there are no gradient factors return a scalar factor
        if len(gradient_factors) == 0:
            return FastFactor(tensor=torch.tensor([0.0], device=self.device, requires_grad=False), labels=[])
        
        # confirm no variables that should have been eliminated are in gradient factors
        should_have_been_eliminated = set([v.label for v in self.elim_order[0:self.elim_order.index(self.matching_var(bucket_var))]])
        
        for factor in gradient_factors:
            for v in factor.labels:
                if type(v) != int:
                    raise(ValueError("Variable ", v, " is not an int"))
                if v in should_have_been_eliminated:
                    raise ValueError("Variable ", v, " should have been eliminated")
        
        
        bucket = self.get_bucket(bucket_var)
        bucket_scope = bucket.get_message_scope()
        # dprint('bucket scope here reads', bucket_scope)

        # Get all variables involved in the gradient factors
        all_vars = set()
        for factor in gradient_factors:
            all_vars.update(factor.labels)

        # Create a new elimination order for the downstream graph
        downstream_elim_order = wtminfill_order(gradient_factors, variables_not_eliminated=bucket_scope)
        
        # Convert the elimination order to Var objects
        downstream_elim_order = [self.matching_var(var) for var in downstream_elim_order]

        # Create the downstream graphical model
        downstream_gm = FastGM(factors=gradient_factors, elim_order=downstream_elim_order, device=self.device, nn_config=self.config)

        return self._wmb_eliminate(downstream_gm, bucket_scope, i_bound, weights)

    def _wmb_eliminate(gm, target_scope, i_bound, weights, combine_factors=False):
        """
        Perform Weighted Mini-Bucket elimination.

        Args:
        gm (FastGM): The graphical model to eliminate.
        target_scope (list): The variables to keep (not eliminate).
        i_bound (int): The maximum allowed scope size for mini-buckets.
        weights (str or list): Weights for WMB.

        Returns:
        FastFactor: The result of WMB elimination.
        """
        # dprint('target scope is ', target_scope)
        if isinstance(weights, str):
            if weights == 'max':
                weight_map = {var.label: 0.0 for var in gm.vars}
            elif weights == 'sum':
                weight_map = {var.label: 1.0 for var in gm.vars}
            else:
                raise ValueError("Unknown weight type. Use 'max', 'sum', or provide a list of weights.")
        else:
            weight_map = {var.label: weight for var, weight in zip(gm.vars, weights)}
        
        result = None
        # dprint('elim order is ', gm.elim_order)
        # dprint()
        # dprint('All bucket factors scopes are:')
        # for key in self.buckets.keys():
        #     dprint("Bucket = ", key)
        #     bucket = self.buckets[key]
        #     dprint('Factors scopes are:')
        #     for factor in bucket.factors:
        #         dprint(factor.labels)
        #     dprint()
                
        for var in gm.elim_order:
            # dprint('var ', var, ' considered')
            if var.label in target_scope:
                continue

            bucket = gm.get_bucket(var)
            
            if not bucket.factors:  # Skip empty buckets
                continue
            
            # debug
            # print('var: ', var.label, end='')
            if bucket.get_width() <= i_bound:
                message = FastGM._compute_weighted_message(bucket.factors, var, weight_map[var.label])
                gm.removeFactors(bucket.factors)
                gm.addFactors([message])
                # debug
                # print('message labels: ', message.labels)
            else:
                mini_buckets = FastGM._create_mini_buckets(bucket.factors, i_bound)
                for i, mini_bucket in enumerate(mini_buckets):
                    mini_weight = weight_map[var.label] / len(mini_buckets)
                    if i == len(mini_buckets) - 1:  # Adjust the last mini-bucket weight
                        mini_weight = weight_map[var.label] - (len(mini_buckets) - 1) * mini_weight
                    mini_message = FastGM._compute_weighted_message(mini_bucket, var, mini_weight)
                    gm.removeFactors(mini_bucket)
                    gm.addFactors([mini_message])
                    # debug
                    # print('mini_message labels: ', mini_message.labels)
                    # dprint(mini_message.labels)

        # After elimination, combine all remaining factors
        remaining_factors = []
        for bucket in gm.buckets.values():
            remaining_factors.extend(bucket.factors)
            # debug
            for factor in bucket.factors:
                if 54 in factor.labels:
                    print('got here')
            
        if not combine_factors:
            return remaining_factors
        
        if remaining_factors:
            result = remaining_factors[0]
            if (result.tensor == float('inf')).any():
                print("inf found")
                print(remaining_factors[0].labels)
                raise ValueError("inf found")
            for factor in remaining_factors[1:]:
                try:
                    result = result * factor
                    # if sum([0 if v in target_scope else 1 for v in factor.labels]) != 0:
                        # dprint('factor scope is ', factor.labels)
                    # dprint('result scope is ', result.labels)
                except:
                    print('err')
                    raise ValueError("scope incorrect")
                if (result.tensor == float('inf')).any():
                    print("inf found")
                    print(factor.labels)
                    raise ValueError("inf found")
        else:
            # If no factors remain, return a scalar factor with value 0 (in log space)
            result = FastFactor(torch.tensor([0.0], device=gm.device, requires_grad=False), [])
        if (result.tensor == float('inf')).any():
            print("inf found")
            raise ValueError("inf found")
        return result

    def _create_mini_buckets(factors, i_bound):
        """
        Partition factors into mini-buckets respecting the i-bound.
        """
        mini_buckets = []
        sorted_factors = sorted(factors, key=lambda f: len(f.labels), reverse=True)
        for factor in sorted_factors:
            placed = False
            for bucket in mini_buckets:
                if len(set.union(*[set(f.labels) for f in bucket], set(factor.labels))) <= i_bound:
                    bucket.append(factor)
                    placed = True
                    break
            if not placed:
                mini_buckets.append([factor])
        return mini_buckets

    def _compute_weighted_message(factors, var, weight):
        # Multiply factors using the FastFactor multiplication method
        product = factors[0]
        for factor in factors[1:]:
            product = product * factor  # Correctly handles label alignment and tensor operations
            if product.tensor.numel() > 2**30:
                print('product is the problem')

        # Proceed with elimination based on the weight
        if weight == 0:  # max-product
            out = FastGM._eliminate_max(product, var)
        elif weight == 1:  # sum-product
            out = product.eliminate([var])
        else:  # weighted sum-product
            out = FastGM._eliminate_weighted_sum(product, var, weight)
        return out

    def _eliminate_max(factor, var):
        """Eliminate a variable using max-product in log10 space."""
        dim = factor.labels.index(var.label)
        max_values, _ = torch.max(factor.tensor, dim=dim)
        return FastFactor(max_values, [label for label in factor.labels if label != var.label])

    def _eliminate_weighted_sum(factor, var, weight):
        dim = factor.labels.index(var.label)
        # Convert from log10 to natural log, perform weighted sum, then convert back to log10
        natural_log_tensor = factor.tensor * math.log(10)
        weighted_sum = torch.logsumexp(natural_log_tensor * weight, dim=dim) / weight
        log10_result = weighted_sum / math.log(10)
        return FastFactor(log10_result, [label for label in factor.labels if label != var.label])
    
    @staticmethod
    def sample_output_function(factors, sum_vars, sample_assignments, device='cuda'):
        sum_vars_set = set(sum_vars)
        
        # Step 1: Create stacked slices for each factor
        stacked_factors = [FastGM._create_stacked_slice(factor, sample_assignments, device)
                        for factor in factors]
        
        # Step 2: Multiply (add in log space) all stacked factors
        result = torch.zeros(sample_assignments.shape[0], device=device)
        for stacked_factor in stacked_factors:
            if stacked_factor.dim() > 1:
                stacked_factor = torch.sum(stacked_factor, dim=tuple(range(1, stacked_factor.dim())))
            result += stacked_factor
        
        return result

    @staticmethod
    def _create_stacked_slice(factor, sample_assignments, device):
        # Create a list to hold slices for each sample
        slices = []

        for assignment in sample_assignments:
            # Create indexing tuple for this sample
            index = tuple(assignment.get(var, slice(None)) for var in factor.labels)
            
            # Extract the slice and add it to the list
            slices.append(factor.tensor[index])

        # Stack all slices
        return torch.stack(slices).to(device)
    
    @staticmethod
    def _get_exact_value(exact_result, assignment):
        index = tuple(assignment.get(i, 0) for i in range(exact_result.dim()))
        return exact_result[index].item()

    @staticmethod
    def generate_sample_assignments(var_dims, non_sum_var_nums, num_samples, device='cuda'):
        # Create a tensor of random integers for each variable
        sample_tensors = {
            var: torch.randint(0, var_dims[var], (num_samples,), device=device)
            for var in non_sum_var_nums
        }
        
        # Combine into a single tensor
        combined_samples = torch.stack([sample_tensors[var] for var in non_sum_var_nums], dim=1)
        
        return combined_samples 
        
    @staticmethod
    def tester_sample_output_function(factors, sum_var_nums, num_samples=100, device='cuda'):
        # Get all variable numbers
        all_var_nums = set()
        for factor in factors:
            all_var_nums.update(factor.labels)
        all_var_nums = list(all_var_nums)

        # Compute exact result
        exact_result = factors[0]
        for factor in factors[1:]:
            exact_result = exact_result * factor
        exact_result = exact_result.eliminate(sum_var_nums)

        # Generate sample assignments
        non_sum_var_nums = [var for var in all_var_nums if var not in sum_var_nums]
        var_dims = {}
        for factor in factors:
            for var in factor.labels:
                if var not in var_dims and var in non_sum_var_nums:
                    var_dims[var] = factor.tensor.shape[factor.labels.index(var)]

        sample_assignments = FastGM.generate_sample_assignments(var_dims, non_sum_var_nums, num_samples, device)

        # Compute sampled result
        sampled_result = FastGM.sample_output_function(factors, sum_var_nums, sample_assignments, device)

        # Compare results
        errors = []
        for i, assignment in enumerate(sample_assignments):
            sampled_value = sampled_result[i].item()
            exact_value = FastGM._get_exact_value(exact_result, assignment)
            error = abs(sampled_value - exact_value)
            errors.append(error)

            print(f"Sample {i}:")
            print(f"  Assignment: {assignment}")
            print(f"  Sampled value: {sampled_value}")
            print(f"  Exact value: {exact_value}")
            print(f"  Error: {error}")
            print()

        # Compute and print average error
        avg_error = sum(errors) / len(errors)
        print(f"Average error: {avg_error}")

        return avg_error
    
    def populate_global_stats(self):
        from nce.utils.stats import get_gm_message_stats
        var_g_avg, rho_avg, _ = get_gm_message_stats(self, self.config.get('ecl'))
        self.config['sigma_g_global'] = var_g_avg ** 0.5
        self.config['rho_global'] = rho_avg

