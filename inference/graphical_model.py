from .factor import FastFactor
from .bucket import FastBucket
from .elimination_order import wtminfill_order
import pyGMs as gm
from pyGMs import wmb
from pyGMs.neuro import *
from pyGMs.graphmodel import eliminationOrder
from pyGMs import Var
import torch
import math
from typing import List


class FastGM:
    def __init__(self, elim_order=None, buckets=None, factors=None, uai_file=None, device="cuda", reference_fastgm=None, nn_config=None):
        self.iB = nn_config['iB']
        self.uai_file = uai_file
        self.device = device
        self.vars = []
        self.elim_order = None
        self.config = nn_config
        if nn_config is not None:
            self.sampling_scheme = nn_config['sampling_scheme']
            self.traced_losses = nn_config['traced_losses']
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
        # Load the UAI file
        ord_file = uai_file + ".vo"
        gm_model = uai_to_GM(uai_file=uai_file, order_file=ord_file)
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
            
    def eliminate_variables(self, elim_vars=None, up_to=None, through=None, all=False, all_but=None):
        if sum(map(bool, [elim_vars, up_to, through, all, all_but])) != 1:
            raise ValueError("Exactly one of elim_vars, up_to, through, or all must be specified")

        if all:
            vars_to_eliminate = self.elim_order
        elif elim_vars:
            vars_to_eliminate = elim_vars
        elif up_to:
            vars_to_eliminate = self.elim_order[:self.elim_order.index(up_to)]
        elif through:
            vars_to_eliminate = self.elim_order[:self.elim_order.index(through) + 1]
        elif all_but:
            vars_to_keep = set(all_but)
            vars_to_eliminate = [var for var in self.elim_order if var not in vars_to_keep]

        # Create a dummy root bucket to collect the final result
        root_bucket = FastBucket(self, 'root', [], self.device, [], isRoot=True)
        
        max_width = 0

        for var in vars_to_eliminate:
            # if type(var) != int:
            #     var = var.label
            current_bucket = self.buckets[var]
            # print(current_bucket.label)
            message = self.process_bucket(current_bucket)
            
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
    
    def process_bucket(self, bucket):
        if bucket.get_width() <= self.iB:
            return self._process_bucket_exact(bucket)
        else:
            return self._process_bucket_nn(bucket)
    
    def _process_bucket_exact(self, bucket):
        return bucket.compute_message_exact()
        # """Process a bucket by multiplying all factors and eliminating the bucket's variable."""
        # if not bucket.factors:
        #     raise ValueError("No factors in the bucket to process")
        
        # message = bucket.factors[0]
        # for factor in bucket.factors[1:]:
        #     message = message * factor

        # # Eliminate the bucket's variable
        # message = message.eliminate(bucket.elim_vars)
        # return message
    
    def _process_bucket_nn(self, bucket):
        pass
    
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
                print(f"  Sends: mess_{bucket['var']}_to_{bucket['sends_to']}({', '.join(map(str, bucket['outgoing_message']))}) to bucket {bucket['sends_to']}")
            else:
                print(f"  Sends: mess_{bucket['var']}_to_root({', '.join(map(str, bucket['outgoing_message']))}) to root")
            print(f"  Width: {bucket['width']}")
            print()

        print(f"Maximum width: {max_width}")

    def get_large_message_buckets(self, iB, debug = False):
        self.calculate_message_scopes()
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
    
    def get_message_gradient(self, bucket_var, gradient_factors=None):
        # function should do elimination up to, but not including bucket_var.
        # function should gather all factors from all buckets that come after bucket_var and create a new fastGM from them.
        # function should eliminate all bucket the variables in the scope of bucket_var's bucket's scope
        self.eliminate_variables(up_to=bucket_var)
        if gradient_factors is None:
            gradient_factors = []
            for var in self.elim_order[self.elim_order.index(self.matching_var(bucket_var))+1:]:
                bucket = self.buckets[var]
                bucket_factors = bucket.factors
                for factor in bucket_factors:
                    gradient_factors.append(factor)
        bucket = self.get_bucket(bucket_var)
        message = bucket.compute_message_exact()
        bucket_scope = bucket.get_message_scope()
        # if no downstream function
        if gradient_factors == []:
            return FastFactor(torch.tensor([0.0], device=self.device), []), message
        downstream_elim_order = wtminfill_order(gradient_factors, variables_not_eliminated=bucket_scope)
        # print("deo is ", downstream_elim_order)
        # device_copy=str(self.device)
        downstream_gm = FastGM(factors=gradient_factors, elim_order=downstream_elim_order, reference_fastgm=self, device=self.device)
        downstream_gm.eliminate_variables(all_but=bucket_scope)
        return downstream_gm.get_joint_distribution(), message

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
        downstream_gm = FastGM(factors=gradient_factors, elim_order=downstream_elim_order, device=self.device)

        return self._wmb_eliminate(downstream_gm, bucket_scope, i_bound, weights)

    def _wmb_eliminate(self, gm, target_scope, i_bound, weights):
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

            if len(bucket.factors[0].labels) <= i_bound:
                message = self._compute_weighted_message(bucket.factors, var, weight_map[var.label])
                gm.removeFactors(bucket.factors)
                gm.addFactors([message])
                # dprint(message.labels)
            else:
                mini_buckets = self._create_mini_buckets(bucket.factors, i_bound)
                for i, mini_bucket in enumerate(mini_buckets):
                    mini_weight = weight_map[var.label] / len(mini_buckets)
                    if i == len(mini_buckets) - 1:  # Adjust the last mini-bucket weight
                        mini_weight = weight_map[var.label] - (len(mini_buckets) - 1) * mini_weight
                    mini_message = self._compute_weighted_message(mini_bucket, var, mini_weight)
                    gm.removeFactors(mini_bucket)
                    gm.addFactors([mini_message])
                    # dprint(mini_message.labels)

        # After elimination, combine all remaining factors
        remaining_factors = []
        for bucket in gm.buckets.values():
            remaining_factors.extend(bucket.factors)
        
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
            result = FastFactor(torch.tensor([0.0], device=self.device), [])
        if (result.tensor == float('inf')).any():
            print("inf found")
            raise ValueError("inf found")
        return result

    def _create_mini_buckets(self, factors, i_bound):
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

    def _compute_weighted_message(self, factors, var, weight):
        # Multiply factors using the FastFactor multiplication method
        product = factors[0]
        for factor in factors[1:]:
            product = product * factor  # Correctly handles label alignment and tensor operations

        # Proceed with elimination based on the weight
        if weight == 0:  # max-product
            out = self._eliminate_max(product, var)
        elif weight == 1:  # sum-product
            out = product.eliminate([var])
        else:  # weighted sum-product
            out = self._eliminate_weighted_sum(product, var, weight)
        return out

    def _eliminate_max(self, factor, var):
        """Eliminate a variable using max-product in log10 space."""
        dim = factor.labels.index(var.label)
        max_values, _ = torch.max(factor.tensor, dim=dim)
        return FastFactor(max_values, [label for label in factor.labels if label != var.label])

    def _eliminate_weighted_sum(self, factor, var, weight):
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
    