from .factor import FastFactor
from .bucket import FastBucket
from .elimination_order import wtminfill_order
from nce.config_schema import prepare_config
from nce.utils.stats import get_message_stats
from nce.training_logger import setup_training_logger
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
    def __init__(self, model=None, elim_order=None, evid=None,buckets=None, factors=None, uai_file=None, device="cuda", reference_fastgm=None, nn_config=None, stats=None):
        if model is not None:
            uai_file = model.file
            try:
                elim_order = model.order
            except (ValueError, Exception):
                elim_order = None  # will be computed from .vo file or min-fill
            try:
                evid = model.evidence
            except (ValueError, Exception):
                evid = None
            self.logSS = getattr(model, 'logSS', None)
        else:
            self.logSS = None
        # Validate, normalize, and flatten the config through the schema layer.
        # prepare_config returns a new plain dict — safe from shared-reference bugs.
        self.config = prepare_config(nn_config) if nn_config else {}

        self.iB = self.config.get('iB', 0)
        self.ecl = self.config.get('ecl', 0)
        self.complexity_limit = self.config.get('complexity_limit', 0)
        self.num_trained = 0
        self.wmb_fw_partitions = 0  # Total forward partitions during backward factor population
        self.uai_file = uai_file
        self.device = device
        self.vars = []
        self.elim_order = None
        self.traced_losses_data = []
        self.message_stats = []
        self.stats=stats # cheater stats with variances
        self.is_primary = True
        self.is_populating_backward_factors = False  # Flag to indicate this GM is a copy used for backward factor population
        self.bucket_complexities = []
        self.track_errors = self.config.get('track_errors', False)
        self.nn_errors = []
        self.error_tracking_data = []  # List of (bucket_label, [(epoch, loss, log_Z_err, abs_log_Z_err), ...])
        self.local_errors = []  # per-NN-cluster signed local error: sum(approx_fw*exact_bw) - sum(exact_fw*exact_bw)
        self.phase_times = {}   # accumulated wall time per phase: 'exact', 'nn_path' (see process_bucket)
        self.per_bucket_training_log = []  # List of dicts per NN bucket: {label, epochs_trained, hidden_sizes, losses, val_losses, [nn_state_dict, normalizing_constant]}
        self.populate_bw_factors = self.config.get('populate_bw_factors', False)
        if self.config:
            self.sampling_scheme = self.config.get('sampling_scheme')
            self.traced_losses = self.config.get('traced_losses', [])
            self.lower_dim = self.config.get('lower_dim', False)
            self.hidden_sizes = self.config.get('hidden_sizes', [])
            self.loss_fn = self.config.get('loss_fn')
            self.optimizer = self.config.get('optimizer')
            self.lr = self.config.get('lr')
            self.lr_decay = self.config.get('lr_decay', 1.0)
            self.patience = self.config.get('patience', 10)
            self.min_lr = self.config.get('min_lr', 1e-8)
            self.num_samples = self.config.get('num_samples')
            self.num_epochs = self.config.get('num_epochs')
            self.batch_size = self.config.get('batch_size')
            self.set_size = self.config.get('set_size')
            self.seed = self.config.get('seed')
            self.gather_message_stats = self.config.get('gather_message_stats', False)

        # Set up training logger (JSONL file) if log_file is configured
        log_file_path = self.config.get('log_file')
        if log_file_path:
            self._training_logger = setup_training_logger(log_file_path)
        else:
            self._training_logger = None

        if uai_file is not None:
            self._load_from_uai(uai_file, elim_order=elim_order, evid=evid)
        elif buckets is not None:
            self.buckets = buckets
            self.elim_order = elim_order
        elif factors is not None:
            # If reference_fastgm is provided, use its vars for correct domain sizes
            # Otherwise, extract from factor tensor shapes
            if reference_fastgm is not None:
                # Get all variable labels from factors
                var_labels = set()
                for factor in factors:
                    if factor.labels:
                        var_labels.update(factor.labels)
                # Use domain sizes from reference_fastgm
                for label in var_labels:
                    ref_var = reference_fastgm.matching_var(label)
                    if ref_var is not None:
                        self.vars.append(Var(label, ref_var.states))
                    else:
                        # Fallback: extract from factor if not in reference
                        for factor in factors:
                            if label in factor.labels:
                                idx = factor.labels.index(label)
                                self.vars.append(Var(label, factor.tensor.shape[idx]))
                                break
            else:
                self._load_vars_from_factors(factors)
            if elim_order is not None:
                self.load_elim_order(elim_order, reference_fastgm)
            else:
                print("Computing elim order")
                self.load_elim_order(wtminfill_order(factors), reference_fastgm)
            self.buckets = self._create_buckets_from_factors(factors)
        else:
            raise ValueError("Either buckets, factors, or a UAI file must be provided")
        
        self.message_scopes = {}
        self.calculate_message_scopes()
        if self.config.get('use_join_tree_merge', False):
            self.merge_join_tree(
                verbose=self.config.get('verbose_merge', True),
                max_merge_bound=self.config.get('max_merge_bound',
                                                self.config.get('max_cluster_size')),
            )
        if self.config.get('use_reduce_nn_merge', False):
            self.reduce_nn_merge(
                verbose=self.config.get('verbose_merge', True),
                max_merge_bound=self.config.get('max_merge_bound',
                                                self.config.get('max_cluster_size')),
                backtrack=self.config.get('reduce_nn_backtrack', False),
            )
        if self.config.get('use_non_subsumption_merge', False):
            self.merge_non_subsumption(
                verbose=self.config.get('verbose_merge', True),
                max_merge_bound=self.config.get('max_merge_bound',
                                                self.config.get('max_cluster_size')),
            )
        if self.config.get('merge_degree', 0) and int(self.config.get('merge_degree', 0)) > 0:
            self.merge_by_degree(
                int(self.config['merge_degree']),
                verbose=self.config.get('verbose_merge', True),
            )
        if self.populate_bw_factors:
            if self.config.get('populate_bw_via_tree_collect', False):
                print("Populating backward factor LISTS via tree-collect (no per-bucket elim)...")
                self.populate_backward_factors_via_tree_collect()
            else:
                print("Populating backward factors using WMB approximations...")
                self.populate_backward_factors_wmb()
        if self.config.get('dope_factors'):
            self.dope_factors()
        if self.config.get('sigma_g_global') is None and ('approx_smg' in self.loss_fn or 'approx_smg' in self.config.get('loss_fn2', '')):
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
    
    def _load_from_uai(self, uai_file, elim_order=None, evid=None):
        import os
        # Load the UAI file
        ord_file = uai_file + ".vo"
        evid_file = uai_file + ".evid"
        # Prefer elim_order (all n vars, from .ord.elim) over .vo file (n-1 vars,
        # skips root).  Only fall back to .vo when no explicit order is given.
        if elim_order is not None:
            gm_model = uai_to_GM(uai_file=uai_file, elim_order=elim_order)
        else:
            gm_model = uai_to_GM(uai_file=uai_file, order_file=ord_file, elim_order=elim_order)
        if os.path.exists(evid_file):
            evid = readEvidence14(evid_file)
            gm_model.condition(evid)
        elif evid is not None:
            gm_model.condition(evid)
        
        self.vars = gm_model.vars
        
        # Convert PyGM factors to FastFactors
        from nce.utils.dtype_utils import get_dtype
        factor_dtype = get_dtype(self.config)
        fast_factors = []
        for factor in gm_model.factors:
            tensor = torch.tensor(factor.table, dtype=factor_dtype).to(self.device)
            labels = [var.label for var in factor.vars]
            fast_factors.append(FastFactor(torch.log10(tensor), labels))
        
        # Set elimination order
        self.elim_order = gm_model.elim_order
        
        # Create buckets from FastFactors
        self.buckets = self._create_buckets_from_factors(fast_factors)

    def _load_vars_from_factors(self, factors):
        var_domains = {}
        for factor in factors:
            if factor.labels:
                for i, label in enumerate(factor.labels):
                    if label not in var_domains:
                        var_domains[label] = factor.tensor.shape[i]
        for label, domain_size in var_domains.items():
            self.vars.append(Var(label, domain_size))

    def _create_buckets_from_factors(self, factors):
        if self.elim_order is None:
            raise ValueError("Elimination order must be set before creating buckets")
        
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

        if unplaced_factors and self.is_primary:
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
    def eliminate_variables(self, elim_vars=None, up_to=None, through=None, all=False, all_but=None, exact=False):
        from .factor_nn import FactorNN
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

        # Determine if this is partial elimination (where we need to preserve scalars in buckets)
        is_partial_elimination = not all

        # Create a dummy root bucket to collect the final result
        root_bucket = FastBucket(self, 'root', [], self.device, [], isRoot=True)

        max_width = 0

        for key in self.message_scopes:
            mess_vars = self.message_scopes[key]
            mess_size = int(np.prod([self.matching_var(var).states for var in mess_vars]))
            # if key in vars_to_eliminate and (len(mess_vars) > self.iB or mess_size > self.ecl):
            #     self.num_trained += 1
        large_buckets = self.get_large_message_buckets(iB=self.config['iB'], ecl=self.config['ecl'])
        num_to_train = len(large_buckets)
        for var in vars_to_eliminate:
            try:
                current_bucket = self.buckets[var]
                result = self.process_bucket(current_bucket, exact=exact)
                # Handle both single messages and lists of messages (from WMB)
                messages = result if isinstance(result, list) else [result]

                for message in messages:
                    if not message.is_nn:
                        assert message.tensor is not None
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
                        # Scalar message handling:
                        # - For partial elimination: send to next bucket to preserve for backward factors
                        # - For full elimination: send to root as before
                        if is_partial_elimination:
                            # Find the next bucket in elimination order that still exists
                            var_idx = self.elim_order.index(var)
                            next_bucket_found = None
                            for next_var in self.elim_order[var_idx + 1:]:
                                if next_var in self.buckets:
                                    next_bucket_found = self.buckets[next_var]
                                    break
                            if next_bucket_found:
                                next_bucket_found.receive_message(message)
                            else:
                                root_bucket.receive_message(message)
                        else:
                            root_bucket.receive_message(message)

                # Remove the eliminated variable's bucket
                del self.buckets[var]

            except Exception as e:
                import traceback
                print("\n" + "="*60)
                print("ERROR during variable elimination")
                print("="*60)
                print(f"\nVariable being eliminated: {var}")
                print(f"Intended message scope: {self.message_scopes.get(var, 'N/A')}")

                # Get bucket info
                if var in self.buckets:
                    bucket = self.buckets[var]
                    print(f"\nBucket {var} info:")
                    print(f"  get_width() = {bucket.get_width()}")
                    print(f"  get_ec() = {bucket.get_ec():,}")
                    print(f"  get_message_complexity() = {bucket.get_message_complexity():,}")
                    print(f"  get_message_scope() = {bucket.get_message_scope()}")
                    print(f"\nConfig thresholds:")
                    print(f"  iB = {self.iB}")
                    print(f"  ecl = {self.ecl:,}")
                    print(f"  complexity_limit = {self.complexity_limit:,}")
                    print(f"\nExact check: width<={self.iB}? {bucket.get_width() <= self.iB}, ec<={self.ecl:,}? {bucket.get_ec() <= self.ecl}")
                    print(f"\nBucket {var} factors ({len(bucket.factors)} total):")
                    for i, f in enumerate(bucket.factors):
                        factor_type = "NN" if f.is_nn else "FastFactor"
                        tensor_info = "tensor=None" if f.tensor is None else f"tensor shape={f.tensor.shape}"
                        factor_complexity = f.get_factor_complexity()
                        print(f"  [{i}] {factor_type}: scope={f.labels} (width={len(f.labels)}), complexity={factor_complexity:,}, {tensor_info}")
                else:
                    print(f"\nBucket {var} not found in self.buckets")

                print("\n" + "-"*60)
                print("Full traceback:")
                print("-"*60)
                traceback.print_exc()
                print("="*60 + "\n")
                raise

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
        # Check if we should use exact computation based on width
        bucket_width = bucket.get_width()
        bucket_ec = bucket.get_ec()
        bucket_msg_complexity = bucket.get_message_complexity()

        if (exact or (bucket_width <= self.iB and bucket_ec <= self.ecl)) or bucket_msg_complexity < self.complexity_limit:
            import time as _time
            _t0 = _time.time()
            output_message = bucket.compute_message_exact()
            self.phase_times['exact'] = self.phase_times.get('exact', 0.0) + (_time.time() - _t0)
            if self.gather_message_stats:
                get_message_stats(self, bucket, output_message)
            # begin debug
            if output_message.tensor.isnan().any():
                raise ValueError(f"Output message for bucket {bucket.label} contains NaN values")
            return output_message
        else:
            # CRITICAL FIX: Check if we're in backward factor population mode first
            # When populating backward factors, always use WMB regardless of config
            if self.is_populating_backward_factors:
                output_messages = bucket.compute_wmb_message(self.iB)
                return output_messages
            elif self.config.get('approximation_method') == 'nn':
                print(f"Bucket {bucket.label}: training NN", flush=True)
                import time as _time
                _t0 = _time.time()
                output_message = bucket.compute_message_nn()
                # NN-path wall time (sample gen + training + message construction);
                # pure training time is separable per bucket from training.jsonl.
                self.phase_times['nn_path'] = self.phase_times.get('nn_path', 0.0) + (_time.time() - _t0)
            elif self.config.get('approximation_method') == 'dt':
                print(f"Bucket {bucket.label}: training DT", flush=True)
                output_message = bucket.compute_message_dt()
            elif self.config.get('approximation_method') == 'quantization':
                num_states = self.config.get('quantization_states', self.ecl)
                loss_fn = self.config.get('loss_fn', 'unnormalized_kl')
                print(f"Bucket {bucket.label}: quantizing (K={num_states})", flush=True)
                output_message = bucket.compute_message_quantization(num_states=num_states, loss_fn=loss_fn)
            elif self.config.get('approximation_method') == 'wmb':
                # Use compute_wmb_message which returns a LIST of messages
                # This keeps mini-bucket messages separate to respect ecl
                output_messages = bucket.compute_wmb_message(self.iB)
                # Track WMB partitions
                self.wmb_fw_partitions += bucket.wmb_stats.get('fw_partitions', 0)

                # Return list of messages - caller will handle each separately
                return output_messages
            else:
                raise ValueError(f"Unknown approximation_method: '{self.config.get('approximation_method')}'")

            # Track NN error if enabled
            if self.track_errors:
                from nce.utils.backward_message import get_backward_message

                # Compute exact message for comparison
                exact_message = bucket.compute_message_exact()

                # Get backward message (uses remaining factors, may include NN approximations)
                exact_backward_mg, _ = get_backward_message(
                    self, bucket.label,
                    iB=100,
                    backward_ecl=2**30,
                    approximation_method='wmb'
                )

                # Compute NN error if backward message is not scalar
                if exact_backward_mg.tensor.numel() > 1:
                    # Convert NN factor to exact factor for multiplication
                    nn_factor_exact = output_message.to_exact() if hasattr(output_message, 'to_exact') else output_message
                    nn_contribution = (nn_factor_exact * exact_backward_mg).sum_all_entries()
                    exact_contribution = (exact_message * exact_backward_mg).sum_all_entries()
                    nn_error = float(nn_contribution - exact_contribution)
                    partition_estimate = float(nn_contribution)
                    self.nn_errors.append((bucket.label, nn_error, partition_estimate))
                    print(f"  NN error for bucket {bucket.label}: {nn_error}")
                    print(f"  Partition function estimate: {partition_estimate}")

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

    def _bucket_scope_at_elim(self, key_var):
        """Variables present in bucket at elimination time.

        For a single-var bucket this is message_scopes[var] ∪ {var}. For a
        merged cluster, we cache the cluster scope on the surviving bucket.
        """
        b = self.buckets[key_var]
        cached = getattr(b, '_scope_at_elim', None)
        if cached is not None:
            return cached
        scope = set(self.message_scopes.get(key_var.label, []))
        scope.update(v.label for v in b.elim_vars)
        return scope

    def merge_join_tree(self, verbose=False, max_merge_bound=None, max_cluster_size=None):
        """Merge buckets along the bucket tree wherever the join-tree subsumption
        condition holds: a child whose scope-at-elim-time contains its parent's
        scope can be merged into the parent at zero added complexity.

        max_merge_bound caps the number of eliminated variables per merged
        cluster (None = unbounded full subsumption).

        Pre-condition: calculate_message_scopes() has been called.

        After this pass, self.buckets contains a mix of single-var buckets and
        merged clusters (FastBucket with len(elim_vars) > 1). Buckets absorbed
        into a parent are removed from self.buckets but stay in self.elim_order
        (the eliminate_variables loop must skip vars not in self.buckets).
        """
        # Backward compatibility: max_cluster_size was the previous name for
        # max_merge_bound. Honor it if the new name was not supplied.
        if max_cluster_size is not None and max_merge_bound is None:
            max_merge_bound = max_cluster_size

        # Initialise scope cache on each bucket
        for key_var, b in self.buckets.items():
            b._scope_at_elim = set(self.message_scopes.get(key_var.label, []))
            b._scope_at_elim.update(v.label for v in b.elim_vars)

        merges = []  # log of (child_elim_var_labels, parent_elim_var_labels_before, after)

        for var in list(self.elim_order):
            if var not in self.buckets:
                continue  # absorbed earlier
            cur_var = var
            while True:
                cur_bucket = self.buckets[cur_var]
                cur_scope = cur_bucket._scope_at_elim
                msg_scope = cur_scope - {v.label for v in cur_bucket.elim_vars}
                parent_bucket = self.find_next_bucket(list(msg_scope), cur_var)
                if parent_bucket is None or parent_bucket is cur_bucket:
                    break
                parent_scope = parent_bucket._scope_at_elim
                # Optional safeguard: cap the number of elim_vars per cluster (merge bound).
                merged_size = len(set(cur_bucket.elim_vars + parent_bucket.elim_vars))
                if max_merge_bound is not None and merged_size > int(max_merge_bound):
                    break
                if cur_scope >= parent_scope:
                    # Absorb cur into parent
                    child_elim_labels = sorted(v.label for v in cur_bucket.elim_vars)
                    parent_elim_before = sorted(v.label for v in parent_bucket.elim_vars)
                    for v in cur_bucket.elim_vars:
                        if v not in parent_bucket.elim_vars:
                            parent_bucket.elim_vars.append(v)
                    parent_bucket.factors = list(cur_bucket.factors) + list(parent_bucket.factors)
                    # New cluster scope is the union; by the condition this equals cur_scope.
                    parent_bucket._scope_at_elim = cur_scope | parent_scope
                    parent_key = next(k for k, v in self.buckets.items() if v is parent_bucket)
                    del self.buckets[cur_var]
                    parent_elim_after = sorted(v.label for v in parent_bucket.elim_vars)
                    merges.append({
                        'child_elim_vars': child_elim_labels,
                        'parent_elim_vars_before': parent_elim_before,
                        'cluster_elim_vars': parent_elim_after,
                        'cluster_scope': sorted(parent_bucket._scope_at_elim),
                    })
                    if verbose:
                        print(f"[merge_join_tree] absorb {child_elim_labels} into "
                              f"{parent_elim_before} → cluster {parent_elim_after} "
                              f"(scope {sorted(parent_bucket._scope_at_elim)})")
                    cur_var = parent_key
                else:
                    break

        self._merges = merges
        if verbose:
            print(f"[merge_join_tree] {len(merges)} merges performed; "
                  f"{len(self.buckets)} buckets/clusters remain "
                  f"(was {len(self.elim_order)} original).")
        return merges

    def merge_non_subsumption(self, verbose=False, max_merge_bound=None):
        """Merge buckets along the bucket tree ONLY where the join-tree subsumption
        condition does NOT hold — the complement of merge_join_tree. Each merge
        strictly grows the cluster scope (costly merges), capped at max_merge_bound
        eliminated vars per cluster. Subsumption (free) merges are skipped.

        Purpose: isolate how much scope-growing merges alone reduce error,
        vs the free subsumption merges.

        Pre-condition: calculate_message_scopes() has been called.
        """
        for key_var, b in self.buckets.items():
            b._scope_at_elim = set(self.message_scopes.get(key_var.label, []))
            b._scope_at_elim.update(v.label for v in b.elim_vars)

        merges = []
        for var in list(self.elim_order):
            if var not in self.buckets:
                continue  # absorbed earlier
            cur_var = var
            while True:
                cur_bucket = self.buckets[cur_var]
                cur_scope = cur_bucket._scope_at_elim
                msg_scope = cur_scope - {v.label for v in cur_bucket.elim_vars}
                parent_bucket = self.find_next_bucket(list(msg_scope), cur_var)
                if parent_bucket is None or parent_bucket is cur_bucket:
                    break
                parent_scope = parent_bucket._scope_at_elim
                merged_size = len(set(cur_bucket.elim_vars + parent_bucket.elim_vars))
                if max_merge_bound is not None and merged_size > int(max_merge_bound):
                    break
                if cur_scope >= parent_scope:
                    break  # subsumption (free) merge -> NOT taken in this mode
                # Non-subsumption merge: absorb cur into parent (scope grows)
                child_elim_labels = sorted(v.label for v in cur_bucket.elim_vars)
                parent_elim_before = sorted(v.label for v in parent_bucket.elim_vars)
                for v in cur_bucket.elim_vars:
                    if v not in parent_bucket.elim_vars:
                        parent_bucket.elim_vars.append(v)
                parent_bucket.factors = list(cur_bucket.factors) + list(parent_bucket.factors)
                parent_bucket._scope_at_elim = cur_scope | parent_scope
                parent_key = next(k for k, v in self.buckets.items() if v is parent_bucket)
                del self.buckets[cur_var]
                merges.append({
                    'child_elim_vars': child_elim_labels,
                    'parent_elim_vars_before': parent_elim_before,
                    'cluster_elim_vars': sorted(v.label for v in parent_bucket.elim_vars),
                    'cluster_scope': sorted(parent_bucket._scope_at_elim),
                })
                if verbose:
                    print(f"[merge_non_subsumption] absorb {child_elim_labels} into "
                          f"{parent_elim_before} → cluster "
                          f"{sorted(v.label for v in parent_bucket.elim_vars)}")
                cur_var = parent_key

        self._merges = merges
        if verbose:
            print(f"[merge_non_subsumption] {len(merges)} merges performed; "
                  f"{len(self.buckets)} buckets/clusters remain "
                  f"(was {len(self.elim_order)} original).")
        return merges

    def merge_by_degree(self, merge_degree, verbose=False):
        """Greedy merge of adjacent NN-eligible buckets along the bucket tree.

        Limits each resulting cluster to at most `merge_degree` elimination
        variables. Only considers NN-eligible buckets (those whose outgoing
        message exceeds ecl or iB). At each step the candidate merge that
        ADDS THE FEWEST scope variables is chosen (subsumption merges add 0
        and therefore win automatically).

        Can be run after merge_join_tree() to combine subsumption-based
        merges with degree-bounded merges, OR standalone.
        """
        ecl = float(self.ecl or float('inf'))
        iB = float(self.iB or float('inf'))

        # Ensure scope-at-elim cache exists on every bucket.
        for key_var, b in self.buckets.items():
            if getattr(b, '_scope_at_elim', None) is None:
                b._scope_at_elim = set(self.message_scopes.get(key_var.label, []))
                b._scope_at_elim.update(v.label for v in b.elim_vars)

        def out_scope(b):
            return b._scope_at_elim - {v.label for v in b.elim_vars}

        def is_nn_eligible(b):
            scope = out_scope(b)
            size = 1
            for label in scope:
                size *= self.matching_var(label).states
            return size > ecl or len(scope) > iB

        merges_done = []
        while True:
            candidates = []
            for var in self.elim_order:
                if var not in self.buckets:
                    continue
                cur = self.buckets[var]
                if not is_nn_eligible(cur):
                    continue
                parent = self.find_next_bucket(list(out_scope(cur)), var)
                if parent is None or parent is cur:
                    continue
                if not is_nn_eligible(parent):
                    continue
                new_size = len(set(cur.elim_vars + parent.elim_vars))
                if new_size > int(merge_degree):
                    continue
                added = len(cur._scope_at_elim | parent._scope_at_elim) - len(parent._scope_at_elim)
                candidates.append((added, self.elim_order.index(var), var, parent))
            if not candidates:
                break
            candidates.sort(key=lambda x: (x[0], x[1]))  # min added scope, then earliest var
            added, _, var, parent = candidates[0]
            cur = self.buckets[var]
            for v in cur.elim_vars:
                if v not in parent.elim_vars:
                    parent.elim_vars.append(v)
            parent.factors = list(cur.factors) + list(parent.factors)
            parent._scope_at_elim = cur._scope_at_elim | parent._scope_at_elim
            del self.buckets[var]
            merges_done.append({
                'child_elim_vars': sorted(v.label for v in cur.elim_vars),
                'parent_elim_vars': sorted(v.label for v in parent.elim_vars),
                'added_scope_vars': added,
            })
            if verbose:
                print(f"[merge_by_degree] absorb {sorted(v.label for v in cur.elim_vars)} "
                      f"into cluster {sorted(v.label for v in parent.elim_vars)} "
                      f"(added {added} scope vars)")

        if not hasattr(self, '_merges') or self._merges is None:
            self._merges = []
        self._merges.extend(merges_done)
        if verbose:
            print(f"[merge_by_degree] {len(merges_done)} merges; "
                  f"{len(self.buckets)} buckets/clusters remain.")
        return merges_done

    def reduce_nn_merge(self, max_merge_bound=None, verbose=False, backtrack=False):
        """Greedy NN-count-reducing merge (NOT subsumption), with optional backtrack.

        Goal: minimize the number of NN-approximated buckets. For each NN
        bucket, absorb its ancestor chain (up the bucket tree) -- the message may
        grow mid-chain (lookahead) -- until the cluster either COLLAPSES to exact
        (output message width <= iB and size <= ecl) or MERGES into another NN;
        either outcome drops the NN count by >=1. Repeat until no NN bucket can be
        reduced within max_merge_bound (cap on eliminated vars per cluster).

        With backtrack=True: the greedy can over-grow clusters when the cap leaves
        head-room (a bigger cap lets it take longer chains, so the same NN count is
        reached with needlessly large clusters -- e.g. rbm_22 reaches 1 NN with a
        24-var cluster at D=24 but a 28-var one at D=28). Backtracking finds the
        minimum NN count reachable within the cap, then re-runs the merge at the
        SMALLEST bound that still achieves it -- the least merging (smallest
        clusters) for the best NN count. The result then depends only on the NN
        count achieved, not on spare cap.

        Each committed chain is a connected path in the bucket tree, so the
        merge is elimination-order valid (same mechanics as merge_join_tree).

        Pre-condition: calculate_message_scopes() has been called.
        """
        D = None if max_merge_bound is None else int(max_merge_bound)
        if not (backtrack and D is not None):
            n_nn, merges = self._reduce_nn_core(D, verbose=verbose)
            if not getattr(self, '_merges', None):
                self._merges = []
            self._merges.extend(merges)
            return merges

        # --- backtrack: smallest bound achieving the best (min) NN count ---
        snap = self._reduce_nn_snapshot()
        k_target, _ = self._reduce_nn_core(D, verbose=False)
        lo, hi, best = 1, D, D                      # K(D') is ~non-increasing in D'
        while lo <= hi:
            mid = (lo + hi) // 2
            self._reduce_nn_restore(snap)
            k_mid, _ = self._reduce_nn_core(mid, verbose=False)
            if k_mid <= k_target:
                best = mid
                hi = mid - 1
            else:
                lo = mid + 1
        self._reduce_nn_restore(snap)
        n_nn, merges = self._reduce_nn_core(best, verbose=False)
        if not getattr(self, '_merges', None):
            self._merges = []
        self._merges.extend(merges)
        if verbose:
            print(f"[reduce_nn_merge:backtrack] cap D={D}: min NN={k_target} first "
                  f"reached at D={best}; using D={best} ({len(merges)} chain-merges, "
                  f"{len(self.buckets)} clusters, {n_nn} NN).")
        return merges

    def _reduce_nn_snapshot(self):
        """Snapshot pristine bucket state so the merge can be re-run at different
        bounds during backtracking. Holds references to the bucket objects (some are
        deleted on merge) plus copies of their mutable elim_vars/factors lists."""
        return {kv: (b, list(b.elim_vars), list(b.factors))
                for kv, b in self.buckets.items()}

    def _reduce_nn_restore(self, snap):
        """Restore buckets to the snapshotted pristine state (undo a trial merge)."""
        self.buckets = {kv: b for kv, (b, _, _) in snap.items()}
        for kv, (b, ev, fc) in snap.items():
            b.elim_vars = list(ev)
            b.factors = list(fc)
            b._scope_at_elim = None              # recomputed by _reduce_nn_core
        return

    def _reduce_nn_core(self, max_merge_bound=None, verbose=False):
        """One greedy NN-count-reducing merge pass at the given bound. Mutates
        self.buckets in place; returns (n_nn_remaining, merges_list)."""
        iB = float(self.iB) if self.iB else float('inf')
        ecl = float(self.ecl) if self.ecl else float('inf')
        D = None if max_merge_bound is None else int(max_merge_bound)
        pos = {var: i for i, var in enumerate(self.elim_order)}
        states = {v.label: self.matching_var(v.label).states for v in self.vars}

        for key_var, b in self.buckets.items():
            b._scope_at_elim = set(self.message_scopes.get(key_var.label, []))
            b._scope_at_elim.update(v.label for v in b.elim_vars)

        # label -> the bucket key_var that eliminates that label (updated on merge),
        # so the parent of a cluster is found in O(message size) not O(#vars).
        owner = {v.label: kv for kv in self.buckets for v in self.buckets[kv].elim_vars}

        def states_prod(labels):
            p = 1.0
            for l in labels:
                p *= states[l]
            return p

        def is_nn(scope, elim_labels):
            out = scope - elim_labels
            return len(out) > iB or states_prod(out) > ecl

        def find_chain(start_var):
            scope = set(self.buckets[start_var]._scope_at_elim)
            elim = {v.label for v in self.buckets[start_var].elim_vars}
            chain = [start_var]; seen = {start_var}
            while True:
                out = scope - elim
                if not (len(out) > iB or states_prod(out) > ecl):
                    return chain                        # collapsed to exact
                # parent = bucket eliminating the earliest (in elim order) message var
                cand = {owner[l] for l in out}
                cand = [kv for kv in cand if kv not in seen]
                if not cand:
                    return None
                pvar = min(cand, key=lambda kv: pos[kv])
                pb = self.buckets[pvar]
                p_elim = {v.label for v in pb.elim_vars}
                if D is not None and len(elim | p_elim) > D:
                    return None
                p_is_nn = is_nn(set(pb._scope_at_elim), p_elim)
                elim |= p_elim; scope |= pb._scope_at_elim
                chain.append(pvar); seen.add(pvar)
                if p_is_nn:
                    return chain                        # merged into another NN

        merges = []
        while True:
            progressed = False
            nn_vars = sorted(
                (kv for kv, b in self.buckets.items()
                 if is_nn(set(b._scope_at_elim), {v.label for v in b.elim_vars})),
                key=lambda kv: pos[kv])
            for kv in nn_vars:
                if kv not in self.buckets:
                    continue
                chain = find_chain(kv)
                if chain and len(chain) > 1:
                    # Per-step tags (P3 enabling-bookkeeping): walking the chain from
                    # its seed upward, record how many scope vars each absorbed parent
                    # adds (0 = subsumption/free step) and the parent's identity, so a
                    # structure pass can tell which free steps exist only because prior
                    # (scope-growing) steps enlarged the child's scope.
                    step_tags = []
                    _cum = set(self.buckets[chain[0]]._scope_at_elim)
                    for other in chain[1:]:
                        _os = set(self.buckets[other]._scope_at_elim)
                        step_tags.append({
                            'parent_key': other.label,
                            'parent_elim_vars': sorted(v.label for v in self.buckets[other].elim_vars),
                            'added_scope_vars': len(_os - _cum),
                        })
                        _cum |= _os
                    key = chain[-1]                     # topmost (latest-eliminated)
                    kb = self.buckets[key]
                    for other in chain[:-1]:
                        ob = self.buckets[other]
                        for v in ob.elim_vars:
                            if v not in kb.elim_vars:
                                kb.elim_vars.append(v)
                            owner[v.label] = key
                        kb.factors = list(ob.factors) + list(kb.factors)
                        kb._scope_at_elim = set(kb._scope_at_elim) | set(ob._scope_at_elim)
                        del self.buckets[other]
                    merges.append({
                        'cluster_elim_vars': sorted(v.label for v in kb.elim_vars),
                        'seed_key': chain[0].label,
                        'steps': step_tags,
                    })
                    progressed = True
                    break
            if not progressed:
                break

        n_nn = sum(1 for kv, b in self.buckets.items()
                   if is_nn(set(b._scope_at_elim), {v.label for v in b.elim_vars}))
        if verbose:
            print(f"[reduce_nn_merge] {len(merges)} chain-merges; "
                  f"{len(self.buckets)} clusters remain, {n_nn} NN.")
        return n_nn, merges

    def print_join_tree(self):
        """Human-readable dump of all buckets/clusters and their scopes."""
        print("Join-tree clusters (in elim order):")
        for var in self.elim_order:
            if var not in self.buckets:
                continue
            b = self.buckets[var]
            elim = sorted(v.label for v in b.elim_vars)
            scope = sorted(getattr(b, '_scope_at_elim', set()))
            n_factors = len(b.factors)
            tag = "cluster" if len(elim) > 1 else "bucket"
            print(f"  {tag} key={var.label} elim={elim} scope={scope} n_factors={n_factors}")
        if getattr(self, '_merges', None):
            print(f"Total merges: {len(self._merges)}")
    
    def show_elimination(self, elim_vars=None, up_to=None, through=None, all=False, complexity=False):
        if sum(map(bool, [elim_vars, up_to, through, all])) != 1:
            print("Showing all eliminations")
            all = True
            # raise ValueError("Exactly one of elim_vars, up_to, through, or all must be specified")

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
            if complexity:
                # Compute factor complexity: sum of table sizes for all
                # factors in the bucket (originals + incoming messages).
                # Can't use bucket.get_message_complexity() since incoming
                # messages haven't been moved into the bucket yet.
                factor_complexity = 0
                # Original factors in the bucket
                for f in bucket_factors:
                    size = 1
                    for v in f.labels:
                        size *= self.matching_var(v).states
                    factor_complexity += size
                # Incoming messages from earlier buckets
                for msg_scope in incoming_messages:
                    size = 1
                    for v in msg_scope:
                        size *= self.matching_var(v).states
                    factor_complexity += size
                bucket_info['message_complexity'] = factor_complexity
                # Compute message size from the scope we already derived,
                # since bucket.get_message_size() relies on factors/messages
                # having been moved during elimination (which hasn't happened).
                msg_size = 1
                for v in outgoing_message_vars:
                    msg_size *= self.matching_var(v).states
                bucket_info['message_size'] = msg_size

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
            if complexity:
                print(f"  Factor complexity: {bucket['message_complexity']:,}")
                print(f"  Message size: {int(bucket['message_size']):,}")
            print()

        print(f"Maximum width: {max_width}")
        if complexity:
            total_complexity = sum(b['message_complexity'] for b in elimination_scheme)
            max_msg_size = max((b['message_size'] for b in elimination_scheme), default=0)
            print(f"Total factor complexity: {total_complexity:,}")
            print(f"Maximum message size: {int(max_msg_size):,}")

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

    def get_adjusted_width(self, ecl=None):
        """
        Calculate the adjusted width of the graphical model.

        The adjusted width is the complexity of the largest message that would be
        sent with exact inference. It's defined as the log2 of the maximum product
        of domain sizes across all message scopes.

        Args:
            ecl: Optional exact computation limit. If provided, counts how many
                 messages have size strictly greater than this limit.

        Returns:
            tuple: (adjusted_width, num_exceeding_ecl) where:
                - adjusted_width: log2 of the maximum product of domain sizes
                - num_exceeding_ecl: count of messages exceeding ecl (or -1 if ecl not provided)

        Examples:
            >>> gm.get_adjusted_width()
            (4.91, -1)  # Largest message has size 2*3*5=30, no ecl provided

            >>> gm.get_adjusted_width(ecl=1000)
            (10.1, 2)  # Largest message has size 1100, 2 messages exceed ecl=1000
        """
        max_complexity = 0
        num_exceeding = 0 if ecl is not None else -1

        for key in self.message_scopes:
            # Compute product of domain sizes for this message scope
            prod = 1
            for idx in self.message_scopes[key]:
                if idx != key:  # Exclude the variable being eliminated
                    var = self.matching_var(idx)
                    prod *= var.states

            # Track the maximum
            max_complexity = max(max_complexity, prod)

            # Count messages exceeding ecl if provided
            if ecl is not None and prod > ecl:
                num_exceeding += 1

        # Return log2 of the maximum complexity and count of exceeding messages
        if max_complexity == 0:
            return 0.0, num_exceeding
        return math.log2(max_complexity), num_exceeding

    def get_large_message_buckets(self, iB=None, ecl=None, debug=False):
        """
        Get buckets with large messages that will need NN approximation.

        A bucket is considered "large" if EITHER:
        - Number of variables in message scope > iB (if iB is provided)
        - Number of elements in message (product of states) > ecl (if ecl is provided)

        Args:
            iB: Maximum number of variables allowed (optional)
            ecl: Maximum number of elements allowed (optional)
            debug: Print debug information

        Returns:
            List of bucket keys that have large messages
        """
        large_buckets = []

        for key in self.message_scopes:
            scope = self.message_scopes[key]
            num_vars = len(scope)

            # Compute message size (product of states for all vars in scope except the eliminated var)
            message_size = 1
            for idx in scope:
                if idx != key:
                    var = self.matching_var(idx)
                    message_size *= var.states

            # Check if bucket is large (either condition triggers)
            is_large = False
            if iB is not None and num_vars > iB:
                is_large = True
            if ecl is not None and message_size > ecl:
                is_large = True

            if is_large:
                large_buckets.append(key)
                if debug:
                    print('------------------------------------')
                    print('Key is ', key)
                    print('Scope is ', scope)
                    print(f'num_vars: {num_vars}, message_size: {message_size}')
                    adj_num_vars = math.log2(message_size) if message_size > 0 else 0
                    print(f'bucket: {key}, num vars: {num_vars}, adj num vars: {adj_num_vars:.2f}')

        return large_buckets
    
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

    def get_all_factors(self) -> List[FastFactor]:
        """
        Returns all factors from all buckets as a list WITHOUT multiplying them.
        Used for batched learning where we don't want to materialize the full product.
        """
        all_factors = []
        for bucket in self.buckets.values():
            for factor in bucket.factors:
                all_factors.append(factor)
        return all_factors

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
                self.buckets[self.matching_var(earliest_var)].factors.append(factor)
                
                # self.buckets[var].factors.append(factor)

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
            return FastFactor(tensor=torch.tensor(0.0, device=self.device, requires_grad=False), labels=[])
        
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

    def _wmb_eliminate(gm, target_scope, i_bound, weights, combine_factors=False, ecl=None):
        """
        Perform Weighted Mini-Bucket elimination.

        Args:
        gm (FastGM): The graphical model to eliminate.
        target_scope (list): The variables to keep (not eliminate).
        i_bound (int): The maximum allowed scope size for mini-buckets. Only used if ecl is None.
        weights (str or list): Weights for WMB.
        combine_factors (bool): Whether to combine remaining factors at the end.
        ecl (int): Exact complexity limit (max tensor entries). If provided, completely overrides i_bound.

        Returns:
        FastFactor: The result of WMB elimination.
        """
        # Use gm.ecl if ecl not explicitly provided
        if ecl is None:
            ecl = getattr(gm, 'ecl', None)

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

        for var in gm.elim_order:
            if var.label in target_scope:
                continue

            bucket = gm.get_bucket(var)

            if not bucket.factors:  # Skip empty buckets
                continue

            # Determine if we need mini-buckets based on ecl (preferred) or i_bound (fallback)
            needs_mini_buckets = False
            if ecl is not None and ecl > 0:
                # Use ecl: check if message complexity exceeds ecl
                bucket_ec = bucket.get_ec()  # Product of domain sizes
                needs_mini_buckets = bucket_ec > ecl
            else:
                # Fallback to i_bound: check variable count
                needs_mini_buckets = bucket.get_width() > i_bound

            if not needs_mini_buckets:
                message = FastGM._compute_weighted_message(bucket.factors, var, weight_map[var.label])
                gm.removeFactors(bucket.factors)
                gm.addFactors([message])
            else:
                mini_buckets = FastGM._create_mini_buckets(bucket.factors, i_bound, gm, ecl)
                for i, mini_bucket in enumerate(mini_buckets):
                    mini_weight = weight_map[var.label] / len(mini_buckets)
                    if i == len(mini_buckets) - 1:  # Adjust the last mini-bucket weight
                        mini_weight = weight_map[var.label] - (len(mini_buckets) - 1) * mini_weight
                    mini_message = FastGM._compute_weighted_message(mini_bucket, var, mini_weight)
                    gm.removeFactors(mini_bucket)
                    gm.addFactors([mini_message])

        # After elimination, combine all remaining factors
        remaining_factors = []
        for bucket in gm.buckets.values():
            remaining_factors.extend(bucket.factors)
            
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
                except Exception:
                    raise ValueError("scope incorrect")
                if (result.tensor == float('inf')).any():
                    print("inf found")
                    print(factor.labels)
                    raise ValueError("inf found")
        else:
            # If no factors remain, return a scalar factor with value 0 (in log space)
            result = FastFactor(torch.tensor(0.0, device=gm.device, requires_grad=False), [])
        if (result.tensor == float('inf')).any():
            print("inf found")
            raise ValueError("inf found")
        return result

    def _create_mini_buckets(factors, i_bound, gm=None, ecl=None):
        """
        Partition factors into mini-buckets respecting ecl (complexity limit).

        When ecl is provided, it completely overrides i_bound - mini-buckets are created
        based solely on the product of domain sizes (tensor entry count), not variable count.

        Args:
            factors: List of factors to partition
            i_bound: Max variables per mini-bucket (only used if ecl is None)
            gm: Graphical model (needed to look up variable states for ecl calculation)
            ecl: Exact complexity limit (max tensor entries). If provided, overrides i_bound.
        """
        mini_buckets = []
        # Sort by tensor size (numel) instead of variable count
        sorted_factors = sorted(factors, key=lambda f: f.tensor.numel(), reverse=True)

        for factor in sorted_factors:
            placed = False
            for bucket in mini_buckets:
                combined_scope = set.union(*[set(f.labels) for f in bucket], set(factor.labels))

                # If ecl is set and we have gm, use complexity (product of domain sizes)
                if ecl is not None and ecl > 0 and gm is not None:
                    combined_complexity = 1
                    for var_label in combined_scope:
                        var = gm.matching_var(var_label)
                        combined_complexity *= var.states
                    if combined_complexity <= ecl:
                        bucket.append(factor)
                        placed = True
                        break
                else:
                    # Fallback to i_bound (variable count)
                    if len(combined_scope) <= i_bound:
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

    def _create_population_copy(self):
        """
        Create a controlled copy of this GM for backward factor population.

        Uses explicit construction instead of deepcopy to avoid config dict sharing issues.
        The copy is configured to use WMB and not create its own copies.

        Returns:
            FastGM: A new GM instance configured for WMB-based backward factor computation
        """
        import copy

        # CRITICAL: Create NEW config dict using dict() constructor, NOT deepcopy
        # This ensures each GM has its own independent config dict
        pop_config = dict(self.config)

        # Configure for population: use WMB, don't populate recursively
        pop_config['populate_bw_factors'] = False  # Prevent recursive population
        pop_config['approximation_method'] = 'wmb'  # Use WMB for backward factors
        pop_config['use_join_tree_merge'] = False  # Don't re-merge in the copy

        # Get all factors from original GM's buckets. Skip vars whose buckets
        # were absorbed into a merged cluster (their factors live on the
        # surviving cluster's bucket already).
        all_factors = []
        for var in self.elim_order:
            if var not in self.buckets:
                continue
            bucket = self.buckets[var]
            for factor in bucket.factors:
                # Deep copy each factor to avoid sharing
                all_factors.append(copy.deepcopy(factor))

        # Create new GM with explicit parameters (NOT deepcopy of entire GM)
        copied_gm = FastGM(
            factors=all_factors,
            elim_order=list(self.elim_order),  # Copy the list
            device=self.device,
            reference_fastgm=self,
            nn_config=pop_config,  # Use the NEW config dict
            stats=self.stats
        )

        # Mark as copy to prevent further copying
        copied_gm.is_primary = False
        copied_gm.is_populating_backward_factors = True

        return copied_gm

    def _wmb_eliminate_to_scope(self, factors, target_scope, bucket_var):
        """
        WMB-eliminate a list of factors down to a target variable scope.

        Returns a list of factors (not a single product) to keep sizes bounded
        by bw_ecl. This mirrors get_backward_message(..., return_factor_list=True)
        but works directly on a factor list without needing a full GM copy.

        Args:
            factors: List of FastFactor objects to eliminate
            target_scope: List of variable labels to keep (not eliminate)
            bucket_var: The Var object for the bucket (used for logging)

        Returns:
            List of FastFactor objects over target_scope variables
        """
        import copy as copy_module

        # Separate scalar factors from non-scalar
        scalar_factors = [f for f in factors if not f.labels]
        non_scalar_factors = [f for f in factors if f.labels]

        # Accumulate scalar constants
        scalar_constant = torch.tensor(0.0, device=self.device, requires_grad=False)
        for sf in scalar_factors:
            scalar_constant = scalar_constant + sf.tensor.squeeze()

        if not non_scalar_factors:
            if scalar_constant != 0.0:
                return [FastFactor(scalar_constant, [])]
            return []

        # Check if any variables actually need to be eliminated
        all_vars = set()
        for f in non_scalar_factors:
            all_vars.update(f.labels)
        vars_to_eliminate = all_vars - set(target_scope)

        if not vars_to_eliminate:
            # Nothing to eliminate — just return factors as-is (plus scalar)
            result = list(non_scalar_factors)
            if scalar_constant != 0.0:
                result[0] = FastFactor(result[0].tensor + scalar_constant, result[0].labels)
            return result

        # Compute elimination order for the variables we need to remove
        elim_order = wtminfill_order(non_scalar_factors, variables_not_eliminated=target_scope)

        # Build a temporary GM for elimination
        bw_ecl = self.config.get('bw_ecl', 0)
        downstream_config = dict(self.config)
        downstream_config['populate_bw_factors'] = False
        downstream_config['approximation_method'] = 'wmb'
        downstream_config['ecl'] = bw_ecl
        # Don't re-merge in the temporary GM
        downstream_config['use_join_tree_merge'] = False
        downstream_config['merge_degree'] = 0

        # Compute effective iB from ecl
        effective_iB = int(math.log2(bw_ecl)) if bw_ecl > 0 else 0
        downstream_config['iB'] = effective_iB

        downstream_gm = FastGM(
            factors=non_scalar_factors,
            elim_order=elim_order,
            reference_fastgm=self,
            device=self.device,
            nn_config=downstream_config,
            stats=self.stats
        )
        downstream_gm.is_primary = False

        # Eliminate all variables except target scope
        downstream_gm.eliminate_variables(all_but=target_scope)

        # Collect remaining factors
        result_factors = downstream_gm.get_all_factors()

        if not result_factors:
            if scalar_constant != 0.0:
                return [FastFactor(scalar_constant, [])]
            return []

        # Add scalar constant to first factor
        if scalar_constant != 0.0:
            first = result_factors[0]
            result_factors[0] = FastFactor(first.tensor + scalar_constant, first.labels)

        return result_factors

    def populate_backward_factors_wmb(self):
        """
        Pre-compute WMB backward factors for all buckets before main elimination.

        This method creates a copy of the graphical model and performs elimination
        using WMB approximations (instead of NNs) when buckets exceed ecl/iB thresholds.
        The resulting factors are stored in each original bucket's approximate_downstream_factors
        field for later use during NN training.

        Key features:
        - Uses WMB approximation when bucket width > iB OR complexity > ecl
        - Keeps WMB factors together when placing in downstream buckets
        - Factors are placed in the earliest bucket that contains any of their variables
        """
        # Create a controlled copy using factory method (NOT deepcopy)
        # This avoids config dict sharing issues
        copied_gm = self._create_population_copy()

        # Get sender-receiver scheme for proper message routing
        scheme_info = copied_gm.get_senders_receivers()
        scheme = {info['var']: info['sends_to'] for info in scheme_info}

        # Optional optimization: skip the per-bucket WMB-eliminate step for
        # buckets that won't use NN approximation in forward elim (i.e.,
        # bucket_ec <= self.ecl ⇒ exact message in forward, no proposal sampling).
        # Non-NN buckets don't read approximate_*_factors, so we can leave them
        # as None and save the O(N) WMB-eliminate work per non-NN bucket.
        skip_non_nn = self.config.get('populate_bw_skip_non_nn', False)
        n_skipped = 0

        # Process buckets in elimination order
        n_total = len(self.elim_order)
        for i, current_var in enumerate(self.elim_order):
            if i % 50 == 0 or i == n_total - 1:
                print(f"  populate_bw_factors: bucket {i+1}/{n_total} (var={current_var.label})", flush=True)
            # Get corresponding buckets from original and copied GMs
            orig_bucket = self.get_bucket(current_var)
            copy_bucket = copied_gm.get_bucket(current_var)

            # NN-eligibility check (forward gate): if bucket_ec > self.ecl in original
            # forward, this bucket will be NN-approximated and may use bw factors via
            # proposal sampling. Else it will be exact and ignores bw factors.
            # copy_bucket.get_ec() reflects scope (var domain product) which is
            # invariant to WMB partitioning, so it's a valid proxy for forward bucket_ec.
            is_nn_eligible = (not skip_non_nn) or (copy_bucket.get_ec() > self.ecl)

            if is_nn_eligible:
                # Gather all downstream factors from copied GM and WMB-eliminate them
                # down to the bucket scope (elim var + message scope).
                # This keeps factor sizes bounded by bw_ecl.
                downstream_factors_raw = []
                for j in range(i+1, len(self.elim_order)):
                    downstream_bucket = copied_gm.get_bucket(self.elim_order[j])
                    for factor in downstream_bucket.factors:
                        downstream_factors_raw.append(factor.to_exact() if hasattr(factor, 'to_exact') else factor)

                bucket_scope = orig_bucket.get_message_scope() + [current_var.label]
                orig_bucket.approximate_downstream_factors = self._wmb_eliminate_to_scope(
                    downstream_factors_raw, bucket_scope, current_var)

                # Save the forward factors: original factors + upstream WMB messages
                upstream_factors = []
                for factor in copy_bucket.factors:
                    upstream_factors.append(factor.to_exact() if hasattr(factor, 'to_exact') else factor)
                orig_bucket.approximate_upstream_factors = upstream_factors
            else:
                # Skip — non-NN bucket won't use these.
                orig_bucket.approximate_downstream_factors = None
                orig_bucket.approximate_upstream_factors = None
                n_skipped += 1

            # Separate factors into those with and without the elimination variable
            has_elim_var = [f for f in copy_bucket.factors if current_var in f.labels]
            no_elim_var = [f for f in copy_bucket.factors if current_var not in f.labels]

            # Temporarily set bucket factors to only those with elimination variable
            copy_bucket.factors = has_elim_var

            if not has_elim_var:
                # No factors to eliminate, just move independent factors
                messages = []
            else:
                # Check if we need approximation (width > iB OR complexity > ecl)
                bucket_width = copy_bucket.get_width()
                bucket_ec = copy_bucket.get_ec()

                bw_ecl = self.config.get('bw_ecl')
                needs_approx = bucket_ec > bw_ecl

                if needs_approx:
                    # Use WMB approximation - returns list of factors
                    messages = copy_bucket.compute_wmb_message(ecl=bw_ecl)
                    # Track forward partitions (upper bound for backward message impact)
                    self.wmb_fw_partitions += copy_bucket.wmb_stats.get('fw_partitions', 0)
                else:
                    # Compute exact message - wrap in list for consistency
                    try:
                        exact_msg = copy_bucket.compute_message_exact()
                        messages = [exact_msg]
                    except Exception as e:
                        print(f"    Warning: Exact computation failed: {e}, falling back to WMB")
                        messages = copy_bucket.compute_wmb_message(ecl=bw_ecl)

            # Combine WMB messages and independent factors
            all_outgoing_factors = messages + no_elim_var

            # Place all outgoing factors in the next bucket
            # IMPORTANT: Keep WMB factors together by finding the earliest bucket
            # that contains ANY variable from ANY of the outgoing factors
            if all_outgoing_factors and scheme[current_var] is not None:
                next_var_label = scheme[current_var]  # This is an int (label), not a Var
                next_bucket = copied_gm.get_bucket(next_var_label)
                next_bucket.factors.extend(all_outgoing_factors)

        if skip_non_nn:
            print(f"  populate_bw skipped {n_skipped}/{n_total} non-NN buckets")

        # Print summary of forward partitions during backward factor population
        if self.wmb_fw_partitions > 0:
            print(f"  WMB forward partitions during backward factor population: {self.wmb_fw_partitions} (upper bound)")

    def populate_backward_factors_via_tree_collect(self):
        """Junction-tree backward distribute that just COLLECTS factor lists.

        For each bucket X:
            approximate_downstream_factors[X] = bw(parent) + parent.originals
                                              + (forward msgs from parent's children except X)

        Stored as a list of FastFactor — NOT eliminated. The lazy WMB-elim
        happens later when a consumer (e.g. get_backward_message during NN
        training) needs the actual bw message.

        Cost: O(N) per-bucket bookkeeping. Per-NN-bucket elim cost amortized to
        when the bucket actually trains (≈ N_NN × WMB-elim of ~depth*branching
        small factors).

        Each factor in approximate_downstream_factors is a Python reference to
        a shared FastFactor object — sharing across buckets means memory cost
        is O(distinct factors) not O(N × bucket_factors).
        """
        copied_gm = self._create_population_copy()
        scheme_info = copied_gm.get_senders_receivers()
        parent_of = {info['var'].label: info['sends_to'] for info in scheme_info}
        children_of = {label: [] for label in parent_of}
        roots = []
        for child, par in parent_of.items():
            if par is None:
                roots.append(child)
            else:
                children_of[par].append(child)

        bw_ecl = self.config.get('bw_ecl', 0)
        n_total = len(self.elim_order)

        # ---- Forward sweep: WMB-elim each bucket, store outgoing msgs and originals ----
        parent_originals = {}     # var_label -> list of FastFactor (= bucket's true originals,
                                   # i.e. factors that did NOT come from any child msg)
        forward_msgs = {}          # var_label -> list of FastFactor (bucket's outgoing msg)

        for i, current_var in enumerate(self.elim_order):
            if i % 100 == 0 or i == n_total - 1:
                print(f"  fwd: bucket {i+1}/{n_total} (var={current_var.label})", flush=True)

            # If this var was absorbed into a merged cluster, self has no
            # bucket for it. The copied_gm still has the original tree.
            # Skip setting upstream on self for this var; cluster's surviving
            # bucket already has correct upstream from its own slot.
            orig_bucket = self.buckets.get(current_var)
            copy_bucket = copied_gm.get_bucket(current_var)

            # Identify originals at this bucket: factors NOT in any child's outgoing msg
            child_msg_ids = set()
            for c_label in children_of.get(current_var.label, []):
                for f in forward_msgs.get(c_label, []):
                    child_msg_ids.add(id(f))
            parent_originals[current_var.label] = [
                f for f in copy_bucket.factors if id(f) not in child_msg_ids
            ]

            # Snapshot upstream (full joint at forward) for orig_bucket
            if orig_bucket is not None:
                orig_bucket.approximate_upstream_factors = list(copy_bucket.factors)

            # WMB-elim the elim var → outgoing msg
            has_elim_var = [f for f in copy_bucket.factors if current_var in f.labels]
            no_elim_var = [f for f in copy_bucket.factors if current_var not in f.labels]
            copy_bucket.factors = has_elim_var

            if has_elim_var:
                bucket_ec = copy_bucket.get_ec()
                if bucket_ec > bw_ecl:
                    msgs = copy_bucket.compute_wmb_message(ecl=bw_ecl)
                    self.wmb_fw_partitions += copy_bucket.wmb_stats.get('fw_partitions', 0)
                else:
                    try:
                        msgs = [copy_bucket.compute_message_exact()]
                    except Exception:
                        msgs = copy_bucket.compute_wmb_message(ecl=bw_ecl)
            else:
                msgs = []

            outgoing = msgs + no_elim_var
            forward_msgs[current_var.label] = outgoing

            parent_label = parent_of.get(current_var.label)
            if parent_label is not None and outgoing:
                copied_gm.get_bucket(parent_label).factors.extend(outgoing)

        # ---- Backward distribute (BFS from root) — collect factor lists, no elim ----
        # Maintain an internal bw-chain dict keyed by var label, populated for
        # every var in the tree (even those whose self-bucket was absorbed by
        # a join-tree merge). Surviving buckets get the chain copied onto
        # their approximate_downstream_factors; absorbed vars don't need it
        # on a bucket (they live inside a cluster) but the chain must still
        # flow through them to their descendants.
        internal_bw = {}
        for r in roots:
            internal_bw[r] = []
            r_var = self.matching_var(r)
            if r_var in self.buckets:
                self.buckets[r_var].approximate_downstream_factors = []

        visited = set(roots)
        queue = list(roots)
        bw_processed = 0

        while queue:
            next_queue = []
            for parent_label in queue:
                parent_bw = internal_bw.get(parent_label, [])
                p_originals = parent_originals.get(parent_label, [])

                for child_label in children_of.get(parent_label, []):
                    if child_label in visited:
                        continue
                    visited.add(child_label)

                    sibling_msgs = []
                    for c in children_of.get(parent_label, []):
                        if c != child_label:
                            sibling_msgs.extend(forward_msgs.get(c, []))

                    child_bw_chain = parent_bw + p_originals + sibling_msgs
                    internal_bw[child_label] = child_bw_chain

                    child_var = self.matching_var(child_label)
                    child_orig_bucket = self.buckets.get(child_var)
                    if child_orig_bucket is not None:
                        child_orig_bucket.approximate_downstream_factors = child_bw_chain

                    next_queue.append(child_label)
                    bw_processed += 1
            queue = next_queue

        print(f"  bwd distribute: {bw_processed}/{n_total} buckets processed (factor lists stored, lazy elim)", flush=True)
        if self.wmb_fw_partitions > 0:
            print(f"  WMB forward partitions during backward factor population: {self.wmb_fw_partitions} (upper bound)")

