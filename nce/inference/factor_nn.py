import torch
import time
from torch import nn
from typing import List
from .factor import FastFactor
import torch.nn.functional as F

class FactorNN(FastFactor):
    """
    A subclass of FastFactor that integrates a neural network for more flexible message computation.
    """
    def __init__(self, net, data_processor, losses=None):
        labels = net.bucket.get_message_scope()
        super().__init__(None, labels)
        self.is_nn = True
        self.net = net
        self.data_processor = data_processor
        self.gm = self.net.gm
        self.device = self.gm.device
        self.domain_sizes = [self.gm.matching_var(label).states for label in self.labels]
        self.losses = losses
        
    def eliminate(self, elim_labels, elimination_scheme='sum'):
        """
        Create a fast factor with labels = self.labels - elim_labels and an empty tensor of the appropriate size.
        Query self.net in batches (to reduce memory usage) and sum the elim_variables.
        Insert the sum into the proper indices of the fast factor tensor.
        Return the fast factor.
        """
        remaining_labels = [label for label in self.labels if label not in elim_labels]
        elim_indices = [self.labels.index(label) for label in elim_labels]
        remaining_indices = [self.labels.index(label) for label in remaining_labels]

        # Compute the shape of the reduced tensor
        domain_sizes = [var.states for var in [self.gm.matching_var(label) for label in self.labels]]
        reduced_shape = [domain_sizes[idx] for idx in remaining_indices]

        # Initialize reduced FastFactor tensor
        reduced_tensor = torch.zeros(reduced_shape, device=self.net.device)

        # Determine batch dimensions and their sizes
        batch_dims = [domain_sizes[idx] for idx in elim_indices]
        batch_assignments = torch.cartesian_prod(*[torch.arange(size, device=self.net.device) for size in batch_dims])

        # Process assignments iteratively
        batch_size = 1024  # Adjust based on memory constraints
        for start in range(0, batch_assignments.size(0), batch_size):
            batch = batch_assignments[start:start + batch_size]

            # Expand batch assignments to include non-eliminated variables
            full_assignments = torch.cartesian_prod(
                *[torch.arange(domain_sizes[idx], device=self.net.device) if idx in elim_indices else torch.tensor([0], device=self.net.device)
                  for idx in range(len(self.labels))]
            )

            # Generate inputs for the network
            inputs = self.prepare_inputs(full_assignments, domain_sizes)

            # Query the neural network
            with torch.no_grad():
                outputs = self.net(inputs)

            # Reshape and reduce along the eliminated dimensions
            output_shape = [domain_sizes[idx] if idx in elim_indices else 1 for idx in range(len(self.labels))]
            outputs = outputs.reshape([batch.shape[0]] + output_shape[len(remaining_indices):])

            if elimination_scheme == 'sum':
                reduced_values = torch.logsumexp(outputs, dim=tuple(range(1, len(output_shape) - len(remaining_indices) + 1)))
            elif elimination_scheme == 'max':
                reduced_values, _ = outputs.max(dim=tuple(range(1, len(output_shape) - len(remaining_indices) + 1)))
            else:
                raise ValueError(f"Unsupported elimination scheme: {elimination_scheme}")

            # Insert reduced values into the reduced tensor
            for i, assignment in enumerate(batch):
                index = tuple(assignment.tolist())
                reduced_tensor[index] = reduced_values[i]

        # Create and return the reduced FastFactor
        return FastFactor(reduced_tensor, remaining_labels)

    def _get_slices(self, assignments, elim_vars, elim_domain_sizes, message_scope):
        """
        Args:
            assignments (torch.tensor): _description_
            elim_vars (list[int]): _description_
            message_scope (list[int]): _description_
            self.labels is a list of variable indices in the NN input

        """
        all_elim_assignments = torch.cartesian_prod(
            *[torch.arange(size, device=self.net.device) for size in elim_domain_sizes])
        if all_elim_assignments.dim() == 1:
            all_elim_assignments = all_elim_assignments.unsqueeze(1)
        n_elim = len(all_elim_assignments)
        n_assign = len(assignments)
        n_labels = len(self.labels)
        elim_var_labels = [var.label for var in elim_vars]
        param_dtype = next(self.net.parameters()).dtype

        # Precompute, for each NN-input label, where its value comes from: a column
        # of the message assignment, or a column of the elim assignment.
        msg_idx = {l: k for k, l in enumerate(message_scope)}
        elim_idx = {l: k for k, l in enumerate(elim_var_labels)}
        col_src = []  # (is_msg, source_col) per NN-input label
        for l in self.labels:
            if l in msg_idx:
                col_src.append((True, msg_idx[l]))
            elif l in elim_idx:
                col_src.append((False, elim_idx[l]))
            else:
                raise ValueError(f"Label {l} not in message_scope or elim_var_labels")

        # Batch over assignments so peak rows in any NN forward ≤ MAX_QUERY_ROWS.
        # Each chunk produces (chunk × n_elim) one-hot rows; the previous code did
        # all (n_assign × n_elim) at once which OOMs when n_elim = 2^K is large.
        MAX_QUERY_ROWS = 65536
        chunk_size = max(1, MAX_QUERY_ROWS // max(1, n_elim))

        chunks_out = []
        for start in range(0, n_assign, chunk_size):
            end = min(n_assign, start + chunk_size)
            chunk = assignments[start:end]
            chunk_n = end - start
            dev = chunk.device
            elim_a = all_elim_assignments.to(dev)

            # Build the (n_elim, chunk_n, n_labels) assignment cube by broadcasting,
            # WITHOUT a Python loop over elim assignments. The old loop was O(n_elim)
            # per chunk -> O(n_assign * n_elim^2 / MAX) overall, which made merged
            # buckets (large 2^#elim) hang for many minutes during sample generation.
            # Row ordering (block-major by elim: i_elim*chunk_n + a) is preserved.
            cube = torch.empty((n_elim, chunk_n, n_labels), dtype=torch.int64, device=dev)
            for i, (is_msg, src) in enumerate(col_src):
                if is_msg:
                    cube[:, :, i] = chunk[:, src].unsqueeze(0).expand(n_elim, chunk_n)
                else:
                    cube[:, :, i] = elim_a[:, src].unsqueeze(1).expand(n_elim, chunk_n)
            nn_assignments = cube.reshape(n_elim * chunk_n, n_labels)

            if self.gm.lower_dim:
                one_hot = torch.cat([F.one_hot(nn_assignments[:, i], num_classes=self.domain_sizes[i])[:, 1:] for i in range(n_labels)], dim=-1)
            else:
                one_hot = torch.cat([F.one_hot(nn_assignments[:, i], num_classes=self.domain_sizes[i]) for i in range(n_labels)], dim=-1)
            one_hot = one_hot.to(dtype=param_dtype, device=self.net.device)

            values = self.data_processor.undo_normalization(self.net(one_hot))
            flat = values.view(n_elim, chunk_n).T.detach()
            if len(elim_domain_sizes) > 1:
                flat = flat.reshape((chunk_n,) + tuple(int(d) for d in elim_domain_sizes))
            chunks_out.append(flat)

        return torch.cat(chunks_out, dim=0)

    def _eval_elim_block(self, assignments, elim_coords, elim_vars, elim_var_labels, message_scope):
        """Evaluate this NN factor at every (message assignment x elim assignment)
        pair, for an explicit BLOCK of elim assignments.

        assignments : (A, len(message_scope))  message-scope assignments
        elim_coords : (B, n_elim_vars)          a block of elim assignments
        returns     : (A, B)                    factor value per pair

        Used by sample_tensor_product_elimination to stream over elim blocks so the
        full 2^#elim grid is never materialized.
        """
        self.net.eval()   # ensure masked-net masking applies at inference
        dev = self.net.device
        assignments = assignments.to(dev)
        elim_coords = elim_coords.to(dev)
        A = assignments.shape[0]
        B = elim_coords.shape[0]
        n_labels = len(self.labels)
        param_dtype = next(self.net.parameters()).dtype

        msg_idx = {l: k for k, l in enumerate(message_scope)}
        elim_idx = {l: k for k, l in enumerate(elim_var_labels)}
        col_src = []
        for l in self.labels:
            if l in msg_idx:
                col_src.append((True, msg_idx[l]))
            elif l in elim_idx:
                col_src.append((False, elim_idx[l]))
            else:
                raise ValueError(f"Label {l} not in message_scope or elim_var_labels")

        # Hard safety net: chunk over the elim-block dim B so the (b*A, n_labels)
        # int64 coord cube + one-hot expansion never exceed a memory bound, no
        # matter how large a block the caller passed. Result is identical.
        try:
            onehot_w = int(sum(int(d) for d in self.domain_sizes))
        except Exception:
            onehot_w = 2 * n_labels
        per_row = n_labels * 8 + onehot_w * 4
        try:
            free, _ = torch.cuda.mem_get_info(dev)
        except Exception:
            free = 2 * 1024 ** 3
        rows_cap = max(A, int(0.20 * free / max(1, per_row * 4)))   # >= one full B-row (A wide)
        bchunk = max(1, rows_cap // max(1, A))

        out = torch.empty((A, B), device=dev)
        for b0 in range(0, B, bchunk):
            b1 = min(B, b0 + bchunk)
            b = b1 - b0
            cube = torch.empty((b, A, n_labels), dtype=torch.int64, device=dev)
            for i, (is_msg, src) in enumerate(col_src):
                if is_msg:
                    cube[:, :, i] = assignments[:, src].unsqueeze(0).expand(b, A)
                else:
                    cube[:, :, i] = elim_coords[b0:b1, src].unsqueeze(1).expand(b, A)
            nn_assignments = cube.reshape(b * A, n_labels)
            if self.gm.lower_dim:
                one_hot = torch.cat([F.one_hot(nn_assignments[:, i], num_classes=self.domain_sizes[i])[:, 1:]
                                     for i in range(n_labels)], dim=-1)
            else:
                one_hot = torch.cat([F.one_hot(nn_assignments[:, i], num_classes=self.domain_sizes[i])
                                     for i in range(n_labels)], dim=-1)
            one_hot = one_hot.to(dtype=param_dtype, device=dev)
            vals = self.data_processor.undo_normalization(self.net(one_hot))
            out[:, b0:b1] = vals.view(b, A).T
        return out.detach()                                      # (A, B)


    def _filter_and_fix_assignments(self, assignments_tensor: torch.Tensor,
                                assignments_column_labels: List[int],
                                summation_assignment: List[int],
                                summation_assignment_labels: List[int]) -> torch.Tensor:
        """
        Maps columns from assignments_tensor to a new tensor based on factor_column_labels.
        If a label in factor_column_labels is not found in assignments_column_labels,
        the corresponding column is populated from summation_assignment using summation_assignment_labels.

        Args:
            assignments_tensor (torch.Tensor): 2D tensor with shape (m, n_assignments)
            assignments_column_labels (list[int]): Labels for columns in assignments_tensor
            factor_column_labels (list[int]): Desired column labels for the output tensor
            summation_assignment (list[int]): List of values to populate columns for unmatched labels
            summation_assignment_labels (list[int]): Labels corresponding to summation_assignment values

        Returns:
            torch.Tensor: 2D tensor with shape (m, len(factor_column_labels))
        """
        
        factor_column_labels = self.labels
        
        # Ensure input tensor is 2D
        assert assignments_tensor.ndim == 2, "assignments_tensor must be 2-dimensional"

        # Prepare output tensor with shape (m, len(self.labels))
        output_tensor = torch.zeros(assignments_tensor.size(0), len(self.labels))

        # Create a mapping from assignments_column_labels to column indices
        assignments_label_to_index = {label: idx for idx, label in enumerate(assignments_column_labels)}

        # Create a mapping from summation_assignment_labels to values
        summation_label_to_value = {label: value for label, value in zip(summation_assignment_labels, summation_assignment.view(-1))}

        # Iterate over self.labels to populate the output tensor
        for i, slice_label in enumerate(self.labels):
            if slice_label in assignments_label_to_index:
                # If the label exists in assignments_column_labels, use the corresponding column
                output_tensor[:, i] = assignments_tensor[:, assignments_label_to_index[slice_label]]
            elif slice_label in summation_label_to_value:
                # If the label exists in summation_assignment_labels, use the corresponding value
                output_tensor[:, i] = summation_label_to_value[slice_label]
            else:
                # If no matching label is found, raise an error
                raise ValueError(f"Label {slice_label} not found in assignments_column_labels or summation_assignment_labels")

        return output_tensor
    
    def to_exact(self):
        """Convert FactorNN to FastFactor."""
        result = FactorNN.nn_to_FastFactor(fastGM=self.gm, jit_file = None, net = self.net, device=self.device, debug=False, data_processor=self.data_processor)
        return result

    def get_factor_complexity(self):
        """
        Calculate the complexity of this NN factor without materializing the tensor.

        For FactorNN, we compute complexity from the scope by taking the product
        of domain sizes for each variable. This avoids calling to_exact() which
        would materialize a potentially huge tensor and cause OOM errors.

        Returns:
            int: Product of domain sizes across all variables in the scope
        """
        complexity = 1
        for label in self.labels:
            var = self.gm.matching_var(label)
            complexity *= var.states
        return complexity

    def order_indices(self):
        assert self.labels == sorted(self.labels), "Labels must be sorted"
    
    @staticmethod    
    def nn_to_FastFactor(fastGM, jit_file = None, net = None, device='cuda', debug=False, data_processor=None):
        if jit_file is None and net is None or jit_file is not None and net is not None:
            raise ValueError("Exactly one of a JIT file or a PyTorch net must be provided")
        if debug:
            start_time = time.time()

        # Load the JIT model
        if debug:
            load_start = time.time()
        if jit_file is not None:
            model = torch.jit.load(jit_file).to(device)
        else:
            model = net
        if debug:
            load_end = time.time()
            print(f"Loading model took {load_end - load_start:.4f} seconds")

        # Get the scope and domain sizes
        scope = net.bucket.get_message_scope()
        # matching_var returns a Var object (not an index), so use .states directly
        domain_sizes = torch.tensor([fastGM.matching_var(v).states for v in scope], device=device)
        
        # Derive dtype from model parameters
        param_dtype = next(model.parameters()).dtype

        domain_list = [int(d) for d in domain_sizes.tolist()]
        total = 1
        for d in domain_list:
            total *= d
        full_onehot = not fastGM.lower_dim

        if debug:
            input_creation_start = time.time()

        # Build the dense factor by STREAMING the NN over assignment chunks.
        # The previous code materialized the full cartesian_prod of assignments and
        # one (total x in_dim) input tensor, then did a single forward pass -- which
        # OOMs for wide-scope NN factors (e.g. ~2^25 assignments -> ~10 GiB ReLU
        # activations) even though the resulting factor itself is only ~MB. We now
        # process MAX_QUERY_ROWS assignments at a time and write into a preallocated
        # output, so peak memory ~ chunk size. Result is identical to the unchunked
        # path. Mirrors the batching already used in _get_slices.
        MAX_QUERY_ROWS = 65536
        cat_arange = [torch.arange(s, device=device) for s in domain_list]
        flat_out = torch.empty(total, dtype=param_dtype, device=device)
        with torch.no_grad():
            for start in range(0, total, MAX_QUERY_ROWS):
                end = min(total, start + MAX_QUERY_ROWS)
                # Unravel flat index -> per-variable coords in C-order (last var
                # fastest), matching torch.cartesian_prod / reshape(domain_sizes).
                rem = torch.arange(start, end, device=device)
                coords = [None] * len(domain_list)
                for j in range(len(domain_list) - 1, -1, -1):
                    d = domain_list[j]
                    coords[j] = rem % d
                    rem = rem // d
                # One-hot encode this chunk: lower_dim drops category 0 (n-1 encoding);
                # full keeps all categories, with size==1 vars copied as their value.
                cols = []
                for i, size in enumerate(domain_list):
                    ci = coords[i].unsqueeze(1)
                    if full_onehot:
                        if size > 1:
                            cols.append((ci == cat_arange[i]).to(param_dtype))
                        else:
                            cols.append(ci.to(param_dtype))
                    elif size > 1:
                        cols.append((ci == cat_arange[i][1:]).to(param_dtype))
                chunk_inputs = (torch.cat(cols, dim=1) if cols
                                else torch.zeros((end - start, 0), dtype=param_dtype, device=device))
                flat_out[start:end] = model(chunk_inputs).reshape(-1)

        outputs = flat_out.reshape(tuple(domain_list))
        if debug:
            print(f"Streaming NN densify ({total} assignments) took "
                  f"{time.time() - input_creation_start:.4f} seconds")

        outputs = data_processor.undo_normalization(outputs)
        # Create and return the FastFactor
        if debug:
            factor_creation_start = time.time()
        fast_factor = FastFactor(outputs, scope)
        if debug:
            factor_creation_end = time.time()
            print(f"Creating FastFactor took {factor_creation_end - factor_creation_start:.4f} seconds")

        if debug:
            end_time = time.time()
            print(f"Total time for nn_to_FastFactor: {end_time - start_time:.4f} seconds")

        return fast_factor


class MessageGenerator:
    """Lightweight, picklable wrapper that regenerates a table-format message from saved NN state.

    Stores only the NN weights, architecture, normalization constants, scope, and domain sizes —
    everything needed to run nn_to_FastFactor without a live FastGM or bucket.

    Usage:
        generator = torch.load('generate_nn.pkl', weights_only=False)
        fast_factor = generator.generate(device='cuda')  # full table-format message
    """

    def __init__(self, net, data_processor, scope, domain_sizes, lower_dim):
        """Capture the minimal state needed to regenerate the message.

        Args:
            net: Trained Net instance (weights will be copied to CPU)
            data_processor: DataPreprocessor with normalization constants
            scope: List of variable labels (from bucket.get_message_scope())
            domain_sizes: List of ints (domain size per scope variable)
            lower_dim: Bool — whether n-1 one-hot encoding was used
        """
        self.state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        self.hidden_sizes = self._infer_hidden_sizes(net)
        self.activation = self._infer_activation(net)
        self.use_linspace_bias = getattr(net, 'use_linspace_bias', False)
        self.scope = list(scope)
        self.domain_sizes = list(domain_sizes)
        self.lower_dim = lower_dim

        # DataPreprocessor state
        self.normalization_mode = data_processor.normalization_mode
        self.normalizing_constant = (
            data_processor.normalizing_constant.detach().cpu()
            if isinstance(data_processor.normalizing_constant, torch.Tensor)
            else data_processor.normalizing_constant
        )
        self.ln_min = data_processor.ln_min
        self.ln_max = data_processor.ln_max
        self.ln_range = data_processor.ln_range

    @staticmethod
    def _infer_hidden_sizes(net):
        """Extract hidden layer sizes from a Net's Sequential."""
        sizes = []
        layers = list(net.network)
        for layer in layers[:-1]:  # skip final Linear
            if isinstance(layer, nn.Linear):
                sizes.append(layer.out_features)
        return sizes

    @staticmethod
    def _infer_activation(net):
        """Detect activation function used in the network."""
        for layer in net.network:
            if isinstance(layer, nn.ReLU):
                return 'relu'
            if isinstance(layer, nn.Tanh):
                return 'tanh'
        return 'tanh'

    def _build_net(self, device):
        """Reconstruct a standalone nn.Sequential from saved state."""
        domain_sizes_t = torch.tensor(self.domain_sizes)
        if self.lower_dim:
            input_size = (domain_sizes_t - 1).sum().item()
        else:
            input_size = domain_sizes_t.sum().item()

        activation_cls = nn.ReLU if self.activation == 'relu' else nn.Tanh

        layers = []
        prev_dim = input_size
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation_cls())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        model = nn.Sequential(*layers).to(device)

        # Load weights into the sequential (skip linspace_bias)
        seq_state = {k: v.to(device) for k, v in self.state_dict.items()
                     if k.startswith('network.')}
        model.load_state_dict({k.removeprefix('network.'): v for k, v in seq_state.items()})
        model.eval()
        return model

    def _build_data_processor(self, device):
        """Reconstruct a minimal DataPreprocessor for undo_normalization."""
        from nce.data.data_preprocessor import DataPreprocessor
        dp = DataPreprocessor.__new__(DataPreprocessor)
        dp.normalization_mode = self.normalization_mode
        dp.normalizing_constant = (
            self.normalizing_constant.to(device)
            if isinstance(self.normalizing_constant, torch.Tensor)
            else self.normalizing_constant
        )
        dp.ln_min = self.ln_min
        dp.ln_max = self.ln_max
        dp.ln_range = self.ln_range
        return dp

    def generate(self, device='cpu'):
        """Regenerate the full table-format FastFactor message approximation.

        Args:
            device: Torch device to run inference on ('cpu' or 'cuda')

        Returns:
            FastFactor with the full message tensor and scope labels
        """
        model = self._build_net(device)
        dp = self._build_data_processor(device)
        domain_sizes = torch.tensor(self.domain_sizes, device=device)

        # Generate all possible assignments
        assignments = torch.cartesian_prod(
            *[torch.arange(s, device=device) for s in domain_sizes]
        )

        param_dtype = next(model.parameters()).dtype

        # One-hot encode
        if self.lower_dim:
            total_inputs = (domain_sizes - 1).sum().item()
            all_inputs = torch.zeros(
                (assignments.shape[0], total_inputs), dtype=param_dtype, device=device
            )
            offset = 0
            for i, size in enumerate(domain_sizes):
                if size > 1:
                    mask = assignments[:, i].unsqueeze(1) == torch.arange(1, size.item(), device=device)
                    all_inputs[:, offset:offset + size - 1] = mask.to(dtype=param_dtype)
                    offset += size - 1
        else:
            total_inputs = domain_sizes.sum().item()
            all_inputs = torch.zeros(
                (assignments.shape[0], total_inputs), dtype=param_dtype, device=device
            )
            offset = 0
            for i, size in enumerate(domain_sizes):
                if size > 1:
                    mask = assignments[:, i].unsqueeze(1) == torch.arange(0, size.item(), device=device)
                    all_inputs[:, offset:offset + size.item()] = mask.to(dtype=param_dtype)
                    offset += size.item()
                else:
                    all_inputs[:, offset] = assignments[:, i].to(dtype=param_dtype)
                    offset += 1

        with torch.no_grad():
            outputs = model(all_inputs).reshape(tuple(self.domain_sizes))

        outputs = dp.undo_normalization(outputs)
        return FastFactor(outputs, self.scope)

    def __repr__(self):
        size = 1
        for d in self.domain_sizes:
            size *= d
        return (f"MessageGenerator(scope={self.scope}, "
                f"domain_sizes={self.domain_sizes}, "
                f"hidden={self.hidden_sizes}, "
                f"table_size={size})")
