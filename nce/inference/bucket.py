from .factor import FastFactor
from .factor_nn import FactorNN
from nce.training_logger import log_bucket_training_start, log_bucket_training_end
from typing import List
import numpy as np
import torch

class FastBucket:
    
    # TODO: Multiply factors in a more sensible order, e.g. subsumed multiplications first
    
    def __init__(self, gm, label, factors, device, elim_vars, isRoot = False):
        self.gm = gm
        self.stats = self.gm.stats if hasattr(self.gm, 'stats') else None
        self.config = gm.config
        self.label = label
        self.factors = factors
        self.device = device
        self.elim_vars = elim_vars
        self.isRoot = isRoot
        self.approximate_downstream_factors: List[FastFactor] = None
        self.sigma_f, self.sigma_g, self.rho = None, None, None
        self.numel = -1

        # WMB statistics tracking
        self.wmb_stats = {
            'fw_partitions': 0,      # Partitions in forward pass (upper bound for bw impact)
            'bw_partitions': 0,      # Partitions when computing backward message
            'num_mini_buckets': 1,   # Number of mini-buckets (1 = exact, >1 = WMB)
        }

        # Assert that all factors are on the specified device type
        for factor in self.factors:
            assert self.device in str(factor.device), f"Factor device {factor.device} does not match bucket device type {self.device}"
        
    def compute_message_exact(self):
        # Multiply all factors
        self.numel = 0
        if not self.factors:
            # Empty bucket: return scalar 0 (contributes nothing when multiplied)
            return FastFactor(torch.tensor(0.0, device=self.device, requires_grad=False), [])
        
        if self.factors[0].is_nn:
            message = self.factors[0].to_exact()
            assert message.tensor is not None
        else:
            message = self.factors[0]
            if self.config.get('exact', False):
                self.numel += message.tensor.numel()
        for factor in self.factors[1:]:
            if self.config.get('exact', False):
                self.numel += factor.tensor.numel()
            if factor.is_nn:
                factor = factor.to_exact()
            message = message * factor
        # Eliminate variables
        try:
            message = message.eliminate(self.elim_vars)
        except Exception as e:
            print(f"Warning: Elimination failed in bucket {self.label} with size {message.tensor.shape if message.tensor is not None else 'None'}: {e}")
            raise e
        width = len(message.labels)
        assert not (message.tensor is None and len(message.labels) > 0), f"{self.label}"
        self.gm.bucket_complexities.append((self.label, width, self.numel, message.tensor.std().item()))
        return message
    
    def compute_message_nn(self, loss_fn='None', loss_fn2=None):
        """
        Enhanced compute_message_nn with memorizer and linear solver options.

        Behavior depends on configuration:
        - use_memorizer=True: Uses explicit Memorizer lookup table (no training)
        - use_linear_solver=True: Uses exact linear solver (no training)
        - init_with_linear_optimum=True: Initializes with linear optimum then trains
        - Otherwise: Standard neural network training with plotting support
        """
        from nce.neural_networks.net import Net, Memorizer
        from nce.neural_networks.train import Trainer

        # Check configuration flags
        use_memorizer = self.config.get('use_memorizer', False)

        # Check if plotting is enabled in config
        plot_messages = self.config.get('plot_messages', False)

        # Compute exact message if plotting is enabled AND logSS <= 6
        exact_message = None
        if plot_messages:
            logSS = getattr(self.gm, 'logSS', 0)
            if logSS <= 6:
                try:
                    print(f"Computing exact message for comparison (Bucket {self.label})...")
                    exact_message = self.compute_message_exact()
                except Exception as e:
                    print(f"Warning: Could not compute exact message for plotting: {e}")

        # Handle memorizer path
        if use_memorizer:
            print(f"Bucket {self.label}: Using Memorizer (lookup table)")

            # Handle "nbe,<epsilon>" string format for num_samples
            num_samples_cfg = self.config.get('num_samples')
            if isinstance(num_samples_cfg, str) and num_samples_cfg.startswith('nbe'):
                if ',' in num_samples_cfg:
                    epsilon = float(num_samples_cfg.split(',')[1])
                else:
                    epsilon = 0.25  # default from NeuroBE Config.h
                nbe_result = self.get_nbe_num_samples(epsilon)
                self.config['num_samples'] = nbe_result['total']
                print(f"Bucket {self.label}: NBE num_samples (eps={epsilon}): total={nbe_result['total']}, train={nbe_result['n_train']}, val={nbe_result['n_val']}")

            # Create a dummy net for initialization (needed for Trainer/dataloader)
            net = Net(self, hidden_sizes=[])
            t = Trainer(net=net, bucket=self, stats=self.stats)

            # Generate validation set first (for normalization and plotting)
            nbe_val_size = max(1, self.config['num_samples'] // 9)
            t.nbe_val_set = t._generate_validation_set_nbe(nbe_val_size)
            print(f"Validation set generated for normalization: {len(t.nbe_val_set[0]['x'])} samples")

            # Handle use_bw_approx mode
            bw_wmb = None
            if self.config.get('use_bw_approx', False):
                from nce.utils.backward_message import get_backward_message

                # Get backward_iB and bw_ecl from config
                backward_iB = self.config.get('backward_iB', self.config.get('iB', 100))
                backward_ecl = self.config.get('bw_ecl', self.config.get('ecl', 2**20))

                # Check if we should use pre-computed backward factors
                use_precomputed = self.gm.populate_bw_factors and self.approximate_downstream_factors is not None
                backward_factors_arg = self.approximate_downstream_factors if use_precomputed else None

                # Determine backward message complexity to decide on mode
                # We need to check if the combined backward message would exceed backward_ecl
                # If so, we must use factor list mode (sample_tensor_product) even with sampling_scheme='all'

                # Get backward message scope
                message_scope = t.dataloader.sample_generator.message_scope
                bw_message_complexity = 1
                for var_label in message_scope:
                    var = self.gm.matching_var(var_label)
                    bw_message_complexity *= var.states

                # Only use full_data_batch mode if:
                # 1. sampling_scheme is 'all' AND
                # 2. backward message complexity doesn't exceed backward_ecl
                can_use_full_data_batch = (
                    t.dataloader.sample_generator.sampling_scheme == 'all' and
                    bw_message_complexity <= backward_ecl
                )

                if can_use_full_data_batch:
                    # Full data batch mode: materialize full backward message tensor (complexity allows it)
                    print("Computing backward message with bw ecl ", backward_ecl)
                    bw_wmb, _ = get_backward_message(
                        self.gm,
                        self.label,
                        backward_factors=backward_factors_arg,  # Use pre-computed if available
                        iB=backward_iB,
                        backward_ecl=backward_ecl,
                        approximation_method='wmb',
                        return_factor_list=False  # Return single factor
                    )

                    # Set as single factor since complexity allows materialization
                    t.dataloader.bw_modifier = bw_wmb
                else:
                    # Batched mode: return factor list to avoid materializing full product (complexity exceeds limit)
                    bw_factors, _ = get_backward_message(
                        self.gm,
                        self.label,
                        backward_factors=backward_factors_arg,  # Use pre-computed if available
                        iB=backward_iB,
                        backward_ecl=backward_ecl,
                        approximation_method='wmb',
                        return_factor_list=True  # Return factor list
                    )

                    # Set as factor list to avoid materialization
                    t.dataloader.bw_factors = bw_factors

            # Calculate and print target_complexity
            target_complexity = self.get_message_complexity()
            if bw_wmb is not None:
                target_complexity += bw_wmb.tensor.numel()
            print(f"Bucket {self.label}: target_complexity = {target_complexity}")

            # Load all data
            x_all, y_all, _ = t.dataloader.load(all=True)

            # Create memorizer
            mem = Memorizer(self, x_all, y_all)

            # Create FactorNN with memorizer (no bw_inv needed - loss handles backward message)
            nn_message_factor = FactorNN(mem, t.data_preprocessor, losses=None)

        else:
            # Standard NN training path
            # Create neural network
            get_hidden_sizes = self.config.get('custom_hidden_sizes')
            if get_hidden_sizes is not None:
                hidden_sizes = get_hidden_sizes(self)
                print(f"Bucket {self.label}: Using custom hidden sizes: {hidden_sizes}")
                net = Net(self, hidden_sizes=hidden_sizes)
            else:
                hidden_sizes = self.config.get('hidden_sizes')

                # Handle "nbe,{b}" or "neurobe,{b}" string format for hidden sizes
                if isinstance(hidden_sizes, str) and hidden_sizes.startswith('neurobe'):
                    # neurobe mode: h = scope_size * b (scope = number of variables in message)
                    if ',' in hidden_sizes:
                        b = int(hidden_sizes.split(',')[1])
                    else:
                        b = 1
                    scope_size = len(self.get_message_scope())
                    h = scope_size * b
                    hidden_sizes = [h, h]
                elif isinstance(hidden_sizes, str) and hidden_sizes.startswith('nbe'):
                    import math
                    # Parse the multiplier b (default 1)
                    if ',' in hidden_sizes:
                        b = int(hidden_sizes.split(',')[1])
                    else:
                        b = 1
                    # Compute h = b * ceil(log2(message_size))
                    message_size = self.get_message_size()
                    h = b * math.ceil(math.log2(message_size)) if message_size > 1 else b
                    hidden_sizes = [h, h]

            # Handle "nbe,<epsilon>" string format for num_samples
            num_samples_cfg = self.config.get('num_samples')
            if isinstance(num_samples_cfg, str) and num_samples_cfg.startswith('nbe'):
                if ',' in num_samples_cfg:
                    epsilon = float(num_samples_cfg.split(',')[1])
                else:
                    epsilon = 0.25  # default from NeuroBE Config.h
                nbe_result = self.get_nbe_num_samples(epsilon)
                self.config['num_samples'] = nbe_result['total']
                print(f"Bucket {self.label}: NBE num_samples (eps={epsilon}): total={nbe_result['total']}, train={nbe_result['n_train']}, val={nbe_result['n_val']}")

            net = Net(self, hidden_sizes=hidden_sizes)
            t = Trainer(net=net, bucket=self, stats=self.stats)

            # Handle use_bw_approx mode
            if self.config.get('use_bw_approx', False):
                from nce.utils.backward_message import get_backward_message

                # Get backward_iB and bw_ecl from config (fallback to regular iB/ecl if not specified)
                backward_iB = self.config.get('backward_iB', self.config.get('iB', 100))
                backward_ecl = self.config.get('bw_ecl', self.config.get('ecl', 2**20))

                # Check if we should use pre-computed backward factors
                use_precomputed = self.gm.populate_bw_factors and self.approximate_downstream_factors is not None
                backward_factors_arg = self.approximate_downstream_factors if use_precomputed else None

                if use_precomputed:
                    print(f"Bucket {self.label}: Using pre-computed WMB backward factors ({len(self.approximate_downstream_factors)} factors)")

                # Determine backward message complexity to decide on mode
                # We need to check if the combined backward message would exceed backward_ecl
                # If so, we must use factor list mode (sample_tensor_product) even with sampling_scheme='all'

                # Get backward message scope
                message_scope = t.dataloader.sample_generator.message_scope
                bw_message_complexity = 1
                for var_label in message_scope:
                    var = self.gm.matching_var(var_label)
                    bw_message_complexity *= var.states

                # Only use full_data_batch mode if:
                # 1. sampling_scheme is 'all' AND
                # 2. backward message complexity doesn't exceed backward_ecl
                can_use_full_data_batch = (
                    t.dataloader.sample_generator.sampling_scheme == 'all' and
                    bw_message_complexity <= backward_ecl
                )

                if can_use_full_data_batch:
                    # Full data batch mode: materialize full backward message tensor (complexity allows it)

                    if use_precomputed:
                        # pyGMs factors ARE the backward message - use directly without re-processing
                        # Multiply factors together (sum in log space) to get single factor
                        from nce.inference.factor import FastFactor
                        print(f"Bucket {self.label}: Using pyGMs backward directly ({len(self.approximate_downstream_factors)} factors, materializing product)")

                        # Multiply all pyGMs factors together
                        bw_wmb = self.approximate_downstream_factors[0]
                        for f in self.approximate_downstream_factors[1:]:
                            bw_wmb = bw_wmb * f

                        t.dataloader.bw_modifier = bw_wmb
                    else:
                        # No pre-computed factors - compute backward message on-the-fly
                        bw_wmb, _ = get_backward_message(
                            self.gm,
                            self.label,
                            backward_factors=None,
                            iB=backward_iB,
                            backward_ecl=backward_ecl,
                            approximation_method='wmb',
                            return_factor_list=False
                        )
                        t.dataloader.bw_modifier = bw_wmb
                else:
                    # Batched mode: use factor list to avoid materializing full product

                    if use_precomputed:
                        # pyGMs factors ARE the backward message - use directly as factor list
                        # sample_tensor_product will evaluate at each sample point
                        print(f"Bucket {self.label}: Using pyGMs backward directly as factor list ({len(self.approximate_downstream_factors)} factors)")
                        t.dataloader.bw_factors = self.approximate_downstream_factors
                    else:
                        # No pre-computed factors - compute backward message on-the-fly
                        bw_factors, _ = get_backward_message(
                            self.gm,
                            self.label,
                            backward_factors=None,
                            iB=backward_iB,
                            backward_ecl=backward_ecl,
                            approximation_method='wmb',
                            return_factor_list=True
                        )
                        t.dataloader.bw_factors = bw_factors

                # Enable backward-aware normalization - the actual normalizing constant
                # will be computed lazily on first load() call using the training samples
                t.data_preprocessor.use_bw_approx = True

            # Emit bucket_training_start event
            if self.gm._training_logger:
                log_bucket_training_start(
                    self.gm._training_logger,
                    bucket_id=self.label,
                    hidden_sizes=hidden_sizes,
                    num_epochs=self.config.get('num_epochs'),
                    loss_fn=self.config.get('loss_fn'),
                    num_samples=self.config.get('num_samples'),
                )

            t.train()
            self.epochs_trained = t.losses[-1][0] + 1 if t.losses else 0
            self.trained_hidden_sizes = hidden_sizes
            if self.config.get('loss_fn2') is not None and self.config.get('num_epochs2') is not None:
                t.train(new_loss_fn=self.config['loss_fn2'], override_epochs=self.config['num_epochs2'])

            # Store per-bucket training info on the FastGM (bucket is deleted after elimination)
            if hasattr(self.gm, 'per_bucket_training_log'):
                entry = {
                    'label': self.label,
                    'epochs_trained': self.epochs_trained,
                    'hidden_sizes': self.trained_hidden_sizes,
                    'losses': t.losses,           # list of (epoch, loss_value) tuples
                    'val_losses': t.val_losses,    # list of (epoch, loss_value) tuples, may be empty
                }
                if self.config.get('save_nn_weights', False):
                    entry['nn_state_dict'] = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                    entry['normalizing_constant'] = t.data_preprocessor.normalizing_constant.cpu().item()
                self.gm.per_bucket_training_log.append(entry)

            # Emit bucket_training_end event
            if self.gm._training_logger:
                final_loss = t.losses[-1][1] if t.losses else None
                log_bucket_training_end(
                    self.gm._training_logger,
                    bucket_id=self.label,
                    epochs_trained=self.epochs_trained,
                    final_loss=final_loss,
                )

            # Synchronize CUDA operations to prevent race conditions
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Create FactorNN with trained network (no bw_inv needed - loss handles backward message)
            nn_message_factor = FactorNN(net, t.data_preprocessor, losses=t.losses)

            # Save error tracking data to FastGM (bucket gets destroyed after elimination)
            if hasattr(t, 'error_tracking_data') and t.error_tracking_data:
                self.gm.error_tracking_data.append((self.label, t.error_tracking_data))

            # Increment num_trained counter on FastGM
            self.gm.num_trained += 1

        # Plot comparison if enabled
        if plot_messages:
            try:
                logSS = getattr(self.gm, 'logSS', 0)

                if logSS <= 6 and exact_message is not None:
                    # Full message comparison - logSS is small enough
                    from nce.utils.plots import plot_fastfactor_comparison
                    print(f"Generating full comparison plot for Bucket {self.label}...")

                    # Convert NN message to exact FastFactor for comparison
                    nn_exact = nn_message_factor.to_exact()
                    if nn_exact.tensor.isnan().any():
                        raise ValueError(f"NN exact message for bucket {self.label} contains NaN values")
                    # Plot comparison
                    plot_title = f"Bucket {self.label}: NN vs Exact Message"
                    plot_fastfactor_comparison(exact_message, nn_exact, title=plot_title, show=True)
                else:
                    # Validation set comparison - logSS > 6
                    from nce.utils.plots import plot_validation_comparison
                    print(f"Generating validation set comparison plot for Bucket {self.label} (logSS={logSS} > 6)...")

                    # Use the same validation set from training
                    val_batch = t.nbe_val_set[0]
                    x_val = val_batch['x']
                    y_val = val_batch['y']
                    bw_val = val_batch.get('bw', val_batch.get('mgh'))

                    # Get predictions
                    with torch.no_grad():
                        y_pred = net(x_val).squeeze()

                    # Plot validation set comparison
                    val_size = len(x_val)
                    plot_title = f"Bucket {self.label}: NN vs True (Validation Set, n={val_size:,})"
                    plot_validation_comparison(
                        y_val, y_pred, bw=bw_val,
                        title=plot_title, show=True,
                        losses=t.losses, val_losses=t.val_losses
                    )

            except Exception as e:
                print(f"Warning: Could not generate comparison plot: {e}")

        return nn_message_factor

    def compute_message_dt(self, loss_fn='None'):
        from nce.neural_networks.decision_tree import DecisionTreeLossOptimizer
        dt = DecisionTreeLossOptimizer(
            bucket=self
        )
        f_hat = dt.fit_and_convert_to_FastFactor()
        plot_messages = self.config.get('plot_messages', False)
        if plot_messages:
            logSS = getattr(self.gm, 'logSS', 0)

            if logSS <= 6:
                try:
                    print(f"Computing exact message for comparison (Bucket {self.label})...")
                    exact_message = self.compute_message_exact()
                    from nce.utils.plots import plot_fastfactor_comparison
                    print(f"Generating comparison plot for Bucket {self.label}...")
                    plot_title = f"Bucket {self.label}: DT vs Exact Message"
                    plot_fastfactor_comparison(exact_message, f_hat, title=plot_title, show=True)
                except Exception as e:
                    print(f"Warning: Could not generate comparison plot: {e}")
            else:
                print(f"Skipping full plot for Bucket {self.label}: logSS={logSS} > 6")
        return f_hat

    def compute_message_quantization(self, num_states: int, loss_fn: str = 'unnormalized_kl'):
        """
        Compute message using optimal K-segment quantization.

        This method computes the exact message, optionally gets a WMB backward
        message approximation, then finds the optimal K-segment quantization
        that minimizes the specified loss function.

        Args:
            num_states: Number of quantization levels (K)
            loss_fn: Loss function - 'unnormalized_kl' or 'mse'/'logspace_mse'

        Returns:
            FastFactor: The quantized message
        """
        from nce.neural_networks.quantization import quantize_message
        from nce.utils.backward_message import get_backward_message

        # Map loss function names
        if loss_fn in ['unnormalized_kl', 'ukl']:
            loss_type = 'ukl'
        else:
            loss_type = 'mse'

        # Compute exact message
        exact_message = self.compute_message_exact()

        # Get backward message if using UKL
        backward_message = None
        if loss_type == 'ukl' and self.config.get('use_bw_approx', False):
            backward_iB = self.config.get('backward_iB', self.config.get('iB', 100))
            backward_ecl = self.config.get('bw_ecl', self.config.get('ecl', 2**20))

            # Check if we should use pre-computed backward factors
            use_precomputed = (
                self.gm.populate_bw_factors and
                self.approximate_downstream_factors is not None
            )
            backward_factors_arg = self.approximate_downstream_factors if use_precomputed else None

            print(f"Bucket {self.label}: Computing WMB backward message for quantization "
                  f"(backward_ecl={backward_ecl}, backward_iB={backward_iB})")

            backward_message, _ = get_backward_message(
                self.gm,
                self.label,
                backward_factors=backward_factors_arg,
                iB=backward_iB,
                backward_ecl=backward_ecl,
                approximation_method='wmb',
                return_factor_list=False
            )

        # Perform quantization
        print(f"Bucket {self.label}: Quantizing message into {num_states} states "
              f"(loss_type={loss_type}, message_size={exact_message.tensor.numel()})")

        quantized_message, info = quantize_message(
            exact_message=exact_message,
            backward_message=backward_message,
            num_states=num_states,
            loss_type=loss_type,
            device=self.device
        )

        print(f"Bucket {self.label}: Quantization complete "
              f"(total_cost={info['total_cost']:.4f})")

        # Plot comparison if enabled
        plot_messages = self.config.get('plot_messages', False)
        if plot_messages:
            logSS = getattr(self.gm, 'logSS', 0)
            if logSS <= 6:
                try:
                    from nce.utils.plots import plot_fastfactor_comparison
                    print(f"Generating comparison plot for Bucket {self.label}...")
                    plot_title = f"Bucket {self.label}: Quantized (K={num_states}) vs Exact"
                    plot_fastfactor_comparison(exact_message, quantized_message,
                                               title=plot_title, show=True)
                except Exception as e:
                    print(f"Warning: Could not generate comparison plot: {e}")

        # Increment num_trained counter on FastGM
        self.gm.num_trained += 1

        return quantized_message

    def compute_message_with_linear_solver(self):
        """
        Compute message using optimal linear MSE solution.
        
        This method creates a Net object, validates it's linear, then uses
        the optimal closed-form solution instead of iterative training.
        
        Returns:
        --------
        FactorNN : The neural network factor with optimal linear parameters
        
        Raises:
        -------
        ValueError : If the Net configuration creates anything but a linear model
        """
        import torch
        from nce.neural_networks.net import Net
        from nce.neural_networks.train import Trainer
        from nce.inference.factor_nn import FactorNN
        from nce.neural_networks.linear_mse_solver import (
            create_linear_net_for_validation,
            solve_optimal_logspace_mse,
            validate_linear_config
        )
        
        # Check if plotting is enabled in config
        plot_messages = self.config.get('plot_messages', False)

        # Compute exact message if plotting is enabled AND logSS <= 6
        exact_message = None
        if plot_messages:
            logSS = getattr(self.gm, 'logSS', 0)
            if logSS <= 6:
                try:
                    print(f"Computing exact message for comparison (Bucket {self.label})...")
                    exact_message = self.compute_message_exact()
                except Exception as e:
                    print(f"Warning: Could not compute exact message for plotting: {e}")

        # Step 1: Validate configuration creates a linear model
        try:
            validate_linear_config(self.config)
        except ValueError as e:
            raise ValueError(f"Cannot use compute_linear_mse_message: {e}")
        
        # Step 2: Create and validate Net object
        try:
            net = create_linear_net_for_validation(self)
        except ValueError as e:
            raise ValueError(f"Net object validation failed: {e}")
        
        # Step 3: Create trainer to get data loading infrastructure
        trainer = Trainer(net=net, bucket=self, stats=self.stats)

        # Generate validation set first (for normalization and plotting)
        nbe_val_size = max(1, self.config['num_samples'] // 9)
        trainer.nbe_val_set = trainer._generate_validation_set_nbe(nbe_val_size)
        print(f"Validation set generated for normalization: {len(trainer.nbe_val_set[0]['x'])} samples")

        # Step 4: Load the full dataset to get ground truth
        x_all, y_all, bw_hat_all = trainer.dataloader.load(all=True)
        # Apply the SAME preprocessing that neural networks use
        # _, y_normalized = trainer.data_preprocessor.convert_data()

        # Step 5: Solve for optimal parameters using logspace MSE
        from nce.neural_networks.linear_mse_solver import solve_optimal_logspace_mse
        # from nce.neural_networks.linear_mse_solver import enhanced_solve_optimal_logspace_mse
        
        # Get regularization from config if available
        regularization = trainer.config.get('weight_decay', 0.0)
        
        # For linear models, we typically use features as input to predict targets
        # This can be customized based on your specific feature engineering needs
        
        results = solve_optimal_logspace_mse(
            X=x_all,
            y=y_all,
            regularization=regularization
        )
        # results = enhanced_solve_optimal_logspace_mse(
        #     X=x_all,
        #     y=y_all,
        #     regularization=regularization
        # )
        
        optimal_weights = results['optimal_weights']
        optimal_bias = results['optimal_bias']
        
        print(f"Bucket {self.label}: Found optimal linear solution")
        
        # Safe printing of metrics
        mse_loss = results['mse_loss']
        r_squared = results['r_squared']
        
        if torch.isnan(mse_loss):
            print(f"  MSE Loss: NaN (likely constant targets)")
        else:
            print(f"  MSE Loss: {mse_loss.item():.6f}")
        
        if torch.isnan(r_squared):
            print(f"  R²: NaN (likely zero variance in targets)")
        else:
            print(f"  R²: {r_squared.item():.4f}")
        
        # Step 7: Set the optimal parameters in the network
        # The Net should have exactly one Linear layer for linear models
        linear_layer = None
        for layer in net.network:
            if isinstance(layer, torch.nn.Linear):
                linear_layer = layer
                break
        
        if linear_layer is None:
            raise RuntimeError("Could not find Linear layer in the network")
        
        # Set optimal parameters
        with torch.no_grad():
            linear_layer.weight.data = optimal_weights.unsqueeze(0)  # Shape: (1, n_features)
            if linear_layer.bias is not None:
                linear_layer.bias.data = optimal_bias.unsqueeze(0)   # Shape: (1,)
            
            # Also set the linspace_bias if it exists
            if hasattr(net, 'linspace_bias'):
                net.linspace_bias.data.fill_(0.0)  # Reset to zero since we have optimal bias
        
        # Step 8: Verify the solution
        with torch.no_grad():
            final_predictions = net(x_all)
            verification_loss = torch.mean((final_predictions.squeeze() - y_all) ** 2)
            if torch.isnan(verification_loss):
                print(f"  Verification MSE: NaN")
            else:
                print(f"  Verification MSE: {verification_loss.item():.6f}")
        
        # Step 9: Return FactorNN with optimally trained network
        # Create FactorNN with optimally trained network
        linear_message_factor = FactorNN(net, trainer.data_preprocessor)
        
        # Plot comparison if enabled
        if plot_messages:
            try:
                logSS = getattr(self.gm, 'logSS', 0)

                if logSS <= 6 and exact_message is not None:
                    # Full message comparison
                    from nce.utils.plots import plot_fastfactor_comparison
                    print(f"Generating full comparison plot for Bucket {self.label}...")

                    linear_exact = linear_message_factor.to_exact()
                    if linear_exact.tensor.isnan().any():
                        raise ValueError(f"Linear exact message for bucket {self.label} contains NaN values")

                    plot_title = f"Bucket {self.label}: Linear Solver vs Exact Message"
                    plot_fastfactor_comparison(exact_message, linear_exact, title=plot_title, show=True)
                else:
                    # Validation set comparison - logSS > 6
                    from nce.utils.plots import plot_validation_comparison
                    print(f"Generating validation set comparison plot for Bucket {self.label} (logSS={logSS} > 6)...")

                    # Use the same validation set from training
                    val_batch = trainer.nbe_val_set[0]
                    x_val = val_batch['x']
                    y_val = val_batch['y']
                    bw_val = val_batch.get('bw', val_batch.get('mgh'))

                    with torch.no_grad():
                        y_pred = net(x_val).squeeze()

                    val_size = len(x_val)
                    plot_title = f"Bucket {self.label}: Linear Solver vs True (Validation Set, n={val_size:,})"
                    plot_validation_comparison(y_val, y_pred, bw=bw_val, title=plot_title, show=True)

            except Exception as e:
                print(f"Warning: Could not generate comparison plot: {e}")

        return linear_message_factor

    def compute_message_nn_with_linear_init(self, loss_fn='None'):
        """
        Enhanced compute_message_nn that can initialize with linear optimum.
        
        This method:
        1. Optionally initializes the network with linear MSE optimal solution
        2. Then trains normally with the specified loss function
        
        Used when config['init_with_linear_optimum'] = True.
        
        Returns:
        --------
        FactorNN : Neural network factor (potentially initialized with linear optimum)
        """
        from nce.neural_networks.net import Net
        from nce.neural_networks.train import Trainer
        from nce.neural_networks.linear_mse_solver import initialize_net_with_linear_optimum
        
        # Create net and trainer as usual
        net = Net(self)
        trainer = Trainer(net=net, bucket=self, stats=self.stats)
        
        # Check if we should initialize with linear optimum
        init_with_linear = self.config.get('init_with_linear_optimum', False)
        
        if init_with_linear:
            print(f"Bucket {self.label}: Initializing with linear MSE optimum before training")
            
            # Load data for initialization
            x_all, y_all, bw_hat_all = trainer.dataloader.load(all=True)
            
            # Initialize with linear optimum
            regularization = trainer.config.get('weight_decay', 0.0)
            try:
                initialize_net_with_linear_optimum(net, x_all, y_all, regularization)
            except Exception as e:
                print(f"Warning: Could not initialize with linear optimum: {e}")
                print("Proceeding with random initialization...")
        
        # Train the network (either from linear initialization or random)
        debug = False
        if debug:
            trainer.loss_fn = trainer._get_loss_fn('gil1c')
            trainer.train()
            trainer.loss_fn = trainer._get_loss_fn('gil1c_linear')
            trainer.train()
        else:
            trainer.train()

        # Increment num_trained counter on FastGM
        self.gm.num_trained += 1

        return FactorNN(net, trainer.data_preprocessor)

    def compute_one_to_one_nn(self):
        from nce.neural_networks.net import Net, BitVectorLookup
        from nce.neural_networks.train import Trainer
        w = self.get_width()
        net = BitVectorLookup(self, w)
        t=Trainer(net=net, bucket=self)
        t.train()
        # Increment num_trained counter on FastGM
        self.gm.num_trained += 1
        return FactorNN(net, t.data_preprocessor)

    def compute_wmb_message(self, iB: int = 100, debug=False, ecl: int = None) -> List[FastFactor]:
        # todo: add weights functionality. Currently just doing mb
        """
        Compute the Weighted Mini-Bucket (WMB) message for the bucket with given i-bound.

        Args:
        iB (int): The i-bound parameter for mini-bucket elimination. Only used if ecl is None.
        debug (bool): Print debug information.
        ecl (int): Exact complexity limit (max tensor entries). If provided, overrides iB.

        Returns:
        List[FastFactor]: The list of factors representing the WMB message.
        """
        # Get elimination variable labels
        elim_var_labels = set(v.label for v in self.elim_vars)

        # Separate factors into those with and without the elimination variable
        # Factors without the elim var are passed through unchanged
        factors_with_elim_var = []
        factors_without_elim_var = []
        for factor in self.factors:
            if elim_var_labels & set(factor.labels):
                factors_with_elim_var.append(factor)
            else:
                factors_without_elim_var.append(factor)

        if debug:
            print(f"Factors with elim var: {len(factors_with_elim_var)}, without: {len(factors_without_elim_var)}")

        # If no factors have the elimination variable, just return all factors unchanged
        if not factors_with_elim_var:
            return self.factors.copy()

        # Step 1: Split factors (only those with elim var) into mini-buckets
        # Temporarily replace self.factors for _create_mini_buckets
        original_factors = self.factors
        self.factors = factors_with_elim_var
        mini_buckets = self._create_mini_buckets(iB, ecl=ecl)
        self.factors = original_factors

        if debug:
            print(f"Number of mini-buckets: {len(mini_buckets)}")

        # Step 2: Compute elimination for each mini-bucket
        wmb_factors = []
        
        EQUAL_WEIGHTS = True

        if EQUAL_WEIGHTS:
            for mb in mini_buckets:
                n = len(mini_buckets)
                if n == 0:
                    raise ValueError("n is zero!")
                combined_factor = mb[0]
                for factor in mb[1:]:
                    combined_factor *= factor
                # CRITICAL FIX: Clone tensor before in-place modification to avoid
                # corrupting original factors when mini-bucket has only 1 element.
                # When len(mb) == 1, combined_factor still references mb[0] (the original),
                # and in-place *= would modify the original tensor, causing issues when
                # factors are shared between graphical models (e.g., during backward message computation).
                combined_factor = FastFactor(combined_factor.tensor.clone(), list(combined_factor.labels))
                # apply weights inside
                combined_factor.tensor *= n
                # do the LSE
                eliminated_factor = combined_factor.eliminate(self.elim_vars)
                # apply weights outside
                eliminated_factor.tensor *= (1/n)
                wmb_factors.append(eliminated_factor)
        else:
            for mb in mini_buckets:
                combined_factor = mb[0]
                for factor in mb[1:]:
                    combined_factor *= factor
                eliminated_factor = combined_factor.eliminate(self.elim_vars)
                wmb_factors.append(eliminated_factor)

        # Add factors that didn't have the elimination variable (pass through unchanged)
        wmb_factors.extend(factors_without_elim_var)

        return wmb_factors

    def compute_message_wmb_single(self, iB: int, debug=False, ecl: int = None) -> FastFactor:
        """
        Compute WMB message by splitting into mini-buckets and combining with weights.
        This properly implements Weighted Mini-Bucket elimination.

        Args:
            iB (int): The i-bound parameter (max bucket width). Only used if ecl is None.
            debug (bool): Print debug information
            ecl (int): Exact complexity limit (max tensor entries). If provided, overrides iB.

        Returns:
            FastFactor: Single factor representing the WMB message
        """
        import math
        import torch

        if debug:
            print(f"Computing WMB message for bucket {self.label} with iB={iB}, ecl={ecl}")

        # Check if we even need mini-buckets - use ecl if provided, otherwise iB
        if ecl is not None and ecl > 0:
            bucket_ec = self.get_ec()
            if bucket_ec <= ecl:
                if debug:
                    print(f"  Bucket ec {bucket_ec} <= ecl {ecl}, using exact elimination")
                return self.compute_message_exact()
        else:
            bucket_width = self.get_width()
            if bucket_width <= iB:
                # Bucket is small enough, just do exact elimination
                if debug:
                    print(f"  Bucket width {bucket_width} <= iB {iB}, using exact elimination")
                return self.compute_message_exact()

        # Need to split into mini-buckets (ecl will override iB if provided)
        mini_buckets = self._create_mini_buckets(iB, ecl=ecl)

        if debug:
            print(f"  Split into {len(mini_buckets)} mini-buckets")

        # Compute weighted messages for each mini-bucket
        # Using uniform weights: each mini-bucket gets weight 1/n
        weight = 1.0 / len(mini_buckets)

        if debug:
            print(f"  Using weight {weight:.4f} per mini-bucket")

        wmb_messages = []
        for i, mb in enumerate(mini_buckets):
            # Multiply factors in the mini-bucket
            if len(mb) == 1:
                combined = mb[0]
            else:
                combined = mb[0]
                for factor in mb[1:]:
                    combined = combined * factor

            # Weighted elimination using lsePower formula from PyGMs:
            # result = (1/weight) * log(sum(exp(values * weight)))
            # In natural log space:
            #   temp = logsumexp(values_ln * weight)
            #   result_ln = temp / weight
            # In log10 space (FastFactor):
            #   values_ln = values_log10 * ln(10)
            #   temp_ln = logsumexp(values_ln * weight)
            #   result_ln = temp_ln / weight
            #   result_log10 = result_ln / ln(10)

            # Convert from log10 to natural log
            ln_tensor = combined.tensor * math.log(10)

            # Get dimensions to eliminate
            elim_dims = [combined.labels.index(var.label) for var in self.elim_vars]

            # Apply weight BEFORE logsumexp
            weighted_tensor = ln_tensor * weight

            # Perform logsumexp elimination
            result_tensor = weighted_tensor
            for dim_idx in sorted(elim_dims, reverse=True):  # Eliminate from right to left
                result_tensor = torch.logsumexp(result_tensor, dim=dim_idx)

            # Divide by weight (multiply by 1/weight)
            result_tensor = result_tensor / weight

            # Convert back to log10
            result_tensor = result_tensor / math.log(10)

            # Create result factor with remaining labels
            remaining_labels = [label for label in combined.labels if label not in [v.label for v in self.elim_vars]]
            wmb_message = FastFactor(result_tensor, remaining_labels)
            wmb_messages.append(wmb_message)

            if debug:
                print(f"  Mini-bucket {i}: eliminated to scope {remaining_labels}")

        # Combine all WMB messages by multiplication (addition in log space)
        if len(wmb_messages) == 0:
            return FastFactor(torch.tensor(0.0, device=self.device), [])
        elif len(wmb_messages) == 1:
            return wmb_messages[0]
        else:
            # Check if combining would exceed ecl BEFORE materializing
            combined_scope = set()
            for msg in wmb_messages:
                combined_scope.update(msg.labels)

            combined_complexity = 1
            for var_label in combined_scope:
                var = self.gm.matching_var(var_label)
                combined_complexity *= var.states

            if combined_complexity > self.gm.ecl:
                raise RuntimeError(
                    f"WMB combined message would exceed ecl: complexity={combined_complexity} > ecl={self.gm.ecl}. "
                    f"This indicates mini-buckets are too large. Try lowering iB or increasing ecl. "
                    f"Combined scope size: {len(combined_scope)} variables from {len(wmb_messages)} mini-buckets."
                )

            combined_message = wmb_messages[0]
            for msg in wmb_messages[1:]:
                combined_message = combined_message * msg

            if debug:
                print(f"  Combined message scope: {combined_message.labels}")

            return combined_message
    
    def _get_nn_input_size(self):
        dimensions = self.get_message_dimension()
        out = 0
        for nstates in dimensions:
            if self.gm.lower_dim:
                out += nstates - 1
            else:
                out += nstates
        return out

    def _create_mini_buckets(self, iB: int, ecl: int = None) -> List[List[FastFactor]]:
        """
        Create mini-buckets from the factors in the bucket based on ecl (complexity limit).

        When ecl is provided, it completely overrides iB - mini-buckets are created
        based solely on the product of domain sizes (tensor entry count), not variable count.

        Args:
        iB (int): The i-bound parameter (number of variables). Only used if ecl is None.
        ecl (int): The exact complexity limit (max tensor entries). If provided, overrides iB.

        Returns:
        List[List[FastFactor]]: A list of mini-buckets, where each mini-bucket is a list of factors.
        """
        # Use ecl from gm if not explicitly provided
        if ecl is None:
            ecl = getattr(self.gm, 'ecl', None)

        mini_buckets = []
        # Sort by tensor size (numel) instead of variable count
        sorted_factors = sorted(self.factors, key=lambda f: f.tensor.numel(), reverse=True)

        for factor in sorted_factors:
            placed = False
            for mb in mini_buckets:
                # Get combined scope if we add this factor to this mini-bucket
                combined_scope = set.union(*[set(f.labels) for f in mb], set(factor.labels))

                # Calculate combined complexity (product of domain sizes)
                combined_complexity = 1
                for var_label in combined_scope:
                    var = self.gm.matching_var(var_label)
                    combined_complexity *= var.states

                # If ecl is set, use it as the sole constraint (override iB)
                if ecl is not None and ecl > 0:
                    if combined_complexity <= ecl:
                        mb.append(factor)
                        placed = True
                        break
                else:
                    # Fallback to iB (variable count) only if ecl not provided
                    combined_width = len(combined_scope)
                    if combined_width <= iB:
                        mb.append(factor)
                        placed = True
                        break
            if not placed:
                mini_buckets.append([factor])

        # Track partitioning statistics
        # N mini-buckets means N-1 partitioning events
        num_partitions = len(mini_buckets) - 1 if len(mini_buckets) > 1 else 0
        self.wmb_stats['num_mini_buckets'] = len(mini_buckets)
        self.wmb_stats['fw_partitions'] = num_partitions

        return mini_buckets

    def send_message(self, bucket: 'FastBucket'):
        """
        Multiply all factors, eliminate variables, and send the resulting message to another bucket.
        """
        # Multiply all factors
        if not self.factors:
            raise ValueError("No factors in the bucket to send message from")
        
        message = self.factors[0]
        for factor in self.factors[1:]:
            message = message * factor

        # Eliminate variables
        message = message.eliminate(self.elim_vars)

        # Send the message to the receiving bucket
        bucket.receive_message(message)

    def receive_message(self, message: FastFactor):
        """
        Receive a message (factor) from another bucket and append it to this bucket's factors.
        """
        # Assert that the incoming message is on the correct device
        # assert str(self.device) in str(message.device), f"Message device {message.device} does not match bucket device {self.device}"

        # Append the message to the factors list
        self.factors.append(message)
        
    @staticmethod
    def compute_nbe_num_samples(w, l, epsilon):
        """Compute NeuroBE number of samples for a bucket.

        Formula: nSamples = floor((pd + ln(1000)) / epsilon)
        where pd = temp * ln(temp/l), temp = (l-1)*w^2 + l*w + 4

        NeuroBE uses 80:20 split: 80% training, 20% validation.

        Args:
            w: bucket width (number of variables in message scope)
            l: max domain size of variables in scope
            epsilon: error tolerance parameter

        Returns:
            dict with keys: 'total', 'n_train', 'n_val'
        """
        import math
        temp = (l - 1) * w**2 + l * w + 4
        pd = temp * math.log(temp / l)
        n_total = int(math.floor((pd + math.log(1000)) / epsilon))
        n_train = int(math.floor(n_total * 0.8))
        n_val = n_total - n_train
        return {'total': n_total, 'n_train': n_train, 'n_val': n_val}

    def get_nbe_num_samples(self, epsilon):
        """Compute NeuroBE num_samples for this bucket using its actual width and domain sizes."""
        w = len(self.get_message_scope())
        dims = self.get_message_dimension()
        l = max(dims) if dims else 2
        return FastBucket.compute_nbe_num_samples(w, l, epsilon)

    def get_message_scope(self):
        scope = set()
        for factor in self.factors:
            scope = scope.union(factor.labels)
        scope.discard(self.label)
        return sorted(list(scope))
    
    def get_width(self):
        return len(self.get_message_scope())
    
    def get_message_dimension(self):
        return [self.gm.matching_var(idx).states for idx in self.get_message_scope()]
    
    def get_message_size(self):
        scopes = self.get_message_dimension()
        return np.prod([float(d) for d in scopes])
    
    def get_message_complexity(self):
        """
        Calculate the total complexity of all factors in this bucket.

        This sums up the complexity of each factor. For regular factors, this is
        the number of elements in the tensor. For NN factors, this is computed
        from the scope without materializing the tensor (avoiding OOM errors).

        Returns:
            int: Total complexity of all factors in the bucket
        """
        complexity = 0
        for factor in self.factors:
            complexity += factor.get_factor_complexity()
        return complexity
    
    def get_ec(self):
        return self.get_message_size()
    
    def get_fw_bw_stats(self):
        from nce.utils.stats import get_fw_bw_correlation
        from nce.utils.backward_message import get_backward_message
        """
        returns sigma_f, sigma_g, rho
        """
        if self.sigma_f is not None:
            return self.sigma_f, self.sigma_g, self.rho
        else:
            g, f = get_backward_message(self.gm, self.label)
            sigma_f = f.tensor.std(unbiased=False)
            sigma_g = g.tensor.std(unbiased=False)
            rho = get_fw_bw_correlation(f, g)
            self.sigma_f, self.sigma_g, self.rho = sigma_f, sigma_g, rho
            return sigma_f, sigma_g, rho

    def compute_bw_sensitivity(self, bw_wmb, exact_backward=None, exact_forward=None):
        """
        Compute backward message sensitivity metric.

        Sensitivity = lse(f+b_wmb) - lse(b_wmb) - lse(f+b) + lse(b)

        Where:
        - f = exact forward message
        - b = exact backward message
        - b_wmb = WMB approximate backward message
        - lse = log-sum-exp (sum_all_entries in log space)

        This measures how much the WMB approximation affects the combined
        forward-backward distribution compared to the exact backward.

        Args:
            bw_wmb: WMB approximate backward message (FastFactor)
            exact_backward: Exact backward message (FastFactor). If None, computed with high ecl.
            exact_forward: Exact forward message (FastFactor). If None, computed.

        Returns:
            float: Sensitivity value. 0 means WMB matches exact backward perfectly.
        """
        from nce.utils.backward_message import get_backward_message

        # Get exact forward message if not provided
        if exact_forward is None:
            exact_forward = self.compute_message_exact()

        # Get exact backward message if not provided (use very high ecl)
        if exact_backward is None:
            exact_backward, _ = get_backward_message(
                self.gm,
                self.label,
                backward_factors=None,
                iB=100,
                backward_ecl=2**30,
                approximation_method='wmb',
                return_factor_list=False
            )

        # Compute sensitivity: lse(f+b_wmb) - lse(b_wmb) - lse(f+b) + lse(b)
        f_bw_wmb = exact_forward * bw_wmb  # f + b_wmb in log space
        f_bw_exact = exact_forward * exact_backward  # f + b in log space

        term1 = f_bw_wmb.sum_all_entries()  # lse(f + b_wmb)
        term2 = bw_wmb.sum_all_entries()     # lse(b_wmb)
        term3 = f_bw_exact.sum_all_entries() # lse(f + b)
        term4 = exact_backward.sum_all_entries()  # lse(b)

        sensitivity = term1 - term2 - term3 + term4
        return float(sensitivity)

    def compute_bw_sensitivity_with_pygms(self, pygms_provider, exact_backward=None, exact_forward=None):
        """
        Compute backward message sensitivity using pyGMs weight-optimized backward.

        This is a convenience method that uses a PyGMsBackwardProvider to get
        the weight-optimized WMB backward message.

        Args:
            pygms_provider: PyGMsBackwardProvider instance with learned weights
            exact_backward: Exact backward message (FastFactor). If None, computed.
            exact_forward: Exact forward message (FastFactor). If None, computed.

        Returns:
            float: Sensitivity value
        """
        # Get pyGMs backward message (as single factor by multiplying the list)
        factor_list = pygms_provider.get_backward_factor_list(self.label)

        if not factor_list:
            # No backward factors - return 0 sensitivity
            return 0.0

        # Multiply factors together to get single backward message
        bw_wmb = factor_list[0]
        for f in factor_list[1:]:
            bw_wmb = bw_wmb * f

        return self.compute_bw_sensitivity(bw_wmb, exact_backward, exact_forward)

