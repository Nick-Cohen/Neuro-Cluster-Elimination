from .factor import FastFactor
from .factor_nn import FactorNN
from typing import List
import numpy as np

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

        # Assert that all factors are on the specified device type
        for factor in self.factors:
            assert self.device in str(factor.device), f"Factor device {factor.device} does not match bucket device type {self.device}"
        
    def compute_message_exact(self):
        # Multiply all factors
        if not self.factors:
            raise ValueError("No factors in the bucket to send message from")
        
        if self.factors[0].is_nn:
            message = self.factors[0].to_exact()
            assert message.tensor is not None
        else:
            message = self.factors[0]
        for factor in self.factors[1:]:
            if factor.is_nn:
                factor = factor.to_exact()
            message = message * factor

        # Eliminate variables
        message = message.eliminate(self.elim_vars)
        assert not (message.tensor is None and len(message.labels) > 0), f"{self.label}"
        return message
    
    def compute_message_nn(self, loss_fn='None', loss_fn2=None):
        """
        Enhanced compute_message_nn with linear solver options and plotting preserved.
        
        Behavior depends on configuration:
        - use_linear_solver=True: Uses exact linear solver (no training)
        - init_with_linear_optimum=True: Initializes with linear optimum then trains
        - Otherwise: Standard neural network training with plotting support
        """
        from nce.neural_networks.net import Net
        from nce.neural_networks.train import Trainer
        
        # Check configuration flags for linear features
        init_with_linear = self.config.get('init_with_linear_optimum', False)
        
        # Otherwise, proceed with neural network approach (with optional linear initialization)
        
        # Check if plotting is enabled in config
        plot_messages = self.config.get('plot_messages', False)
        
        # Compute exact message if plotting is enabled
        exact_message = None
        if plot_messages:
            try:
                print(f"Computing exact message for comparison (Bucket {self.label})...")
                exact_message = self.compute_message_exact()
            except Exception as e:
                print(f"Warning: Could not compute exact message for plotting: {e}")
                plot_messages = False
        
        # Create neural network
        net = Net(self)
        t = Trainer(net=net, bucket=self, stats=self.stats)

        t.train()
        if self.config.get('loss_fn2') is not None and self.config.get('num_epochs2') is not None:
            t.train(new_loss_fn=self.config['loss_fn2'], override_epochs=self.config['num_epochs2'])

        # Create FactorNN with trained network
        nn_message_factor = FactorNN(net, t.data_preprocessor)
        
        # Plot comparison if enabled and exact message was computed
        if plot_messages and exact_message is not None:
            try:
                from nce.utils.plots import plot_fastfactor_comparison
                print(f"Generating comparison plot for Bucket {self.label}...")
                
                # Convert NN message to exact FastFactor for comparison
                nn_exact = nn_message_factor.to_exact()
                if nn_exact.tensor.isnan().any():
                    raise ValueError(f"NN exact message for bucket {self.label} contains NaN values")
                # Plot comparison
                plot_title = f"Bucket {self.label}: NN vs Exact Message"
                plot_fastfactor_comparison(exact_message, nn_exact, title=plot_title, show=True)
                
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
            try:
                print(f"Computing exact message for comparison (Bucket {self.label})...")
                exact_message = self.compute_message_exact()
            except Exception as e:
                print(f"Warning: Could not compute exact message for plotting: {e}")
                plot_messages = False
            try:
                from nce.utils.plots import plot_fastfactor_comparison
                print(f"Generating comparison plot for Bucket {self.label}...")
                
                # Plot comparison
                plot_title = f"Bucket {self.label}: NN vs Exact Message"
                plot_fastfactor_comparison(exact_message, f_hat, title=plot_title, show=True)
                
            except Exception as e:
                print(f"Warning: Could not generate comparison plot: {e}")
        return f_hat

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
        
        # Compute exact message if plotting is enabled
        exact_message = None
        if plot_messages:
            try:
                print(f"Computing exact message for comparison (Bucket {self.label})...")
                exact_message = self.compute_message_exact()
            except Exception as e:
                print(f"Warning: Could not compute exact message for plotting: {e}")
                plot_messages = False
        
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
        
        # Step 4: Load the full dataset to get ground truth
        x_all, y_all, mg_hat_all = trainer.dataloader.load(all=True)
        # Apply the SAME preprocessing that neural networks use
        # _, y_normalized = trainer.data_preprocessor.convert_data()
        
        # print(f"y_normalized stats:")
        # print(f"  Shape: {y_normalized.shape}")
        # print(f"  Min: {y_normalized.min().item():.6f}")
        # print(f"  Max: {y_normalized.max().item():.6f}")
        # print(f"  Mean: {y_normalized.mean().item():.6f}")
        # print(f"  Std: {y_normalized.std().item():.6f}")
        # print(f"  Unique values: {len(torch.unique(y_normalized))}")
        
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
        
        # Plot comparison if enabled and exact message was computed
        if plot_messages and exact_message is not None:
            try:
                from nce.utils.plots import plot_fastfactor_comparison
                print(f"Generating comparison plot for Bucket {self.label}...")
                
                # Convert linear solver message to exact FastFactor for comparison
                linear_exact = linear_message_factor.to_exact()
                
                if linear_exact.tensor.isnan().any():
                    raise ValueError(f"Linear exact message for bucket {self.label} contains NaN values")
                
                # Plot comparison
                plot_title = f"Bucket {self.label}: Linear Solver vs Exact Message"
                plot_fastfactor_comparison(exact_message, linear_exact, title=plot_title, show=True)
                
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
            x_all, y_all, mg_hat_all = trainer.dataloader.load(all=True)
            
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
        
        return FactorNN(net, trainer.data_preprocessor)
        
    def compute_dummy_nn(self):
        from nce.neural_networks.net import Net, Memorizer
        from nce.neural_networks.train import Trainer
        net = Net(self)
        t=Trainer(net=net, bucket=self)
        # use trainer to make dataloader
        x_all, y_all, _ = t.dataloader.load(all=True)
        mem = Memorizer(self, x_all, y_all)
        return FactorNN(mem, t.data_preprocessor)
    
    def compute_one_to_one_nn(self):
        from nce.neural_networks.net import Net, BitVectorLookup
        from nce.neural_networks.train import Trainer
        w = self.get_width()
        net = BitVectorLookup(self, w)
        t=Trainer(net=net, bucket=self)
        t.train()
        return FactorNN(net, t.data_preprocessor)

    def compute_wmb_message(self, iB: int, debug=False) -> List[FastFactor]:
        # todo: add weights functionality. Currently just doing mb
        """
        Compute the Weighted Mini-Bucket (WMB) message for the bucket with given i-bound.
        
        Args:
        iB (int): The i-bound parameter for mini-bucket elimination.
        
        Returns:
        List[FastFactor]: The list of factors representing the WMB message.
        """
        # Step 1: Split factors into mini-buckets
        mini_buckets = self._create_mini_buckets(iB)
        if debug:
            print(f"Number of mini-buckets: {len(mini_buckets)}")
        
        # Step 2: Compute weighted elimination for each mini-bucket
        wmb_factors = []
        first_bucket = True
        for mb in mini_buckets:
            if len(mb) == 1:
                wmb_factors.append(mb[0])
            else:
                combined_factor = mb[0]
                for factor in mb[1:]:
                    combined_factor *= factor
                eliminated_factor = combined_factor.eliminate(self.elim_vars) if first_bucket else combined_factor.eliminate(self.elim_vars, elimination_scheme='sum')
                first_bucket = False
                wmb_factors.append(eliminated_factor)
        
        return wmb_factors
    
    def _get_nn_input_size(self):
        dimensions = self.get_message_dimension()
        out = 0
        for nstates in dimensions:
            if self.gm.lower_dim:
                out += nstates - 1
            else:
                out += nstates
        return out

    def _create_mini_buckets(self, iB: int) -> List[List[FastFactor]]:
        """
        Create mini-buckets from the factors in the bucket based on the i-bound.
        
        Args:
        iB (int): The i-bound parameter for mini-bucket creation.
        
        Returns:
        List[List[FastFactor]]: A list of mini-buckets, where each mini-bucket is a list of factors.
        """
        mini_buckets = []
        sorted_factors = sorted(self.factors, key=lambda f: len(f.vars), reverse=True)
        
        for factor in sorted_factors:
            placed = False
            for mb in mini_buckets:
                if len(set.union(*[set(f.vars) for f in mb], set(factor.vars))) <= iB:
                    mb.append(factor)
                    placed = True
                    break
            if not placed:
                mini_buckets.append([factor])
        
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
        return np.prod(scopes)
    
    def get_ec(self):
        return self.get_message_size()
    
    def get_fw_bw_stats(self):
        from nce.utils.stats import get_fw_bw_correlation
        from nce.utils.message_gradient import get_message_gradient
        """
        returns sigma_f, sigma_g, rho
        """
        if self.sigma_f is not None:
            return self.sigma_f, self.sigma_g, self.rho
        else:
            g, f = get_message_gradient(self.gm, self.label)
            sigma_f = f.tensor.std(unbiased=False)
            sigma_g = g.tensor.std(unbiased=False)
            rho = get_fw_bw_correlation(f, g)
            self.sigma_f, self.sigma_g, self.rho = sigma_f, sigma_g, rho
            return sigma_f, sigma_g, rho
        
     