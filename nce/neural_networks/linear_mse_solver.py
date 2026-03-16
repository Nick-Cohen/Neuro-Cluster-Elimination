# File: nce/neural_networks/linear_mse_solver.py
"""
Optimal solver for linear models with MSE loss in bucket elimination.

This module provides functionality to find the globally optimal parameters
for linear models that minimize standard MSE loss in log space.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
import warnings


class LinearMSEOptimalSolver:
    """
    Enhanced LinearMSEOptimalSolver that automatically handles singular matrices.
    
    This solver detects singularity and automatically chooses the best solution method:
    - Non-singular: Standard Cholesky decomposition
    - Singular: Minimum norm solution via pseudoinverse
    - Ill-conditioned: Small regularization to stabilize
    """
    
    def __init__(self, regularization: float = 0.0, fit_intercept: bool = True, 
                 singularity_threshold: float = 1e-10, condition_threshold: float = 1e12,
                 auto_regularization: bool = True, verbose: bool = False):
        """
        Initialize the robust solver.
        
        Parameters:
        -----------
        regularization : float
            L2 regularization parameter (Ridge regression)
        fit_intercept : bool
            Whether to fit an intercept term
        singularity_threshold : float
            Threshold for detecting singular matrices (based on smallest eigenvalue)
        condition_threshold : float
            Threshold for detecting ill-conditioned matrices
        auto_regularization : bool
            Whether to automatically apply regularization for ill-conditioned matrices
        verbose : bool
            Whether to print diagnostic information
        """
        self.regularization = regularization
        self.fit_intercept = fit_intercept
        self.singularity_threshold = singularity_threshold
        self.condition_threshold = condition_threshold
        self.auto_regularization = auto_regularization
        self.verbose = verbose
        
        # Solution metadata
        self.weights_ = None
        self.bias_ = None
        self.is_solved = False
        self.solution_method_ = None
        self.condition_number_ = None
        self.rank_ = None
        self.effective_regularization_ = None
        
    def _diagnose_matrix(self, XtX: torch.Tensor) -> Dict[str, Any]:
        """Diagnose the condition of the X^T X matrix."""
        try:
            # Compute eigenvalues for condition number and rank
            eigenvals = torch.linalg.eigvals(XtX).real
            eigenvals = eigenvals[eigenvals >= 0]  # Keep only non-negative
            
            if len(eigenvals) == 0:
                return {
                    'is_singular': True,
                    'is_ill_conditioned': True,
                    'condition_number': float('inf'),
                    'rank': 0,
                    'min_eigenval': 0.0,
                    'max_eigenval': 0.0,
                    'eigenvals': eigenvals,
                    'null_space_dim': XtX.shape[0]
                }
            
            min_eigenval = eigenvals.min().item()
            max_eigenval = eigenvals.max().item()
            
            # Determine rank (number of eigenvalues above threshold)
            rank = (eigenvals > self.singularity_threshold).sum().item()
            null_space_dim = XtX.shape[0] - rank
            
            # Determine condition number
            if min_eigenval > self.singularity_threshold:
                condition_number = max_eigenval / min_eigenval
            else:
                condition_number = float('inf')
            
            is_singular = min_eigenval <= self.singularity_threshold
            is_ill_conditioned = condition_number > self.condition_threshold
            
            # Additional diagnostics
            det = torch.det(XtX).item()
            trace = torch.trace(XtX).item()
            
            return {
                'is_singular': is_singular,
                'is_ill_conditioned': is_ill_conditioned,
                'condition_number': condition_number,
                'rank': rank,
                'null_space_dim': null_space_dim,
                'min_eigenval': min_eigenval,
                'max_eigenval': max_eigenval,
                'determinant': det,
                'trace': trace,
                'eigenvals': eigenvals,
                'frobenius_norm': torch.norm(XtX, 'fro').item()
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not diagnose matrix condition: {e}")
            return {
                'is_singular': True,
                'is_ill_conditioned': True,
                'condition_number': float('inf'),
                'rank': 0,
                'null_space_dim': XtX.shape[0],
                'error': str(e)
            }
    
    def _solve_cholesky(self, XtX: torch.Tensor, Xty: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Try to solve using Cholesky decomposition."""
        try:
            L = torch.linalg.cholesky(XtX)
            params = torch.cholesky_solve(Xty.unsqueeze(1), L).squeeze()
            
            if torch.isnan(params).any() or torch.isinf(params).any():
                return None, False
            
            return params, True
        except Exception:
            return None, False
    
    def _solve_pseudoinverse(self, X_aug: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        FIXED VERSION: Solve using pseudoinverse with proper bias handling
        """
        try:
            n_samples, n_features_aug = X_aug.shape  # FIXED: Use X_aug instead of undefined X
            
            # Handle intercept vs no-intercept cases properly
            if self.fit_intercept:
                # X_aug = [ones_column, X_features], so extract X_features
                X_features = X_aug[:, 1:]  # Remove intercept column
                n_features = n_features_aug - 1
                
                if n_features > 0:
                    # Method 1: Solve for weights and bias simultaneously using pseudoinverse
                    # This is the mathematically correct approach for singular systems
                    params = torch.linalg.pinv(X_aug, rtol=self.singularity_threshold) @ y
                    
                    # Validate solution
                    if torch.isnan(params).any() or torch.isinf(params).any():
                        # Fallback: separate computation
                        return self._solve_pseudoinverse_fallback(X_features, y)
                    
                    return params, True
                else:
                    # Edge case: no features, just bias
                    bias = y.mean()
                    params = bias.unsqueeze(0)  # Just the bias term
                    return params, True
            else:
                # No intercept case - just solve for weights
                params = torch.linalg.pinv(X_aug, rtol=self.singularity_threshold) @ y
                
                if torch.isnan(params).any() or torch.isinf(params).any():
                    return None, False
                    
                return params, True
                
        except Exception as e:
            if self.verbose:
                print(f"Pseudoinverse failed: {e}")
            return None, False
        
    def _solve_pseudoinverse_fallback(self, X_features: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Fallback method: solve for weights and bias separately
        This handles cases where the joint pseudoinverse fails
        """
        try:
            n_features = X_features.shape[1]
            
            if n_features > 0:
                # Center the targets and solve for weights
                y_mean = y.mean()
                y_centered = y - y_mean
                
                # Solve: X_features @ weights = y_centered
                weights = torch.linalg.pinv(X_features, rtol=self.singularity_threshold) @ y_centered
                
                # Compute optimal bias: minimize ||y - (X @ weights + bias)||²
                # The optimal bias is: bias = mean(y - X @ weights)
                residuals = y - X_features @ weights
                bias = residuals.mean()
                
                # Combine parameters [bias, weights]
                params = torch.cat([bias.unsqueeze(0), weights])
                
                return params, True
            else:
                # No features case
                bias = y.mean()
                params = bias.unsqueeze(0)
                return params, True
                
        except Exception:
            return None, False
    
    def _solve_regularized(self, XtX: torch.Tensor, Xty: torch.Tensor, 
                        regularization: float) -> Tuple[torch.Tensor, bool]:
        """
        ENHANCED: Solve with regularization, handling intercept properly
        """
        try:
            # Add regularization matrix
            reg_matrix = regularization * torch.eye(XtX.shape[0], device=XtX.device, dtype=XtX.dtype)
            
            # CRITICAL: Don't regularize the intercept term (first element if fit_intercept=True)
            if self.fit_intercept:
                reg_matrix[0, 0] = 0  # Don't regularize bias term
            
            XtX_reg = XtX + reg_matrix
            
            # Try Cholesky first
            params, success = self._solve_cholesky(XtX_reg, Xty)
            if success:
                return params, True
            
            # Fall back to pseudoinverse of regularized system
            try:
                params = torch.linalg.pinv(XtX_reg, rtol=self.singularity_threshold) @ Xty
                if not (torch.isnan(params).any() or torch.isinf(params).any()):
                    return params, True
            except Exception:
                pass
                
            return None, False
            
        except Exception:
            return None, False
    
    def solve(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve for optimal linear parameters with automatic singularity handling.
        
        The method automatically detects matrix condition and chooses the best approach:
        1. Singular matrix -> Minimum norm solution via pseudoinverse
        2. Ill-conditioned matrix -> Small regularization + Cholesky
        3. Well-conditioned matrix -> Standard Cholesky
        
        Parameters:
        -----------
        X : torch.Tensor, shape (n_samples, n_features)
            Feature matrix
        y : torch.Tensor, shape (n_samples,)
            Target values
            
        Returns:
        --------
        weights : torch.Tensor
            Optimal weights
        bias : torch.Tensor
            Optimal bias (0 if fit_intercept=False)
        """
        # Convert to float and ensure proper shapes
        X = X.float()
        y = y.float().squeeze()
        n_samples, n_features = X.shape
        
        if self.verbose:
            print(f"Input dimensions: X=({n_samples}, {n_features}), y=({y.shape[0]},)")
            print(f"Fit intercept: {self.fit_intercept}")
        
        # Add intercept column if needed
        if self.fit_intercept:
            ones_col = torch.ones(n_samples, 1, device=X.device, dtype=X.dtype)
            X_aug = torch.cat([ones_col, X], dim=1)
        else:
            X_aug = X
        
        if self.verbose:
            print(f"Augmented dimensions: X_aug=({X_aug.shape[0]}, {X_aug.shape[1]})")
        
        # Compute normal equation components
        XtX = X_aug.T @ X_aug
        Xty = X_aug.T @ y
        
        # Diagnose matrix condition
        diagnosis = self._diagnose_matrix(XtX)
        self.condition_number_ = diagnosis['condition_number']
        self.rank_ = diagnosis['rank']
        
        if self.verbose:
            print(f"Matrix diagnosis:")
            print(f"  Rank: {diagnosis['rank']}/{XtX.shape[0]}")
            print(f"  Condition number: {diagnosis['condition_number']:.2e}")
            print(f"  Singular: {diagnosis['is_singular']}")
            print(f"  Ill-conditioned: {diagnosis['is_ill_conditioned']}")
        
        # Choose solution method based on matrix condition
        params = None
        success = False
        
        # Method 1: Handle singular matrices with pseudoinverse
        if diagnosis['is_singular']:
            if self.verbose:
                print("Using pseudoinverse for singular matrix (minimum norm solution)")
            params, success = self._solve_pseudoinverse(X_aug, y)
            if success:
                self.solution_method_ = "pseudoinverse"
                self.effective_regularization_ = 0.0
        
        # Method 2: Handle ill-conditioned matrices with regularization
        elif diagnosis['is_ill_conditioned'] and self.auto_regularization:
            if self.verbose:
                print("Matrix is ill-conditioned, trying regularization")
            
            # Try increasing levels of regularization
            regularizations = [self.regularization, 1e-8, 1e-6, 1e-4, 1e-3]
            for reg in regularizations:
                if reg <= 0 and reg == self.regularization:
                    continue
                    
                params, success = self._solve_regularized(XtX, Xty, reg)
                if success:
                    self.solution_method_ = f"regularized_cholesky"
                    self.effective_regularization_ = reg
                    if self.verbose:
                        print(f"  Success with regularization = {reg:.1e}")
                    break
            
            # If regularization fails, fall back to pseudoinverse
            if not success:
                if self.verbose:
                    print("  Regularization failed, falling back to pseudoinverse")
                params, success = self._solve_pseudoinverse(X_aug, y)
                if success:
                    self.solution_method_ = "pseudoinverse_fallback"
                    self.effective_regularization_ = 0.0
        
        # Method 3: Standard Cholesky for well-conditioned matrices
        else:
            if self.regularization > 0:
                params, success = self._solve_regularized(XtX, Xty, self.regularization)
                if success:
                    self.solution_method_ = "regularized_cholesky"
                    self.effective_regularization_ = self.regularization
            else:
                params, success = self._solve_cholesky(XtX, Xty)
                if success:
                    self.solution_method_ = "cholesky"
                    self.effective_regularization_ = 0.0
            
            # Fallback to pseudoinverse if needed
            if not success:
                if self.verbose:
                    print("Standard methods failed, using pseudoinverse")
                params, success = self._solve_pseudoinverse(X_aug, y)
                if success:
                    self.solution_method_ = "pseudoinverse_fallback"
                    self.effective_regularization_ = 0.0
        
        # Final check and extract weights/bias
        if params is None or not success:
            if self.verbose:
                print("All solution methods failed, using fallback")
            # Ultimate fallback: zero weights, mean bias
            self.weights_ = torch.zeros(n_features, device=X.device, dtype=X.dtype)
            self.bias_ = y.mean() if self.fit_intercept else torch.tensor(0.0, device=X.device)
            self.solution_method_ = "fallback"
            self.effective_regularization_ = None
        else:
            # Extract weights and bias from solution
            if self.fit_intercept:
                self.bias_ = params[0]
                self.weights_ = params[1:]
                if self.verbose:
                    print(f"Extracted bias: {self.bias_.item():.6f}")
                    print(f"Extracted weights shape: {self.weights_.shape}")
            else:
                self.bias_ = torch.tensor(0.0, device=X.device)
                self.weights_ = params
        
        if self.verbose:
            print(f"Solution method: {self.solution_method_}")
            if self.effective_regularization_ is not None:
                print(f"Effective regularization: {self.effective_regularization_:.1e}")
        
        self.is_solved = True
        return self.weights_, self.bias_
    
    def test_bias_computation(n_samples=100, n_features=5, rank_deficient=True, verbose=True):
        """
        Test function to validate bias computation in different scenarios
        """
        print("Testing Linear MSE Solver Bias Computation...")
        
        # Generate test data
        torch.manual_seed(42)
        
        if rank_deficient:
            # Create rank-deficient X matrix (singular case)
            X_base = torch.randn(n_samples, 2)
            # Make some columns linear combinations of others (creates singularity)
            X = torch.cat([X_base, X_base[:, 0:1] * 2, X_base[:, 1:2] * 0.5, torch.randn(n_samples, n_features-4)], dim=1)
        else:
            # Create full-rank X matrix
            X = torch.randn(n_samples, n_features)
        
        # True parameters
        true_weights = torch.randn(n_features)
        true_bias = torch.tensor(2.5)
        
        # Generate y with noise
        y = X @ true_weights + true_bias + 0.1 * torch.randn(n_samples)
        
        # Test solver
        solver = LinearMSEOptimalSolver(fit_intercept=True, verbose=verbose)
        weights, bias = solver.solve(X, y)
        
        # Compute predictions and MSE
        predictions = solver.predict(X)
        mse = solver.compute_mse_loss(X, y)
        
        print(f"\nResults:")
        print(f"True bias: {true_bias.item():.6f}, Estimated bias: {bias.item():.6f}")
        print(f"Bias error: {abs(true_bias.item() - bias.item()):.6f}")
        print(f"MSE: {mse.item():.6f}")
        print(f"Solution method: {solver.solution_method_}")
        print(f"Matrix rank: {solver.rank_}/{X.shape[1]+1}")  # +1 for intercept
        
        return solver, weights, bias, mse
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions using solved parameters."""
        if not self.is_solved:
            raise RuntimeError("Must solve for parameters first")
        return X @ self.weights_ + self.bias_
    
    def compute_mse_loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss."""
        predictions = self.predict(X)
        return torch.mean((predictions - y.squeeze()) ** 2)
    
    def get_solution_info(self) -> Dict[str, Any]:
        """Get information about the solution method used."""
        if not self.is_solved:
            return {"error": "Not yet solved"}
        
        return {
            "solution_method": self.solution_method_,
            "condition_number": self.condition_number_,
            "matrix_rank": self.rank_,
            "effective_regularization": self.effective_regularization_,
            "is_minimum_norm": "pseudoinverse" in self.solution_method_
        }


def solve_optimal_logspace_mse(X: torch.Tensor, y: torch.Tensor, 
                              regularization: float = 0.0) -> Dict[str, torch.Tensor]:
    """
    Find optimal linear parameters for logspace MSE loss.
    
    This function computes the globally optimal solution for a linear model
    that minimizes standard MSE loss in log space.
    
    Parameters:
    -----------
    X : torch.Tensor, shape (n_samples, n_features)
        Input features 
    y : torch.Tensor, shape (n_samples,)
        Target values (both X and y should be in log space)
    regularization : float
        L2 regularization strength
        
    Returns:
    --------
    results : dict
        Dictionary containing optimal weights, bias, loss, and other metrics
    """
    return enhanced_linear_mse_solver_with_rank_handling(X, y, regularization)
    # Solve for optimal parameters using robust solver
    solver = LinearMSEOptimalSolver(regularization=regularization, fit_intercept=True)
    opt_weights, opt_bias = solver.solve(X, y)
    
    # Compute metrics
    predictions = solver.predict(X)
    mse_loss = solver.compute_mse_loss(X, y)
    
    # Additional metrics with NaN handling
    mae_error = torch.mean(torch.abs(predictions - y))
    
    # Safe R² calculation
    y_var = torch.var(y)
    if y_var > 1e-8:  # Avoid division by very small numbers
        r_squared = 1 - mse_loss / y_var
        # Clamp R² to reasonable range
        r_squared = torch.clamp(r_squared, -10.0, 1.0)
    else:
        r_squared = torch.tensor(0.0, device=y.device)  # Return tensor, not scalar
    
    return {
        'optimal_weights': opt_weights,
        'optimal_bias': opt_bias,
        'mse_loss': mse_loss,
        'predictions': predictions,
        'mae_error': mae_error,
        'r_squared': r_squared
    }


def validate_linear_config(config: Dict[str, Any]) -> None:
    """
    Validate that the configuration represents a linear model.
    
    Parameters:
    -----------
    config : dict
        Neural network configuration dictionary
        
    Raises:
    -------
    ValueError : If the configuration is not for a linear model
    """
    hidden_sizes = config.get('hidden_sizes', [])
    
    if not isinstance(hidden_sizes, list):
        raise ValueError(f"hidden_sizes must be a list, got {type(hidden_sizes)}")
    
    if len(hidden_sizes) > 0:
        raise ValueError(
            f"Configuration specifies a non-linear model with hidden_sizes={hidden_sizes}. "
            f"For linear models, hidden_sizes must be an empty list []."
        )


def create_linear_net_for_validation(bucket):
    """
    Create a Net instance and validate it's linear.
    
    This is used by compute_linear_mse_message to ensure the configuration
    creates a linear model before proceeding with optimal solving.
    """
    from nce.neural_networks.net import Net
    
    # Validate config first
    validate_linear_config(bucket.config)
    
    # Create Net instance
    net = Net(bucket)
    
    # Additional validation: check the actual network structure
    # For a linear model, should only have one Linear layer
    linear_layers = [layer for layer in net.network if isinstance(layer, nn.Linear)]
    
    if len(linear_layers) != 1:
        raise ValueError(
            f"Expected exactly 1 Linear layer for linear model, "
            f"but Net created {len(linear_layers)} Linear layers. "
            f"This suggests the configuration is not for a linear model."
        )
    
    # Check for activation functions (should be none for pure linear)
    activation_layers = [layer for layer in net.network 
                        if not isinstance(layer, nn.Linear)]
    
    if len(activation_layers) > 0:
        raise ValueError(
            f"Linear model should not have activation functions, "
            f"but found {len(activation_layers)} activation layers: {activation_layers}"
        )
    
    return net


def initialize_net_with_linear_optimum(net, X: torch.Tensor, y: torch.Tensor, 
                                     regularization: float = 0.0) -> None:
    """
    Initialize a neural network with the linear MSE optimal solution.
    
    This sets the weights and biases of the network to the globally optimal
    linear solution, providing a warm start for further training.
    
    Parameters:
    -----------
    net : torch.nn.Module
        Neural network to initialize
    X : torch.Tensor, shape (n_samples, n_features)
        Input features
    y : torch.Tensor, shape (n_samples,)
        Target values
    regularization : float
        L2 regularization parameter
    """
    # Get optimal linear solution
    results = solve_optimal_logspace_mse(X, y, regularization)
    optimal_weights = results['optimal_weights']
    optimal_bias = results['optimal_bias']
    
    # Find the linear layers in the network to initialize
    with torch.no_grad():
        for layer in net.modules():
            if isinstance(layer, torch.nn.Linear):
                # For the first Linear layer, set optimal weights and bias
                if layer.weight.shape[1] == optimal_weights.shape[0]:  # Input dimension matches
                    # Set first row to optimal weights, others to small random values
                    layer.weight.data[0, :] = optimal_weights
                    if layer.weight.shape[0] > 1:  # If multiple output units
                        # Initialize other rows with small random values
                        torch.nn.init.normal_(layer.weight.data[1:, :], mean=0, std=0.01)
                    
                    if layer.bias is not None:
                        layer.bias.data[0] = optimal_bias
                        if layer.bias.shape[0] > 1:
                            # Initialize other biases with small values
                            torch.nn.init.normal_(layer.bias.data[1:], mean=0, std=0.01)
                    break
        
        # Also initialize linspace_bias if it exists
        if hasattr(net, 'linspace_bias'):
            net.linspace_bias.data.fill_(0.0)
    
    print(f"Initialized network with linear MSE optimum:")
    print(f"  Optimal weights: {optimal_weights}")
    print(f"  Optimal bias: {optimal_bias}")
    print(f"  MSE loss: {results['mse_loss'].item():.6f}")
    
def diagnose_linear_solver_failure(X: torch.Tensor, y: torch.Tensor, 
                                  data_preprocessor=None, bucket_label="Unknown") -> Dict[str, Any]:
    """
    Comprehensive diagnostic for linear solver failures like Bucket 34
    """
    print(f"\n🔍 DIAGNOSING LINEAR SOLVER FAILURE - {bucket_label}")
    print("=" * 60)
    
    diagnostics = {
        'bucket_label': bucket_label,
        'data_issues': {},
        'matrix_issues': {},
        'normalization_issues': {},
        'solver_issues': {},
        'recommendations': []
    }
    
    # 1. DATA DIAGNOSTICS
    print("📊 DATA DIAGNOSTICS:")
    n_samples, n_features = X.shape
    print(f"  Data shape: X=({n_samples}, {n_features}), y=({y.shape[0]})")
    
    # Check for basic data issues
    x_has_nan = torch.isnan(X).any()
    y_has_nan = torch.isnan(y).any()
    x_has_inf = torch.isinf(X).any()
    y_has_inf = torch.isinf(y).any()
    
    diagnostics['data_issues'] = {
        'x_has_nan': x_has_nan.item(),
        'y_has_nan': y_has_nan.item(),
        'x_has_inf': x_has_inf.item(),
        'y_has_inf': y_has_inf.item(),
        'n_samples': n_samples,
        'n_features': n_features
    }
    
    print(f"  X has NaN: {x_has_nan}, Y has NaN: {y_has_nan}")
    print(f"  X has Inf: {x_has_inf}, Y has Inf: {y_has_inf}")
    
    if x_has_nan or y_has_nan or x_has_inf or y_has_inf:
        diagnostics['recommendations'].append("❌ CRITICAL: Data contains NaN/Inf values")
        return diagnostics
    
    # Check data ranges and variance
    y_mean = y.mean().item()
    y_std = y.std().item()
    y_var = y.var().item()
    y_min = y.min().item()
    y_max = y.max().item()
    
    print(f"  Y stats: mean={y_mean:.6f}, std={y_std:.6f}, min={y_min:.6f}, max={y_max:.6f}")
    
    # Check for constant targets (zero variance)
    if y_var < 1e-8:
        diagnostics['recommendations'].append("❌ CRITICAL: Y has zero variance (constant targets)")
        return diagnostics
    
    # Check X statistics
    x_means = X.mean(dim=0)
    x_stds = X.std(dim=0)
    x_mins = X.min(dim=0)[0]
    x_maxs = X.max(dim=0)[0]
    
    # Check for constant features
    constant_features = (x_stds < 1e-8).sum().item()
    if constant_features > 0:
        print(f"  ⚠️  WARNING: {constant_features} constant features detected")
        diagnostics['data_issues']['constant_features'] = constant_features
    
    # 2. MATRIX CONDITION DIAGNOSTICS
    print("\n🔢 MATRIX CONDITION DIAGNOSTICS:")
    
    # Add intercept column
    ones_col = torch.ones(n_samples, 1, device=X.device, dtype=X.dtype)
    X_aug = torch.cat([ones_col, X], dim=1)
    
    # Compute normal equation matrix
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y
    
    # Compute condition diagnostics
    try:
        eigenvals = torch.linalg.eigvals(XtX).real
        eigenvals = eigenvals[eigenvals >= 0]
        
        if len(eigenvals) == 0:
            condition_number = float('inf')
            rank = 0
            min_eigenval = 0.0
        else:
            min_eigenval = eigenvals.min().item()
            max_eigenval = eigenvals.max().item()
            rank = (eigenvals > 1e-10).sum().item()
            condition_number = max_eigenval / max(min_eigenval, 1e-16)
        
        det = torch.det(XtX).item()
        
        diagnostics['matrix_issues'] = {
            'condition_number': condition_number,
            'rank': rank,
            'expected_rank': X_aug.shape[1],
            'determinant': det,
            'min_eigenval': min_eigenval,
            'is_singular': min_eigenval <= 1e-10,
            'is_ill_conditioned': condition_number > 1e12
        }
        
        print(f"  Condition number: {condition_number:.2e}")
        print(f"  Matrix rank: {rank}/{X_aug.shape[1]} (expected full rank)")
        print(f"  Determinant: {det:.2e}")
        print(f"  Min eigenvalue: {min_eigenval:.2e}")
        
        if min_eigenval <= 1e-10:
            print("  ❌ MATRIX IS SINGULAR")
            diagnostics['recommendations'].append("Matrix is singular - features are linearly dependent")
        elif condition_number > 1e12:
            print("  ⚠️  MATRIX IS ILL-CONDITIONED")
            diagnostics['recommendations'].append("Matrix is ill-conditioned - may need regularization")
        
    except Exception as e:
        print(f"  ❌ Failed to compute matrix diagnostics: {e}")
        diagnostics['matrix_issues']['error'] = str(e)
    
    # 3. NORMALIZATION DIAGNOSTICS
    print("\n🔄 NORMALIZATION DIAGNOSTICS:")
    if data_preprocessor is not None:
        if hasattr(data_preprocessor, 'fdb') and data_preprocessor.fdb:
            print("  Using FDB normalization")
            if hasattr(data_preprocessor, 'fdb_y_mean') and data_preprocessor.fdb_y_mean is not None:
                print(f"  FDB Y mean: {data_preprocessor.fdb_y_mean.item():.6f}")
            else:
                print("  ❌ FDB Y mean not set!")
                diagnostics['recommendations'].append("FDB normalization constants not initialized")
        else:
            print("  Using standard normalization")
            if hasattr(data_preprocessor, 'y_max') and data_preprocessor.y_max is not None:
                print(f"  Y max: {data_preprocessor.y_max.item():.6f}")
            else:
                print("  ❌ Y max not set!")
                diagnostics['recommendations'].append("Standard normalization constants not initialized")
    
    # 4. SOLVER TESTING
    print("\n🛠️  SOLVER TESTING:")
    
    # Test different solution methods
    solver_results = {}
    
    # Test 1: Standard solver
    try:
        from nce.neural_networks.linear_mse_solver import LinearMSEOptimalSolver
        solver = LinearMSEOptimalSolver(fit_intercept=True, verbose=False)
        weights, bias = solver.solve(X, y)
        predictions = solver.predict(X)
        mse = torch.mean((predictions - y)**2).item()
        r_squared = 1 - mse / y_var
        
        solver_results['standard'] = {
            'success': True,
            'mse': mse,
            'r_squared': r_squared,
            'method': solver.solution_method_,
            'weights_norm': torch.norm(weights).item(),
            'bias': bias.item() if hasattr(bias, 'item') else bias
        }
        print(f"  Standard solver: MSE={mse:.6f}, R²={r_squared:.6f}, method={solver.solution_method_}")
        
    except Exception as e:
        solver_results['standard'] = {'success': False, 'error': str(e)}
        print(f"  Standard solver failed: {e}")
    
    # Test 2: Regularized solver
    try:
        solver_reg = LinearMSEOptimalSolver(regularization=1e-6, fit_intercept=True, verbose=False)
        weights_reg, bias_reg = solver_reg.solve(X, y)
        predictions_reg = solver_reg.predict(X)
        mse_reg = torch.mean((predictions_reg - y)**2).item()
        r_squared_reg = 1 - mse_reg / y_var
        
        solver_results['regularized'] = {
            'success': True,
            'mse': mse_reg,
            'r_squared': r_squared_reg,
            'method': solver_reg.solution_method_,
            'weights_norm': torch.norm(weights_reg).item(),
            'bias': bias_reg.item() if hasattr(bias_reg, 'item') else bias_reg
        }
        print(f"  Regularized solver: MSE={mse_reg:.6f}, R²={r_squared_reg:.6f}, method={solver_reg.solution_method_}")
        
    except Exception as e:
        solver_results['regularized'] = {'success': False, 'error': str(e)}
        print(f"  Regularized solver failed: {e}")
    
    # Test 3: Simple mean predictor baseline
    mean_predictions = torch.full_like(y, y_mean)
    mean_mse = torch.mean((mean_predictions - y)**2).item()
    print(f"  Mean predictor baseline: MSE={mean_mse:.6f}")
    
    solver_results['mean_baseline'] = {'mse': mean_mse}
    diagnostics['solver_issues'] = solver_results
    
    # 5. GENERATE RECOMMENDATIONS
    print("\n💡 RECOMMENDATIONS:")
    
    # Check if solver is worse than mean predictor
    if 'standard' in solver_results and solver_results['standard']['success']:
        if solver_results['standard']['r_squared'] < -5:
            diagnostics['recommendations'].append("❌ CRITICAL: Solver performs much worse than mean predictor")
        if abs(solver_results['standard']['bias'] - y_mean) > 10 * y_std:
            diagnostics['recommendations'].append("❌ CRITICAL: Bias term is unreasonably large")
        if solver_results['standard']['weights_norm'] > 1000:
            diagnostics['recommendations'].append("⚠️  WARNING: Weights have very large norm - possible overfitting")
    
    # Print all recommendations
    if diagnostics['recommendations']:
        for rec in diagnostics['recommendations']:
            print(f"  {rec}")
    else:
        print("  ✅ No major issues detected")
    
    return diagnostics

def fix_linear_solver_issues(X: torch.Tensor, y: torch.Tensor, 
                            diagnostics: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply fixes based on diagnostic results
    """
    print(f"\n🔧 APPLYING FIXES BASED ON DIAGNOSTICS")
    print("=" * 50)
    
    X_fixed = X.clone()
    y_fixed = y.clone()
    
    # Fix 1: Remove constant features
    if 'constant_features' in diagnostics['data_issues'] and diagnostics['data_issues']['constant_features'] > 0:
        print("  Removing constant features...")
        x_stds = X.std(dim=0)
        non_constant_mask = x_stds >= 1e-8
        X_fixed = X_fixed[:, non_constant_mask]
        print(f"  Removed {(~non_constant_mask).sum()} constant features")
    
    # Fix 2: Add regularization if matrix is ill-conditioned or singular
    regularization = 0.0
    if (diagnostics['matrix_issues']['is_singular'] or 
        diagnostics['matrix_issues']['is_ill_conditioned']):
        regularization = 1e-6
        print(f"  Adding regularization: {regularization}")
    
    # Fix 3: Use robust solver with enhanced error handling
    try:
        from nce.neural_networks.linear_mse_solver import LinearMSEOptimalSolver
        
        solver = LinearMSEOptimalSolver(
            regularization=regularization,
            fit_intercept=True,
            singularity_threshold=1e-10,
            condition_threshold=1e12,
            auto_regularization=True,
            verbose=True
        )
        
        print("  Solving with enhanced robust solver...")
        weights, bias = solver.solve(X_fixed, y_fixed)
        
        # Validate solution
        predictions = solver.predict(X_fixed)
        mse = torch.mean((predictions - y_fixed)**2).item()
        r_squared = 1 - mse / y_fixed.var().item()
        
        print(f"  Fixed solution: MSE={mse:.6f}, R²={r_squared:.6f}")
        print(f"  Solution method: {solver.solution_method_}")
        
        if r_squared > -1:  # Much better than the -10 we saw
            print("  ✅ Fix appears successful!")
        else:
            print("  ❌ Fix may not be sufficient")
        
        return weights, bias
        
    except Exception as e:
        print(f"  ❌ Enhanced solver also failed: {e}")
        
        # Ultimate fallback: return mean predictor
        print("  Using mean predictor as ultimate fallback")
        weights = torch.zeros(X_fixed.shape[1])
        bias = y_fixed.mean()
        return weights, bias

def enhanced_solve_optimal_logspace_mse(X: torch.Tensor, y: torch.Tensor, 
                                      data_preprocessor=None,
                                      bucket_label="Unknown",
                                      regularization: float = 0.0) -> Dict[str, Any]:
    """
    Enhanced version with comprehensive diagnostics and fixes
    """
    print(f"🎯 ENHANCED LINEAR SOLVER - {bucket_label}")
    
    # Run diagnostics first
    diagnostics = diagnose_linear_solver_failure(X, y, data_preprocessor, bucket_label)
    
    # Apply fixes if needed
    if diagnostics['recommendations']:
        print("  Issues detected, applying fixes...")
        weights, bias = fix_linear_solver_issues(X, y, diagnostics)
    else:
        # Use standard solver
        from nce.neural_networks.linear_mse_solver import LinearMSEOptimalSolver
        solver = LinearMSEOptimalSolver(regularization=regularization, fit_intercept=True)
        weights, bias = solver.solve(X, y)
    
    # Compute final metrics
    if len(weights.shape) == 0:
        weights = weights.unsqueeze(0)
    if len(bias.shape) == 0:
        bias = bias.unsqueeze(0)
    
    predictions = X @ weights + bias
    mse_loss = torch.mean((predictions - y) ** 2)
    mae_error = torch.mean(torch.abs(predictions - y))
    
    y_var = torch.var(y)
    if y_var > 1e-8:
        r_squared = 1 - mse_loss / y_var
        r_squared = torch.clamp(r_squared, -10.0, 1.0)
    else:
        r_squared = torch.tensor(0.0)
    
    results = {
        'optimal_weights': weights,
        'optimal_bias': bias,
        'mse_loss': mse_loss,
        'predictions': predictions,
        'mae_error': mae_error,
        'r_squared': r_squared,
        'diagnostics': diagnostics
    }
    
    print(f"🏁 FINAL RESULTS: MSE={mse_loss.item():.6f}, R²={r_squared.item():.6f}")
    
    return results

# Usage example for debugging Bucket 34:
def debug_bucket_34_specifically():
    """
    Specific debugging function for the failing Bucket 34 case
    """
    print("🔍 BUCKET 34 SPECIFIC DEBUGGING")
    print("="*50)
    
    # This would be used like:
    # results = enhanced_solve_optimal_logspace_mse(
    #     X=x_all, 
    #     y=y_all, 
    #     data_preprocessor=trainer.data_preprocessor,
    #     bucket_label="34"
    # )
    
    print("To debug Bucket 34, replace the solve_optimal_logspace_mse call with:")
    print("enhanced_solve_optimal_logspace_mse(X, y, data_preprocessor, '34')")
    
    return None

def solve_rank_deficient_system(X: torch.Tensor, y: torch.Tensor, 
                               tolerance: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Robust solver specifically for rank-deficient systems like Bucket 34
    
    Uses SVD-based pseudoinverse with proper rank detection and regularization
    """
    print("🔧 SOLVING RANK-DEFICIENT SYSTEM")
    
    n_samples, n_features = X.shape
    
    # Step 1: Add intercept column
    ones_col = torch.ones(n_samples, 1, device=X.device, dtype=X.dtype)
    X_aug = torch.cat([ones_col, X], dim=1)  # Shape: (n_samples, n_features + 1)
    
    print(f"  Augmented matrix shape: {X_aug.shape}")
    
    # Step 2: Use SVD for robust rank detection and solution
    try:
        # Compute SVD of X_aug
        U, S, Vt = torch.linalg.svd(X_aug, full_matrices=False)
        
        print(f"  SVD shapes: U={U.shape}, S={S.shape}, Vt={Vt.shape}")
        print(f"  Singular values range: {S.min().item():.2e} to {S.max().item():.2e}")
        
        # Step 3: Determine effective rank
        rank = (S > tolerance).sum().item()
        full_rank = S.shape[0]
        
        print(f"  Effective rank: {rank}/{full_rank} (tolerance={tolerance})")
        
        # Step 4: Compute pseudoinverse using SVD
        # For SVD: X = U @ diag(S) @ Vt
        # Pseudoinverse: X^+ = V @ diag(1/S) @ U^T (only for non-zero singular values)
        
        # Create inverse of singular values (only for non-zero ones)
        S_inv = torch.zeros_like(S)
        S_inv[S > tolerance] = 1.0 / S[S > tolerance]
        
        # Construct pseudoinverse: X^+ = Vt^T @ diag(S_inv) @ U^T
        X_pinv = Vt.T @ torch.diag(S_inv) @ U.T
        
        print(f"  Pseudoinverse shape: {X_pinv.shape}")
        
        # Step 5: Solve for parameters
        params = X_pinv @ y
        
        print(f"  Parameters shape: {params.shape}")
        print(f"  Bias: {params[0].item():.6f}")
        print(f"  Weights norm: {torch.norm(params[1:]).item():.6f}")
        
        # Step 6: Extract weights and bias
        bias = params[0]
        weights = params[1:]
        
        # Step 7: Compute predictions and metrics
        predictions = X_aug @ params
        mse = torch.mean((predictions - y) ** 2)
        mae = torch.mean(torch.abs(predictions - y))
        
        y_var = torch.var(y)
        if y_var > 1e-8:
            r_squared = 1 - mse / y_var
        else:
            r_squared = torch.tensor(0.0)
        
        print(f"  Results: MSE={mse.item():.6f}, R²={r_squared.item():.6f}")
        
        # Step 8: Return results
        info = {
            'method': 'svd_pseudoinverse',
            'effective_rank': rank,
            'full_rank': full_rank,
            'condition_number': (S.max() / S[S > tolerance].min()).item() if rank > 0 else float('inf'),
            'singular_values': S.detach().cpu().numpy(),
            'tolerance_used': tolerance,
            'mse': mse.item(),
            'r_squared': r_squared.item()
        }
        
        return weights, bias, info
        
    except Exception as e:
        print(f"  ❌ SVD solution failed: {e}")
        
        # Ultimate fallback: Ridge regression with strong regularization
        return solve_with_strong_regularization(X, y)

def solve_with_strong_regularization(X: torch.Tensor, y: torch.Tensor, 
                                   reg_strength: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Fallback solver using strong Ridge regularization
    """
    print(f"🛡️  FALLBACK: Strong Ridge regularization (λ={reg_strength})")
    
    n_samples, n_features = X.shape
    
    # Add intercept
    ones_col = torch.ones(n_samples, 1, device=X.device, dtype=X.dtype)
    X_aug = torch.cat([ones_col, X], dim=1)
    
    # Compute normal equations with regularization
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y
    
    # Add regularization (don't regularize intercept)
    reg_matrix = reg_strength * torch.eye(XtX.shape[0], device=X.device, dtype=X.dtype)
    reg_matrix[0, 0] = 0  # Don't regularize intercept
    
    XtX_reg = XtX + reg_matrix
    
    try:
        # Solve regularized system
        params = torch.linalg.solve(XtX_reg, Xty)
        
        bias = params[0]
        weights = params[1:]
        
        # Compute metrics
        predictions = X_aug @ params
        mse = torch.mean((predictions - y) ** 2)
        r_squared = 1 - mse / torch.var(y)
        
        print(f"  Ridge results: MSE={mse.item():.6f}, R²={r_squared.item():.6f}")
        
        info = {
            'method': 'ridge_regularization',
            'regularization': reg_strength,
            'mse': mse.item(),
            'r_squared': r_squared.item()
        }
        
        return weights, bias, info
        
    except Exception as e:
        print(f"  ❌ Ridge regularization failed: {e}")
        
        # Ultimate fallback: mean predictor
        weights = torch.zeros(n_features, device=X.device, dtype=X.dtype)
        bias = y.mean()
        
        info = {
            'method': 'mean_predictor_fallback',
            'mse': torch.var(y).item(),
            'r_squared': 0.0
        }
        
        return weights, bias, info

def enhanced_linear_mse_solver_with_rank_handling(X: torch.Tensor, y: torch.Tensor, 
                                                 regularization: float = 0.0) -> Dict[str, Any]:
    """
    Enhanced solver that properly handles rank-deficient matrices like Bucket 34
    """
    print("🎯 ENHANCED LINEAR SOLVER WITH RANK HANDLING")
    print("=" * 50)
    
    # First, try the rank-deficient solver
    try:
        weights, bias, info = solve_rank_deficient_system(X, y)
        
        # Check if the solution is reasonable
        if info['r_squared'] > -1.0:  # Much better than the -10 we were getting
            print("✅ SVD-based solution successful!")
            
            return {
                'optimal_weights': weights,
                'optimal_bias': bias,
                'mse_loss': torch.tensor(info['mse']),
                'predictions': X @ weights + bias,
                'mae_error': torch.mean(torch.abs(X @ weights + bias - y)),
                'r_squared': torch.tensor(info['r_squared']),
                'solver_info': info
            }
    
    except Exception as e:
        print(f"❌ SVD solver failed: {e}")
    
    # Fallback to regularized solver
    try:
        weights, bias, info = solve_with_strong_regularization(X, y, 1e-3)
        
        return {
            'optimal_weights': weights,
            'optimal_bias': bias,
            'mse_loss': torch.tensor(info['mse']),
            'predictions': X @ weights + bias,
            'mae_error': torch.mean(torch.abs(X @ weights + bias - y)),
            'r_squared': torch.tensor(info['r_squared']),
            'solver_info': info
        }
        
    except Exception as e:
        print(f"❌ All solvers failed: {e}")
        
        # Return mean predictor
        weights = torch.zeros(X.shape[1], device=X.device, dtype=X.dtype)
        bias = y.mean()
        
        return {
            'optimal_weights': weights,
            'optimal_bias': bias,
            'mse_loss': torch.var(y),
            'predictions': torch.full_like(y, bias),
            'mae_error': torch.mean(torch.abs(y - bias)),
            'r_squared': torch.tensor(0.0),
            'solver_info': {'method': 'mean_predictor', 'mse': torch.var(y).item(), 'r_squared': 0.0}
        }