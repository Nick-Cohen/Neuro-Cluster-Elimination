"""
Example usage of the QuantizedDecisionTreeFactor with various loss functions.

This example demonstrates how to use the decision tree implementation
as a drop-in replacement for neural networks in the message passing context.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import time

# Assuming the QuantizedDecisionTreeFactor is imported
# from decision_tree_factor import QuantizedDecisionTreeFactor, create_decision_tree_factor, mg_sampled_loss_decision_tree

class MockBucket:
    """Mock bucket class for testing purposes."""
    
    def __init__(self, scope, domain_sizes, device='cpu'):
        self.scope = scope
        self.domain_sizes = domain_sizes
        self.device = device
        self.gm = MockGM()
        
    def get_message_scope(self):
        return self.scope
    
    def get_message_dimension(self):
        return self.domain_sizes

class MockGM:
    """Mock GraphModel for testing."""
    
    def __init__(self):
        self.vars = {}
        
    def matching_var(self, label):
        return label

class MockVar:
    """Mock variable class."""
    
    def __init__(self, states):
        self.states = states


def create_test_data(bucket, num_samples: int = 1000) -> tuple:
    """
    Create test data for training the decision tree.
    
    Parameters:
    -----------
    bucket : MockBucket
        The bucket defining the scope and dimensions
    num_samples : int
        Number of training samples to generate
        
    Returns:
    --------
    tuple : (X, y, bw_hat)
        Training features, targets, and message gradients
    """
    domain_sizes = bucket.get_message_dimension()
    total_states = np.prod(domain_sizes)
    
    # Create all possible assignments
    all_assignments = []
    for i in range(total_states):
        assignment = []
        temp = i
        for dim_size in reversed(domain_sizes):
            assignment.append(temp % dim_size)
            temp //= dim_size
        all_assignments.append(list(reversed(assignment)))
    
    all_assignments = np.array(all_assignments)
    
    # Create one-hot features
    def onehot_single(Xi):
        ret = np.zeros((Xi.shape[0], Xi.max() + 1))
        ret[np.arange(Xi.shape[0]), Xi.flatten()] = 1
        return ret
    
    def onehot(X):
        return np.hstack(tuple(onehot_single(X[:, i:i+1]) for i in range(X.shape[1])))
    
    features = onehot(all_assignments)
    
    # Generate synthetic target values (log space)
    # Create a complex function to approximate
    np.random.seed(42)
    true_function = np.random.randn(total_states) * 2.0
    
    # Add some structure based on variable interactions
    for i, assignment in enumerate(all_assignments):
        # Add interaction terms
        interaction = np.sum(assignment) % 3 - 1
        true_function[i] += interaction * 0.5
        
        # Add some nonlinear terms
        if len(assignment) > 1:
            true_function[i] += np.sin(assignment[0] + assignment[1]) * 0.3
    
    # Convert to tensors
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(true_function, dtype=torch.float32).view(-1, 1)
    
    # Generate mock message gradients (for mg_sampled_loss)
    bw_hat = torch.randn_like(y) * 0.1
    
    return X, y, bw_hat


def compare_loss_functions(bucket, X, y, bw_hat=None):
    """
    Compare different loss functions for decision tree training.
    
    Parameters:
    -----------
    bucket : MockBucket
        The bucket defining the scope
    X, y, bw_hat : torch.Tensor
        Training data
        
    Returns:
    --------
    dict : Results from different loss functions
    """
    results = {}
    
    # Configuration for decision trees
    base_config = {
        'max_leaf_nodes': 20,
        'sf': 1.0,
        'sb': 1.5,
        'Sx': 0.1,
        'num_iterations': 50,
        'convergence_tolerance': 0.001
    }
    
    loss_functions = ['logspace_mse', 'linspace_mse', 'mg_sampled']
    
    for loss_fn in loss_functions:
        print(f"\n=== Training with {loss_fn} ===")
        
        # Create decision tree
        dt_config = base_config.copy()
        dt_factor = QuantizedDecisionTreeFactor(bucket, dt_config)
        
        # Train the decision tree
        start_time = time.time()
        dt_factor.train_on_batch_data(X, y, bw_hat, loss_type=loss_fn)
        training_time = time.time() - start_time
        
        # Make predictions
        predictions = dt_factor.predict(X)
        
        # Compute MSE
        mse = torch.mean((predictions - y)**2).item()
        
        # Get unique values (quantization levels)
        unique_vals = dt_factor.get_unique_values()
        
        results[loss_fn] = {
            'dt_factor': dt_factor,
            'predictions': predictions,
            'mse': mse,
            'training_time': training_time,
            'num_unique_values': len(unique_vals),
            'unique_values': unique_vals
        }
        
        print(f"Training time: {training_time:.3f}s")
        print(f"MSE: {mse:.6f}")
        print(f"Number of unique values: {len(unique_vals)}")
        print(f"Unique values range: [{unique_vals.min():.3f}, {unique_vals.max():.3f}]")
    
    return results


def plot_results(X, y, results):
    """Plot comparison of different loss functions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # True values vs predictions for each method
    for i, (loss_fn, result) in enumerate(results.items()):
        row = i // 2
        col = i % 2
        
        if row < 2 and col < 2:
            ax = axes[row, col]
            
            y_true = y.detach().numpy().flatten()
            y_pred = result['predictions'].detach().numpy().flatten()
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{loss_fn}\nMSE: {result["mse"]:.6f}, '
                        f'Unique: {result["num_unique_values"]}')
            ax.grid(True, alpha=0.3)
    
    # Use the last subplot for a summary
    if len(results) == 3:
        ax = axes[1, 1]
        
        # Bar plot of MSEs
        loss_names = list(results.keys())
        mses = [results[name]['mse'] for name in loss_names]
        unique_counts = [results[name]['num_unique_values'] for name in loss_names]
        
        x_pos = np.a