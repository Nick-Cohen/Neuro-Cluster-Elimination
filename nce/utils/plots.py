import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_fastfactor_comparison(exact_factor, approx_factor, message_gradient=None, title="FastFactor Comparison", show=True):
    """
    Plot comparison between exact and approximate FastFactors
    
    Args:
        exact_factor: FastFactor with exact values
        approx_factor: FastFactor with approximate values  
        title: Plot title
        show: Whether to show the plot
    """
    # Order indices for both factors
    exact_factor.order_indices()
    approx_factor.order_indices()
    if message_gradient is not None:
        message_gradient.order_indices()
    
    # Reshape tensors to 1D
    mt = exact_factor.tensor.reshape(-1)
    mht = approx_factor.tensor.reshape(-1)
    mgt = message_gradient.tensor.reshape(-1) if message_gradient is not None else None
    
    # Calculate differences
    abs_difs = torch.abs(mt - mht)
    difs = mt - mht
    
    # Convert to numpy for plotting
    Y1 = mt.cpu().detach().numpy()
    Y2 = mht.cpu().detach().numpy()
    Y3 = mgt.cpu().detach().numpy() if mgt is not None else None

    # Get sorted indices based on exact values
    start, stop = 0, len(Y1)
    sorted_indices = np.argsort(Y1)
    
    # Sort both arrays using the same indices
    Y1_sorted = Y1[sorted_indices]
    Y2_sorted = Y2[sorted_indices]
    Y3_sorted = Y3[sorted_indices] if Y3 is not None else None

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.title(title)
    
    X = np.arange(len(Y1))
    plt.plot(X[start:stop], Y2_sorted[start:stop], label='m_hat (approx)', lw=0.5, color='orange')
    plt.plot(X[start:stop], Y1_sorted[start:stop], label='m (exact)', lw=0.5, color='blue')
    if Y3_sorted is not None:
        plt.plot(X[start:stop], Y3_sorted[start:stop], label='m_gradient', lw=0.5, color='green', alpha=0.3)
    
    plt.xlabel('Sorted Index')
    plt.ylabel('Message Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    
    return plt