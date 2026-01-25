import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_fastfactor_comparison(exact_factor, approx_factor, message_gradient=None, title="FastFactor Comparison", show=True, sort_indices=True, show_loss_curve=True, n_biggest=None, pred_alpha=None):
    """
    Plot comparison between exact and approximate FastFactors

    Args:
        exact_factor: FastFactor with exact values
        approx_factor: FastFactor with approximate values
        message_gradient: Optional message gradient factor to overlay
        title: Plot title
        show: Whether to show the plot
        sort_indices: Whether to sort by exact values
        show_loss_curve: Whether to show the loss curve
        n_biggest: If specified, only plot the n largest values (by exact factor)
        pred_alpha: Transparency for predicted values (0-1). If None, uses 1.0 for n_biggest mode or <50k points, 0.01 for larger datasets
    """
    # Order indices for both factors
    # see if approx_factor has data field losses

    losses = getattr(approx_factor, 'losses', None)
    exact_factor.order_indices()
    approx_factor = approx_factor.to_exact()
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
    sorted_indices = np.argsort(Y1)

    # Sort both arrays using the same indices
    Y1_sorted = Y1[sorted_indices]
    Y2_sorted = Y2[sorted_indices]
    Y3_sorted = Y3[sorted_indices] if Y3 is not None else None

    # Apply n_biggest filter (take last n elements since sorted ascending)
    if n_biggest is not None and n_biggest < len(Y1_sorted):
        Y1_sorted = Y1_sorted[-n_biggest:]
        Y2_sorted = Y2_sorted[-n_biggest:]
        Y3_sorted = Y3_sorted[-n_biggest:] if Y3_sorted is not None else None

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.title(title + (f" (top {n_biggest})" if n_biggest else ""))

    X = np.arange(len(Y1_sorted))
    if n_biggest is not None:
        # Line plot when filtering to n_biggest
        alpha = pred_alpha if pred_alpha is not None else 1.0
        plt.plot(X, Y2_sorted, label='m_hat (approx)', lw=0.5, color='orange', alpha=alpha)
    else:
        # Scatter plot for full data
        # Use alpha=1 for small datasets (<50k points), otherwise 0.01 for visibility
        if pred_alpha is not None:
            alpha = pred_alpha
        elif len(Y1_sorted) < 50000:
            alpha = 1.0
        else:
            alpha = 0.01
        plt.scatter(X, Y2_sorted, label='m_hat (approx)', alpha=alpha, color='orange', s=.1)
    plt.plot(X, Y1_sorted, label='m (exact)', lw=0.5, color='blue')
    if Y3_sorted is not None:
        plt.plot(X, Y3_sorted, label='m_gradient', lw=0.5, color='green', alpha=0.3)
    
    plt.xlabel('Sorted Index')
    plt.ylabel('Message Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    
    # plot loss curve
    val_losses = getattr(approx_factor, 'val_losses', None)
    if show_loss_curve and (losses is not None or val_losses is not None):
        plt.figure(figsize=(8, 5))
        if losses is not None and len(losses) > 0:
            x = [losses[i][0] for i in range(len(losses))]
            y = [losses[i][1] for i in range(len(losses))]
            plt.plot(x, y, label='Training Loss', alpha=0.7)
        if val_losses is not None and len(val_losses) > 0:
            x_val = [val_losses[i][0] for i in range(len(val_losses))]
            y_val = [val_losses[i][1] for i in range(len(val_losses))]
            plt.plot(x_val, y_val, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return plt


def plot_validation_comparison(y_true, y_pred, bw=None, title="Validation Set Comparison", show=True, show_loss_curve=True, losses=None, val_losses=None):
    """
    Plot comparison between true and predicted values on a validation set.

    Used when message complexity is too large to materialize the full message tensor.

    Args:
        y_true: True message values (normalized, natural log space)
        y_pred: Predicted message values (normalized, natural log space)
        bw: Optional backward message values
        title: Plot title
        show: Whether to show the plot
        show_loss_curve: Whether to show the loss curve
        losses: Training losses list of (epoch, loss) tuples
        val_losses: Validation losses list of (epoch, loss) tuples
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        Y1 = y_true.cpu().detach().numpy()
    else:
        Y1 = np.array(y_true)

    if isinstance(y_pred, torch.Tensor):
        Y2 = y_pred.cpu().detach().numpy()
    else:
        Y2 = np.array(y_pred)

    # Get sorted indices based on true values
    sorted_indices = np.argsort(Y1)

    # Sort both arrays using the same indices
    Y1_sorted = Y1[sorted_indices]
    Y2_sorted = Y2[sorted_indices]

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.title(title)

    X = np.arange(len(Y1_sorted))
    plt.plot(X, Y2_sorted, label='predicted', lw=0.5, color='orange')
    plt.plot(X, Y1_sorted, label='true', lw=0.5, color='blue')

    plt.xlabel('Sorted Index')
    plt.ylabel('Message Value (normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if show:
        plt.show()

    # plot loss curve
    if show_loss_curve and (losses is not None or val_losses is not None):
        plt.figure(figsize=(8, 5))
        if losses is not None and len(losses) > 0:
            x = [losses[i][0] for i in range(len(losses))]
            y = [losses[i][1] for i in range(len(losses))]
            plt.plot(x, y, label='Training Loss', alpha=0.7)
        if val_losses is not None and len(val_losses) > 0:
            x_val = [val_losses[i][0] for i in range(len(val_losses))]
            y_val = [val_losses[i][1] for i in range(len(val_losses))]
            plt.plot(x_val, y_val, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return plt