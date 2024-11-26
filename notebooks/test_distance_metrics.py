#%%

from utils.distance_metrics import *
from fastElim import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

#%%

def noise_FF(fastFactor, std=1):
    noise = torch.randn_like(fastFactor.tensor) * std
    return FastFactor(fastFactor.tensor + noise, fastFactor.labels)

data_info = ["l1", "grad_informed_l1", "logspace_l1", "logspace_grad_informed_l1", "l1_with_cancellation", "grad_informed_l1_with_cancellation", "mse", "grad_informed_mse", "logspace_mse", "logspace_grad_informed_mse"]

def generate_error_data(mess, mg, mg_hat, max_stdev, num_samples):
    device = mess.tensor.device
    
    error_data = []
    
    for i in range(num_samples):
        # Generate a random standard deviation between 0 and max_stdev
        np.random.seed(i)
        stdev = np.random.uniform(0, max_stdev)
        
        # Generate noise and add it to the message
        noise = torch.randn_like(mess.tensor) * stdev
        noised_mess = FastFactor(mess.tensor + noise, mess.labels)
        
        # Compute various distance metrics
        distances = [
            l1(mess, noised_mess),
            grad_informed_l1(mess, noised_mess, mg_hat),
            logspace_l1(mess, noised_mess),
            logspace_grad_informed_l1(mess, noised_mess, mg_hat),
            l1_with_cancellation(mess, noised_mess),
            grad_informed_l1_with_cancellation(mess, noised_mess, mg_hat),
            mse(mess, noised_mess),
            grad_informed_mse(mess, noised_mess, mg_hat),
            logspace_mse(mess, noised_mess),
            logspace_grad_informed_mse(mess, noised_mess, mg_hat)
        ]
        distances = [float(d) for d in distances]
        
        # Compute true error
        true_error = true_err(noised_mess, mess, mg)
        error_data.append((distances,abs(true_error)))
    
    return error_data

def generate_sampled_error_data(mess, mg, mg_hat, max_stdev, num_mess_samples, num_samples_from_message):
    device = mess.tensor.device
    
    error_data = []
    
    for i in range(num_mess_samples):
        # Generate a random standard deviation between 0 and max_stdev
        np.random.seed(i)
        stdev = np.random.uniform(0, max_stdev)
        
        # Generate noise and add it to the message
        noise = torch.randn_like(mess.tensor) * stdev
        noised_mess = FastFactor(mess.tensor + noise, mess.labels)
        
        # Compute various distance metrics
        distances = [
            l1(mess, noised_mess),
            grad_informed_l1(mess, noised_mess, mg_hat),
            logspace_l1(mess, noised_mess),
            logspace_grad_informed_l1(mess, noised_mess, mg_hat),
            l1_with_cancellation(mess, noised_mess),
            grad_informed_l1_with_cancellation(mess, noised_mess, mg_hat),
            mse(mess, noised_mess),
            grad_informed_mse(mess, noised_mess, mg_hat),
            logspace_mse(mess, noised_mess),
            logspace_grad_informed_mse(mess, noised_mess, mg_hat)
        ]
        distances = [float(d) for d in distances]
        sampled_distances = [
            sampled_distance(l1, num_samples_from_message, mess, noised_mess),
            sampled_distance(grad_informed_l1, num_samples_from_message, mess, noised_mess, mg_hat),
            sampled_distance(logspace_l1, num_samples_from_message, mess, noised_mess),
            sampled_distance(logspace_grad_informed_l1, num_samples_from_message, mess, noised_mess, mg_hat),
            sampled_distance(l1_with_cancellation, num_samples_from_message, mess, noised_mess),
            sampled_distance(grad_informed_l1_with_cancellation, num_samples_from_message, mess, noised_mess, mg_hat),
            sampled_distance(mse, num_samples_from_message, mess, noised_mess),
            sampled_distance(grad_informed_mse, num_samples_from_message, mess, noised_mess, mg_hat),
            sampled_distance(logspace_mse, num_samples_from_message, mess, noised_mess),
            sampled_distance(logspace_grad_informed_mse, num_samples_from_message, mess, noised_mess, mg_hat)
        ]
        sampled_distances = [float(d) for d in sampled_distances]
        
        # Compute true error
        true_error = true_err(noised_mess, mess, mg)
        error_data.append((distances, sampled_distances, abs(true_error)))
    
    return error_data

def save_error_metric_graphs(output_path, data, data_info, problem_name, idx):
    os.makedirs(output_path, exist_ok=True)

    has_sampled_data = len(data[0]) == 3  # Check if data includes sampled distances

    if has_sampled_data:
        error_metrics = [item[0] for item in data]
        sampled_error_metrics = [item[1] for item in data]
        true_errors = [item[2] for item in data]
    else:
        error_metrics = [item[0] for item in data]
        true_errors = [item[1] for item in data]

    n_metrics = len(data_info)
    n_cols = 3  # You can adjust this for different layouts
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(f'Error Metrics vs True Error - {problem_name} (idx: {idx})', fontsize=16)

    for i, (metric_name, ax) in enumerate(zip(data_info, axes.flatten())):
        x = [metrics[i] for metrics in error_metrics]
        y = true_errors
        # remove -inf pairs
        good_indices = [i for i in range(len(x)) if x[i] != float('-inf')]
        x = [x[i] for i in good_indices]
        y = [y[i] for i in good_indices]
        
        ax.scatter(x, y, alpha=0.5, s=10, color='blue', label='Unsampled')
        
        if has_sampled_data:
            x_sampled = [metrics[i] for metrics in sampled_error_metrics]
            ax.scatter(x_sampled, y, alpha=0.5, s=10, color='red', label='Sampled')

        ax.set_xlabel(metric_name)
        ax.set_ylabel('True Error')
        ax.set_title(metric_name, fontsize=10)
        
        # Adjust axis limits to show all data points
        all_x = x + (x_sampled if has_sampled_data else [])
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(y), np.max(y)
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax.set_xlim(x_min - 0.1*x_range, x_max + 0.1*x_range)
        ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)

        ax.legend(fontsize='small')

    # Remove any unused subplots
    for j in range(i + 1, len(axes.flatten())):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    filename = f'{problem_name}_idx{idx}_error_metrics.png'
    plt.savefig(os.path.join(output_path, filename), dpi=300)
    plt.close()

    print(f"Graphs for {problem_name} (idx: {idx}) saved to {output_path}")
#%%

# uai_file = "/home/cohenn1/SDBE/width20-30/pedigree18.uai"
uai_file = "/home/cohenn1/SDBE/width20-30/grid20x20.f2.uai"
# uai_file = "/home/cohenn1/SDBE/width_under_20_problems/grid10x10.f10.uai"
ib = 5
# ib = 'inf'
problem_name = "grid20x20.f2-ib" + str(ib)
# problem_name = "grid10x10.f10-ib" + str(ib)
output_path = "/home/cohenn1/SDBE/PyTorchGMs/graphs"
grid20f2_idxs = [30, 106, 213, 123, 331]
idx = 106
#%%
device = 'cpu'
fastgm = FastGM(uai_file=uai_file, device=device)
# fastgm.eliminate_variables(up_to=37)
# factors = fastgm.buckets[37].factors

fastgm_copy = FastGM(uai_file=uai_file, device='cpu')
mg, mess = fastgm.get_message_gradient(idx)
# print("mg scope is ", mg.labels)
# print("mess scope is ", mess.labels)
#%%
mg_hat = fastgm_copy.get_wmb_message_gradient(bucket_var=idx, i_bound=ib, weights='max')
# assert set(mg_hat.labels) == set(mg.labels)
# for v in mg_hat.labels:
#     assert(v in mg.labels, f"v={v} not in mg.labels")
# %%
# data = generate_error_data(mess, mg, mg_hat, max_stdev=10, num_samples=100)
# generate_sampled_error_data(mess, mg, mg_hat, max_stdev, num_mess_samples, num_samples_from_message)
data = generate_sampled_error_data(mess, mg, mg_hat, max_stdev=10, num_mess_samples=100, num_samples_from_message=1000)
# %%
save_error_metric_graphs(output_path, data, data_info, problem_name, idx)
# %%
