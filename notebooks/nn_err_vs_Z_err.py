#%%
from fastElim import *
import torch
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
sys.path.append('/home/cohenn1/SDBE/PyGMs/pyGMs')

# Path to the parent directory of the package
package_parent_dir = '/home/cohenn1/SDBE/PyGMs'
package_name = 'pyGMs'

if package_parent_dir not in sys.path:
    sys.path.append(package_parent_dir)
import pyGMs as gm
from pyGMs import wmb
from pyGMs.neuro import *
from pyGMs.graphmodel import eliminationOrder
from pyGMs import Var

def print_cuda_memory():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def nn_err(approx_message, exact_message):
    mess_z_hat = approx_message.eliminate('all')
    mess_z = exact_message.eliminate('all')
    return mess_z_hat.tensor.item() - mess_z.tensor.item()

def true_err(approx_message, exact_message, message_gradient, Z=None, verbose=False):
    Z_hat = (approx_message * message_gradient).sum_all_entries()
    Z = (exact_message * message_gradient).sum_all_entries()
    if verbose:
        print("Z_hat is", Z_hat)
        print("Z is", Z)
    return Z_hat - Z
    
def weighted_nn_err(approx_message, exact_message, approx_gradient):
    return true_err(approx_message, exact_message, approx_gradient)

def noisy_message_errors(exact_message, message_gradient, noise_std=0.1):
    # Check that exact_message and message_gradient are on the same device
    if exact_message.tensor.device != message_gradient.tensor.device:
        raise ValueError("exact_message and message_gradient must be on the same device")

    # Use the device of the input tensors
    device = exact_message.tensor.device

    # Ensure we're working with tensors on the correct device
    exact_tensor = exact_message.tensor.to(device)
    gradient_tensor = message_gradient.tensor.to(device)

    # Generate Gaussian noise on the correct device
    noise = torch.randn_like(exact_tensor) * noise_std

    # Add noise to the exact message (in log space)
    noisy_tensor = exact_tensor + noise
    noisy_message = FastFactor(noisy_tensor, exact_message.labels)

    # Calculate nn_error
    nn_error = noisy_message.eliminate('all').tensor.item() - exact_message.eliminate('all').tensor.item()

    # Calculate true_error
    Z_hat = (noisy_message * message_gradient).eliminate('all').tensor.item()
    Z = (exact_message * message_gradient).eliminate('all').tensor.item()
    true_error = Z_hat - Z

    return nn_error, true_error

def xml_to_fastfactor(file_path, device='cuda'):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract information from the XML
    n_samples = int(root.get('n'))
    output_scope = [int(x) for x in root.get('outputfnscope').split(';')]
    domain_sizes = [int(x) for x in root.get('outputfnvariabledomainsizes').split(';')]
    is_log_space = root.get('datainlogspace') == 'Y'
    
    # Create a tensor to hold the factor values
    tensor_shape = tuple(domain_sizes)
    factor_tensor = torch.zeros(tensor_shape, device=device)
    
    # Fill the tensor with values from the XML
    for sample in root.findall('sample'):
        signature = tuple(int(x) for x in sample.get('signature').split(';'))
        value = float(sample.get('value'))
        factor_tensor[signature] = value
    
    # If the data is not in log space, convert it
    if not is_log_space:
        factor_tensor = torch.log10(factor_tensor)
    
    # Create and return the FastFactor
    return FastFactor(factor_tensor, output_scope)

def time_PYGMs(uai_file):
    start = time.time()
    ord_file = uai_file + ".vo"
    gm = uai_to_GM(uai_file=uai_file, order_file=ord_file)
    z(gm)
    return time.time() - start

def create_random_fastfactor(labels, mean=0.0, stdev=1.0, device='cuda'):
    """
    Create a random FastFactor with specified labels, mean, and standard deviation.
    All variables are assumed to have domain size 2.

    Args:
    labels (list): List of variable labels.
    mean (float): Mean of the normal distribution for generating random values.
    stdev (float): Standard deviation of the normal distribution for generating random values.
    device (str): The device to create the tensor on ('cuda' or 'cpu').

    Returns:
    FastFactor: A FastFactor object with random values.
    """
    # Determine the shape of the tensor
    shape = tuple([2] * len(labels))

    # Generate random values from a normal distribution
    random_values = np.random.normal(mean, stdev, size=shape)

    # Convert to log space
    log_values = np.log10(np.abs(random_values))

    # Convert to PyTorch tensor
    tensor = torch.tensor(log_values, dtype=torch.float32, device=device)

    # Create and return the FastFactor
    return FastFactor(tensor, labels)

def test_weighted_nn_err(num_samples=100, max_noise_std=0.1, device='cpu'):
    # Define labels
    mess_labels = [0, 2, 4, 6, 8, 9, 19, 11, 12, 13, 14, 15]
    partition1_labels = [1, 2, 3, 4, 5, 6, 7, 99]
    partition2_labels = [99, 9, 19, 11, 12, 13, 14, 15]
    part1_processed_labels = [1, 2, 3, 4, 5, 6, 7]

    # Convert labels to Var objects
    mess_vars = [Var(idx, 2) for idx in mess_labels]
    partition1_vars = [Var(idx, 2) for idx in partition1_labels]
    partition2_vars = [Var(idx, 2) for idx in partition2_labels]
    part1_processed_vars = [Var(idx, 2) for idx in part1_processed_labels]
   
    # Create FastFactors
    exact_message = create_random_fastfactor(labels=mess_vars, mean=0.0, stdev=1.0, device=device)
    partition1 = create_random_fastfactor(labels=partition1_vars, mean=0.0, stdev=1.0, device=device)
    partition2 = create_random_fastfactor(labels=partition2_vars, mean=0.0, stdev=1.0, device=device)

    # Process partition1
    part1_processed, _ = torch.max(partition1.tensor, dim=-1)
    part1_processed = FastFactor(part1_processed, labels=part1_processed_vars)

    # Process partition2
    part2_processed = partition2.eliminate([Var(99, 2)])

    # Compute mg and mg_hat
    mg = (partition1 * partition2).eliminate([Var(99, 2)])
    mg_hat = part1_processed * part2_processed

    data = []
    noise_levels = np.linspace(0, max_noise_std, num_samples)

    for noise_std in noise_levels:
        for _ in range(10):  # Generate multiple samples for each noise level
            # Add noise to the exact message
            noise = torch.randn_like(exact_message.tensor) * noise_std
            noisy_message = FastFactor(exact_message.tensor + noise, labels=mess_vars)

            # Calculate errors
            nn_error = nn_err(noisy_message, exact_message)
            weighted_nn_error = weighted_nn_err(noisy_message, exact_message, mg_hat)
            true_error = true_err(noisy_message, exact_message, mg)

            data.append((nn_error, weighted_nn_error, true_error))

    return data

def plot_errors(data):
    nn_errors, weighted_nn_errors, true_errors = zip(*data)

    plt.figure(figsize=(15, 6))

    # Plot nn_err vs true_error
    plt.subplot(1, 2, 1)
    plt.scatter(nn_errors, true_errors, alpha=0.5)
    plt.xlabel('NN Error')
    plt.ylabel('True Error')
    plt.title('NN Error vs True Error')
    min_err = min(min(nn_errors), min(true_errors))
    max_err = max(max(nn_errors), max(true_errors))
    plt.plot([min_err, max_err], [min_err, max_err], 'r--', label='y=x')
    plt.legend()

    # Plot weighted_nn_err vs true_error
    plt.subplot(1, 2, 2)
    plt.scatter(weighted_nn_errors, true_errors, alpha=0.5)
    plt.xlabel('Weighted NN Error')
    plt.ylabel('True Error')
    plt.title('Weighted NN Error vs True Error')
    min_err = min(min(weighted_nn_errors), min(true_errors))
    max_err = max(max(weighted_nn_errors), max(true_errors))
    plt.plot([min_err, max_err], [min_err, max_err], 'r--', label='y=x')
    plt.legend()

    plt.tight_layout()
    plt.show()

def check_approximate_tensor(exact_tensor, approx_tensor, factor_threshold=1.5, absolute_threshold=2):
    # Ensure tensors are on the same device
    device = exact_tensor.device
    approx_tensor = approx_tensor.to(device)

    # Calculate relative difference
    relative_diff = torch.abs(exact_tensor - approx_tensor) / torch.abs(exact_tensor)
    
    # Check if relative difference is within factor threshold
    within_factor = relative_diff <= (factor_threshold - 1)
    
    # Check if absolute difference is within absolute threshold
    absolute_diff = torch.abs(exact_tensor - approx_tensor)
    within_absolute = absolute_diff <= absolute_threshold
    
    # Combine conditions (either within factor OR within absolute threshold)
    is_close = torch.logical_or(within_factor, within_absolute)
    
    # Check if all elements satisfy the condition
    all_close = torch.all(is_close)
    
    # Calculate percentage of elements that are close
    percentage_close = torch.mean(is_close.float()) * 100
    
    return all_close.item(), percentage_close.item()






# %%

# jit_file = "/home/cohenn1/SDBE/Analysis/big_num_samples_experiement2/grid20x20.f2_iB_25_nSamples_10000_ecl_10_run_4/bucket-output-fn30/1000/nn_1000_0.jit"
uai_file = "/home/cohenn1/SDBE/width20-30/pedigree18.uai"
idx = 37
# fastgm = FastGM(uai_file=uai_file, device='cpu')
# mg_hat = fastgm.get_wmb_message_gradient(idx, 20, weights='max')

device = 'cpu'
fastgm = FastGM(uai_file=uai_file, device=device)
fastgm.eliminate_variables(up_to=37)
factors = fastgm.buckets[37].factors
elim_vars = [37]

#%%
FastGM.tester_sample_output_function(factors, elim_vars, device=device)


# fastgm_copy = FastGM(uai_file=uai_file, device='cpu')
# mg, mess = fastgm.get_message_gradient(37)
# mg_hat = fastgm_copy.get_wmb_message_gradient(37, 15, weights='max')
# #%%
# Z = (mg * mess).eliminate('all').tensor.item()
# Z_hat = (mg_hat * mess).eliminate('all').tensor.item()
# print("Z is", Z)
# print("Z_hat is", Z_hat)


# %%
