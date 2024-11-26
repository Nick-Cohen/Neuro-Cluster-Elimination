from .factor import FastFactor
import time
import torch

def nn_to_FastFactor(idx, fastGM, jit_file = None, net = None, device='cuda', debug=False):
    if jit_file is None and net is None or jit_file is not None and net is not None:
        raise ValueError("Exacgtly one of a JIT file or a PyTorch net must be provided")
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
    scope = fastGM.message_scopes[idx]
    domain_sizes = torch.tensor([fastGM.vars[fastGM.matching_var(v)].states for v in scope], device=device)
    
    # Calculate the total number of inputs
    total_inputs = (domain_sizes - 1).sum().item()
    
    if debug:
        input_creation_start = time.time()
    
    # Generate all possible assignments efficiently
    assignments = torch.cartesian_prod(*[torch.arange(size, device=device) for size in domain_sizes])
    
    # Create the input tensor efficiently
    all_inputs = torch.zeros((assignments.shape[0], total_inputs), device=device)
    
    offset = 0
    for i, size in enumerate(domain_sizes):
        if size > 1:
            mask = assignments[:, i].unsqueeze(1) == torch.arange(1, size, device=device)
            all_inputs[:, offset:offset+size-1] = mask.float()
            offset += size - 1

    if debug:
        input_creation_end = time.time()
        print(f"Creating input tensor took {input_creation_end - input_creation_start:.4f} seconds")

    # Query the model for all inputs
    if debug:
        query_start = time.time()
    with torch.no_grad():
        outputs = model(all_inputs).reshape(tuple(domain_sizes.tolist()))
    if debug:
        query_end = time.time()
        print(f"Querying model took {query_end - query_start:.4f} seconds")

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
