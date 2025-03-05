from NCE.inference import FastBucket
import torch
import torch.nn.functional as F
from .data_preprocessor import DataPreprocessor
from torch.utils.data import Dataset, DataLoader

class DataLoader:
    def __init__(self, bucket: FastBucket, sample_generator = None, data_preprocessor = None, sampled_assignments_vectors: torch.Tensor = None, sampled_mess_values: torch.Tensor = None, sampled_mg_values: torch.Tensor = None, filepath: str = None, lower_dim = True):
        if filepath is not None:
            self.assignments, self,values, self.mg_hat = self.load_data(filepath)
        else:
            self.assignments = sampled_assignments_vectors
            self.values = sampled_mess_values
            self.mg_hat = sampled_mg_values
        # self.num_samples = len(self.assignments)
        # self.domain_sizes = torch.IntTensor(list(self.assignments.shape))
        self.bucket = bucket
        self.sample_generator = sample_generator
        self.message_size = self.sample_generator.message_size
        self.data_preprocessor = data_preprocessor
        sampling_scheme = self.sample_generator.gm.config['sampling_scheme']
        self.grad_informed = (sampling_scheme != 'uniform') or sampling_scheme != 'all'
        
    # def __len__(self):
    #     """Required: Returns the total number of samples"""
    #     return len(self.assignments)
    
    def __getitem__(self, idx):
        if self.mg_hat is not None:
            return {
                'input': self.assignments[idx],
                'target': self.values[idx],
                'mg_hat': self.mg_hat[idx]
            }
        return {
            'input': self.assignments[idx],
            'target': self.values[idx]
        }
        
    def shuffle_data(self):
        indices = torch.randperm(len(self))
    
    def load_data(filepath: str):
        # In case I want to presave datasets to save time in testing
        pass
    
    def load(self, num_samples = 0, all=False):
        # generate the samples with the sample generator
        if self.sample_generator is None:
            raise ValueError("No sample generator provided")
        if all:
            assignments = self.sample_generator.sample_assignments(sampling_scheme='all')
        else:
            assignments = self.sample_generator.sample_assignments(num_samples) # config in sg gives sample scheme
        mess_values = self.sample_generator.compute_message_values(assignments)
        assert not mess_values.requires_grad
        if self.grad_informed:
            mg_values = self.sample_generator.compute_gradient_values(assignments)
        # format the samples with the data preprocessor    
        return self.data_preprocessor.one_hot_encode(self.bucket, assignments), *self.data_preprocessor._normalize_logspace2(y_vals=mess_values, mg_vals=mg_values)
    
    def load_batches(self, batch_size, num_batches):
        num_samples = num_batches * batch_size
        data = self.load(num_samples)
        batches = []
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch = {
                'x': data[0][start:end],
                'y': data[1][start:end],
                'mgh': data[2][start:end]
            }
            batches.append(batch)
        return batches
    
    # generate complete validation set with all assignments for testing
    def load_all(self, num_samples = 0, grad_informed = True, all=True):
        data = self.load(num_samples, all)
        return [{
            'x': data[0],
            'y': data[1],
            'mgh': data[2]
        }]
    
def create_data_loaders(signatures, values, mg_hat=None, batch_size=32, split_point=0.8):
    
    #debug
    device = 'cpu'
    
    if mg_hat is not None:
        mg_hat_arg = mg_hat.to(device)
    else:
        mg_hat_arg = None
    dataset = Data(signatures.to(device), values.to(device), mg_hat_arg)
    
    # Calculate split sizes (e.g., 80% train, 20% validation)
    train_size = int(split_point * len(dataset))
    val_size = len(dataset) - train_size
    
    # Random split
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # debug
    train_dataset
    val_dataset
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


# def one_hot_encode(self, signatures: torch.IntTensor, lower_dim = True):
#     # transforms (num_samples, num_vars) tensor to (num_samples, sum(domain_sizes)) one hot encoding

#     num_samples, num_vars = signatures.shape

#     if lower_dim: # send n domain variables to n-1 vector
#         one_hot_encoded_samples = torch.cat([F.one_hot(signatures[:, i], num_classes=self.domain_sizes[i])[:, 1:] for i in range(num_vars)], dim=-1)
#     else:
#         one_hot_encoded_samples = torch.cat([F.one_hot(signatures[:, i], num_classes=self.domain_sizes[i]) for i in range(num_vars)], dim=-1)
#     return one_hot_encoded_samples

# def renormalize_logspace_tensor(tensor, base=10):
#     lnbase = torch.log(torch.tensor(base))
#     logbasesumexpbase = torch.logsumexp(lnbase * tensor.reshape(-1), dim=0) / lnbase
#     normalizing_constant = logbasesumexpbase - torch.log10(torch.tensor(tensor.numel()))
#     return tensor - normalizing_constant, normalizing_constant

# def sample_tensor(tensor, num_samples, device='cpu', debug=True):
#     if debug:
#         print("Debug mode: Using trivial sampling")
#     # Flatten the tensor to sample indices uniformly
#     flattened_tensor = tensor.reshape(-1)
    
#     # Get the total number of elements in the flattened tensor
#     total_elements = flattened_tensor.numel()
    
#     if debug:
#         # Debug line: Use a trivial permutation (keep original order)
#         sampled_flat_indices = torch.arange(min(num_samples, total_elements))
#     else:
#         # Sample num_samples unique indices uniformly from the flattened tensor
#         sampled_flat_indices = torch.randperm(total_elements)[:num_samples]
    
#     # Manually convert flat indices back to multi-dimensional indices
#     sampled_indices = []
#     for flat_idx in sampled_flat_indices:
#         flat_idx = flat_idx.item()  # Convert tensor to a plain Python integer
#         multi_dim_idx = []
#         for dim in reversed(tensor.shape):
#             flat_idx, idx_in_dim = divmod(flat_idx, dim)
#             multi_dim_idx.insert(0, idx_in_dim)  # Reversed order to maintain proper dimensions
#         sampled_indices.append(multi_dim_idx)
    
#     # Convert sampled_indices to a tensor of shape [num_samples, num_dims]
#     sampled_indices = torch.tensor(sampled_indices, device=device)
    
#     # Gather the values corresponding to the sampled indices
#     sampled_values = tensor[tuple(sampled_indices.t())]  # Unpack indices to match the original tensor's shape
    
#     # Return the sampled indices and values
#     return sampled_indices, sampled_values

# #%%
# class Data_with_mg_tester(NN_Data):
#     def __init__(self, full_message, mg_hat, num_samples, logspace=False):
#         self.file_name = 'none'
#         self.domain_sizes = torch.IntTensor(list(full_message.tensor.shape))
#         self.logspace = logspace
#         self.device = full_message.device
#         if full_message.labels != mg_hat.labels:
#             full_message.order_indices()
#             mg_hat.order_indices()
#         self.input_vectors, self.values, self.mg_hat = self.get_random_subset(full_message, mg_hat, num_samples)
#         # trivial test set for now
#         self.input_vectors_test = self.input_vectors
#         self.values_test = self.values
#         self.mg_hat_test = self.mg_hat
    
#         self.num_samples = num_samples
#         self.num_samples_test = num_samples
#         self.max_value = float('-inf')
        
    
#     def get_random_subset(self, full_message, mg_hat, num_samples):
#         indices, logspace_mess_samples = sample_tensor(full_message.tensor, num_samples, device=self.device)
#         formatted_indices = [tuple(x.item() for x in sample) for sample in indices]
#         self.signatures_test = indices
#         logspace_mg_hat_samples = mg_hat.tensor[tuple(indices.t())]
#         self.max_mess, self.max_mg_hat = logspace_mess_samples.max(), logspace_mg_hat_samples.max()
#         self.max_value = self.max_mess
#         if not self.logspace:
#             renormalized_mess, self.normalizing_constant_mess = renormalize_logspace_tensor(logspace_mess_samples,base=10)
#             renormalized_mg_hat, self.normalizing_constant_mg_hat = renormalize_logspace_tensor(logspace_mg_hat_samples,base=10)
#             linspace_renormalized_mess, linspace_renormalized_mg_hat = torch.pow(10, renormalized_mess), torch.pow(10, renormalized_mg_hat)
#             return self.one_hot_encode(indices).float(), linspace_renormalized_mess, linspace_renormalized_mg_hat
            
            
#             linear_space_mess = torch.pow(10, logspace_mess_samples - self.max_mess)
#             mean_ls_mess = linear_space_mess.mean()
#             adjusted_linear_space_mess = linear_space_mess /mean_ls_mess
#             self.normalizing_constant = self.max_mess + torch.log10(mean_ls_mess)
#             linear_space_mg_hat = torch.pow(10, logspace_mg_hat_samples - self.max_mg_hat)
#             adjusted_linear_space_mg_hat = linear_space_mg_hat / linear_space_mg_hat.mean()
#             self.mg_hat_normalizing_constant = self.max_mg_hat - t.log10(linear_space_mg_hat.mean())
#             return self.one_hot_encode(indices).float(), adjusted_linear_space_mess, adjusted_linear_space_mg_hat
#         else:
#             return self.one_hot_encode(indices).float(), logspace_mess_samples, None

# def generate_loss_fn_data(mess, mg_hat, num_networks_trained, num_samples_from_message, loss_fn, epochs=100):
#     Z_estimates = []
    
#     if loss_fn == 'test':
#         data = [Data_with_mg_tester(mess, mg_hat, num_samples_from_message, logspace=False) for _ in range(num_networks_trained)]
#         for i,d in enumerate(data):
#             print("Training network ", i+1)
#             net = Memorizer(d, device='cpu')
#             # net.train_model(
#             #     X=d.input_vectors,
#             #     Y=d.values,
#             #     batch_size=512,
#             #     early_stopping=False
#             # )
#             converted_nn = FastFactor.nn_to_FastFactor(idx, fastgm, net=net,device='cpu', debug=False)
#             converted_nn.to_logspace(d.normalizing_constant)
#             (converted_nn * mg).sum_all_entries()
#             Z_estimates.append((converted_nn * mg).sum_all_entries())
            
#     if loss_fn == 'grad_informed_l1':
#         data = [Data_with_mg_tester(mess, mg_hat, num_samples_from_message, logspace=False) for _ in range(num_networks_trained)]
#         for i,d in enumerate(data):
#             print("Training network ", i+1)
#             net = Net(d, epochs=epochs, loss_fn='grad_informed_l1_with_cancellation')
#             net.train_model(
#                 X=d.input_vectors,
#                 Y=d.values,
#                 batch_size=1024,
#                 early_stopping=False
#             )
#             converted_nn = FastFactor.nn_to_FastFactor(idx, fastgm, net=net,device='cpu', debug=False)
#             converted_nn.to_logspace(d.max_mess)
#             (converted_nn * mg).sum_all_entries()
#             Z_estimates.append((converted_nn * mg).sum_all_entries())
            
#     if loss_fn == 'grad_informed_l1_with_cancellation':
#         data = [Data_with_mg_tester(mess, mg_hat, num_samples_from_message, logspace=False) for _ in range(num_networks_trained)]
#         for i,d in enumerate(data):
#             print("Training network ", i+1)
#             net = Net(d, epochs=epochs, loss_fn='grad_informed_l1_with_cancellation')
#             net.train_model(
#                 X=d.input_vectors,
#                 Y=d.values,
#                 batch_size=1024,
#                 early_stopping=False
#             )
#             converted_nn = FastFactor.nn_to_FastFactor(idx, fastgm, net=net,device='cpu', debug=False)
#             converted_nn.to_logspace(d.max_mess)
#             # (converted_nn * mg).sum_all_entries()
#             Z_estimates.append((converted_nn * mg).sum_all_entries())
            
#     if loss_fn == 'grad_informed_mse':
#         data = [Data_with_mg_tester(mess, mg_hat, num_samples_from_message, logspace=False) for _ in range(num_networks_trained)]
#         for i,d in enumerate(data):
#             print("Training network ", i+1)
#             net = Net(d, epochs=epochs, loss_fn='grad_informed_mse')
#             net.train_model(
#                 X=d.input_vectors,
#                 Y=d.values,
#                 batch_size=1024,
#                 early_stopping=False
#             )
#             converted_nn = FastFactor.nn_to_FastFactor(idx, fastgm, net=net,device='cpu', debug=False)
#             converted_nn.to_logspace(d.max_mess)
#             # (converted_nn * mg).sum_all_entries()
#             Z_estimates.append((converted_nn * mg).sum_all_entries())
            
#     elif loss_fn == 'logspace_mse':
#         data = [Data_with_mg_tester(mess, mg_hat, num_samples_from_message, logspace=True) for _ in range(num_networks_trained)]
#         for i,d in enumerate(data):
#             print("Training network ", i+1)
#             net = Net(d, epochs=epochs, loss_fn='mse')
#             net.train_model(
#                 X=d.input_vectors,
#                 Y=d.values,
#                 batch_size=512,
#                 early_stopping=False
#             )
#             converted_nn = FastFactor.nn_to_FastFactor(idx, fastgm, net=net,device='cpu', debug=False)
#             (converted_nn * mg).sum_all_entries()
#             Z_estimates.append((converted_nn * mg).sum_all_entries())        
    
#     return Z_estimates
