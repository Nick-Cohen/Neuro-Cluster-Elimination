#%%
import sys
import os
from utils.distance_metrics import *
from fastElim import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# Get the absolute path of the other folder
other_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/cohenn1/SDBE/Super_Buckets/ARP/NN'))

# Add the path to sys.path
sys.path.append(other_folder_path)

# Now you can import the file

from NN_Train import *
#%%
ib = 5
Z = 303.0859680175781
uai_file = "/home/cohenn1/SDBE/width_under_20_problems/grid10x10.f10.uai"
# uai_file = "/home/cohenn1/SDBE/width20-30/grid20x20.f2.uai"
problem_name = "grid10x10.f10-ib" + str(ib)
# problem_name = "grid20x20.f2-ib" + str(ib)
output_path = "/home/cohenn1/SDBE/PyTorchGMs/graphs"
grid20f2_idxs = [30, 106, 213, 123, 331]
idx = 74
#%%
device = 'cuda'
fastgm = FastGM(uai_file=uai_file, device=device)

#%%

fastgm_copy = FastGM(uai_file=uai_file, device=device)
mg, mess = fastgm.get_message_gradient(idx)
# test_mess = copy.deepcopy(mess)
# for i in range(test_mess.tensor.numel()):
#     test_mess.tensor.view(-1)[i] = i
mg_hat = fastgm_copy.get_wmb_message_gradient(bucket_var=idx, i_bound=ib, weights='max')
mess.order_indices()
mg_hat.order_indices()

#%%


def renormalize_logspace_tensor(tensor, base=10):
    lnbase = torch.log(torch.tensor(base))
    logbasesumexpbase = torch.logsumexp(lnbase * tensor.reshape(-1), dim=0) / lnbase
    normalizing_constant = logbasesumexpbase - torch.log10(torch.tensor(tensor.numel()))
    return tensor - normalizing_constant, normalizing_constant

def sample_tensor(tensor, num_samples, device='cpu', debug=True):
    if debug:
        print("Debug mode: Using trivial sampling")
    # Flatten the tensor to sample indices uniformly
    flattened_tensor = tensor.reshape(-1)
    
    # Get the total number of elements in the flattened tensor
    total_elements = flattened_tensor.numel()
    
    if debug:
        # Debug line: Use a trivial permutation (keep original order)
        sampled_flat_indices = torch.arange(min(num_samples, total_elements))
    else:
        # Sample num_samples unique indices uniformly from the flattened tensor
        sampled_flat_indices = torch.randperm(total_elements)[:num_samples]
    
    # Manually convert flat indices back to multi-dimensional indices
    sampled_indices = []
    for flat_idx in sampled_flat_indices:
        flat_idx = flat_idx.item()  # Convert tensor to a plain Python integer
        multi_dim_idx = []
        for dim in reversed(tensor.shape):
            flat_idx, idx_in_dim = divmod(flat_idx, dim)
            multi_dim_idx.insert(0, idx_in_dim)  # Reversed order to maintain proper dimensions
        sampled_indices.append(multi_dim_idx)
    
    # Convert sampled_indices to a tensor of shape [num_samples, num_dims]
    sampled_indices = torch.tensor(sampled_indices, device=device)
    
    # Gather the values corresponding to the sampled indices
    sampled_values = tensor[tuple(sampled_indices.t())]  # Unpack indices to match the original tensor's shape
    
    # Return the sampled indices and values
    return sampled_indices, sampled_values

#%%
class Data_with_mg_tester(NN_Data):
    def __init__(self, full_message, mg_hat, num_samples, logspace=False):
        self.file_name = 'none'
        self.domain_sizes = torch.IntTensor(list(full_message.tensor.shape))
        self.logspace = logspace
        self.device = full_message.device
        if full_message.labels != mg_hat.labels:
            full_message.order_indices()
            mg_hat.order_indices()
        self.input_vectors, self.values, self.mg_hat = self.get_random_subset(full_message, mg_hat, num_samples)
        # trivial test set for now
        self.input_vectors_test = self.input_vectors
        self.values_test = self.values
        self.mg_hat_test = self.mg_hat
    
        self.num_samples = num_samples
        self.num_samples_test = num_samples
        self.max_value = float('-inf')
        
    
    def get_random_subset(self, full_message, mg_hat, num_samples):
        indices, logspace_mess_samples = sample_tensor(full_message.tensor, num_samples, device=self.device)
        formatted_indices = [tuple(x.item() for x in sample) for sample in indices]
        self.signatures_test = indices
        logspace_mg_hat_samples = mg_hat.tensor[tuple(indices.t())]
        self.max_mess, self.max_mg_hat = logspace_mess_samples.max(), logspace_mg_hat_samples.max()
        self.max_value = self.max_mess
        if not self.logspace:
            renormalized_mess, self.normalizing_constant_mess = renormalize_logspace_tensor(logspace_mess_samples,base=10)
            renormalized_mg_hat, self.normalizing_constant_mg_hat = renormalize_logspace_tensor(logspace_mg_hat_samples,base=10)
            linspace_renormalized_mess, linspace_renormalized_mg_hat = torch.pow(10, renormalized_mess), torch.pow(10, renormalized_mg_hat)
            return self.one_hot_encode(indices).float(), linspace_renormalized_mess, linspace_renormalized_mg_hat
            
            
            linear_space_mess = torch.pow(10, logspace_mess_samples - self.max_mess)
            mean_ls_mess = linear_space_mess.mean()
            adjusted_linear_space_mess = linear_space_mess /mean_ls_mess
            self.normalizing_constant = self.max_mess + t.log10(mean_ls_mess)
            linear_space_mg_hat = torch.pow(10, logspace_mg_hat_samples - self.max_mg_hat)
            adjusted_linear_space_mg_hat = linear_space_mg_hat / linear_space_mg_hat.mean()
            self.mg_hat_normalizing_constant = self.max_mg_hat - t.log10(linear_space_mg_hat.mean())
            return self.one_hot_encode(indices).float(), adjusted_linear_space_mess, adjusted_linear_space_mg_hat
        else:
            return self.one_hot_encode(indices).float(), logspace_mess_samples, None

def generate_loss_fn_data(mess, mg_hat, num_networks_trained, num_samples_from_message, loss_fn, epochs=100):
    Z_estimates = []
    
    if loss_fn == 'test':
        data = [Data_with_mg_tester(mess, mg_hat, num_samples_from_message, logspace=False) for _ in range(num_networks_trained)]
        for i,d in enumerate(data):
            print("Training network ", i+1)
            net = Memorizer(d, device='cpu')
            # net.train_model(
            #     X=d.input_vectors,
            #     Y=d.values,
            #     batch_size=512,
            #     early_stopping=False
            # )
            converted_nn = FastFactor.nn_to_FastFactor(idx, fastgm, net=net,device='cpu', debug=False)
            converted_nn.to_logspace(d.normalizing_constant)
            (converted_nn * mg).sum_all_entries()
            Z_estimates.append((converted_nn * mg).sum_all_entries())
            
    if loss_fn == 'grad_informed_l1':
        data = [Data_with_mg_tester(mess, mg_hat, num_samples_from_message, logspace=False) for _ in range(num_networks_trained)]
        for i,d in enumerate(data):
            print("Training network ", i+1)
            net = Net(d, epochs=epochs, loss_fn='grad_informed_l1_with_cancellation')
            net.train_model(
                X=d.input_vectors,
                Y=d.values,
                batch_size=1024,
                early_stopping=False
            )
            converted_nn = FastFactor.nn_to_FastFactor(idx, fastgm, net=net,device='cpu', debug=False)
            converted_nn.to_logspace(d.max_mess)
            (converted_nn * mg).sum_all_entries()
            Z_estimates.append((converted_nn * mg).sum_all_entries())
            
    if loss_fn == 'grad_informed_l1_with_cancellation':
        data = [Data_with_mg_tester(mess, mg_hat, num_samples_from_message, logspace=False) for _ in range(num_networks_trained)]
        for i,d in enumerate(data):
            print("Training network ", i+1)
            net = Net(d, epochs=epochs, loss_fn='grad_informed_l1_with_cancellation')
            net.train_model(
                X=d.input_vectors,
                Y=d.values,
                batch_size=1024,
                early_stopping=False
            )
            converted_nn = FastFactor.nn_to_FastFactor(idx, fastgm, net=net,device='cpu', debug=False)
            converted_nn.to_logspace(d.max_mess)
            # (converted_nn * mg).sum_all_entries()
            Z_estimates.append((converted_nn * mg).sum_all_entries())
            
    if loss_fn == 'grad_informed_mse':
        data = [Data_with_mg_tester(mess, mg_hat, num_samples_from_message, logspace=False) for _ in range(num_networks_trained)]
        for i,d in enumerate(data):
            print("Training network ", i+1)
            net = Net(d, epochs=epochs, loss_fn='grad_informed_mse')
            net.train_model(
                X=d.input_vectors,
                Y=d.values,
                batch_size=1024,
                early_stopping=False
            )
            converted_nn = FastFactor.nn_to_FastFactor(idx, fastgm, net=net,device='cpu', debug=False)
            converted_nn.to_logspace(d.max_mess)
            # (converted_nn * mg).sum_all_entries()
            Z_estimates.append((converted_nn * mg).sum_all_entries())
            
    elif loss_fn == 'logspace_mse':
        data = [Data_with_mg_tester(mess, mg_hat, num_samples_from_message, logspace=True) for _ in range(num_networks_trained)]
        for i,d in enumerate(data):
            print("Training network ", i+1)
            net = Net(d, epochs=epochs, loss_fn='mse')
            net.train_model(
                X=d.input_vectors,
                Y=d.values,
                batch_size=512,
                early_stopping=False
            )
            converted_nn = FastFactor.nn_to_FastFactor(idx, fastgm, net=net,device='cpu', debug=False)
            (converted_nn * mg).sum_all_entries()
            Z_estimates.append((converted_nn * mg).sum_all_entries())        
    
    return Z_estimates
#%%
# Z_hats_gil1 = generate_loss_fn_data(mess, mg, num_networks_trained=2, num_samples_from_message=1024, loss_fn='grad_informed_l1_with_cancellation', epochs=10000)

# Z_hats_test = generate_loss_fn_data(mess, mg, num_networks_trained=2, num_samples_from_message=1024, loss_fn='grad_informed_l1', epochs=10000)

logspace = True
d = Data_with_mg_tester(mess, mg, 1024, logspace=logspace)
# make data have values with mean 0
values_mean = d.values.mean()
d.values = d.values - values_mean
# make data values have variance 1
values_std = d.values.std()
d.values = d.values / values_std
# make mg_hat have mean 1
if not logspace:
    mg_hat_mean = d.mg_hat.mean()
    d.mg_hat = d.mg_hat / mg_hat_mean

# #debug
# for i in range(d.values.numel()):
#     d.values.view(-1)[i] = i

# # testZ = torch.log10((d.values * d.mg_hat).sum()) + d.normalizing_constant_mg_hat
# # testZ = d.normalizing_constant_mess + torch.log10((d.values * d.mg_hat).sum()) + d.normalizing_constant_mg_hat
# # print('Test Z:', testZ)
#%%
# # loss_fn = 'grad_informed_l1_with_cancellation'
# loss_fn = 'grad_informed_l1_with_cancellation'
loss_fn = 'logspace_mse'
# # loss_fn = 'grad_informed_mse'
net = Net(d, epochs=2000, lr = 0.1, loss_fn=loss_fn, hidden_size=1000)
net.train_model(
    X=d.input_vectors,
    Y=d.values,
    batch_size=1024,
    early_stopping=False
)
#%%
# net = Memorizer(d, device=device)
converted_nn = FastFactor.nn_to_FastFactor(idx, fastgm, net=net,device=device, debug=False)
# under forcing variance to 1
converted_nn.tensor = converted_nn.tensor * values_std
# undo forcing mean to 0
converted_nn.tensor = converted_nn.tensor + values_mean


# #%%
if not loss_fn == 'logspace_mse':
    converted_nn.to_logspace(d.normalizing_constant_mess)
print((converted_nn * mg).sum_all_entries())

# #%%
# # Examine max value
# print('Learned message max value is ', converted_nn.tensor.reshape(-1)[713].item(), ' at index 713')
# print('True value at index 713 is ', mess.tensor.reshape(-1)[713].item())

# #%%
# # Examine first 20 values
# print('Learned message values: ', converted_nn.tensor.reshape(-1)[:20])
# print('True message values: ', mess.tensor.reshape(-1)[:20])

# #%%

# Z_hats_gil1 = generate_loss_fn_data(mess, mg, num_networks_trained=2, num_samples_from_message=1024, loss_fn='grad_informed_l1', epochs=1000)

# #%%
# Z_hats_gil1c = generate_loss_fn_data(mess, mg, num_networks_trained=2, num_samples_from_message=1024, loss_fn='grad_informed_l1_with_cancellation', epochs=10)

# #%%

# Z_hats_mse = generate_loss_fn_data(mess, mg_hat, num_networks_trained=2, num_samples_from_message=1024, loss_fn='logspace_mse', epochs=10000)
# #%%
# print('Z_hats_gil1:', Z_hats_gil1)
# print('Z_hats_mse:', Z_hats_mse)
# print('True Z:', Z)





# %%
# loss_fn = 'logspace_mse'
# net = Net(data, loss_fn=loss_fn, epochs=1000)
# # (self, X, Y, batch_size=2048, save_validation=True, verbose_loss=False, mg=None)
# net.train_model(
#     X=data.input_vectors,
#     Y=data.values,
#     batch_size=512
# )

# net.save_model('/home/cohenn1/SDBE/Analysis/test_grad_informed_l1.jit')
# #%%
# file_path = '/home/cohenn1/SDBE/Analysis/test_grad_informed_l1.jit'
# net.save_model(file_path)
# # %%
# nn_to_FastFactor(jit_file, idx, fastGM, device='cuda', debug=False)
# converted_nn = FastFactor.nn_to_FastFactor(idx, fastgm, device='cpu', jit_file = file_path, debug=False)
# converted_nn.to_logspace(data.max_mess)
# # %%
# (converted_nn * mg).sum_all_entries()
# %%

#%%

# %%

# d = Data_with_mg_tester(mess, mg, 1024, logspace=True)

# #debug
# for i in range(d.values.numel()):
#     d.values.view(-1)[i] = i -511.5
    
#%%

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleTrainer:
    def __init__(self, model, learning_rate=0.001, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.device = device

    def train(self, inputs, targets, num_epochs=100, batch_size=32):
        targets = targets.unsqueeze(1)  # Add extra dimension

        dataset = torch.utils.data.TensorDataset(inputs, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_inputs, batch_targets in dataloader:
                batch_inputs, batch_targets = batch_inputs.to(self.device), batch_targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.loss_fn(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            if epoch % 10 == 0:
                self.print_sample_predictions(inputs[:5], targets[:5])
            
            if torch.isnan(loss):
                print("NaN loss detected. Stopping training.")
                break

    def print_sample_predictions(self, inputs, targets):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs.to(self.device))
            print("\nSample predictions:")
            for i in range(5):
                print(f"Input: {inputs[i]}")
                print(f"Target: {targets[i].item():.4f}, Prediction: {outputs[i].item():.4f}")
            print()
        self.model.train()

    def validate(self, inputs, targets):
        targets = targets.unsqueeze(1)  # Add extra dimension
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs.to(self.device))
            loss = self.loss_fn(outputs, targets.to(self.device))
        return loss.item()

# Usage
print("Input shape:", d.input_vectors.shape)
print("Input min/max:", d.input_vectors.min().item(), d.input_vectors.max().item())
print("Target shape:", d.values.shape)
print("Target min/max:", d.values.min().item(), d.values.max().item())

net = SimpleNN(input_size=d.input_vectors.shape[1], hidden_size=100, output_size=1)
trainer = SimpleTrainer(net, learning_rate=0.01, device='cuda' if torch.cuda.is_available() else 'cpu')
trainer.train(d.input_vectors, d.values, num_epochs=10000, batch_size=1024)
#%%
