from inference import FastGM
from inference import FastFactor
from sampling import SampleGenerator
from inference import populate_gradient_factors
from data import DataLoader
from data import create_data_loaders
from data import DataPreprocessor
from neural_networks import Trainer
from neural_networks.losses import *
import numpy as np
import torch
import torchinfo
import torch.nn as nn
from neural_networks import Net
import matplotlib.pyplot as plt
import pickle

device = 'cuda'

ib = 5
# Z = 303.0859680175781
# Z = -78.13997650146484 #pedigree18 doped -5
Z = 58.5306282043457 # rbm20
# Z = 291.7322692871094 # grid20x20f2
# uai_file = "/home/cohenn1/SDBE/width_under_20_problems/grid10x10.f10.uai"
# uai_file = "/home/cohenn1/SDBE/width20-30/grid20x20.f2.uai"
# uai_file = "/home/cohenn1/SDBE/width20-30/pedigree18.uai"
uai_file = "/home/cohenn1/SDBE/width20-30/rbm_20.uai"
# problem_name = "grid10x10.f10-ib" + str(ib)
# problem_name = "grid20x20.f2-ib" + str(ib)  
problem_name = "rbm_20.uai-ib" + str(ib)  
output_path = "/home/cohenn1/SDBE/PyTorchGMs/graphs"
# grid20f2_idxs = [30, 106, 213, 123, 331]
# idx = 74 # for grid 20, ib10
idx = 23

fastgm = FastGM(uai_file=uai_file, device=device)
fastgm.dope_factors(-5)
# fastgm.get_large_message_buckets(20)

# populate_gradient_factors(fastgm,iB=10)
bucket=fastgm.buckets[idx]
fastgm_copy = FastGM(uai_file=uai_file, device=device)
mg, mess = fastgm.get_message_gradient(idx)
gradient_factors = fastgm.get_gradient_factors(idx)
# mg_hat = fastgm_copy.get_wmb_message_gradient(bucket_var=idx, i_bound=ib, weights='max')
mess.order_indices()
# mg_hat.order_indices()

fastgm.config = {
    'num_samples': -1,
    'sampling_scheme': 'mg',
    'sampling_scheme': 'uniform'
}
sg = SampleGenerator(gm=fastgm, bucket=bucket, mess = mess, mg = mg)
sample_assignments = sg.sample_assignments(1000)
sample_values = sg.compute_message_values(sample_assignments)
sample_mg_values = sg.compute_gradient_values(sample_assignments, [mg])

data_preprocessor = DataPreprocessor(y=sample_values, mg=sample_mg_values, is_logspace=True, device = device)
dl = DataLoader(bucket=bucket, sample_generator=sg, data_preprocessor=data_preprocessor)
batches=dl.load_batches(10,5,mgh_factors=[mg]) # used to populate normalization constants
y, mgh = data_preprocessor.convert_data()
x = data_preprocessor.one_hot_encode(bucket, sample_assignments, lower_dim=True)

losses = [
    'logspace_mse',
    'from_logspace_gil1',
    'from_logspace_gil1c',
    'from_logspace_l1',
    'from_logspace_l2',
    'from_logspace_gil2',
    # 'combined_gil1_ls_mse'
]

# Z = 291.7323913574219

def evaluate_loss_fn(data_loader, config, Z, mg):
    net = Net(input_size=len(batches[0]['x'][0]), hidden_sizes=[100,100], use_gradient_values=True, device=device)
    t=Trainer(net, dataloader=dl, config=config)
    try:
        traced_losses_data = t.train(mgh_factors=[mg], traced_loss_fns=config['traced_loss_fns'])
        mess_hat = FastFactor.nn_to_FastFactor(idx, fastgm, data_preprocessor, net=net, device=device)
        return (mess_hat*mg).sum_all_entries() - Z, traced_losses_data
    except:
        mess_hat = FastFactor.nn_to_FastFactor(idx, fastgm, data_preprocessor, net=net, device=device)
        return (mess_hat*mg).sum_all_entries() - Z
    
    results = []
scheduler_config = {
    'factor': 0.5,
    'patience': 10,
    'min_lr': 1e-8
}
results = []
scheduler_config = {
    'factor': 0.5,
    'patience': 10,
    'min_lr': 1e-8
}
test_config = {
    'loss_fn': 'logspace_mse',
    'traced_loss_fns': losses,
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'device': device,
    'num_samples': 102400000,
    'batch_size': 4096,
    'num_epochs': 1,
    'set_size': 409600,
    'scheduler_config': scheduler_config
}

traced_loss_data = {loss_fn: [] for loss_fn in losses}
num_iters = 10
num_loop_iters = num_iters * len(losses)
i = 0
for _ in range(num_iters):
    for loss_fn in losses:
        i += 1
        print(f"Running iteration {i}/{num_loop_iters}")
        # for loss_fn in ['logspace_mse']:
        test_config['loss_fn'] = loss_fn
        print('testing ', 'loss_fn =', loss_fn)
        err, traced_loss_data_dict = evaluate_loss_fn(dl, test_config, Z, mg)
        traced_loss_data[loss_fn].append((err, traced_loss_data_dict))
        print('err: ', err)
        
# pickle the traced_losses_data

with open("/home/cohenn1/NCE/notebooks/traced_loss_data_rbm20_mg_distribution.out3.pickle", "wb") as file:
    pickle.dump(traced_loss_data, file)