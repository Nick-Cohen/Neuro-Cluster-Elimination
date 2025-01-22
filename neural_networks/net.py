import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler
import xml.etree.ElementTree as ET
import time
from typing import List, Dict, Any, Tuple, IO
import argparse
import math
import sys
import matplotlib.pyplot as plt
from adabelief_pytorch import AdaBelief

class Net(nn.Module):
    def __init__(self, bucket):
        nn_config = bucket.config
        input_size = bucket._get_nn_input_size()
        hidden_sizes = nn_config['hidden_sizes']
        device = nn_config['device']
        seed = nn_config['seed']
        if 'memorizer' in nn_config:
            self.memorizer = nn_config['memorizer']
                 
        
        if seed is not None:
            torch.manual_seed(seed)
            if device == 'cuda':
                torch.cuda.manual_seed(seed)
                
        super().__init__()
        
        self.device = device
        
        layers = []
        prev_dim = input_size
        # use no bias for any layers
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Softplus())
            prev_dim = hidden_dim
       
        layers.append(nn.Linear(prev_dim, 1))
       
        self.network = nn.Sequential(*layers)
        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def get_sum_grad(self) -> torch.Tensor:
        total_grad = torch.tensor(0.0, device=self.device)
        
        for param in self.parameters():
            if param.grad is not None:
                total_grad += abs(param.grad.sum())
        
        return total_grad
    
    
class Memorizer(Net):
    def __init__(self, nn_config, x, y):
        # Call parent nn.Module constructor first
        super().__init__(nn_config)
        
        # Store device and sample info
        self.device = nn_config.get('device', 'cuda')
        self.nsamples = len(x)
        
        # Create a dictionary to store all input-output pairs
        self.memory = {}
        
        # Populate the memory dictionary
        for i in range(self.nsamples):
            input_vector = tuple(x[i].tolist())
            value = y[i].item()
            self.memory[input_vector] = value
        
        # Create a simple linear layer for unseen inputs
        self.linear = nn.Linear(len(x[0]), 1)
        self.to(self.device)

    def forward(self, x):
        outputs = []
        for input_vector in x:
            input_tuple = tuple(input_vector.tolist())
            if input_tuple in self.memory:
                outputs.append(self.memory[input_tuple])
            else:
                # For unseen inputs, use the linear layer
                outputs.append(self.linear(input_vector).item())
        return torch.tensor(outputs, device=self.device).view(-1, 1)

    def train_model(self, X, Y, batch_size=None, save_validation=False, verbose_loss=False):
        # No training needed for memorization
        pass

    def save_model(self, file_path=None):
        if file_path is None:
            file_path = "memorizer.pt"
        
        # Save the memory dictionary and linear layer
        torch.save({
            'memory': self.memory,
            'linear_state_dict': self.linear.state_dict()
        }, file_path)
        
    @classmethod
    def load_model(cls, file_path, nn_config, x_shape):
        checkpoint = torch.load(file_path)
        # Create a dummy input tensor of the right shape
        x = torch.zeros((1, x_shape))
        y = torch.zeros(1)
        memorizer = cls(nn_config, x, y)
        memorizer.memory = checkpoint['memory']
        memorizer.linear.load_state_dict(checkpoint['linear_state_dict'])
        return memorizer