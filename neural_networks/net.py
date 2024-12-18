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
    def __init__(self, nn_config):
        
        input_size = nn_config['input_size']
        hidden_sizes = nn_config['hidden_sizes']
        use_gradient_values = nn_config['use_gradient_values']
        device = nn_config['device']
        seed = nn_config['seed']
        if 'memorizer' in nn_config:
            self.memorizer = nn_config['memorizer']
                 
        
        if seed is not None:
            torch.manual_seed(seed)
            if device == 'cuda':
                torch.cuda.manual_seed(seed)
                
        super().__init__()
        
        self.use_gradient_values = use_gradient_values
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