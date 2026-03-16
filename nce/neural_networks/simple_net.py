import torch
import torch.nn.functional as F
import torch.nn as nn
class SimpleNet(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(0.0).to(device))  # Single scalar parameter

    def forward(self, x):
        return self.param.expand(x.shape[0]) 