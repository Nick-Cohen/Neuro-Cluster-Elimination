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
# from adabelief_pytorch import AdaBelief

class Net(nn.Module):
    def __init__(self, bucket, hidden_sizes=None):
        self.bucket = bucket
        self.gm = self.bucket.gm
        nn_config = bucket.config
        input_size = bucket._get_nn_input_size()
        if hidden_sizes is None:
            hidden_sizes = nn_config['hidden_sizes']
        device = nn_config['device']
        seed = nn_config.get('seed', None)
        self.use_linspace_bias = nn_config.get('use_linspace_bias', False)
        if 'memorizer' in nn_config:
            self.memorizer = nn_config['memorizer']
                 
        
        if seed is not None:
            torch.manual_seed(seed)
            if device == 'cuda':
                torch.cuda.manual_seed(seed)
                
        super().__init__()

        self.device = device

        # Handle 'bias_only' mode: learn only a single constant (bias term)
        self.bias_only = (hidden_sizes == 'bias_only')
        if self.bias_only:
            hidden_sizes = []  # Treat as linear model, then freeze weights

        # Configurable activation function (default: tanh for backward compat)
        activation_name = nn_config.get('activation', 'tanh')
        if activation_name == 'relu':
            activation_cls = nn.ReLU
        else:
            activation_cls = nn.Tanh

        # Masked net (NeuroBE-style): a value head + a sigmoid "is-nonzero" mask
        # head over a shared trunk. At inference, configs the mask predicts as
        # zero are set to -inf (true zero) instead of relying on factor doping.
        self.masked_net = nn_config.get('masked_net', False)
        self._last_mask_logit = None

        layers = []
        prev_dim = input_size
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_cls())
            prev_dim = hidden_dim

        if self.masked_net:
            # trunk = shared hidden layers; separate value + mask heads
            self.trunk = nn.Sequential(*layers)
            self.value_head = nn.Linear(prev_dim, 1)
            self.mask_head = nn.Linear(prev_dim, 1)
            self.network = None
        else:
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)


        # Add a learnable bias parameter for linspace mode
        self.linspace_bias = nn.Parameter(torch.zeros(1, device=device))

        self.to(device)

        # Xavier/Glorot normal initialization for all layers
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0)

        # For bias_only mode: freeze weights to zero, only bias is trainable
        if self.bias_only:
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    layer.weight.data.fill_(0.0)
                    layer.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.masked_net:
            h = self.trunk(x)
            network_output = self.value_head(h)
            self._last_mask_logit = self.mask_head(h)
        else:
            network_output = self.network(x)

        if self.use_linspace_bias:
            # If we're using linspace bias, we need to combine with the output
            # using logsumexp. First, reshape the outputs to ensure compatibility.
            batch_size = network_output.shape[0]
            
            # Expand the bias to match the batch size
            expanded_bias = self.linspace_bias.expand(batch_size, 1)
            
            # Stack the network output and the bias together along a new dimension
            combined = torch.stack([network_output, expanded_bias], dim=-1)
            
            # Apply logsumexp along the last dimension
            final_output = torch.logsumexp(combined, dim=-1, keepdim=True)

            out = final_output
        else:
            # If not using linspace bias, just return the network output directly
            out = network_output

        # Masked net: at inference (eval), force configs the mask predicts as
        # zero to -inf (true zero) -- this is value*mask in log-space. During
        # training we return the raw value; the BCE mask loss reads _last_mask_logit.
        if self.masked_net and not self.training:
            maskp = torch.sigmoid(self._last_mask_logit)
            out = out.masked_fill(maskp < 0.5, float('-inf'))
        return out
    
    def get_sum_grad(self) -> torch.Tensor:
        total_grad = torch.tensor(0.0, device=self.device)
        
        for param in self.parameters():
            if param.grad is not None:
                total_grad += abs(param.grad.sum())
        
        return total_grad


class WMBResidualNet(nn.Module):
    """Training-time wrapper that reconstructs the message from a residual net.

    Used only when ``wmb_residual`` is enabled. The inner ``Net`` predicts the
    *residual* in normalized units; the WMB base value at each sample rides
    along as the LAST column of the input ``x`` (appended by
    ``DataLoader.load``), already scaled into the same normalized units.

        forward(x) = self.scale * inner(x[:, :-1]) + x[:, -1:]

    ``scale`` is 1.0 under ``wmb_residual_norm='message'`` (the normaliser stays
    fitted to y, so the inner net's units already are y's units). Under
    ``wmb_residual_norm='residual'`` a second normaliser is fitted to the
    residual r, the inner net outputs ``r_norm`` in *residual* units, and

        y_norm = (r*ln10 + base*ln10 - off_y)/scale_y
               = r_norm*(scale_r/scale_y) + (off_r + base*ln10 - off_y)/scale_y

    so ``scale = scale_r/scale_y`` and the trailing column carries the whole
    affine offset. ``DataLoader.load`` sets both, once, on the first load.

    Because every ``self.net(...)`` call site in ``Trainer`` receives batches
    built by ``DataLoader.load``, wrapping the net makes *all* losses, all
    validation losses and all early-stopping checks operate on the
    reconstructed message ``y_hat = NN_residual + wmb_base`` against the
    original target ``y`` -- which also keeps the ``neurobe_weighted_mse``
    importance weights keyed on the true message ``y`` rather than on the
    residual.

    The factor emitted downstream is ``FactorNN(inner, ...)`` (the residual
    alone), multiplied back by the WMB base factors that the cluster emits
    alongside it.
    """

    def __init__(self, inner: nn.Module, scale: float = 1.0):
        super().__init__()
        self.inner = inner
        # Plain float, not a buffer/parameter: it is a fixed unit conversion set
        # once from the fitted normaliser stats, never trained.
        self.scale = float(scale)

    @property
    def masked_net(self):
        return getattr(self.inner, 'masked_net', False)

    @property
    def device(self):
        return self.inner.device

    @property
    def bucket(self):
        return self.inner.bucket

    @property
    def gm(self):
        return self.inner.gm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.inner(x[:, :-1]) + x[:, -1:]


class Memorizer(Net):
    def __init__(self, bucket, all_x, all_y):
        # Call parent nn.Module constructor first
        super().__init__(bucket)

        self.nsamples = len(all_x)
        
        # Create a dictionary to store all input-output pairs
        self.memory = {}
        
        # Populate the memory dictionary
        for i in range(self.nsamples):
            input_vector = tuple(all_x[i].tolist())
            value = all_y[i].item()
            self.memory[input_vector] = value
        
        # Create a simple linear layer for unseen inputs
        self.linear = nn.Linear(len(all_x[0]), 1)
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
    
class BitVectorLookup(Net):
    def __init__(self, bucket, input_size):
        # Call parent nn.Module constructor first
        super().__init__(bucket)
        # 32 possible inputs (2^5), each maps to 1 parameter
        self.input_size = input_size
        num_params = 2 ** input_size
        self.lookup = nn.Embedding(num_params, 1).to(self.device)

    def forward(self, x):
        # Convert 5-bit vector to integer index
        # x shape: (batch_size, 5) with values 0 or 1
        indices = self.bits_to_index(x)
        # Get the parameter for each input
        return self.lookup(indices).squeeze(-1)  # Remove last dim to get (batch_size,)
    
    def bits_to_index(self, bits):
        # Convert binary vector to decimal index
        # bits: (batch_size, 5)
        powers = torch.tensor([2**i for i in range(self.input_size)], device=bits.device, dtype=bits.dtype)
        return torch.sum(bits * powers, dim=1).long()