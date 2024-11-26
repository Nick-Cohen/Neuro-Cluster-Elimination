import torch
import torch.nn.functional as F
from typing import List, Tuple
from inference.bucket import FastBucket

class DataPreprocessor:
    def __init__(self, y: torch.Tensor, mg: torch.Tensor, is_logspace: bool, device = None) -> None:
        self.y = y
        self.mg = mg
        if device is None:
            self.device = mg.device
        else:
            self.device = device
        self.is_logspace = is_logspace
        
        # these values, once set, are fixed. This may cause an issue if significantly larger values are sampled later
        self.y_max = None
        self.mg_max = None
    
    def convert_data(self, mess_normalizing_constant = None, mg_normalizing_constant = None) -> Tuple[torch.Tensor]:
        if not self.is_logspace:
            return self._convert_to_lin_space()
        else:
            return self._normalize_logspace(mess_normalizing_constant, mg_normalizing_constant)
    
    def _convert_to_lin_space(self, mess_normalizing_constant = None, mg_normalizing_constant = None):
        # update the max values if new data has larger max by factor of 32,000
        if mess_normalizing_constant is not None or mg_normalizing_constant is not None:
            raise ValueError("Normalizing constant changes are not yet implemented correctly")
        self.y_max = self.y.max() if mess_normalizing_constant is None or mess_normalizing_constant + 5 < self.y.max() else mess_normalizing_constant
        self.mg_max = self.mg.max() if mg_normalizing_constant is None or mg_normalizing_constant + 5 < self.mg.max() else mg_normalizing_constant
        
        # subtract the maxes
        self.y = self.y - self.y_max
        self.mg = self.mg - self.mg_max
        
        # exponentiate
        self.y = torch.pow(10, self.y)
        self.mg = torch.pow(10, self.mg)
        
        # make the mean 0 of just y, mg needs to keep proportionality
        self.y_mean = self.y.mean()
        self.y = self.y - self.y_mean
        
        # make the std 1 of just y
        self.y_std = self.y.std()
        self.y = self.y / self.y_std
        
        # make average value of mg to standardize mg loss weights to an average of 1
        self.mg_mean = self.mg.mean()
        self.mg = self.mg / self.mg_mean
        assert torch.all(self.mg >= 0)
        
        # to convert NN input back:
        # multiply by std, add mean, exponentiate, add max
        
        return self.y, self.mg
    
    def convert_back_message_logspace(self, outputs):
        outputs = outputs * self.y_std
        outputs = outputs + self.y_mean
        outputs[outputs < 0] = 0
        outputs = torch.log10(outputs)
        outputs = outputs + self.y_max
        return outputs
    
    # just needed for testing
    def convert_back_mg_logspace(self, mg_hat):
        mg_hat = mg_hat * self.mg_mean
        mg_hat = torch.log10(mg_hat)
        return mg_hat
    
    def _normalize_logspace(self, mess_normalizing_constant = None, mg_normalizing_constant = None):
        if mess_normalizing_constant is not None or mg_normalizing_constant is not None:
            raise ValueError("Normalizing constant changes are not yet implemented")
        self._normalize_logspace_message(mess_normalizing_constant)
        self._normalize_logspace_mg(mg_normalizing_constant)
        return self.y, self.mg
    
    def _normalize_logspace2(self, y_vals, mg_vals):
        self._set_normalizing_constants(y_vals, mg_vals)
        return self._normalize_logspace_message2(y_vals), self._normalize_logspace_mg2(mg_vals)
    
    def _set_normalizing_constants(self, y_vals, mg_vals):
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        if self.y_max is None:
            self.y_max = y_vals.max() * ln10
        if self.mg_max is None:
            self.mg_max = mg_vals.max() * ln10
        return self.y_max, self.mg_max
    
    def _normalize_logspace_message(self, normalizing_constant = None):
        # convert to log base e
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        self.y = self.y * ln10
        
        # calculate normalizing constant
        if normalizing_constant is None:
            self.y_max = self.y.max()
        elif normalizing_constant < self.y.max():
            self.y_max = self.y.max()
        else:
            self.y_max = normalizing_constant
        self.y -= self.y_max
        
    def _normalize_logspace_message2(self, y_vals):
        # convert to log base e
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        y_vals = y_vals * ln10
        
        # calculate normalizing constant
        y_vals -= self.y_max
        return y_vals
    
    def _normalize_logspace_mg(self, normalizing_constant = None):
        # convert to log base e
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        self.mg = self.mg * ln10
        
        # calculate normalizing constant
        if normalizing_constant is None:
            self.mg_max = self.mg.max()
        elif normalizing_constant < self.mg.max():
            self.mg_max = self.mg.max()
        else:
            self.mg_max = normalizing_constant
        self.mg -= self.mg_max
        
    def _normalize_logspace_mg2(self, mg_vals):
        # convert to log base e
        ln10 = torch.log(torch.tensor(10.0)).to(self.device)
        mg_vals = mg_vals * ln10
        
        # calculate normalizing constant
        mg_vals -= self.mg_max
        return mg_vals
        
    
    def one_hot_encode(self, bucket: FastBucket, assignments: torch.IntTensor, lower_dim = True):
        domain_sizes = bucket.get_message_size()
        num_samples, num_vars = assignments.shape
        
        if lower_dim: # send n domain variables to n-1 vector
            one_hot_encoded_samples = torch.cat([F.one_hot(assignments[:, i], num_classes=domain_sizes[i])[:, 1:] for i in range(num_vars)], dim=-1)
        else:
            one_hot_encoded_samples = torch.cat([F.one_hot(assignments[:, i], num_classes=domain_sizes[i]) for i in range(num_vars)], dim=-1)
        return one_hot_encoded_samples.float().to(self.device)