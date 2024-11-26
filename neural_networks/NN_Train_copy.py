# To run: python NN-Train.py --samples samplesFileName.xml --nn_path pathToSavedNN --done_path pathToFileIndicatingTrainingIsDone
# nn_path is 'nn-samples-varIndex1;varIndex2;... .pt' by default
# done_path is 'training-complete-varIndex1;varIndex2;... .pt' by default



# Load samples and domain size from file, also potentially is_log_space is_masked
# Convert samples to one-hot
# Train NN
# Write NN weights to file

#%%
import torch as t
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

def log10SumExp(tensor):
    max_val = max(tensor)
    return t.log10(t.sum(t.pow(10, tensor - max_val))) + max_val

def mean_adjusting_for_log_space(tensor):
    return log10SumExp(tensor) - t.log10(t.tensor(t.numel(tensor)))

class NN_Data:

    def __init__(self, file_name = None, processed_samples = None, values = None, device = 'cpu'):
        self.file_name = file_name
        # self.num_samples: int
        # self.features_per_sample: int
        # self.is_log_space: bool
        # self.max: float
        # self.min: float
        # self.sum: float # log sum of non-logspace values
        # self.signatures: t.Tensor
        # self.input_vectors: t.Tensor
        # self.values: t.Tensor
        # self.is_positive: t.BoolTensor # (num_samples,1) indicates if table has non-zero value: True or zero: False
        # self.domain_sizes: t.IntTensor
        self.device = device
        self.max_value = None
        if file_name is not None:
            try:
                 self.parse_samples()
            except IOError:
                print(f"Error reading file %s", self.file_name)

    def parse_samples(self):
        tree = ET.parse(self.file_name)
        root = tree.getroot()

        signatures_list = []
        values_list = []
        num_samples = 0
        for sample in root.iter('sample'):
            num_samples += 1
            signature = sample.get('signature')
            # Convert signature string '0;1;0;1;...' to a list of integers [0, 1, 0, 1, ...]
            signature = list(map(int, signature.split(';')))
            signatures_list.append(signature)

            value = float(sample.get('value'))
            values_list.append(value)
        self.num_samples = num_samples

        # Convert lists to PyTorch tensors
        signatures_tensor = t.tensor(signatures_list).to(self.device)
        values_tensor = t.tensor(values_list).to(self.device)

        # Replace -inf with a large negative number
        # values_tensor[values_tensor == float('-inf')] = -1e10
        
        self.max_value = float(max(values_tensor)) # take max value exponentiated out of log space

        self.signatures, self.values = signatures_tensor, values_tensor
        # self.values[self.values == float('-inf')] = -1e10
            
        self.is_positive = self.values.ne(float('-inf'))
        

        # Get 'outputfnvariabledomainsizes' attribute and convert it to a list of integers
        domain_sizes = [int(x) for x in root.get('outputfnvariabledomainsizes').split(';')]
        self.domain_sizes = t.IntTensor(domain_sizes).to(self.device)
        # self.input_vectors = (self.one_hot_encode(self.signatures).float()).to(self.device)
        
        # Split into test and validation
        total_samples = self.num_samples
        split_point = self.num_samples - min(5000, math.ceil(0.8 * total_samples))  # min of 20% or 5000 for validation

        # Splitting the data into training and test sets
        self.signatures = signatures_tensor[:split_point]
        self.values = values_tensor[:split_point]
        self.input_vectors = self.one_hot_encode(self.signatures).float().to(self.device)[:split_point]
        
        self.signatures_test = signatures_tensor[split_point:]
        self.values_test = values_tensor[split_point:]
        self.input_vectors_test = self.one_hot_encode(self.signatures_test).float().to(self.device)
        
        # Update the number of samples for both training and test sets
        # self.remove_duplicates()
        self.num_samples = len(self.input_vectors)
        self.num_samples_test = len(self.input_vectors_test)

    def remove_duplicates(self):
        def hash_sample(sample):
            return hash(tuple(sample.tolist()))

        # Hashing the training data
        train_hashes = set(hash_sample(sample) for sample in self.signatures)

        # Filtering out duplicates from test data
        unique_indices = [i for i, test_sample in enumerate(self.input_vectors_test)
                          if hash_sample(test_sample) not in train_hashes]

        self.values_test = self.values_test[unique_indices]
        self.input_vectors_test = self.input_vectors_test[unique_indices]
        
    def reverse_transform(self, tensor):
        output = tensor * self.mean_transformation_constant
        output = t.log10(tensor)
        output = output + self.max_value
        output[t.isnan(output)] = 0
        return output

    def one_hot_encode(self, signatures: t.IntTensor, lower_dim = True):
        # transforms (num_samples, num_vars) tensor to (num_samples, sum(domain_sizes)) one hot encoding

        num_samples, num_vars = signatures.shape

        if lower_dim: # send n domain variables to n-1 vector
            one_hot_encoded_samples = t.cat([F.one_hot(signatures[:, i], num_classes=self.domain_sizes[i])[:, 1:] for i in range(num_vars)], dim=-1)
        else:
            one_hot_encoded_samples = t.cat([F.one_hot(signatures[:, i], num_classes=self.domain_sizes[i]) for i in range(num_vars)], dim=-1)
        return one_hot_encoded_samples
    
    
# %%
class Net(nn.Module):

    def __init__(self, nn_data: NN_Data, lr = 0.01, epochs = 1000, device = None, loss_fn = None, hidden_size=100):
        super(Net, self).__init__()
        self.nn_data = nn_data
        self.max_value = nn_data.max_value
        self.num_samples = self.nn_data.num_samples
        self.epochs = epochs
        self.has_constraints = float('-inf') in nn_data.values
        self.is_logspace = True
        if device is None:
            self.device = nn_data.device
        else:
            self.device = device
        if loss_fn == 'grad_informed_l1_with_cancellation':
            self.loss_fn = self.grad_informed_l1_with_cancellation
            self.is_logspace = False
        elif loss_fn == 'grad_informed_l1':
            self.loss_fn = self.grad_informed_l1
            self.is_logspace = False
        elif loss_fn == 'grad_informed_mse':
            self.loss_fn = self.grad_informed_mse
            self.is_logspace = False
        elif loss_fn == 'l1':
            # self.loss_fn = self.l1
            self.loss_fn = nn.L1Loss()
            self.is_logspace = False
        elif loss_fn == 'logspace_mse':
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.MSELoss()
        if self.loss_fn == self.grad_informed_l1_with_cancellation or self.loss_fn == self.grad_informed_l1 or self.loss_fn == self.grad_informed_mse:
            self.is_grad_informed = True
        else:
            self.is_grad_informed = False
        self.values = nn_data.values.float().to(self.device)
        input_size, _ = len(nn_data.input_vectors[0]), len(nn_data.input_vectors[0])*2
        # hidden_size = 1024
        # hidden_size = 100
        output_size = 1
        self.fc1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.fc2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.fc3 = nn.Linear(hidden_size, output_size).to(self.device)
        self.parameters_v = list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc3.parameters())
        
        if self.has_constraints:
            self.test_values_filtered = self.nn_data.values_test[self.nn_data.values_test != float('-inf')]
            self.test_X_filtered = self.nn_data.input_vectors_test[self.nn_data.values_test != float('-inf')]
            self.classifier_fc1 = nn.Linear(input_size, hidden_size).to(self.device)
            self.classifier_fc2 = nn.Linear(hidden_size, hidden_size).to(self.device)
            self.classifier_fc3 = nn.Linear(hidden_size, output_size).to(self.device)
            self.parameters_c = list(self.classifier_fc1.parameters()) + list(self.classifier_fc2.parameters()) + list(self.classifier_fc3.parameters())
            self.classifier_values = (~t.isinf(self.values)).float()
            self.classifier_values_test = (~t.isinf(self.nn_data.values_test.float().to(self.device))).float()
            self.es_classifier = False # initialize early stopping condition
            self.es_val_predictor = False
        # Initialize classifier attributes to dummy fc1, this is probably not efficient
        if not self.has_constraints:
            # dummy layers that aren't used but necessary for serialization
            self.classifier_fc1 = nn.Linear(1, 1).to(self.device)
            self.classifier_fc2 = nn.Linear(1, 1).to(self.device)
            self.classifier_fc3 = nn.Linear(1, 1).to(self.device)
            # self.classifier_fc1 = t.tensor(1)
            # self.classifier_fc2 = t.tensor(1)
            # self.classifier_fc3 = t.tensor(1)
        
        # self.activation_function = nn.Softplus().to(self.device)
        # debug
        self.activation_function = nn.ReLU().to(self.device)
 
        # self.loss_fn = nn.MSELoss()
        # self.loss_fn = self.nbe_loss
        # self.loss_fn = self.klish_loss
        if self.has_constraints:
            self.loss_fn2 = nn.BCEWithLogitsLoss() # delete this eventually
            self.loss_fn_c = nn.BCEWithLogitsLoss()
        
        # self.optimizer = t.optim.SGD(self.parameters(), lr=lr)
        self.optimizer = t.optim.Adam(self.parameters(), lr=lr)
        # self.optimizer = t.optim.Adam(self.parameters_v)
        # self.optimizer = AdaBelief(
        #     self.parameters(),
        #     lr=1e-3,
        #     eps=1e-16,
        #     betas=(0.9, 0.999),
        #     weight_decouple=True,
        #     rectify=False,
        #     print_change_log=False  # Add this to suppress the warning
        # )
        # self.optimizer = t.optim.SGD(self.parameters_v, lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        if self.has_constraints:
            self.optimizer_c = t.optim.Adam(self.parameters_c)
        
        # Inialize validation loss twice previous and once previous, used for early stopping
        self.val_loss2 = sys.float_info.max
        self.val_loss1 = self.val_loss2/2
        if self.has_constraints:
            self.val_loss2_c = sys.float_info.max
            self.val_loss1_c = self.val_loss2_c/2
            
        # Initialize storage for gradient norms
        self.gradient_norms = {name: [] for name, _ in self.named_parameters()}

    def forward(self, x, force_positive=t.tensor(False)):
        # value predictor
        out = self.fc1(x)
        out = self.activation_function(out)
        out = self.fc2(out)
        out = self.activation_function(out)
        out = self.fc3(out)
        if not self.is_logspace:
            out = out
            # out = self.activation_function(out)
        # binary classifier
        # debug = False
        # if debug:
        if self.has_constraints and not force_positive:
            out2 = self.classifier_fc1(x)
            out2 = self.activation_function(out2)
            out2 = self.classifier_fc2(out2)
            out2 = self.activation_function(out2)
            out2 = self.classifier_fc3(out2)
            out2 = t.sigmoid(out2)
            out2 = t.round(out2) # Masked Net
            # out2 = t.log10(out2)
            neg_inf = t.tensor(float('-inf')).to(out2.device)
            out2 = t.where(out2 == 0, neg_inf, t.tensor(0.))
            # print('out dtype =', out.dtype)
            # print('out2 dtype =', out2.dtype)
            # print(f'out2 is {out2}')
            # sys.exit(1)
            out = out + out2              
        return out
    
    def forward_classifier(self, x):
        out = self.classifier_fc1(x)
        out = self.activation_function(out)
        out = self.classifier_fc2(out)
        out = self.activation_function(out)
        out = self.classifier_fc3(out)
        return out
    
    def forward_train_with_constraints(self, x_filtered, x_unfiltered):
        # value predictor
        out = self.fc1(x_filtered)
        out = self.activation_function(out)
        out = self.fc2(out)
        out = self.activation_function(out)
        out = self.fc3(out)
        # binary classifier
        out2 = self.classifier_fc1(x_unfiltered)
        out2 = self.activation_function(out2)
        out2 = self.classifier_fc2(out2)
        out2 = self.activation_function(out2)
        out2 = self.classifier_fc3(out2)
        return out, out2
    
    def validate_model(self, print_loss=True, type=""):
        # debug
        # print("print_loss is ", print_loss)
        self.eval()  # set the model to evaluation mode
        with t.no_grad():
            if not self.has_constraints:
                test_X = self.nn_data.input_vectors_test
                test_Y = self.nn_data.values_test.float().to(self.device)
                mg_hat = self.nn_data.mg_hat if self.is_grad_informed else None
                num_test_samples = self.nn_data.num_samples_test
                
                

                total_val_loss = 0
                pred = self(test_X)
                if self.is_grad_informed:
                    val_loss = self.loss_fn(pred.view(-1), test_Y.view(-1), self.nn_data.mg_hat.view(-1))
                else:
                    val_loss = self.loss_fn(pred, test_Y.view(-1, 1))
                total_val_loss += val_loss.item()
                avg_val_loss = total_val_loss / num_test_samples
                if print_loss:
                    # err = t.abs(pred - test_Y.view(-1, 1))
                    # mean_loss = mean_adjusting_for_log_space(err).item()
                    # print('Validation mean loss (no log space): {:.12f}'.format(mean_loss), end=' ')
                    print('Validation Loss: {:.12f}'.format(avg_val_loss))
                    if math.isnan(avg_val_loss):
                        print("total_val_loss is ", total_val_loss, "num_test_samples is ", num_test_samples)
                return avg_val_loss
            else:
                num_test_samples_classifier = self.nn_data.num_samples_test
                num_test_samples_filtered = len(self.test_values_filtered)
                total_val_loss_classifier = 0
                total_val_loss_filtered = 0
                val_predictions, binary_predictions = self.forward_train_with_constraints(self.test_X_filtered, self.nn_data.input_vectors_test)
                val_loss_classifier = self.loss_fn2(binary_predictions, self.classifier_values_test.view(-1, 1))
                val_loss_filtered = self.loss_fn(val_predictions, self.test_values_filtered.view(-1, 1))
                # nn.BCEWithLogitsLoss(pred_constraint, constraints_values_batch)
                total_val_loss_classifier += val_loss_classifier.item()
                total_val_loss_filtered += val_loss_filtered.item()
                avg_val_loss_classifier = total_val_loss_classifier / num_test_samples_classifier
                if avg_val_loss_classifier == float('-inf') or avg_val_loss_classifier == float('inf'):
                    print('inf loss')
                    # print(binary_predictions)
                    print(self.classifier_values_test.view(-1, 1))
                    sys.exit()
                if num_test_samples_filtered == 0:
                    print("num_test_samples_filtered is zero! ", self.nn_data.file_name)
                    avg_val_loss_filtered = 123456789
                else:
                    avg_val_loss_filtered = total_val_loss_filtered / num_test_samples_filtered
                if print_loss:
                    print('Validation Loss Std: {:.12f}'.format(avg_val_loss_filtered), end=' ')
                    print('Validation Loss Consistency: {:.12f}'.format(avg_val_loss_classifier))
                    # err = t.abs(pred - test_Y.view(-1, 1))
                    # mean_loss = mean_adjusting_for_log_space(err).item()
                    # print('Validation mean loss (no log space): {:.12f}'.format(mean_loss), end=' ')
                return avg_val_loss_filtered, avg_val_loss_classifier

    def klish_loss(self, y_hat, y):
        # print(t.sum(t.log10(t.mean((y_hat - self.max_value)**10))  + self.max_value))
        # exit(1)
        # return t.sum((y-y_hat)**2)
        return t.sum((y-y_hat)**2) + t.sum(t.log10(t.mean(10**(y_hat - self.max_value)))  + self.max_value)
    
    def nbe_loss(self, y_hat, y):
        # print(t.pow(1.5, y_hat - self.nn_data.max_value))
        # return t.sum(t.pow((y - y_hat),2))
        # print(t.sum(t.abs(y_hat/self.nn_data.max_value) * t.pow((y - y_hat),2)))
        # print(t.pow(1.1, t.max(y,y_hat) - self.nn_data.max_value))
        # exit(1)
        return t.sum(t.pow(1.1, t.max(y,y_hat) - self.nn_data.max_value) * t.pow((y - y_hat),2))
        # return t.sum(t.pow(10, y_hat - self.nn_data.max_value) * t.pow((y - y_hat),2))
    
    # adding term to try to punish overestimation bias
    def custom_loss2(self, y_hat, y):
        base = 2.0
        return t.sum((y_hat - y)**2 * base ** (y_hat - y))
        # return t.sum((y_hat - y)**2 * base ** (t.max(y_hat,y) - y))
        
    def grad_informed_l1_with_cancellation(self, y_hat, y, mg_hat):
        # this function assumes that y, y_hat, and mg_hat are all outside of log space and have maxes divided out
        return t.abs(t.sum((y_hat.view(-1) - y.view(-1)) * mg_hat.view(-1)))
    
    def grad_informed_l1(self, y_hat, y, mg_hat):
        # this function assumes that y, y_hat, and mg_hat are all outside of log space and have maxes divided out
        return t.sum(t.abs(y_hat.view(-1) - y.view(-1)) * mg_hat.view(-1))
    
    def l1(self, y_hat, y, mg_hat=None):
        print(["y_hat: " + "{:.2f}".format(e.item()) for e in y_hat.view(-1)[:5]])
        print(["y   : " + "{:.2f}".format(e.item()) for e in y.view(-1)[:5]])
        return t.sum(t.abs(y_hat.view(-1) - y.view(-1)))
    
    def grad_informed_mse(self, y_hat, y, mg_hat):
        # this function assumes that y, y_hat, and mg_hat are all outside of log space and have maxes divided out
        return t.sum(((y_hat.view(-1) - y) * mg_hat)**2)
    
    def train_model(self, X, Y, batch_size=2048, save_validation=True, verbose_loss=False, early_stopping=True):
        # use other train method
        if self.has_constraints:
            res = self.train_model_with_constraints(X,Y,batch_size=batch_size, verbose_loss=verbose_loss)
            if save_validation:
                self.save_validation_results_to_file()
            return res

        if self.num_samples < 256:
            batch_size = self.num_samples
            # batch_size = self.num_samples // 10
        tiempo = time.time()
        epochs = self.epochs
        num_batches = int(self.num_samples // batch_size)
        # print(epochs,num_batches)
        previous_loss = 10e10
        early_stopping_counter = 0
        
        # fixed batches
        batches = []
        # # Get mini-batch
        for i in range(num_batches):
            X_batch = X[i*batch_size : (i+1)*batch_size]
            values_batch = Y[i*batch_size : (i+1)*batch_size]
            if self.is_grad_informed:
                mg_hat_batch = self.nn_data.mg_hat[i*batch_size : (i+1)*batch_size]
            else:
                mg_hat_batch = None
            if not self.has_constraints:
                batches.append((X_batch,values_batch, mg_hat_batch))
        for epoch in range(epochs):
            for i in range(num_batches):
                X_batch, values_batch, mg_hat_batch = batches[i]

                # Predict value
                pred_values = self(X_batch)

                # Compute loss
                if self.is_grad_informed:
                    loss = self.loss_fn(pred_values, values_batch, mg_hat_batch)
                else:
                    loss = self.loss_fn(pred_values, values_batch.view(-1, 1))
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                
                # Collect gradient norms
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        self.gradient_norms[name].append(grad_norm)

                # Update weights
                self.optimizer.step()
                
                #debug
                # self.scheduler.step()

            
            if (epoch+1) % 10 == 0:
                # Early stopping
                if early_stopping and self.early_stopping(verbose_loss=verbose_loss):
                    print("Early stopping triggered.")
                    print("Train time is ", time.time() - tiempo)
                    self.validate_model(batch_size)
                    self.display_debug_info()
                    if save_validation:
                            self.save_validation_results_to_file()
                    return True
                else:
                    with t.no_grad():
                        args = (self(self.nn_data.input_vectors_test).view(-1), self.nn_data.values_test.view(-1))
                        if self.is_grad_informed:
                            args = args + (self.nn_data.mg_hat,)
                        val_loss = self.loss_fn(*args)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, val_loss.item()))
        print("Train time is ", time.time() - tiempo)
        self.display_debug_info()
        self.validate_model(batch_size)
        if save_validation:
            self.save_validation_results_to_file()
        return True
             
        """ unfixed batches        
        # Create a TensorDataset
        if self.is_grad_informed:
            dataset = TensorDataset(X, Y, self.nn_data.mg_hat)
        else:
            dataset = TensorDataset(X, Y)

        # Create a DataLoader with shuffling
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            # Every 10 epochs, update the DataLoader to reshuffle data
            if epoch % 10 == 0:
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for tup in data_loader:
                if self.is_grad_informed:
                    X_batch, values_batch, mg_hat_batch = tup
                else:
                    X_batch, values_batch = tup
                    mg_hat_batch = None
                X_batch, values_batch, mg_hat_batch 
                # Move data to device if necessary
                X_batch = X_batch.to(self.device)
                values_batch = values_batch.to(self.device)
                mg_hat_batch = mg_hat_batch.to(self.device) if self.is_grad_informed else None
                
                # Your training code goes here
                self.optimizer.zero_grad()
                
                # Forward pass
                pred_values = self.forward(X_batch)
                
                # Compute loss
                if self.is_grad_informed:
                    loss = self.loss_fn(pred_values, values_batch.view(-1, 1), mg_hat_batch)
                else:
                    loss = self.loss_fn(pred_values, values_batch.view(-1, 1))
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
            # self.scheduler.step() # debug
            current_lr = self.optimizer.param_groups[0]['lr']
            early_stopping = False # debug
            if (epoch+1) % 10 == 0:
                if early_stopping and self.early_stopping(verbose_loss=verbose_loss):
                    print("Early stopping triggered.")
                    print("Train time is ", time.time() - tiempo)
                    self.validate_model(batch_size)
                    self.display_debug_info()
                    if save_validation:
                            self.save_validation_results_to_file()
                    return True
                else:
                    val_loss = self.validate_model(batch_size)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print('Epoch [{}/{}], Loss: {:.4f}, LR: {:.4f}'.format(epoch+1, epochs, val_loss, current_lr))
        print("Train time is ", time.time() - tiempo)
        self.display_debug_info()
        val_loss = self.validate_model(batch_size)
        if save_validation:
            self.save_validation_results_to_file()
        return True
        """
                
        
        
    def train_model_with_constraints(self, X, Y, batch_size=2048, verbose_loss=False):
        # if self.num_samples < 256:
        #     batch_size = self.num_samples // 10
        tiempo = time.time()
        epochs = self.epochs
        previous_loss = 10e10
        early_stopping_counter = 0
        # data for predicting the specific value
        X_v = X[Y != float('-inf')]
        Y_v = Y[Y != float('-inf')]
        # deta for the classifier
        X_c = X
        Y_c = self.classifier_values
        # Get batch size
        num_batches_v = int(len(X_v) // batch_size)
        if num_batches_v == 0:
            return False
        num_batches_c = int(self.num_samples // batch_size)
        print(f'num_batches_v is {num_batches_v} and num_batches_c is {num_batches_c} and len(X_v) is {len(X_v)} and len(X_c) is {len(X_c)}')
        # Get mini-batches
        batches_v = []
        batches_c = []
        for i in range(num_batches_v):
            X_batch = X_v[i*batch_size : (i+1)*batch_size]
            values_batch = Y_v[i*batch_size : (i+1)*batch_size]
            batches_v.append((X_batch,values_batch))
        for i in range(num_batches_c):
            X_batch = X_c[i*batch_size : (i+1)*batch_size]
            values_batch = Y_c[i*batch_size : (i+1)*batch_size]
            batches_c.append((X_batch,values_batch))
            
        early_stop_v = False
        early_stop_c = False
        
        loss_v = t.tensor(float('inf'))
        loss_c = t.tensor(float('inf'))
        
        # Train loop
        for epoch in range(epochs):
            for i in range(num_batches_v):
                if self.es_val_predictor:
                    break
                self.optimizer.zero_grad()
                # self.optimizer_c.zero_grad()
                x_b, y_b = batches_v[i]
                pred = self.forward(x_b, force_positive=True)
                loss_v = self.loss_fn(pred, y_b.view(-1,1))
                
                # Compute gradient via backprop
                loss_v.backward()
                # if epoch % 10 == 0 and i == 0:
                #     print(f'Loss is {loss_v.item()}')
                
                # Update weights
                self.optimizer.step()
                
                # for name, parameter in self.named_parameters():
                #     print(f"{name} gradient: \n{parameter.grad}")
                # sys.exit(1)
                
                
                
                
                
            for i in range(num_batches_c):
                if self.es_classifier:
                    break
                self.optimizer.zero_grad()
                self.optimizer_c.zero_grad()
                x_b, y_b = batches_c[i]
                logits = self.forward_classifier(x_b)
                loss_c = self.loss_fn_c(logits, y_b.view(-1,1))
                
                # Compute gradient via backprop
                loss_c.backward()
                
                # if epoch == 10:
                #     for name, parameter in self.named_parameters():
                #         print(f"Gradient of {name} is {parameter.grad}")
                #     sys.exit(1)
                # print("Before")
                # for name, param in self.named_parameters():
                #     print(f"{name}: {param}")
                # Update weights
                self.optimizer_c.step()

                # print("After")
                # for name, param in self.named_parameters():
                #     print(f"{name}: {param}")
                    
                # sys.exit(1)
                
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch: {epoch+1}')
                # Early stopping
                if self.early_stopping(verbose_loss=verbose_loss):
                    print("Early stopping triggered.")
                    print("Train time is ", time.time() - tiempo)
                    self.validate_model(batch_size)
                    self.display_debug_info()
                    return True
                if self.has_constraints:
                    # print('Epoch [{}/{}], Loss: {:.4f}, XEntropy loss: {:.4f}'.format(epoch+1, epochs, loss_v.item(), loss_c.item()))
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss_v.item()))
                else:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss_v.item()))
        print("Train time is ", time.time() - tiempo)
        self.display_debug_info()
        self.validate_model(batch_size)
        return True
        
    def early_stopping(self, verbose_loss=False):
        if not self.has_constraints:
            current_validation_loss = self.validate_model(print_loss=verbose_loss)
            if self.val_loss2 < self.val_loss1 and self.val_loss1 < current_validation_loss:
                return True
            else:
                self.val_loss2 = self.val_loss1
                self.val_loss1 = current_validation_loss
                return False
        else:
            current_validation_loss, current_validation_loss_c = self.validate_model(print_loss=verbose_loss)
            # print(f"val_loss2: {self.val_loss2}, val_loss1: {self.val_loss1}, current_validation_loss: {current_validation_loss}, val_loss2_c: {self.val_loss2_c}, val_loss1_c: {self.val_loss1_c}, current_validation_loss_c: {current_validation_loss_c}")
            # if self.val_loss2 < self.val_loss1 and self.val_loss1 < current_validation_loss and self.val_loss2_c < self.val_loss1_c and self.val_loss1_c < current_validation_loss_c:
            
            #early stopping val predictor condition
            if self.val_loss2 <= 1.000001 * self.val_loss1 and self.val_loss1 <= 1.000001 * current_validation_loss:
                self.es_val_predictor = True
            #early stopping classifier condition
            if self.val_loss2_c <= 1.000001 * self.val_loss1_c and self.val_loss1_c <= 1.000001 * current_validation_loss_c:
                self.es_classifier = True
            
            if self.es_classifier and self.es_val_predictor:
                return True
            else:
                self.val_loss2 = self.val_loss1
                self.val_loss2_c = self.val_loss1_c
                self.val_loss1 = current_validation_loss
                self.val_loss1_c = current_validation_loss_c
                return False
              
    def prediction_error(self, input_vector, value):
        self.eval()
        with t.no_grad():
            pred_value = self(input_vector)
            exp_pred, exp_value = t.exp(pred_value), t.exp(value)
            # print(exp_pred,exp_value, self.loss_fn(exp_pred,exp_value))
            assert(self.loss_fn(exp_pred, exp_value).device.type==self.device) #################################
            return self.loss_fn(exp_pred, exp_value)

    def plot_gradients(self):

        plt.figure(figsize=(12, 8))
        for name, grad_norms in self.gradient_norms.items():
            plt.plot(grad_norms, label=name)

        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save_model(self, file_path=None):
        if file_path is None:
            file_path = "nn-weights" + self.nn_data.file_name[7:-3] + "pt"
        scripted_model = t.jit.script(self)
        scripted_model.save(file_path)
    
    def save_validation_results_to_file(self, file_path=None):
        if file_path is None:
            file_path = "validation" + self.nn_data.file_name[7:-4] + ".txt"
        self.eval()  # Set the model to evaluation mode
        with t.no_grad():
            predictions = self(self.nn_data.input_vectors_test)
            true_values = self.nn_data.values_test
            assignments = self.nn_data.signatures_test
            with open(file_path, 'w') as file:
                file.write("Assignment,True Value,Predicted Value\n")
                for idx in range(len(assignments)):
                    assignment = ';'.join(map(str, assignments[idx].tolist()))
                    true_value = true_values[idx].item()
                    predicted_value = predictions[idx].item()
                    file.write(f"{assignment},{true_value},{predicted_value}\n")
        print(f"Validation results saved to {file_path}")


    def get_full_NN_message(self, full_table_data: NN_Data, model=None):
        input_vectors = t.cat((full_table_data.input_vectors, full_table_data.input_vectors_test), dim=0)
        size = self.nn_data.domain_sizes
        size = tuple(size.tolist())
        if model is None:
            return self(input_vectors).reshape(size)
        else:
            return model(input_vectors).reshape(size)

    def display_debug_info(self):
        print(f'Sample file is {self.nn_data.file_name}')
        input_length = len(self.nn_data.input_vectors)
        print("First ", input_length, " value predictions are")
        for i in range(min(100, input_length)):
            try:
                predicted = self.forward(self.nn_data.input_vectors[i], force_positive=False).item()
                true_value = self.nn_data.values[i].item()
                print(f'Predicted: {predicted}, True value: {true_value}')
            except IndexError as e:
                print(f"IndexError at index {i}: {e}")
            except Exception as e:
                print(f"Error at index {i}: {e}")
                
class Memorizer(Net):
    def __init__(self, nn_data: NN_Data, device='cuda'):
        super().__init__(nn_data, device=device)
        
        # Create a dictionary to store all input-output pairs
        self.memory = {}
        
        # Populate the memory dictionary
        for i in range(self.nn_data.num_samples):
            input_vector = tuple(self.nn_data.input_vectors[i].tolist())
            value = self.nn_data.values[i].item()
            self.memory[input_vector] = value
        
        # Create a simple linear layer for unseen inputs
        self.linear = nn.Linear(len(nn_data.input_vectors[0]), 1)

    def forward(self, x):
        outputs = []
        for input_vector in x:
            input_tuple = tuple(input_vector.tolist())
            if input_tuple in self.memory:
                outputs.append(self.memory[input_tuple])
            else:
                # For unseen inputs, use the linear layer
                outputs.append(self.linear(input_vector).item())
        return t.tensor(outputs, device=self.device).view(-1, 1)

    def train_model(self, X, Y, batch_size=None, save_validation=False, verbose_loss=False):
        # No training needed for memorization
        pass

    def save_model(self, file_path=None):
        if file_path is None:
            file_path = "memorizer-" + self.nn_data.file_name[7:-3] + "pt"
        
        # Save the memory dictionary and linear layer
        t.save({
            'memory': self.memory,
            'linear_state_dict': self.linear.state_dict()
        }, file_path)
        
    @classmethod
    def load_model(cls, file_path, nn_data):
        checkpoint = t.load(file_path)
        memorizer = cls(nn_data)
        memorizer.memory = checkpoint['memory']
        memorizer.linear.load_state_dict(checkpoint['linear_state_dict'])
        return memorizer
                

#%%
def main(file_name, nn_save_path, skip_training=False):
    data = NN_Data(file_name, device='cuda')
    
    gpu = True
    if gpu:
        print('Using GPU')
        nn = Net(data, epochs=1000)
        # nn = Net(data, epochs=1000000, has_constraints=False)
        if not skip_training:
            nn.train_model(data.input_vectors, data.values, batch_size=100)
        else:
            print('SKIPPING TRAINING')
        nn_cpu = Net(data, device='cpu')
        
        # copy weights to cpu model
        model_weights = nn.state_dict()
        cpu_model_weights = {k: v.cpu() for k, v in model_weights.items()}
        nn_cpu.load_state_dict(cpu_model_weights)
    else:
        print('debug test')
        sys.exit()
        nn_cpu = Net(data, epochs=1000000)
        nn_cpu.train_model(data.input_vectors, data.values)
    
    if nn_save_path is not None:
        nn_cpu.save_model(nn_save_path)
    else:
        nn_cpu.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network.')
    parser.add_argument('--samples', type=str, required=True, help='Path to the input file')
    parser.add_argument('--nn_path', type=str, default=None, help='Path to save the trained neural network')
    parser.add_argument('--done_path', type=str, default=None, help='Path to save the training done indicator file')
    args = parser.parse_args()
    main(args.samples, args.nn_path, skip_training=True)
    # Indicate training is done by creating a status file
    if args.done_path == None:
        done_path = 'training-complete-'+args.samples[8:-4]+'.txt'
    else:
        done_path = args.done_path
    with open(done_path, 'w') as f:
        f.write('Training complete')

#%%
# if True:
#     file_name = "/home/cohenn1/SDBE/_grad_space/testResults/__full_tables/NNs_and_samples/grid20x20.f2_iB_25_nSamples_10000_ecl_10_run_4/bucket-output-fn0.xml"
#     data = NN_Data(file_name, device='cuda')
#     net = Net(data, epochs=1000)
    # net.train_model(data.input_vectors, data.values)
# %%
