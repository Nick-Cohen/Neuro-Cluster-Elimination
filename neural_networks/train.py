from .losses import *
from NCE.data import *
from NCE.sampling import *
# from NCE.inference.graphical_model import *
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys


class Trainer:
    # def __init__(self, model, dataloader, config, mgh_factors = None):
    def __init__(self, net, bucket):
        self.config = bucket.gm.config
        self.bucket = bucket
        self.net = net
        self.sample_generator, self.data_preprocessor, self.dataloader = self._make_dataloader()
        self.message_size = self.dataloader.message_size
        self.debug = self.config['debug']
        self.tracked = {'parameters': [], 'gradients': []}
        self.mgh_factors = [self._get_mgh()] # TODO: will need to grab list of factors in the future
        
        # Set optimizer
        if self.config['optimizer'] == 'adam' or self.config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), 
                lr=self.config['lr']
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.net.parameters(), 
                lr=self.config['lr']
            )
        # define scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',           # Minimize loss
            factor=self.config['lr_decay'],           # Reduce LR by this factor
            patience=self.config['patience'],          # Number of epochs with no improvement before reducing LR
            verbose=True,         # Print updates
            min_lr=self.config['min_lr']          # Minimum learning rate
        )
            
        self.loss_fn = self._get_loss_fn(self.config['loss_fn'])
   
    def _make_dataloader(self):
        sg = SampleGenerator(gm=self.bucket.gm, bucket=self.bucket, random_seed=self.config['seed'])
        sample_assignments = sg.sample_assignments(1000)
        sample_values = sg.compute_message_values(sample_assignments)
        sample_mg_values = sg.compute_gradient_values(sample_assignments, [self._get_mgh()]) # TODO: will need to accept list of factors in the future
        data_preprocessor = DataPreprocessor(y=sample_values, mg=sample_mg_values, is_logspace=True, device = self.config['device'])
        return sg, data_preprocessor, DataLoader(self.bucket, sample_generator=sg, data_preprocessor=data_preprocessor)
    
    def _get_mgh_factors(self):
        pass
    
    def _get_mgh(self):
        from inference import FastGM
        fastgm_copy = FastGM(uai_file=self.bucket.gm.uai_file, device=self.config['device'])
        mg_hat = fastgm_copy.get_wmb_message_gradient(bucket_var=self.bucket.label, i_bound=self.config['iB_backwards'], weights='max')
        return mg_hat

    def _get_val_set(self):
        if self.config['val_set'] is None:
            return None
        elif self.config['val_set'] == 'all':
            return self.dataloader.load_all(mg_hat_factors = self.mgh_factors, all=True)          
   
    def _get_loss_fn(self, loss_fn_name):
        if loss_fn_name == "mse" or loss_fn_name == "MSE":
            return nn.MSELoss()
        elif loss_fn_name == "gil1":
            return gil1
        elif loss_fn_name == "gil1c":
            return gil1c
        elif loss_fn_name == "logspace_mse":
            return logspace_mse
        elif loss_fn_name == "l1":
            return l1
        elif loss_fn_name == "gil2c":
            return gil2c
        elif loss_fn_name == "from_logspace_gil1":
            return from_logspace_gil1
        elif loss_fn_name == "logspace_l1":
            return from_logspace_l1
        elif loss_fn_name == "from_logspace_l1":
            return from_logspace_l1
        elif loss_fn_name == "from_logspace_mse" or loss_fn_name == "from_logspace_l2":
            return from_logspace_mse
        elif loss_fn_name == "from_logspace_gil2":
            return from_logspace_gil2
        elif loss_fn_name == "gil1c":
            return gil1c
        elif loss_fn_name == "z_err":
            return z_err
        elif loss_fn_name == "combined_gil1_ls_mse":
            return combined_gil1_ls_mse
        elif loss_fn_name == "logspace_mse_pathIS":
            return logspace_mse_pathIS
        elif loss_fn_name == "l1c":
            return l1c
        elif loss_fn_name == "huber_gil1c":
            return huber_gil1c
        elif loss_fn_name == "weighted_logspace_mse":
            return weighted_logspace_mse
        elif loss_fn_name == "weighted_logspace_mse_pedigree":
            return weighted_logspace_mse_pedigree
        
        else:
            raise ValueError(f"Loss function {self.config['loss_fn']} not recognized")
      
    def _get_weights(self, mg_hat_batch):
        if self.dataloader.sample_generator.sampling_scheme == 'mg':
            message_size = self.dataloader.bucket.get_message_size()
            p_dist = 1 / message_size
            q_dist = torch.exp(mg_hat_batch)
            return torch.exp(mg_hat_batch)
        elif self.dataloader.sample_generator.sampling_scheme == 'path':
            p_dist = 1 / message_size
            q_dist = torch.exp(mg_hat_batch)
            return torch.exp(mg_hat_batch)
        
    def train_batch(self, x_batch, y_batch, mg_hat_batch=None):
        self.net.train()
        # Zero the parameter gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.net(x_batch)
        
        # Compute loss
        mg_hat_batch.detach()
        weights = self._get_weights(mg_hat_batch)
        loss = self.loss_fn(outputs.squeeze(), y_batch, mg_hat_batch)
        # if mg_hat_batch is not None or self.loss_fn == logspace_mse:
        #     mg_hat_batch.detach()
        #     loss = self.loss_fn(outputs.squeeze(), y_batch, mg_hat_batch)
        # else:
        #     # print(outputs.shape,y_batch.shape)
        #     loss = self.loss_fn(outputs.squeeze(), y_batch)
            
        # Backward pass and optimize
        
        if self.debug:
            self.tracked['parameters'].append([p.data.clone() for p in self.net.parameters()])
        loss.backward()
        if self.debug:
            self.tracked['gradients'].append([p.grad.clone() for p in self.net.parameters()])
        
        print("Batch loss: ", loss.item(), " grad sum: ", self.net.get_sum_grad().item())
        
        if self.debug:
            for name, param in self.net.named_parameters():
                if param.grad is not None:
                    print(f"Before Step - {name}: Grad: {param.grad}, Param: {param.data}")
            self.optimizer.step()
            for name, param in self.net.named_parameters():
                print(f"After Step - {name}: Param: {param.data}")
        else:
            self.optimizer.step()
        return loss
    
    def train_epoch(self, batches):
        losses = []
        for batch in batches:
            x_batch, y_batch, mgh_batch = batch['x'], batch['y'], batch['mgh']
            # print('first ten mg_hat values: ', mgh_batch[:10])
            losses.append(self.train_batch(x_batch, y_batch, mgh_batch))
        if self.loss_fn == logspace_mse:
                loss = sum(losses) / len(losses)
        else:
            if self.debug:
                print('losses are ', losses)
            loss = self._aggregate_batch_losses(losses)
            if self.debug:
                print('loss is ', loss)
        return loss
    
    def _evaluate_batch(self, loss_fn_name, x_batch, y_batch, mg_hat_batch=None):
        with torch.no_grad():
            if self.debug:
                for param_group in self.optimizer.param_groups:
                    print(f"Learning rate: {param_group['lr']}")
            self.net.eval()
            # Forward pass
            # if self.debug:
            #     print('debug')
            #     print('x_batch is ', x_batch)
            outputs = self.net(x_batch)
            # Compute loss
            loss_fn = self._get_loss_fn(loss_fn_name)
            # if self.debug:
            #     print("Loss Function Type:", type(loss_fn))
            #     print("Arguments Passed: outputs, y_batch, mg_hat_batch")
            #     print(outputs.shape, y_batch.shape, mg_hat_batch.shape)
            loss = loss_fn(outputs.squeeze(), y_batch, mg_hat_batch)
            return loss
    
    def evaluate_epoch(self, loss_fns, batches):
        with torch.no_grad():
            out = []
            for loss_fn_name in loss_fns:
                losses = []
                for batch in batches:
                    x_batch, y_batch, mgh_batch = batch['x'], batch['y'], batch['mgh']
                    outputs = self.net(x_batch)
                    
                    losses.append(self._evaluate_batch(loss_fn_name, x_batch, y_batch, mgh_batch))
                if self._get_loss_fn(loss_fn_name) == logspace_mse:
                    # print('loss_fn_name is ', loss_fn_name)
                    # print('self._get_loss_fn(loss_fn_name)', ' returns ', self._get_loss_fn(loss_fn_name))
                    loss = sum(losses) / len(losses)
                    # print('logspace_mse loss: ', loss.item())
                else:
                    loss = self._aggregate_batch_losses(losses)
                out.append(loss)
            return out
    
    def print_epoch_losses(self, loss_fns, batches, losses=None):
        if losses is None:
            losses = self.evaluate_epoch(loss_fns, batches)
        for (loss_fn_name,loss) in zip(loss_fns, losses):
            print(f'{loss_fn_name} loss: {loss.item()}')
    
    def train(self):
        dataloader = self.dataloader
        mgh_factors = self.mgh_factors
        traced_loss_fns = self.config['traced_losses']
        val_set = self._get_val_set()
        
        traced_losses_data = []
        num_samples = self.config['num_samples']
        batch_size = self.config['batch_size']
        num_epochs = self.config['num_epochs']
        set_size = self.config['set_size']
        num_sets = num_samples // set_size
        num_batches_per_set = set_size // batch_size
        if self.dataloader.sample_generator.sampling_scheme == 'all':
            print("Overwriting batch size and num sets for full data batches...")
            set_size = self.message_size
            batch_size = self.message_size
            num_samples = self.message_size
            num_sets = 1
            num_batches_per_set = 1
        if set_size % batch_size != 0:
            print('Warning: set_size is not a multiple of batch_size. Only using ', batch_size * num_batches_per_set, ' samples per set.')
        if num_samples % set_size != 0:
            print('Warning: num_samples is not a multiple of set_size. Only using ', num_sets * batch_size * num_batches_per_set, ' total samples.')
        
        # initial losses------------
        initialize_loss = True
        #---------------------------
        
        # Main train loop-------------------------------
        for s in range(num_sets):
            set_batches = self.dataloader.load_batches(batch_size, num_batches_per_set, mgh_factors = mgh_factors)
            # print first 3 inputs
            # debug
            # print(set_batches[0]['x'][:3])
            for epoch in range(num_epochs):
                # initial losses------------
                if initialize_loss:
                    initialize_loss = False
                    if val_set is None:
                        all_losses = self.evaluate_epoch(traced_loss_fns, set_batches)
                    else:
                        all_losses = self.evaluate_epoch(traced_loss_fns, val_set)
                    traced_losses_data.append([0] + [loss.item() for loss in all_losses])
                #---------------------------
                loss = self.train_epoch(set_batches)
                print(f"Set {s+1}/{num_sets}, Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
                
                # track different losses-------------------
                if val_set is None:
                    all_losses = self.evaluate_epoch(traced_loss_fns, set_batches)
                else:
                    all_losses = self.evaluate_epoch(traced_loss_fns, val_set)
                num_samples_trained_on = (s * num_epochs + epoch + 1) * set_size
                traced_losses_data.append([num_samples_trained_on] + [loss.item() for loss in all_losses])
                # self.print_epoch_losses(traced_loss_fns, set_batches, losses=all_losses)
                #-------------------------------------------
                
                # see if learning rate should be decreased
                self.scheduler.step(loss)
                # stop training if loss is at minimum
                current_lr = self.optimizer.param_groups[0]['lr']
                if current_lr <= self.config['min_lr']:
                    print('Learning rate is at minimum. Stopping training.')
                    return traced_losses_data
        return traced_losses_data
                    
    def _aggregate_batch_losses(self, losses, is_logspace = True):
        # print('losses are ', losses)
        if is_logspace:
            return torch.logsumexp(torch.tensor(losses), dim=0) - torch.log(torch.tensor(len(losses)))
           
    def train_epoch_depricated(self):
        self.net.train()
        epoch_loss = 0.0
        batch_size = self.config['batch_size']
        for batch_idx, batch in enumerate(self.dataloader):
            inputs = batch['input'].to(self.config['device'])
            targets = batch['target'].to(self.config['device'])
            if 'mg_hat' in batch:
                mg_hat = batch['mg_hat'].to(self.config['device'])
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.net(inputs)
            
            # print first ten predictions and targets
            # if batch_idx == 0:
            #     print("predictions, targets")
            #     for i in range(10):
            #         print(outputs[i].item(), targets[i].item())
            
            # Compute loss
            if 'mg_hat' in batch:
                loss = self.loss_fn(outputs.squeeze(), targets, mg_hat)
            else:
                loss = self.loss_fn(outputs.squeeze(), targets)
                
            # Backward pass and optimize
            loss.backward()
            
            # print('first 5 params', self.net.network[0].weight[0][:5])
            # print('pred of x[713] , pred = ', outputs[713].item())
            
            # self.visualize_first_layer()
            
            self.optimizer.step()
            
            # Accumulate batch loss
            epoch_loss += loss.item()
        
        return epoch_loss / len(self.dataloader)
         
    def train_depricated(self):
        """Main training loop"""
        best_val_loss = float('inf')
        early_stopping_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train one epoch
            train_loss = self.train_epoch()
            
            # Validate
            # val_loss = self.validate()
            
            # Print epoch results
            print(f'Train Loss: {train_loss:.6f}')
            # print(f'Val Loss: {val_loss:.6f}')
            
            # # Save checkpoint if best model
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     early_stopping_counter = 0
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': self.net.state_dict(),
            #         'optimizer_state_dict': self.optimizer.state_dict(),
            #         'train_loss': train_loss,
            #         'val_loss': val_loss,
            #     }, self.config['training']['checkpoint_path'])
            #     print('Saved new best model checkpoint')
            # else:
            #     early_stopping_counter += 1
            
            # # Early stopping
            # if early_stopping_counter >= self.config['training']['patience']:
            #     print(f'Early stopping triggered after {epoch+1} epochs')
            #     break
            
    def validate(self):
        pass
    
    def visualize_first_layer(self):
        # Assuming the first layer is a Conv2D layer or Linear layer
        first_layer = list(self.net.children())[0][0]  # Get the first layer

        # Plot weights (assuming Conv2D, modify if Linear)
        weights = first_layer.weight.data
        plot_weights_as_grid(weights, title="First Layer Weights")

        # # After the backward pass, gradients will be available
        # if first_layer.weight.grad is not None:
        #     gradients = first_layer.weight.grad
        #     plot_gradients_as_grid(gradients, title="First Layer Gradients")
        # else:
        #     print("No gradients available yet. Perform a backward pass first.")