_flag0 = False

from .losses import *
from collections import deque
from nce.data import *
from nce.sampling import *
from nce.data.data_loader import shuffle_batches
# from NCE.inference.graphical_model import *
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
# from torchviz import make_dot
import matplotlib.pyplot as plt
import sys
from tqdm.notebook import tqdm

def should_use_convex_early_stopping(config):
    """
    Check if convex early stopping should be activated based on config.
    
    Args:
        config: The nn_config dictionary
        
    Returns:
        bool: True if should use convex early stopping
    """
    return (config.get('convex_early_stopping', False) and 
            (config.get('loss_fn') == 'logspace_mse_fdb' or config.get('loss_fn') == 'linspace_mse_fdb' or config.get('loss_fn') == 'weighted_logspace_mse') and
            config.get('hidden_sizes', []) == [])

class Trainer:
    def __init__(self, net, bucket, loss_fn=None, stats=None):
        self.config = bucket.gm.config
        self.bucket = bucket
        self.stats=stats
        for factor in bucket.factors:
            assert not(factor.tensor is None and not factor.is_nn), f"{bucket.label}"
        self.lower_dim = self.config['lower_dim']
        self.net = net
        self.sample_generator, self.data_preprocessor, self.dataloader = self._make_dataloader()
        self.message_size = self.dataloader.message_size
        self.debug = self.config['debug']
        self.tracked = {'parameters': [], 'gradients': []}
        # self.mgh_factors = [self._get_mgh()] # TODO: will need to grab list of factors in the future
        
        # Set optimizer
        if net is not None:
            self.set_optimizer(self.config['optimizer'])
        # if self.config['optimizer'] == 'adam' or self.config['optimizer'] == 'Adam':
        #     self.optimizer = torch.optim.Adam(
        #         self.net.parameters(), 
        #         lr=self.config['lr']
        #     )
        # else:
        #     self.optimizer = torch.optim.SGD(
        #         self.net.parameters(), 
        #         lr=self.config['lr'],
        #         momentum = self.config['momentum']
        #     )
        # define scheduler
        # Inverse time decay schedule
        def inverse_time_decay(set):
            return self.config['inverse_time_decay_constant'] / (self.config['inverse_time_decay_constant'] + set)

        # LR scheduler using LambdaLR
        if self.config.get('optimizer') == 'muon':
            self.use_scheduler = True
        else:
            self.use_scheduler = False
        if net is not None and self.use_scheduler:
            # self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=inverse_time_decay)
            if isinstance(self.optimizer, list):
                self.muon_scheduler = torch.optim.lr_scheduler.StepLR(self.muon_optimizer, step_size=1000)
                self.adamw_scheduler = torch.optim.lr_scheduler.StepLR(self.adamw_optimizer, step_size=1000)
                self.scheduler = [self.muon_scheduler, self.adamw_scheduler]
            else:
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000)
        if False:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min',           # Minimize loss
                factor=self.config['lr_decay'],           # Reduce LR by this factor
                patience=self.config['patience'],          # Number of epochs with no improvement before reducing LR
                verbose=False,         # Print updates
                min_lr=self.config['min_lr']          # Minimum learning rate
            )
        
        if loss_fn is None:
            self.loss_fn = self._get_loss_fn(self.config['loss_fn'])
        else:
            self.loss_fn = loss_fn
        self.normalizer = torch.tensor(0.0, device=self.config['device'])
         
    def train(self, new_loss_fn=None, override_epochs=0, override_optimizer=None):
        
        original_optimizer = None
        # if override_optimizer is not None:
        #     # Store current optimizer state
        #     original_optimizer = {
        #         'optimizer': self.optimizer,
        #         'optimizer_name': self.config.get('optimizer', 'adam')
        #     }
            
        #     # Set new optimizer
        #     print(f"Overriding optimizer to: {override_optimizer}")
        #     self.set_optimizer(override_optimizer)
        # debug
        # print("Using optimizer: ", self.optimizer)
            
        if new_loss_fn is not None:
            self.loss_fn = self._get_loss_fn(new_loss_fn)
            old_loss_fn_name = self.config['loss_fn']
            self.config['loss_fn'] = new_loss_fn
        if self.loss_fn == linspace_mse_fdb:
            self._initialize_linspace_model_fdb()
        
        early_stopper = None
        if should_use_convex_early_stopping(self.config):
            early_stopper = SimpleConvexEarlyStopping(
                patience=self.config.get('convex_patience', 20),
                min_delta=self.config.get('convex_min_delta', 1e-8),
                verbose=self.config.get('debug', True)
            )
            if self.config.get('debug', True):
                print("Convex early stopping activated for linear logspace_mse_fdb")
        
        dataloader = self.dataloader
        traced_loss_fns = self.config['traced_losses']
        val_set = self._get_val_set()
        
        traced_losses_data = []
        num_samples = self.config['num_samples']
        batch_size = self.config['batch_size']
        if override_epochs > 0:
            num_epochs = override_epochs
        else:
            num_epochs = self.config['num_epochs']
        set_size = self.config['set_size']
        num_sets = num_samples // set_size
        num_batches_per_set = set_size // batch_size
        if self.dataloader.sample_generator.sampling_scheme == 'all':
            print("Overwriting batch size and num sets for full data batches...")
            set_size = self.message_size
            num_samples = self.message_size
            num_sets = 1
            num_batches_per_set = self.config['num_batches_per_set']
            batch_size = self.message_size // num_batches_per_set
        if set_size % batch_size != 0:
            print('Warning: set_size is not a multiple of batch_size. Only using ', batch_size * num_batches_per_set, ' samples per set.')
        if num_samples % set_size != 0:
            print('Warning: num_samples is not a multiple of set_size. Only using ', num_sets * batch_size * num_batches_per_set, ' total samples.')
        
        # initial losses------------
        initialize_loss = True
        #---------------------------
        
        # Try AMP and currently NOT cosine annealing
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)   # set False if not using AMP
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        # Main train loop-------------------------------
        num_progress_steps = num_sets * num_epochs
        with tqdm(total=num_progress_steps, desc="Bucket "+str(self.bucket.label) + " training") as pbar:
            for s in range(num_sets):
                set_batches = self.dataloader.load_batches(batch_size, num_batches_per_set)
                
                def debug_uniformity_histogram():
                    print('debugging uniformity')
                    data = {}
                    for batch in set_batches:
                        for i in range(len(batch['x'])):
                            if batch['x'][i] in data:
                                data[batch['x'][i]] += 1
                            else:
                                data[batch['x'][i]] = 1
                            
                # print first 3 inputs
                # debug
                # print(set_batches[0]['x'][:3])
                for epoch in range(num_epochs):
                    #debug
                    # if epoch > 0:
                    #     set_batches = shuffle_batches(set_batches)
                    #debug-------------------------------------
                    # set_batches_debug = set_batches[:-1]
                    # initial losses------------
                    if initialize_loss:
                        initialize_loss = False
                        if val_set is None:
                            all_losses = self.evaluate_epoch(traced_loss_fns, set_batches)
                        else:
                            all_losses = self.evaluate_epoch(traced_loss_fns, val_set)
                        traced_losses_data.append([0] + [loss.item() for loss in all_losses])
                    
                    
                    #debug---------------------------
                    loss = self.train_epoch(set_batches)
                    
                    if early_stopper is not None:
                        epoch_number = s * num_epochs + epoch
                        if early_stopper(loss, epoch_number):
                            if self.config.get('debug', True):
                                print(f'Convex early stopping triggered at epoch {epoch_number}')
                            self.bucket.gm.traced_losses_data.append((self.bucket.label, traced_losses_data))
                            return traced_losses_data
                    
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
                    # self.scheduler.step(loss)
                    # if use_scheduler:
                    #     self.scheduler.step(s*num_epochs+epoch)
                    # stop training if loss is at minimum
                    # current_lr = self.optimizer.param_groups[0]['lr']
                    postfix = {
                        "Loss": f"{loss.item():.5f}",
                        #"LR": f"{current_lr:.5f}",
                    }
                    pbar.set_postfix(postfix)
                    pbar.update(1)
                    # if current_lr <= self.config['min_lr'] * 10:
                    #     print('Learning rate is at minimum. Stopping training.')
                    #     self.bucket.gm.traced_losses_data.append((self.bucket.label, traced_losses_data))
                    #     return traced_losses_data
        self.bucket.gm.traced_losses_data.append((self.bucket.label, traced_losses_data))
        # for retraining after one loss, reverts config back
        if new_loss_fn is not None:
            self.config['loss_fn'] = old_loss_fn_name
            self.loss_fn = self._get_loss_fn(old_loss_fn_name)
            
        return traced_losses_data
              
    def train_batch(self, x_batch, y_batch, mg_hat_batch=None):
        self.net.train()
        # Zero the parameter gradients
        if isinstance(self.optimizer, list):
            for opt in self.optimizer:
                opt.zero_grad()
        else:
            self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.net(x_batch)
        
        # Compute loss
        # mg_hat_batch = mg_hat_batch.detach()
        # weights = self._get_weights(mg_hat_batch)
        #debug for squared loss
        # if self.loss_fn == w_gil1c or self.loss_fn == gil1c_linear:
        #     loss = self.loss_fn(outputs.squeeze(), y_batch, mg_hat_batch, self.normalizer)
        # else:
        
        
        # if mg_hat_batch is not None or self.loss_fn == logspace_mse:
        #     mg_hat_batch.detach()
        #     loss = self.loss_fn(outputs.squeeze(), y_batch, mg_hat_batch)
        # else:
        #     # print(outputs.shape,y_batch.shape)
        #     loss = self.loss_fn(outputs.squeeze(), y_batch)
            
        # Backward pass and optimize
        
        # if self.debug:
        #     self.tracked['parameters'].append([p.data.clone() for p in self.net.parameters()])
        # debug-------------------------
        # print(f"Loss grad_fn before backward: {loss.grad_fn}")
        # for name, param in self.net.named_parameters():
        #     print(f"Param {name} requires_grad: {param.requires_grad}, grad_fn: {param.grad_fn}")
        #     print(f"Loss memory address: {id(loss)}")
        # print(f"Loss requires_grad: {loss.requires_grad}")
        # print(f"Loss grad_fn: {loss.grad_fn}")

        # debug-------------------------
        # with torch.autograd.detect_anomaly():
        # loss.backward(retain_graph=True)

        # try with scaler-----------------------------------------
        
        if self.config.get('optimizer') == "muon":
            with torch.cuda.amp.autocast(enabled=True):
                loss = self.loss_fn(outputs.reshape(-1), y_batch, mg_hat_batch)
                self.scaler.scale(loss).backward()
        else:
            loss = self.loss_fn(outputs.reshape(-1), y_batch, mg_hat_batch)
            loss.backward()

        # loss.backward(retain_graph=False)
        #-----------------------------------------------------------
        if self.debug:
            self.tracked['gradients'].append([p.grad.clone() for p in self.net.parameters()])
        
        # print("Batch loss: ", loss.item(), " grad sum: ", self.net.get_sum_grad().item())
        
        # if self.debug:
        #     for name, param in self.net.named_parameters():
        #         if param.grad is not None:
        #             print(f"Before Step - {name}: Grad: {param.grad}, Param: {param.data}")
        #     self.optimizer.step()
        #     for name, param in self.net.named_parameters():
        #         print(f"After Step - {name}: Param: {param.data}")
        if True:
            max_norm = 1  # maximum allowed norm of gradients
            # debug
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm)
            # don't need optimizer step with the scaler
            # self.optimizer.step()
            
            # step with scaler
            if self.config.get('optimizer') == 'muon':
                if isinstance(self.optimizer, list):
                    for opt in self.optimizer:
                        self.scaler.step(opt)
                    self.scaler.update()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                if isinstance(self.scheduler, list):
                    for sched in self.scheduler:
                        sched.step()  # or create separate schedulers
                else:
                    self.scheduler.step()
            else:
                self.optimizer.step()
            loss_copy = loss.cpu().item()
            del x_batch, y_batch, mg_hat_batch, loss
            torch.cuda.empty_cache()
        return loss_copy
    
    def train_epoch(self, batches):
        losses = []
        for batch in batches:
            x_batch, y_batch, mgh_batch = batch['x'], batch['y'], batch['mgh']
            # print('first ten mg_hat values: ', mgh_batch[:10])f
            losses.append(self.train_batch(x_batch, y_batch, mgh_batch))
            # Free references to batch tensors
            del x_batch, y_batch, mgh_batch
            torch.cuda.empty_cache()  # If running on GPU
            
        if self.loss_fn == logspace_mse:
                loss = sum(losses) / len(losses)
                loss = torch.Tensor([loss]).to(self.config['device'])
        else:
            if self.debug:
                print('losses are ', losses)
            # debug, uncomment below after debugging
            loss = self._aggregate_batch_losses(losses)
            if self.debug:
                print('loss is ', loss)
        return loss
    
    def _make_dataloader(self):
        sg = SampleGenerator(gm=self.bucket.gm, bucket=self.bucket, random_seed=self.config['seed'])
        sample_assignments = sg.sample_assignments(1000)
        sample_values = sg.compute_message_values(sample_assignments)
        sample_mg_values = sg.compute_gradient_values(sample_assignments) 
        fdb_setting = self.config.get('fdb', False)
        data_preprocessor = DataPreprocessor(
            y=sample_values, 
            mg=sample_mg_values, 
            is_logspace=True, 
            lower_dim=self.lower_dim, 
            device=self.config['device'],
            fdb=fdb_setting  # Pass the fdb setting to DataPreprocessor
        )
        return sg, data_preprocessor, DataLoader(self.bucket, sample_generator=sg, data_preprocessor=data_preprocessor)
    
    # def _get_mgh_factors(self):
    #     return self.bucket.approximate_downstream_factors
    
    # def _get_mgh(self):
        from inference import FastGM
        fastgm_copy = FastGM(uai_file=self.bucket.gm.uai_file, device=self.config['device'], nn_config = self.config)
        mg_hat = fastgm_copy.get_wmb_message_gradient(bucket_var=self.bucket.label, i_bound=self.config['iB_backwards'], weights='max')
        return mg_hat

    def _get_val_set(self):
        if self.config['val_set'] is None:
            return None
        elif self.config['val_set'] == 'all':
            return self.dataloader.load_all()          
   
    def _get_loss_fn(self, loss_fn_name):
        if "approx_smg" in loss_fn_name:
            f = self.bucket.compute_message_exact()
            sigma_f = f.tensor.std(unbiased=False)
            sigma_g = self.config['sigma_g_global']
            rho = self.config['rho_global']
            num_bw_samples = int(loss_fn_name.split(',')[-1]) if ',' in loss_fn_name else -1
            if num_bw_samples < 0:
                raise ValueError("Must specify number of backward samples")
            else:
                seed = 0 if self.config.get('approximation_method', '') == 'dt' else None
                return lambda out, targ, mgh=None: mg_sampled_loss_fdb(out, targ, mgh, sigma_f=sigma_f, sigma_g=sigma_g, rho=rho, num_bw_samples=num_bw_samples, seed=seed)

        elif "mg_sampled_loss_fdb_recompute" in loss_fn_name or "mg_sampled_loss_loo_fdb" in loss_fn_name:
            try:
                sigma_f, sigma_g, rho = self.bucket.get_fw_bw_stats()
            except Exception as e:
                print(f"Error occurred while getting forward/backward stats for {self.bucket.label}: {e}")
                sigma_f, sigma_g, rho = None, None, None
            num_bw_samples = int(loss_fn_name.split(',')[-1]) if ',' in loss_fn_name else -1
            # begin debug
            # print("Using rho: ", rho, " sigma_f: ", sigma_f, " sigma_g: ", sigma_g, " num_bw_samples: ", num_bw_samples)
            # end debug
            if "mg_sampled_loss_fdb_recompute" in loss_fn_name:
                loss = mg_sampled_loss_fdb
            elif "mg_sampled_loss_loo_fdb" in loss_fn_name:
                print('depricated loss')
                loss = mg_sampled_loss_loo_fdb
            else:
                loss = None
                raise ValueError(f"Loss function {loss_fn_name} not recognized")
            seed = 0 if self.config.get('approximation_method', '') == 'dt' else None
            if num_bw_samples < 0:
                print("------Using default num_bw_samples------")
                return lambda out, targ, mgh=None: loss(out, targ, mgh, sigma_f=sigma_f, sigma_g=sigma_g, rho=rho, seed=seed)
            else:
                return lambda out, targ, mgh=None: loss(out, targ, mgh, sigma_f=sigma_f, sigma_g=sigma_g, rho=rho, num_bw_samples=num_bw_samples, seed=seed)
            # return mg_sampled_loss_fdb
        elif loss_fn_name == "mg_sampled_loss_fdb":
            print('depricated loss')
            sigma_f = self.stats[self.bucket.label]['output_message_std']
            sigma_g = self.stats[self.bucket.label]['mg_std']
            rho = self.stats[self.bucket.label]['correlation']
            return lambda out, targ, mgh=None: mg_sampled_loss_fdb(out, targ, mgh, sigma_f=sigma_f, sigma_g=sigma_g, rho=rho)
            # return mg_sampled_loss_fdb     
        elif loss_fn_name == "logspace_mse_fdb":
            return logspace_mse_fdb
        elif loss_fn_name == "linspace_mse_fdb":
            return linspace_mse_fdb
        elif loss_fn_name == "unnormalized_kl":
            return unnormalized_kl
        elif "power_exponential" in loss_fn_name:
            alpha = float(loss_fn_name.split(',')[-1])
            return lambda outputs, targets, mg_hat=None: power_exponential(outputs, targets, mg_hat, alpha=alpha)
        elif loss_fn_name == "mse" or loss_fn_name == "MSE":
            return nn.MSELoss()
        elif loss_fn_name == "gil1":
            return gil1
        elif loss_fn_name == "gil1c":
            return gil1c
        elif loss_fn_name == "w_gil1c":
            return w_gil1c
        elif loss_fn_name == "gil1c_linear":
            return gil1c_linear
        elif loss_fn_name == "logspace_mse":
            return logspace_mse
        elif loss_fn_name == "l1":
            return l1
        elif loss_fn_name == "gil2":
            return gil2
        elif loss_fn_name == "gil2c":
            return gil2c
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
        
    def _aggregate_batch_losses(self, losses, is_logspace = True):
        # print('losses are ', losses)
        losses = torch.tensor(losses, dtype=torch.float32, device=self.config['device']).detach()
        if is_logspace:
            return torch.logsumexp(losses, dim=0) - torch.log(torch.tensor(len(losses)))
        
    def set_optimizer(self, name):
        if name == 'adam' or name == 'Adam':
            self.optimizer = optim.Adam(
                self.net.parameters(), 
                lr=self.config['lr']
            )
        elif name == 'muon':
            from muon import Muon
            
            # Separate parameters for Muon and AdamW
            muon_params = [p for p in self.net.parameters() if p.ndim >= 2]
            adamw_params = [p for p in self.net.parameters() if p.ndim < 2]
            
            # Create separate optimizers
            self.muon_optimizer = Muon(muon_params, lr=0.02, momentum=0.95)
            self.adamw_optimizer = optim.AdamW(adamw_params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
            
            # Store both optimizers (need to step both)
            self.optimizer = [self.muon_optimizer, self.adamw_optimizer]
        elif name == 'sgd' or name == 'SGD':
            self.optimizer = optim.SGD(
                self.net.parameters(), 
                lr=self.config['lr'],
                momentum = self.config['momentum']
            )
           
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
            
            if isinstance(self.optimizer, list):
                for opt in self.optimizer:
                    opt.step()
            else:
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
    
    def graph_vs_exact(approx, exact):
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
        
    def _initialize_linspace_model_fdb(self):
        """
        Initialize the model by adjusting the final network bias term to match partition functions.
        
        This method loads every input, compares the partition function of the predictions 
        and targets by logsumexp'ing both, then adjusts the final bias term of the network
        to make the partition functions match.
        
        Simple approach: bias_adjustment = log_Z_targets - log_Z_predictions
        Then add this difference to the regular bias term of the final layer.
        """
        print(f"Initializing model with partition function matching for bucket {self.bucket.label}")
        
        # Set model to evaluation mode
        self.net.eval()
        
        # Collect all predictions and targets
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            # Load all data from the dataloader
            all_data = self.dataloader.load_all()
            
            for batch in all_data:
                inputs = batch['x'].to(self.config['device'])
                targets = batch['y'].to(self.config['device'])
                
                # Get current predictions from the network
                predictions = self.net(inputs).squeeze()
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        print(f"Loaded {len(all_predictions)} samples for partition function matching")
        
        # Compute log partition functions of the messages
        log_Z_targets = torch.logsumexp(all_targets, dim=0)
        log_Z_predictions = torch.logsumexp(all_predictions, dim=0)
        
        # Calculate the bias adjustment needed
        bias_adjustment = log_Z_targets - log_Z_predictions
        
        print(f"Message partition function analysis:")
        print(f"  log Z (exact message/targets): {log_Z_targets.item():.6f}")
        print(f"  log Z (approximate message/predictions): {log_Z_predictions.item():.6f}")
        print(f"  Partition function error: {bias_adjustment.item():.6f}")
        
        # Find and adjust the final layer bias term
        with torch.no_grad():
            # Look for the final linear layer in the network
            final_layer = None
            for layer in reversed(list(self.net.modules())):
                if isinstance(layer, torch.nn.Linear):
                    final_layer = layer
                    break
            
            if final_layer is not None and final_layer.bias is not None:
                # Add the bias adjustment to the existing bias
                final_layer.bias.data += bias_adjustment.item()
                print(f"  Adjusted final layer bias by: {bias_adjustment.item():.6f}")
                print(f"  New final layer bias: {final_layer.bias.data.item():.6f}")
            else:
                print("  Warning: Could not find final layer with bias term to adjust")
                return bias_adjustment.item()
        
        # Verify the adjustment worked
        with torch.no_grad():
            # Get new predictions with the adjusted bias
            new_predictions = []
            for batch in all_data:
                inputs = batch['x'].to(self.config['device'])
                batch_preds = self.net(inputs).squeeze()
                new_predictions.append(batch_preds)
            
            new_predictions = torch.cat(new_predictions, dim=0)
            log_Z_predictions_adjusted = torch.logsumexp(new_predictions, dim=0)
            final_error = abs(log_Z_predictions_adjusted - log_Z_targets).item()
            
            print(f"  log Z (adjusted predictions): {log_Z_predictions_adjusted.item():.6f}")
            print(f"  Final partition function error: {final_error:.8f}")
            
            if final_error < 1e-6:
                print("  ✓ Partition function matching successful!")
            else:
                print(f"  ⚠ Partition function matching incomplete (error: {final_error:.8f})")
        
        # Return the model to training mode
        self.net.train()
        
        return bias_adjustment.item()
        
class SimpleConvexEarlyStopping:
    """
    Simple early stopping for convex optimization.
    Just checks if the loss has converged (stopped improving).
    """
    
    def __init__(self, patience=20, min_delta=1e-8, min_loss=1e-7, verbose=True):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            verbose: Whether to print convergence messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_loss = min_loss
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.wait = 0
        self.loss_history = deque(maxlen=10)
        
    def __call__(self, loss, epoch):
        """
        Check if training should stop.
        
        Args:
            loss: Current training loss
            epoch: Current epoch number
            
        Returns:
            bool: True if should stop, False otherwise
        """
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
            
        self.loss_history.append(loss)

        if loss < self.min_loss:
            if self.verbose:
                print(f"Epoch {epoch}: Loss {loss:.8f} is below minimum loss threshold {self.min_loss:.8f}. Stopping training.")
                print(f"Final loss: {loss:.8f}, Best loss: {self.best_loss:.8f}")
            return True

        # Check for improvement
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
            if self.verbose:
                print(f"Epoch {epoch}: New best loss: {loss:.8f}")
        else:
            self.wait += 1
            
        # Check for convergence
        if self.wait >= self.patience:
            if self.verbose:
                print(f"Epoch {epoch}: Convergence detected after {self.patience} epochs without improvement")
                print(f"Final loss: {loss:.8f}, Best loss: {self.best_loss:.8f}")
            return True
            
        return False

