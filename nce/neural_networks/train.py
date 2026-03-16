from .losses import *
from collections import deque
from nce.data import *
from nce.sampling import *
from nce.training_logger import log_epoch_loss, log_val_loss, log_early_stopping
# from NCE.inference.graphical_model import *
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
# from torchviz import make_dot
import matplotlib.pyplot as plt
import sys
from tqdm.notebook import tqdm

def get_error_tracking_epochs(num_epochs):
    """
    Generate sorted, deduplicated list of checkpoint epochs for error tracking.

    Fixed base list: [0, 1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    then every 5000 up to num_epochs, plus the final epoch.

    Args:
        num_epochs: Total number of training epochs.

    Returns:
        Sorted list of unique checkpoint epoch numbers.
    """
    base = [0, 1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    epochs = [e for e in base if e <= num_epochs]
    # Add every 5000 after 10000
    e = 15000
    while e <= num_epochs:
        epochs.append(e)
        e += 5000
    # Add final epoch if not already present
    if num_epochs not in epochs:
        epochs.append(num_epochs)
    return sorted(set(epochs))


def should_use_convex_early_stopping(config):
    """
    Check if convex early stopping should be activated based on config.

    Args:
        config: The nn_config dictionary

    Returns:
        bool: True if should use convex early stopping
    """
    loss_fn = config.get('loss_fn', '')
    # Check if loss function supports early stopping
    supported_losses = (
        loss_fn == 'logspace_mse_fdb' or
        loss_fn == 'linspace_mse_fdb' or
        loss_fn == 'weighted_logspace_mse' or
        loss_fn == 'ukf_sequential' or
        'ukf_seq' in loss_fn
    )

    hidden_sizes = config.get('hidden_sizes', [])
    is_linear = (hidden_sizes == [] or hidden_sizes == 'bias_only')
    return (config.get('convex_early_stopping', False) and
            supported_losses and
            is_linear)

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
        self.losses = []
        self.val_losses = []
        # self.bw_factors = [self._get_mgh()] # TODO: will need to grab list of factors in the future

        # For elp_least_squares_v2: track expansion point
        self.expansion_point = None
        self.dL_moving_avg = None
        self.d2L_moving_avg = None
        self.momentum = 0.9  # For moving average of derivatives
        
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
        # NEW: Support config-based LR scheduling for any optimizer
        lr_schedule_type = self.config.get('lr_schedule', 'none')
        if self.config.get('optimizer') == 'muon':
            self.use_scheduler = True
        elif lr_schedule_type != 'none':
            self.use_scheduler = True
        else:
            self.use_scheduler = False

        if net is not None and self.use_scheduler:
            num_epochs = self.config.get('num_epochs', 10000)

            if lr_schedule_type == 'cosine':
                # Cosine annealing - smooth decay to eta_min
                eta_min = self.config.get('lr_schedule_eta_min', 1e-6)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=num_epochs, eta_min=eta_min
                )
            elif lr_schedule_type == 'onecycle':
                # OneCycle - warmup then decay, good for escaping local minima
                max_lr = self.config.get('lr_schedule_max_lr', self.config['lr'] * 10)
                pct_start = self.config.get('lr_schedule_pct_start', 0.1)
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer, max_lr=max_lr, total_steps=num_epochs,
                    pct_start=pct_start, anneal_strategy='cos'
                )
            elif isinstance(self.optimizer, list):
                # Muon with multiple optimizers
                self.muon_scheduler = torch.optim.lr_scheduler.StepLR(self.muon_optimizer, step_size=1000)
                self.adamw_scheduler = torch.optim.lr_scheduler.StepLR(self.adamw_optimizer, step_size=1000)
                self.scheduler = [self.muon_scheduler, self.adamw_scheduler]
            else:
                # Default: StepLR
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000)
        if loss_fn is None:
            self.loss_fn = self._get_loss_fn(self.config['loss_fn'])
        else:
            self.loss_fn = loss_fn
        self.normalizer = torch.tensor(0.0, device=self.config['device'])
         
    def train(self, new_loss_fn=None, override_epochs=-1, override_optimizer=None):
        from nce.inference.factor_nn import FactorNN
        from nce.utils.plots import plot_fastfactor_comparison


        original_optimizer = None
        # if override_optimizer is not None:
        #     # Store current optimizer state
        #     original_optimizer = {
        #         'optimizer': self.optimizer,
        #         'optimizer_name': self.config.get('optimizer', 'adam')
        #     }
            
        #     # Set new optimizer
        #     self.set_optimizer(override_optimizer)

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
                print(f"Convex early stopping activated for linear {self.config.get('loss_fn')}")
        
        dataloader = self.dataloader
        traced_loss_fns = self.config['traced_losses']
        val_set = None

        use_nbe_early_stopping = self.config.get('nbe_early_stopping', False)
        nbe_val_set = None
        nbe_val_losses = []
        nbe_warmup_epochs = 5  # Don't check early stopping until after this many epochs

        # NeuroBE patience-based early stopping (distinct from NBE early stopping)
        use_neurobe_early_stopping = self.config.get('neurobe_early_stopping', False)
        neurobe_stop_iter = self.config.get('neurobe_stop_iter', 2)
        neurobe_patience_count = 0
        neurobe_prev_best = float('inf')

        # IMPORTANT: Initialize normalizing constant from TRAINING data, not validation data
        # This ensures that max(y + bw) from training is used to prevent overflow in UKL loss
        # We load a small training sample first just to trigger the normalization computation
        if self.dataloader.sample_generator.sampling_scheme == 'all':
            init_batches = self.dataloader.load_all()
        else:
            batch_size = self.config['batch_size']
            init_set_size = self.config.get('set_size') or self.config['num_samples']
            num_batches_per_set = init_set_size // batch_size
            stratify = self.config.get('stratify_samples', False)
            init_batches = self.dataloader.load_batches(batch_size, num_batches_per_set, stratify_samples=stratify)
        if self.data_preprocessor.normalizing_constant is not None:
            print(f"Initialized normalizing constant from training data: {self.data_preprocessor.normalizing_constant:.4f}")
        elif self.data_preprocessor.normalization_mode == 'minmax_01':
            print(f"Initialized minmax_01 normalization: ln_min={self.data_preprocessor.ln_min:.4f}, ln_max={self.data_preprocessor.ln_max:.4f}")
        if self.data_preprocessor.bw_normalizing_constant is not None:
            print(f"  bw_normalizing_constant (bw at argmax(y+bw)): {self.data_preprocessor.bw_normalizing_constant:.4f}")

        # Compute global_max_targets for UKL numerical stability
        # CRITICAL: This must be computed ONCE from all training data and used for ALL batches
        # Using per-batch max causes gradient inconsistency and training divergence
        if self.config.get('loss_fn', '') == 'unnormalized_kl':
            with torch.no_grad():
                all_y = torch.cat([b['y'] for b in init_batches], dim=0)
                all_bw = torch.cat([b['bw'] for b in init_batches], dim=0) if init_batches[0].get('bw') is not None else None
                if all_bw is not None and self.data_preprocessor.bw_normalizing_constant is not None:
                    # targets = y + (bw - bw_normalizing_constant)
                    targets_for_max = all_y + all_bw - self.data_preprocessor.bw_normalizing_constant
                else:
                    targets_for_max = all_y
                self.data_preprocessor.global_max_targets = targets_for_max.max().item()
                print(f"  global_max_targets (for UKL numerical stability): {self.data_preprocessor.global_max_targets:.4f}")

        # Now generate validation set (which will use the same normalizing constant)
        if self.dataloader.sample_generator.sampling_scheme == 'all':
            nbe_val_set = self.dataloader.load_all()
        else:
            nbe_val_size = max(1, self.config['num_samples'] // 9)
            nbe_val_set = self._generate_validation_set_nbe(nbe_val_size)
        self.nbe_val_set = nbe_val_set  # Store for later use (plotting, etc.)

        if use_nbe_early_stopping:
            nbe_warmup_epochs = self.config.get('nbe_warmup_epochs', nbe_warmup_epochs)
            print(f"NBE early stopping enabled: validation set size = {len(nbe_val_set[0]['x'])}, warmup = {nbe_warmup_epochs} epochs")
        else:
            # Validation set was generated but won't be used for early stopping
            print(f"Validation set generated: {len(nbe_val_set[0]['x'])} samples")

        # For mini-batch learning, create validation set for early stopping
        use_validation_early_stopping = False
        if self.config.get('use_validation_early_stopping', False):
            use_validation_early_stopping = True
            val_set = self._generate_validation_set()
            print(f"Validation-based early stopping enabled: checking every 10 epochs on {len(val_set[0]['x'])} samples")

        traced_losses_data = []
        num_samples = self.config['num_samples']
        batch_size = self.config['batch_size']
        if override_epochs >= 0:
            num_epochs = override_epochs
        else:
            num_epochs = self.config['num_epochs']
        set_size = self.config.get('set_size') or num_samples

        # Check for full data batch mode BEFORE using set_size
        if self.dataloader.sample_generator.sampling_scheme == 'all':
            set_size = int(self.message_size)
            num_samples = int(self.message_size)
            num_sets = 1
            # Support batch_size='all' to use entire message as one batch
            if self.config['batch_size'] == 'all':
                batch_size = int(self.message_size)
            else:
                batch_size = self.config['batch_size']
            num_batches_per_set = (set_size + batch_size - 1) // batch_size
        else:
            # If num_samples < set_size (e.g. NBE adaptive sampling gives fewer samples than the
            # default set_size of 50000), clamp set_size to num_samples so we get at least 1 set.
            if num_samples < set_size:
                print(f'Note: num_samples ({num_samples}) < set_size ({set_size}). Clamping set_size to num_samples.')
                set_size = num_samples
            num_sets = num_samples // set_size
            num_batches_per_set = set_size // batch_size
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

        # Determine if we need to pass scaling_factor to data loader
        scaling_factor = None
        if self.config['loss_fn'] == 'scaled_mse' or self.config.get('loss_fn2') == 'scaled_mse':
            # Get forward/backward stats for scaling
            try:
                sigma_f, sigma_g, rho = self.bucket.get_fw_bw_stats()
                scaling_factor = (sigma_f ** 2) / (sigma_f ** 2 + sigma_g ** 2 + 1e-12)
            except Exception as e:
                print(f"Warning: Could not get fw/bw stats for scaled_mse: {e}")
                scaling_factor = None

        # Note: scaling_factor logic removed - normalization is now computed once at init

        # Initialize validation-based early stopping tracking
        val_loss_history = []  # Store (epoch, val_loss) tuples for validation-based early stopping

        # Display initial state before any training if display_intermediate is enabled
        if self.config.get('display_intermediate'):
            try:
                from nce.inference.factor_nn import FactorNN
                from nce.utils.plots import plot_fastfactor_comparison, plot_validation_comparison
                logSS = getattr(self.bucket.gm, 'logSS', 0)
                if logSS <= 6:
                    # Full message comparison
                    intermediate_factor = FactorNN(self.net, self.data_preprocessor)
                    intermediate_factor = intermediate_factor.to_exact()
                    exact_message_here = self.bucket.compute_message_exact()
                    plot_fastfactor_comparison(exact_message_here, intermediate_factor, title=f'Bucket {self.bucket.label} BEFORE training (epoch 0)')
                else:
                    # Validation set comparison
                    val_batch = self.nbe_val_set[0]
                    x_val = val_batch['x']
                    y_val = val_batch['y']
                    with torch.no_grad():
                        y_pred = self.net(x_val).squeeze()
                    plot_validation_comparison(
                        y_val, y_pred,
                        title=f'Bucket {self.bucket.label} BEFORE training (epoch 0) (Validation Set)',
                        show=True, show_loss_curve=False
                    )
            except Exception as e:
                print(f"WARNING: display_intermediate (initial) failed: {e}")

        # --- Error tracking setup ---
        error_tracking = self.config.get('error_tracking', False)
        if error_tracking:
            if self.dataloader.sample_generator.sampling_scheme != 'all':
                raise ValueError("error_tracking requires sampling_scheme='all'")
            from nce.utils.backward_message import get_backward_message
            print(f"[Error Tracking] Computing exact forward and backward messages for bucket {self.bucket.label}...")
            exact_fw = self.bucket.compute_message_exact()
            exact_bw, _ = get_backward_message(
                self.bucket.gm, self.bucket.label,
                iB=100, backward_ecl=2**30,
                approximation_method='wmb',
                return_factor_list=False
            )
            exact_contribution = (exact_fw * exact_bw).sum_all_entries()
            self.error_tracking_data = []
            error_tracking_epochs = set(get_error_tracking_epochs(num_epochs))
            print(f"[Error Tracking] Checkpoints: {sorted(error_tracking_epochs)}")

        with tqdm(total=num_progress_steps, desc="Bucket "+str(self.bucket.label) + " training") as pbar:
            for s in range(num_sets):
                stratify = self.config.get('stratify_samples', False)
                if self.dataloader.sample_generator.sampling_scheme == 'all':
                    all_data = self.dataloader.load_all()[0]
                    x_all, y_all, bw_all = all_data['x'], all_data['y'], all_data['bw']
                    set_batches = []
                    for i in range(0, len(x_all), batch_size):
                        end_idx = min(i + batch_size, len(x_all))
                        set_batches.append({
                            'x': x_all[i:end_idx],
                            'y': y_all[i:end_idx],
                            'bw': bw_all[i:end_idx] if bw_all is not None else None
                        })
                else:
                    set_batches = self.dataloader.load_batches(batch_size, num_batches_per_set, stratify_samples=stratify)

                def debug_uniformity_histogram():
                    print('debugging uniformity')
                    data = {}
                    for batch in set_batches:
                        for i in range(len(batch['x'])):
                            if batch['x'][i] in data:
                                data[batch['x'][i]] += 1
                            else:
                                data[batch['x'][i]] = 1
                            
                loss1000 = 1000000

                # --- Error tracking: epoch 0 (before any training) ---
                if error_tracking and 0 in error_tracking_epochs:
                    with torch.no_grad():
                        initial_loss = self.compute_epoch_loss(set_batches, self.loss_fn).item()
                        approx_factor_0 = FactorNN(self.net, self.data_preprocessor)
                        approx_exact_0 = approx_factor_0.to_exact()
                        approx_contribution_0 = (approx_exact_0 * exact_bw).sum_all_entries()
                        log_z_err_0 = approx_contribution_0 - exact_contribution
                        self.error_tracking_data.append((0, initial_loss, log_z_err_0, abs(log_z_err_0)))
                        print(f"[Error Tracking] Epoch 0: loss={initial_loss:.6e}, log_Z_err={log_z_err_0:.6f}, |log_Z_err|={abs(log_z_err_0):.6f}")

                for epoch in range(num_epochs):
                    if self.config.get('display_intermediate') and (epoch % self.config['display_intermediate'] == 0) and epoch > 0:
                        plot = False # debug
                        try:
                            logSS = getattr(self.bucket.gm, 'logSS', 0)
                            if logSS <= 6:
                                # Full message comparison
                                intermediate_factor = FactorNN(self.net, self.data_preprocessor)
                                intermediate_factor = intermediate_factor.to_exact()
                                exact_message_here = self.bucket.compute_message_exact()
                                plot_fastfactor_comparison(exact_message_here, intermediate_factor, title=f'Bucket {self.bucket.label} Intermediate after {s*num_epochs+epoch} epochs', pred_alpha=0.0)
                            else:
                                # Validation set comparison - use stored validation set
                                from nce.utils.plots import plot_validation_comparison
                                val_batch = self.nbe_val_set[0]
                                x_val = val_batch['x']
                                y_val = val_batch['y']
                                with torch.no_grad():
                                    y_pred = self.net(x_val).squeeze()
                                plot_validation_comparison(
                                    y_val, y_pred,
                                    title=f'Bucket {self.bucket.label} Intermediate after {s*num_epochs+epoch} epochs (Validation Set)',
                                    show=True, show_loss_curve=True,
                                    losses=self.losses, val_losses=self.val_losses
                                )
                        except Exception as e:
                            print(f"WARNING: display_intermediate failed: {e}")

                        # UKF Sequential: Display statistics if using this loss
                        if hasattr(self, 'ukf_Lmu') and self.ukf_Lmu is not None:
                            print(f"\n{'='*70}")
                            print(f"UKF Sequential Statistics at Epoch {s*num_epochs+epoch}")
                            print(f"{'='*70}")

                            # Display Gaussian statistics
                            mu0 = self.ukf_Lmu[0].item()
                            mu1 = self.ukf_Lmu[1].item()
                            print(f"Mean vector (Lmu):")
                            print(f"  E[Φ]  (exact log Z)  : {mu0:.6f}")
                            print(f"  E[Φ̂] (approx log Z) : {mu1:.6f}")

                            print(f"\nCovariance matrix (Lsig):")
                            print(f"  Var[Φ]        : {self.ukf_Lsig[0, 0].item():.6f}")
                            print(f"  Var[Φ̂]       : {self.ukf_Lsig[1, 1].item():.6f}")
                            print(f"  Cov[Φ, Φ̂]    : {self.ukf_Lsig[0, 1].item():.6f}")
                            print(f"  Correlation   : {self.ukf_Lsig[0, 1].item() / (torch.sqrt(self.ukf_Lsig[0, 0] * self.ukf_Lsig[1, 1]).item() + 1e-12):.6f}")

                            # Compute loss decomposition
                            bias_squared = (mu0 - mu1)**2
                            variance_term = (self.ukf_Lsig[0, 0].item() + self.ukf_Lsig[1, 1].item() - 2 * self.ukf_Lsig[0, 1].item())
                            total_loss = bias_squared + variance_term

                            print(f"\nLoss Decomposition:")
                            print(f"  Bias² term            : {bias_squared:.6f}")
                            print(f"  Variance term         : {variance_term:.6f}")
                            print(f"  Total loss (E[(Φ-Φ̂)²]): {total_loss:.6f}")

                            # Compute derived metrics
                            expected_error = abs(mu0 - mu1)
                            rmse = total_loss ** 0.5
                            partition_ratio = expected_error  # exp(expected_error) for actual ratio

                            print(f"\nDerived Metrics:")
                            print(f"  Expected log partition error (|E[Φ] - E[Φ̂]|): {expected_error:.6f}")
                            print(f"  RMSE (√loss)                                 : {rmse:.6f}")
                            print(f"  Partition function ratio (exp({expected_error:.3f}))      : {torch.exp(torch.tensor(expected_error)).item():.3f}x")

                            # Display forward/backward stats
                            print(f"\nForward/Backward Message Statistics:")
                            print(f"  σ_f (forward std)  : {self.ukf_sigma_f:.6f}")
                            print(f"  σ_g (backward std) : {self.ukf_sigma_g:.6f}")
                            print(f"  ρ (correlation)    : {self.ukf_rho:.6f}")

                            print(f"{'='*70}\n")

                        # Z_hat = (intermediate_factor * self.bucket.mg).sum_all_entries()
                        # Z_here = (exact_message_here * self.bucket.mg).sum_all_entries()
                        # Z_original = (self.bucket.exact_message * self.bucket.mg).sum_all_entries()
                    else:
                        plot = False
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
                    
                    
                    loss = self.train_epoch(set_batches, plot=plot, epoch=epoch)
                    global_epoch_num = s * num_epochs + epoch
                    self.losses.append((global_epoch_num, loss.item()))
                    if self.bucket.gm._training_logger:
                        log_epoch_loss(self.bucket.gm._training_logger, self.bucket.label, global_epoch_num, loss.item())

                    # Step scheduler once per epoch (not per batch)
                    if self.use_scheduler and hasattr(self, 'scheduler') and self.scheduler is not None:
                        self.scheduler.step()

                    # --- Error tracking: check if this epoch is a checkpoint ---
                    if error_tracking:
                        epochs_completed = epoch + 1
                        if epochs_completed in error_tracking_epochs:
                            with torch.no_grad():
                                approx_factor_et = FactorNN(self.net, self.data_preprocessor)
                                approx_exact_et = approx_factor_et.to_exact()
                                approx_contribution_et = (approx_exact_et * exact_bw).sum_all_entries()
                                log_z_err_et = approx_contribution_et - exact_contribution
                                current_loss_et = self.losses[-1][1]
                                self.error_tracking_data.append((epochs_completed, current_loss_et, log_z_err_et, abs(log_z_err_et)))
                                print(f"[Error Tracking] Epoch {epochs_completed}: loss={current_loss_et:.6e}, log_Z_err={log_z_err_et:.6f}, |log_Z_err|={abs(log_z_err_et):.6f}")

                    # Validation-based early stopping for mini-batch learning
                    if use_validation_early_stopping and not self.config['skip_early_stopping'] and epoch > 0 and epoch % 10 == 0:
                        # Evaluate validation loss every 10 epochs
                        with torch.no_grad():
                            val_loss = self.compute_epoch_loss(val_set, self.loss_fn)
                        val_loss_value = val_loss.item()
                        val_loss_history.append((epoch, val_loss_value))

                        if epoch % 100 == 0:
                            print(f'Epoch {epoch}: Validation loss = {val_loss_value:.6e}')

                        # Check if validation loss is very low - stop immediately
                        if val_loss_value < 0.0001:
                            print(f'Validation loss {val_loss_value:.6e} is below 0.0001 threshold. Stopping training at epoch {epoch}.')
                            if self.bucket.gm._training_logger:
                                log_early_stopping(self.bucket.gm._training_logger, self.bucket.label, epoch, "val_loss_below_0.0001", val_loss_value)
                            return traced_losses_data

                        # Check for improvement every 100 epochs (10 validation checks)
                        if epoch > 0 and epoch % 100 == 0:
                            # Get validation losses from 100 epochs ago
                            baseline_val_loss = None
                            for e, v in val_loss_history:
                                if e == epoch - 100:
                                    baseline_val_loss = v
                                    break

                            if baseline_val_loss is not None:
                                # Get last 10 validation losses
                                recent_val_losses = [v for e, v in val_loss_history[-10:]]

                                # Check if any recent validation loss improved by >= 1% vs baseline
                                any_improved = any(v <= 0.99 * baseline_val_loss for v in recent_val_losses)

                                if not any_improved:
                                    # No significant improvement detected
                                    if val_loss_value < 0.01:
                                        # Loss is good enough, stop
                                        print(f'Validation loss {val_loss_value:.6e} is below 0.01 threshold. Stopping training at epoch {epoch}.')
                                        if self.bucket.gm._training_logger:
                                            log_early_stopping(self.bucket.gm._training_logger, self.bucket.label, epoch, "val_loss_below_0.01_no_improvement", val_loss_value)
                                        return traced_losses_data
                                    else:
                                        # Loss is still high - check for weak improvement
                                        any_weak_improvement = any(v <= 0.9995 * baseline_val_loss for v in recent_val_losses)

                                        if not any_weak_improvement:
                                            # No improvement at all, even with weak threshold - stop
                                            print(f'Validation loss {val_loss_value:.6e} not decreasing after {epoch} epochs (no weak improvement).')
                                            if self.bucket.gm._training_logger:
                                                log_early_stopping(self.bucket.gm._training_logger, self.bucket.label, epoch, "val_loss_no_weak_improvement", val_loss_value)
                                            return traced_losses_data
                                        else:
                                            print(f'Epoch {epoch}: Validation loss {val_loss_value:.6e} still high but showing weak improvement. Continuing...')

                    # Standard early stopping for non-validation mode
                    elif not use_validation_early_stopping and not self.config['skip_early_stopping'] and epoch > 0:
                        # Check if loss is very low - stop immediately
                        if loss.item() < 0.0001:
                            print(f'Loss {loss.item():.6e} is below 0.0001 threshold. Stopping training at epoch {epoch}.')
                            if self.bucket.gm._training_logger:
                                log_early_stopping(self.bucket.gm._training_logger, self.bucket.label, epoch, "loss_below_0.0001", loss.item())
                            return traced_losses_data

                        # Check every 1000 epochs for improvement
                        if epoch % 1000 == 0:
                            baseline = loss1000                 # loss from 1000 epochs ago
                            recent_vals = [v for _, v in self.losses[-10:]]  # last 10 logged losses
                            current_loss = loss.item()

                            # Did any of the last 10 improve by >=1% vs baseline?
                            any_improved = any(v <= 0.99 * baseline for v in recent_vals)

                            if not any_improved and baseline < float('inf'):
                                # No significant improvement detected
                                if current_loss < 0.01:
                                    # Loss is good enough, stop
                                    print(f'Loss {current_loss:.6e} is below 0.01 threshold. Stopping training at epoch {epoch}.')
                                    if self.bucket.gm._training_logger:
                                        log_early_stopping(self.bucket.gm._training_logger, self.bucket.label, epoch, "loss_below_0.01_no_improvement", current_loss)
                                    return traced_losses_data
                                else:
                                    # Loss is still high - check for weak improvement
                                    # Use weaker threshold (0.05% improvement instead of 1%)
                                    any_weak_improvement = any(v <= 0.9995 * baseline for v in recent_vals)

                                    if not any_weak_improvement:
                                        # No improvement at all, even with weak threshold - stop
                                        print(f'Loss {current_loss:.6e} not decreasing after {epoch} epochs (no weak improvement).')
                                        if self.bucket.gm._training_logger:
                                            log_early_stopping(self.bucket.gm._training_logger, self.bucket.label, epoch, "loss_no_weak_improvement", current_loss)
                                        return traced_losses_data
                                    else:
                                        # Weak improvement detected and loss is high - be patient, continue training
                                        print(f'Epoch {epoch}: Loss {current_loss:.6e} still high but showing weak improvement. Continuing...')

                            # set baseline for the next 1000-epoch window
                            loss1000 = loss.item()
                    if early_stopper is not None:
                        epoch_number = s * num_epochs + epoch
                        if early_stopper(loss, epoch_number):
                            if self.config.get('debug', True):
                                print(f'Convex early stopping triggered at epoch {epoch_number}')
                            self.bucket.gm.traced_losses_data.append((self.bucket.label, traced_losses_data))
                            if self.bucket.gm._training_logger:
                                log_early_stopping(self.bucket.gm._training_logger, self.bucket.label, epoch_number, "convex_early_stopping", loss.item())
                            return traced_losses_data

                    # NBE early stopping check
                    if use_nbe_early_stopping and nbe_val_set is not None:
                        global_epoch = s * num_epochs + epoch

                        # Compute validation loss on entire set (no batching)
                        with torch.no_grad():
                            val_batch = nbe_val_set[0]
                            x_val = val_batch['x']
                            y_val = val_batch['y']
                            # Support both old 'mgh' key and new 'bw' key
                            bw_val = val_batch.get('bw', val_batch.get('mgh'))
                            outputs_val = self.net(x_val).squeeze()
                            if bw_val is not None:
                                nbe_val_loss = self.loss_fn(outputs_val, y_val, bw_val)
                            else:
                                nbe_val_loss = self.loss_fn(outputs_val, y_val)
                            # Handle case where loss returns per-sample values
                            if nbe_val_loss.dim() > 0:
                                nbe_val_loss = nbe_val_loss.mean()
                            nbe_val_loss_value = nbe_val_loss.item()
                        nbe_val_losses.append(nbe_val_loss_value)
                        # Store validation losses on self for access in plotting
                        self.val_losses.append((global_epoch, nbe_val_loss_value))
                        if self.bucket.gm._training_logger:
                            log_val_loss(self.bucket.gm._training_logger, self.bucket.label, global_epoch, nbe_val_loss_value)

                        # Simple early stopping: stop if validation loss increased 3 times in a row
                        if global_epoch >= nbe_warmup_epochs and len(nbe_val_losses) >= 4:
                            # Check if loss increased for 3 consecutive epochs
                            v_curr = nbe_val_losses[-1]
                            v_prev1 = nbe_val_losses[-2]
                            v_prev2 = nbe_val_losses[-3]
                            v_prev3 = nbe_val_losses[-4]

                            # Stop if: v_curr > v_prev1 > v_prev2 > v_prev3 (3 consecutive increases)
                            if v_curr > v_prev1 and v_prev1 > v_prev2 and v_prev2 > v_prev3:
                                print(f'NBE early stopping at epoch {global_epoch}: '
                                      f'val_loss increased 3 times in a row: {v_prev3:.6e} -> {v_prev2:.6e} -> {v_prev1:.6e} -> {v_curr:.6e}')
                                self.bucket.gm.traced_losses_data.append((self.bucket.label, traced_losses_data))
                                if self.bucket.gm._training_logger:
                                    log_early_stopping(self.bucket.gm._training_logger, self.bucket.label, global_epoch, "nbe_3_consecutive_increases", v_curr)
                                return traced_losses_data

                        # OLD two-phase early stopping (temporarily disabled):
                        # Phase 1 (high loss > threshold): Strict - compare running averages over window
                        # Phase 2 (low loss <= threshold): Lenient - 3 consecutive epochs of non-improvement
                        # nbe_plateau_threshold = self.config.get('nbe_plateau_threshold', 0.1)
                        # nbe_plateau_window = self.config.get('nbe_plateau_window', 25)
                        # nbe_plateau_min_improvement = self.config.get('nbe_plateau_min_improvement', 0.01)
                        # if global_epoch >= nbe_warmup_epochs:
                        #     v_curr = nbe_val_losses[-1]
                        #     if v_curr > nbe_plateau_threshold:
                        #         # Phase 1: High loss - use strict plateau detection
                        #         if len(nbe_val_losses) >= 2 * nbe_plateau_window:
                        #             recent_avg = sum(nbe_val_losses[-nbe_plateau_window:]) / nbe_plateau_window
                        #             previous_avg = sum(nbe_val_losses[-2*nbe_plateau_window:-nbe_plateau_window]) / nbe_plateau_window
                        #             improvement = (previous_avg - recent_avg) / (abs(previous_avg) + 1e-10)
                        #             if improvement < nbe_plateau_min_improvement:
                        #                 return traced_losses_data
                        #     else:
                        #         # Phase 2: Low loss - use lenient 3-consecutive-epochs criteria
                        #         if len(nbe_val_losses) >= 4:
                        #             v_prev1 = nbe_val_losses[-2]
                        #             v_prev2 = nbe_val_losses[-3]
                        #             v_base = nbe_val_losses[-4]
                        #             if v_curr > v_base and v_prev1 > v_base and v_prev2 > v_base:
                        #                 return traced_losses_data

                    # NeuroBE patience-based early stopping check
                    if use_neurobe_early_stopping and nbe_val_set is not None:
                        global_epoch_nb = s * num_epochs + epoch

                        # Compute validation loss
                        with torch.no_grad():
                            val_batch_nb = nbe_val_set[0]
                            x_val_nb = val_batch_nb['x']
                            y_val_nb = val_batch_nb['y']
                            bw_val_nb = val_batch_nb.get('bw', val_batch_nb.get('mgh'))
                            outputs_val_nb = self.net(x_val_nb).squeeze()
                            if bw_val_nb is not None:
                                neurobe_val_loss = self.loss_fn(outputs_val_nb, y_val_nb, bw_val_nb)
                            else:
                                neurobe_val_loss = self.loss_fn(outputs_val_nb, y_val_nb)
                            if neurobe_val_loss.dim() > 0:
                                neurobe_val_loss = neurobe_val_loss.mean()
                            neurobe_val_loss_value = neurobe_val_loss.item()

                        # Patience counter: reset on improvement, increment otherwise
                        if neurobe_val_loss_value < neurobe_prev_best:
                            neurobe_prev_best = neurobe_val_loss_value
                            neurobe_patience_count = 0
                        else:
                            neurobe_patience_count += 1

                        if neurobe_patience_count > neurobe_stop_iter:
                            print(f'NeuroBE patience early stopping at epoch {global_epoch_nb}: '
                                  f'count {neurobe_patience_count} > stop_iter {neurobe_stop_iter}, '
                                  f'best_val_loss={neurobe_prev_best:.6e}, current={neurobe_val_loss_value:.6e}')
                            self.bucket.gm.traced_losses_data.append((self.bucket.label, traced_losses_data))
                            if self.bucket.gm._training_logger:
                                log_early_stopping(self.bucket.gm._training_logger, self.bucket.label, global_epoch_nb, "neurobe_patience", neurobe_val_loss_value)
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
                    train_loss_str = f"{loss.item():.5f}" if loss.item() >= 0.01 else f"{loss.item():.4e}"
                    postfix = {"TrLoss": train_loss_str}
                    # Add validation loss if NBE early stopping is enabled
                    if use_nbe_early_stopping and nbe_val_losses:
                        val_loss_str = f"{nbe_val_losses[-1]:.5f}" if nbe_val_losses[-1] >= 0.01 else f"{nbe_val_losses[-1]:.4e}"
                        postfix["ValLoss"] = val_loss_str
                    pbar.set_postfix(postfix)
                    pbar.update(1)
                    # if current_lr <= self.config['min_lr'] * 10:
                    #     self.bucket.gm.traced_losses_data.append((self.bucket.label, traced_losses_data))
                    #     return traced_losses_data
        self.bucket.gm.traced_losses_data.append((self.bucket.label, traced_losses_data))

        # Expose neurobe patience state for observability
        if use_neurobe_early_stopping:
            self.neurobe_patience_count = neurobe_patience_count
            self.neurobe_prev_best = neurobe_prev_best

        # for retraining after one loss, reverts config back
        if new_loss_fn is not None:
            self.config['loss_fn'] = old_loss_fn_name
            self.loss_fn = self._get_loss_fn(old_loss_fn_name)

        return traced_losses_data
              
    def train_batch(self, x_batch, y_batch, bw_hat_batch=None, plot_message=False, epoch=None):
        self.net.train()

        # Move data to device
        device = self.config['device']
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if bw_hat_batch is not None:
            bw_hat_batch = bw_hat_batch.to(device)

        # Zero the parameter gradients
        if isinstance(self.optimizer, list):
            for opt in self.optimizer:
                opt.zero_grad()
        else:
            self.optimizer.zero_grad()

        # Forward pass
        outputs = self.net(x_batch)

        # UKF Sequential: Periodically recompute Lmu, Lsig
        if hasattr(self, 'ukf_resample_period'):
            if self.ukf_batch_count % self.ukf_resample_period == 0:
                # Recompute Gaussian statistics
                from .ukf_helpers import estimate_gaussian

                # Compute adjusted backward message parameters
                # Ensure all scalars are tensors on the correct device
                sf = torch.tensor(self.ukf_sigma_f + 1e-12, device=outputs.device, dtype=outputs.dtype)
                sb = torch.tensor(self.ukf_sigma_g + 1e-12, device=outputs.device, dtype=outputs.dtype)
                Sx = torch.tensor(self.ukf_rho * self.ukf_sigma_f * self.ukf_sigma_g, device=outputs.device, dtype=outputs.dtype)
                al_ = 1.0 + Sx / (sf ** 2)
                sb_ = torch.sqrt((1.0 - Sx**2 / (sb**2 * sf**2)) * sb**2)
                mb = (al_ - 1.0) * (y_batch - y_batch.mean())

                # Estimate p(Φ, Φ̂)
                Lmu_, Lsig_ = estimate_gaussian(
                    y_batch.detach(),
                    outputs.detach(),
                    mb,
                    sb_,
                    n_samp=self.ukf_m_per
                )

                # Update stored values (with exponential smoothing if not first time)
                if self.ukf_Lmu is None:
                    self.ukf_Lmu = Lmu_
                    self.ukf_Lsig = Lsig_
                else:
                    # Exponential smoothing: gamma = 0.9 (like the notebook)
                    gamma = 0.9
                    self.ukf_Lmu = gamma * self.ukf_Lmu + (1 - gamma) * Lmu_
                    # For covariance, also account for change in mean
                    self.ukf_Lsig = (gamma * self.ukf_Lsig +
                                    (1 - gamma) * (Lsig_ + (Lmu_ - self.ukf_Lmu)[:,None] * (Lmu_ - self.ukf_Lmu)[None,:]))

            self.ukf_batch_count += 1

        if plot_message:
            from nce.utils.plots import plot_fastfactor_comparison
            from nce.inference.factor import FastFactor
            # Compute exact message and NN approximation for full comparison
            exact_message = self.bucket.compute_message_exact()
            nn_factor = self.bucket.compute_message_nn(train=False)
            nn_exact = nn_factor.to_exact()
            # Attach losses to the approx factor for loss curve display
            nn_exact.losses = self.losses
            nn_exact.val_losses = self.val_losses
            plot_fastfactor_comparison(
                exact_message, nn_exact,
                title=f'Bucket {self.bucket.label} at epoch {epoch}',
                show_loss_curve=True
            )
            
            
        
        # Compute loss
        # bw_hat_batch = bw_hat_batch.detach()
        # weights = self._get_weights(bw_hat_batch)
        #debug for squared loss
        # if self.loss_fn == w_gil1c or self.loss_fn == gil1c_linear:
        #     loss = self.loss_fn(outputs.squeeze(), y_batch, bw_hat_batch, self.normalizer)
        # else:
        
        
        # if bw_hat_batch is not None or self.loss_fn == logspace_mse:
        #     bw_hat_batch.detach()
        #     loss = self.loss_fn(outputs.squeeze(), y_batch, bw_hat_batch)
        # else:
        #     loss = self.loss_fn(outputs.squeeze(), y_batch)
            
        # Backward pass and optimize
        
        # if self.debug:
        #     self.tracked['parameters'].append([p.data.clone() for p in self.net.parameters()])
        # debug-------------------------
        # with torch.autograd.detect_anomaly():
        # loss.backward(retain_graph=True)

        # try with scaler-----------------------------------------
        
        if self.config.get('optimizer') == "muon":
            with torch.cuda.amp.autocast(enabled=True):
                if 'elp_least_squares_v2' in str(self.loss_fn):
                    loss_result = self.loss_fn(outputs.reshape(-1), y_batch, bw_hat_batch, expansion_point=self.expansion_point)
                    if isinstance(loss_result, tuple):
                        loss, self.expansion_point, dL, d2L = loss_result

                        # Update moving averages of derivatives
                        if self.dL_moving_avg is None:
                            self.dL_moving_avg = dL.detach()
                            self.d2L_moving_avg = d2L.detach()
                        else:
                            self.dL_moving_avg = self.momentum * self.dL_moving_avg + (1 - self.momentum) * dL.detach()
                            self.d2L_moving_avg = self.momentum * self.d2L_moving_avg + (1 - self.momentum) * d2L.detach()
                    else:
                        loss = loss_result
                else:
                    loss = self.loss_fn(outputs.reshape(-1), y_batch, bw_hat_batch)
                self.scaler.scale(loss).backward()
        else: # not using muon
            if 'elp_least_squares_v2' in str(self.loss_fn):
                loss_result = self.loss_fn(outputs.reshape(-1), y_batch, bw_hat_batch, expansion_point=self.expansion_point)
                if isinstance(loss_result, tuple):
                    loss, self.expansion_point, dL, d2L = loss_result

                    # Update moving averages of derivatives
                    if self.dL_moving_avg is None:
                        self.dL_moving_avg = dL.detach()
                        self.d2L_moving_avg = d2L.detach()
                    else:
                        self.dL_moving_avg = self.momentum * self.dL_moving_avg + (1 - self.momentum) * dL.detach()
                        self.d2L_moving_avg = self.momentum * self.d2L_moving_avg + (1 - self.momentum) * d2L.detach()
                else:
                    loss = loss_result
            else:
                loss = self.loss_fn(outputs.reshape(-1), y_batch, bw_hat_batch)
            loss.backward()
        #-----------------------------------------------------------
        if self.debug:
            self.tracked['gradients'].append([p.grad.clone() if p.grad is not None else None for p in self.net.parameters()])
        
        # Gradient clipping to prevent overshooting
        grad_clip_norm = self.config.get('grad_clip_norm', None)
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip_norm)

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
                    sched.step()
            else:
                self.scheduler.step()
        else:
            self.optimizer.step()
            # NOTE: Scheduler stepping moved to epoch level (after train_epoch)
            # to fix OneCycleLR step count issue when using multiple batches per epoch
        loss_copy = loss.cpu().item()
        del x_batch, y_batch, bw_hat_batch, loss
        torch.cuda.empty_cache()
        return loss_copy
    
    def train_epoch(self, batches, plot=False, epoch=None):
        losses = []
        for batch in batches:
            x_batch, y_batch = batch['x'], batch['y']
            # Support both old 'mgh' key and new 'bw' key for backward compatibility
            bw_batch = batch.get('bw', batch.get('mgh'))
            losses.append(self.train_batch(x_batch, y_batch, bw_batch, plot_message=plot, epoch=epoch))
            # Free references to batch tensors
            del x_batch, y_batch, bw_batch
            torch.cuda.empty_cache()  # If running on GPU

        if self.loss_fn == logspace_mse:
            loss = sum(losses) / len(losses)
            loss = torch.Tensor([loss]).to(self.config['device'])
        else:
            if self.debug:
                print('losses are ', losses)
            # Determine if loss is in log-space or linear-space
            # UKL and related losses are linear-space (should be summed)
            loss_fn_name = self.config.get('loss_fn', '')
            is_logspace = loss_fn_name not in ['unnormalized_kl', 'scaled_ukl', 'unnormalized_kl_old']
            loss = self._aggregate_batch_losses(losses, is_logspace=is_logspace)
            if self.debug:
                print('loss is ', loss)
        return loss
    
    def _make_dataloader(self):
        """Create sample generator, data preprocessor, and data loader for training.

        The normalizing constant is computed lazily on first load() call,
        using the actual training samples rather than a separate initialization sample.
        This ensures accurate normalization without redundant computation.

        Returns:
            Tuple of (SampleGenerator, DataPreprocessor, DataLoader)
        """
        sg = SampleGenerator(gm=self.bucket.gm, bucket=self.bucket, random_seed=self.config['seed'])

        use_bw_approx = self.config.get('use_bw_approx', False)

        normalization_mode = self.config.get('normalization_mode', 'logspace_mean')

        # Create preprocessor with deferred normalization (y=None)
        # The normalizing constant will be computed on first load() call
        data_preprocessor = DataPreprocessor(
            y=None,  # Deferred - will be set on first load()
            bw=None,
            lower_dim=self.lower_dim,
            device=self.config['device'],
            use_bw_approx=use_bw_approx,
            normalization_mode=normalization_mode,
        )

        return sg, data_preprocessor, DataLoader(self.bucket, sample_generator=sg, data_preprocessor=data_preprocessor)

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
                return lambda out, targ, mgh=None: elp(out, targ, mgh, sigma_f=sigma_f, sigma_g=sigma_g, rho=rho, num_bw_samples=num_bw_samples, seed=seed)

        elif "elp_recompute" in loss_fn_name or "elp_loo" in loss_fn_name:
            try:
                sigma_f, sigma_g, rho = self.bucket.get_fw_bw_stats()
            except Exception as e:
                print(f"Error occurred while getting forward/backward stats for {self.bucket.label}: {e}")
                sigma_f, sigma_g, rho = None, None, None
            num_bw_samples = int(loss_fn_name.split(',')[-1]) if ',' in loss_fn_name else -1
            if "elp_recompute" in loss_fn_name:
                loss = elp
            elif "elp_loo" in loss_fn_name:
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
        elif loss_fn_name == "elp":
            print('depricated loss name format - please use elp_recompute,<num_samples>')
            sigma_f = self.stats[self.bucket.label]['output_message_std']
            sigma_g = self.stats[self.bucket.label]['mg_std']
            rho = self.stats[self.bucket.label]['correlation']
            return lambda out, targ, mgh=None: elp(out, targ, mgh, sigma_f=sigma_f, sigma_g=sigma_g, rho=rho)     
        elif loss_fn_name == "logspace_mse_fdb":
            return logspace_mse_fdb
        elif loss_fn_name == "linspace_mse_fdb":
            # Enable exponential preprocessing mode
            self.data_preprocessor.exp_preprocessing = True
            mse = nn.MSELoss()
            # Wrap to ignore bw_hat_batch parameter
            return lambda out, targ, mgh=None: mse(out, targ)
        elif loss_fn_name == "scaled_mse":
            # Enable exponential preprocessing mode with scaling
            self.data_preprocessor.exp_preprocessing = True
            mse = nn.MSELoss()
            # Wrap to ignore bw_hat_batch parameter
            return lambda out, targ, mgh=None: mse(out, targ)
        elif loss_fn_name == "unnormalized_kl":
            # Return lambda that reads bw_normalizing_constant and global_max_targets from
            # data_preprocessor at call time. These are computed after training data is loaded.
            # CRITICAL: global_max_targets must be used for ALL batches to prevent divergence
            return lambda out, targ, bw=None: unnormalized_kl(
                out, targ, bw,
                bw_normalizing_constant=self.data_preprocessor.bw_normalizing_constant,
                max_val=self.data_preprocessor.global_max_targets
            )
        elif loss_fn_name == "scaled_ukl":
            # Get forward/backward stats for scaling
            try:
                sigma_f, sigma_g, rho = self.bucket.get_fw_bw_stats()
            except Exception as e:
                print(f"Warning: Could not get fw/bw stats for scaled_ukl: {e}")
                sigma_f, sigma_g = None, None

            # Return lambda that calls unnormalized_kl with scaling parameters
            return lambda out, targ, mgh=None: unnormalized_kl(out, targ, mgh, sigma_f=sigma_f, sigma_g=sigma_g)
        elif "power_exponential" in loss_fn_name:
            alpha = float(loss_fn_name.split(',')[-1])
            return lambda outputs, targets, bw_hat=None: power_exponential(outputs, targets, bw_hat, alpha=alpha)
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
        elif loss_fn_name == "neurobe_weighted_mse":
            # Closure reads preprocessor stats at call time (after initialization)
            # neurobe_weighted_mse signature: (outputs, targets, ln_min, ln_max, sum_ln)
            # bw_hat is unused — neurobe_weighted_mse doesn't use backward messages
            def _neurobe_weighted_mse(outputs, targets, bw_hat=None):
                return neurobe_weighted_mse(
                    outputs, targets,
                    ln_min=self.data_preprocessor.ln_min,
                    ln_max=self.data_preprocessor.ln_max,
                    sum_ln=self.data_preprocessor.sum_ln,
                )
            return _neurobe_weighted_mse
        elif loss_fn_name == "weighted_logspace_mse":
            return weighted_logspace_mse
        elif loss_fn_name == "weighted_logspace_mse_pedigree":
            return weighted_logspace_mse_pedigree
        elif "elp_least_squares" in loss_fn_name:
            # Expected log partition least squares loss
            # Syntax: elp_least_squares,<num_bw_samples>
            try:
                sigma_f, sigma_g, rho = self.bucket.get_fw_bw_stats()
            except Exception as e:
                print(f"Error occurred while getting forward/backward stats for {self.bucket.label}: {e}")
                sigma_f, sigma_g, rho = None, None, None
            num_bw_samples = int(loss_fn_name.split(',')[-1]) if ',' in loss_fn_name else 100
            seed = 0 if self.config.get('approximation_method', '') == 'dt' else None
            return lambda out, targ, mgh=None: elp_least_squares(out, targ, mgh, sigma_f=sigma_f, sigma_g=sigma_g, rho=rho, num_bw_samples=num_bw_samples, seed=seed)

        elif loss_fn_name == "ukf_sequential" or "ukf_seq" in loss_fn_name:
            # UKF sequential loss - requires special training mode
            # Syntax: ukf_sequential or ukf_sequential,<resample_period>,<m_per>
            parts = loss_fn_name.split(',')
            resample_period = int(parts[1]) if len(parts) > 1 else 10
            m_per = int(parts[2]) if len(parts) > 2 else 500

            # Get forward/backward stats
            try:
                sigma_f, sigma_g, rho = self.bucket.get_fw_bw_stats()
            except Exception as e:
                print(f"Error occurred while getting forward/backward stats for {self.bucket.label}: {e}")
                sigma_f, sigma_g, rho = None, None, None

            # Store parameters for training loop
            self.ukf_resample_period = resample_period
            self.ukf_m_per = m_per
            self.ukf_Lmu = None  # Will be computed during training
            self.ukf_Lsig = None
            self.ukf_batch_count = 0
            self.ukf_sigma_f = sigma_f
            self.ukf_sigma_g = sigma_g
            self.ukf_rho = rho

            # Return lambda that will use these stored values
            return lambda out, targ, mgh=None: ukf_sequential(
                out, targ, mgh,
                Lmu=self.ukf_Lmu,
                Lsig=self.ukf_Lsig,
                sigma_f=self.ukf_sigma_f,
                sigma_g=self.ukf_sigma_g,
                rho=self.ukf_rho
            )

        else:
            raise ValueError(f"Loss function {loss_fn_name} not recognized")

    def _generate_validation_set(self):
        """Generate validation set for mini-batch early stopping.

        Uses uniform sampling with size = min(100k, message_size).

        Returns:
            List with single batch dict containing validation data
        """
        # Determine validation set size: 100k or full message if smaller
        val_size = min(100000, self.message_size)

        # Check if full message is smaller than requested size
        if self.message_size <= val_size:
            # Use all assignments
            return self.dataloader.load_all()
        else:
            # Sample uniformly
            # Temporarily save original sampling scheme
            original_scheme = self.dataloader.sample_generator.sampling_scheme
            self.dataloader.sample_generator.sampling_scheme = 'uniform'

            # Generate validation data
            x, y, bw = self.dataloader.load(num_samples=val_size, all=False)

            # Restore original sampling scheme
            self.dataloader.sample_generator.sampling_scheme = original_scheme

            # Return as list with single batch dict
            return [{'x': x, 'y': y, 'bw': bw}]

    def _generate_validation_set_nbe(self, val_size):
        """Generate validation set for NBE early stopping.

        Args:
            val_size: Number of samples for validation set

        Returns:
            List with single batch dict containing validation data
        """
        # Cap at message size
        val_size = min(val_size, self.message_size)

        # Check if full message is smaller than requested size
        if self.message_size <= val_size:
            # Use all assignments
            return self.dataloader.load_all()
        else:
            # Sample uniformly
            # Temporarily save original sampling scheme
            original_scheme = self.dataloader.sample_generator.sampling_scheme
            self.dataloader.sample_generator.sampling_scheme = 'uniform'

            # Generate validation data
            x, y, bw = self.dataloader.load(num_samples=val_size, all=False)

            # Restore original sampling scheme
            self.dataloader.sample_generator.sampling_scheme = original_scheme

            # Return as list with single batch dict
            return [{'x': x, 'y': y, 'bw': bw}]

    def compute_epoch_loss(self, batches, loss_fn):
        """Compute total loss across all batches (for validation).

        Args:
            batches: List of batch dicts with 'x', 'y', 'bw' keys
            loss_fn: Loss function to evaluate

        Returns:
            Total loss as torch.Tensor
        """
        total_loss = 0.0
        total_samples = 0

        for batch in batches:
            x_batch = batch['x']
            y_batch = batch['y']
            # Support both old 'mgh' key and new 'bw' key
            bw_batch = batch.get('bw', batch.get('mgh'))

            # Forward pass
            outputs = self.net(x_batch).reshape(-1)  # Flatten to [batch_size] to match y_batch and bw_batch shapes

            # Compute loss
            if bw_batch is not None:
                loss = loss_fn(outputs, y_batch, bw_batch)
            else:
                loss = loss_fn(outputs, y_batch)

            # Accumulate
            total_loss += loss.item() * len(x_batch)
            total_samples += len(x_batch)

        # Return average loss
        return torch.tensor(total_loss / total_samples if total_samples > 0 else 0.0)

    def _get_weights(self, bw_hat_batch):
        if self.dataloader.sample_generator.sampling_scheme == 'mg':
            message_size = self.dataloader.bucket.get_message_size()
            p_dist = 1 / message_size
            q_dist = torch.exp(bw_hat_batch)
            return torch.exp(bw_hat_batch)
        elif self.dataloader.sample_generator.sampling_scheme == 'path':
            p_dist = 1 / message_size
            q_dist = torch.exp(bw_hat_batch)
            return torch.exp(bw_hat_batch)

    def _evaluate_batch(self, loss_fn_name, x_batch, y_batch, bw_hat_batch=None):
        with torch.no_grad():
            if self.debug:
                for param_group in self.optimizer.param_groups:
                    print(f"Learning rate: {param_group['lr']}")
            self.net.eval()
            # Forward pass
            outputs = self.net(x_batch)
            # Compute loss
            loss_fn = self._get_loss_fn(loss_fn_name)
            loss = loss_fn(outputs.squeeze(), y_batch, bw_hat_batch)
            return loss
    
    def evaluate_epoch(self, loss_fns, batches):
        with torch.no_grad():
            out = []
            for loss_fn_name in loss_fns:
                losses = []
                for batch in batches:
                    x_batch, y_batch = batch['x'], batch['y']
                    # Support both old 'mgh' key and new 'bw' key
                    bw_batch = batch.get('bw', batch.get('mgh'))
                    outputs = self.net(x_batch)

                    losses.append(self._evaluate_batch(loss_fn_name, x_batch, y_batch, bw_batch))
                if self._get_loss_fn(loss_fn_name) == logspace_mse:
                    loss = sum(losses) / len(losses)
                else:
                    # Determine if loss is in log-space or linear-space
                    is_logspace = loss_fn_name not in ['unnormalized_kl', 'scaled_ukl', 'unnormalized_kl_old']
                    loss = self._aggregate_batch_losses(losses, is_logspace=is_logspace)
                out.append(loss)
            return out
    
    def print_epoch_losses(self, loss_fns, batches, losses=None):
        if losses is None:
            losses = self.evaluate_epoch(loss_fns, batches)
        for (loss_fn_name,loss) in zip(loss_fns, losses):
            print(f'{loss_fn_name} loss: {loss.item()}')
        
    def _aggregate_batch_losses(self, losses, is_logspace=True):
        """Aggregate batch losses into epoch loss.

        Args:
            losses: List of per-batch loss values
            is_logspace: If True, use logsumexp (for log-space losses like log-likelihood)
                        If False, use simple sum (for linear-space losses like UKL)
        """
        losses = torch.tensor(losses, dtype=torch.float32, device=self.config['device']).detach()
        if is_logspace:
            return torch.logsumexp(losses, dim=0) - torch.log(torch.tensor(len(losses)))
        else:
            # For linear-space losses (like unnormalized_kl), just sum them
            return torch.sum(losses)
        
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
            if 'bw_hat' in batch:
                bw_hat = batch['bw_hat'].to(self.config['device'])
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.net(inputs)
            
            # Compute loss
            if 'bw_hat' in batch:
                loss = self.loss_fn(outputs.squeeze(), targets, bw_hat)
            else:
                loss = self.loss_fn(outputs.squeeze(), targets)
                
            # Backward pass and optimize
            loss.backward()

            # Gradient clipping to prevent overshooting
            grad_clip_norm = self.config.get('grad_clip_norm', None)
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip_norm)

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
            # else:
            #     early_stopping_counter += 1
            
            # # Early stopping
            # if early_stopping_counter >= self.config['training']['patience']:
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
        #     pass

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

