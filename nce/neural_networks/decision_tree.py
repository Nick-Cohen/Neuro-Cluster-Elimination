import numpy as np
from sklearn.tree import DecisionTreeRegressor
import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS

## Useful helper function

def lse(msg, ax=None): return np.log(np.exp(msg-np.max(msg)).sum(axis=ax))+np.max(msg)
#def phi(m, axis=None): return np.log(np.sum(np.exp(m-m.max()),axis,keepdims=True))+m.max()
def phi(m, axis=None): return np.log(np.sum(np.exp(m-np.max(m)),axis,keepdims=True))+np.max(m)
def sig(z): return 1./(1.+np.exp(-z))


class DecisionTreeLossOptimizer:
    """
    General decision tree optimizer that can work with different loss functions.
    Implements iterative gradient-based updates using decision trees.
    """
    
    def __init__(self, bucket=None, num_leaves=10, num_samples=1000, num_batch=10, num_iterations=100, 
                 learning_rate=0.25, momentum=0.25, convergence_threshold=0.01, random_seed=0):
        """
        Parameters:
        -----------
        loss_fn : callable
            Loss function that takes (predictions, targets, **kwargs) and returns scalar loss
        bucket : FastBucket, optional
            Bucket to load assignments from. If provided, assignments will be loaded automatically.
        num_leaves : int
            Maximum number of leaves in each decision tree (K parameter)
        num_iterations : int
            Maximum number of optimization iterations
        learning_rate : float
            Step size for tree predictions (0.25 in Alex's implementation)
        momentum : float
            Step size toward exact solution (0.25 in Alex's implementation) 
        convergence_threshold : float
            Stop when max change in s is below this threshold
        random_seed : int
            Random seed for reproducibility
        """
        from .train import Trainer
        trainer = Trainer(net=None, bucket=bucket)
        self.config = bucket.gm.config
        self.loss_fn = trainer._get_loss_fn(self.config.get('loss_fn'))
        self.sigma_f, self.sigma_g, self.rho = 0,0,0
        if 'elp' in self.config.get('loss_fn') or 'approx_smg' in self.config.get('loss_fn'):
            var_names = self.loss_fn.__code__.co_freevars
            var_values = [cell.cell_contents for cell in self.loss_fn.__closure__]
            closure_vars = dict(zip(var_names, var_values))
            self.sigma_f = float(closure_vars.get('sigma_f', 0))
            self.sigma_g = float(closure_vars.get('sigma_g', 0))
            self.rho = float(closure_vars.get('rho', 0))
            
        self.bucket = bucket
        self.num_leaves = self.config.get('num_leaves', num_leaves)
        # load num samples and batches if using smg
        self.num_iterations = self.config.get('num_iterations', num_iterations)
        self.learning_rate = self.config.get('dt_lr', learning_rate)
        self.momentum = self.config.get('dt_momentum', momentum)
        self.random_seed = self.config.get('dt_random_seed', random_seed)
        self.convergence_threshold = self.config.get('dt_convergence_threshold', convergence_threshold)

        # For tracking optimization
        self.history = {'losses': [], 'convergence': []}
        
        # Load assignments and values if bucket is provided
        if self.bucket is not None:
            self._load_bucket_data()
            
    def fit(self, K=10, sf=1., sb=1., Sx=0., bias=0., plot_incremental=False):
        K = self.num_leaves
        msg = self.true_values.cpu().numpy()
        if 'elp' in self.config.get('loss_fn') or 'approx_smg' in self.config.get('loss_fn'):
            Sx = self.rho * self.sigma_f * self.sigma_g
            dt = self.fit_smg(K=K, sf=self.sigma_f, sb=self.sigma_g, Sx=Sx, bias=bias, plot_incremental=plot_incremental)
        else:
            dt = DecisionTreeRegressor(max_leaf_nodes=K if K>1 else 2, min_samples_leaf=1 if K>1 else len(msg), random_state=0)
            weights = self.get_weights()
            dt.fit(self.dX, msg, sample_weight=weights)

        mhat = dt.predict(self.dX)
        mhat = torch.tensor(mhat, dtype=self.true_values.dtype, device=self.true_values.device)
        mhat = self.convert_back_to_original(mhat)
        return mhat
        # elif 'logspace' in self.config.get('loss_fn'):
        #     return self.fit_log_mse()
        # elif 'linspace_mse' in self.config.get('loss_fn'):
        #     return self.fit_lin_mse()
        # elif 'weighted_logspace_mse' in self.config.get('loss_fn'):
        #     return self.fit_weighted_log_mse()
        # elif 'unnormalized_kl' in self.config.get('loss_fn'):
        #     return self.fit_ukl()
        # else:
        #     raise NotImplementedError(f"Loss function {self.config.get('loss_fn')} not supported yet")
        
    def fit_smg(self, K=10, sf=1., sb=1., Sx=0., bias=0., plot_incremental=True, return_extras=True):
        K = self.num_leaves if K is None else K
        # dY=self.true_values.cpu().numpy()
        # msg = self.true_values.cpu().numpy()
        # dX = self.assignments.cpu().numpy()

        f = 1.*self.dY; s=f+0.; c=s+0.;
        sf = sf+1e-12; sb = sb + 1e-12; rngf = np.std(f.flatten())+1e-12;
        al_ = (1.+Sx/sf**2)
        sb_ = np.sqrt( (1. - Sx**2/sb**2/sf**2)*sb**2 )
        num_iter = self.num_iterations
        num_samples, num_batch = 1000, 10
        print(f'     ... alpha: {np.round(al_,3)}, sb_: {np.round(sb_,3)}, bias: {np.round(bias,3)}, K: {K} (sf:{np.round(sf,2)},sb:{np.round(sb,2)},Sx:{np.round(Sx,2)})')
        #bs = sb_*np.random.randn(num_samples,len(f))
        self.dt = None
        
        dL,d2Ld = 0.,0.
        for it in range(num_iter):
            dL_,d2Ld_ = 0.,0.
            np.random.seed(it)
            for _ in range(num_samples//num_batch):
                bs = sb_*np.random.randn(num_batch,f.size) + (al_-1.)*(f.reshape(1,-1)-f.mean())  # Fixed 9/19/25
                DP = (phi(f.reshape(1,-1)+bs,axis=1)-phi(s.reshape(1,-1)+bs,axis=1)+bias)
                psb = np.exp(s.reshape(1,-1)+bs - phi(s.reshape(1,-1)+bs,axis=1))
                psb = np.maximum(psb, 1./len(f))
                dL_ += -2.*( DP * psb ).mean(0,keepdims=True)/(num_samples//num_batch)
                d2Ld_ += 2.*(psb**2 - DP*(psb-psb**2)).mean(0,keepdims=True)/(num_samples//num_batch)
            #DP = (phi(al_*f.reshape(1,-1)+bs,axis=1)-phi(al_*s.reshape(1,-1)+bs,axis=1))
            #psb = np.exp(al_*s.reshape(1,-1)+bs - phi(al_*s.reshape(1,-1)+bs,axis=1))
            #dL = -2.*( DP * psb ).mean(0, keepdims=True)
            #d2Ld = 2.*(psb**2 - DP*(psb-psb**2)).mean(0,keepdims=True) +1e-8
            d2Ld_ = np.maximum(d2Ld_, 1./len(f)**2);  ### Force d2L to be positive definite
            if it==0: dL,d2Ld = dL_,d2Ld_
            else:     dL,d2Ld = (.9*dL+.1*dL_, .9*d2Ld+.1*d2Ld_)  # make stats evolve slowly

            #s_adj = (s.reshape(1,-1) - dL/d2Ld)    # regression target w/ weights
            s_adj = (s.reshape(1,-1) - np.clip(dL/d2Ld, -rngf/(it+1),rngf/(it+1)) ) 
            #s_adj += 1e-6*np.random.randn(*s_adj.shape)   # dodge numerical roundoff issues

            self.dt = DecisionTreeRegressor(max_leaf_nodes=K if K>1 else 2, min_samples_leaf=1 if K>1 else len(self.dY),random_state=0)
            self.dt.fit(self.dX,s_adj[0,:], sample_weight=d2Ld.squeeze()/d2Ld.sum())
            c = self.dt.predict(self.dX)[None,:];
            #plt.plot(f.flatten(),'b-'); plt.plot(s.flatten(),'r-'); plt.plot(s_adj.flatten(),'m-',alpha=0.5); plt.plot(c.flatten(),'c:',alpha=0.5); plt.show();
            sprev = s+0.
            #s = s + (c-s)*.25 + (f[None,:]-s)*.25
            #s = s + (c-s)*.25 + 0*(f[None,:]-s)*.25 ## change 9/19/25-runB
            s = s + (c-s)*.5 + 0*(f[None,:]-s)*.25 ## change 9/19/25-runC
            if np.abs(s-sprev).max()<.01: break
        
        # for it in range(num_iter):
        #     mx = np.max(s)
        #     dL = np.exp(s-mx)-np.exp(f-mx)
        #     d2Ld = np.exp(s-mx) # diagonal of Hessian
        #     d2Ld = np.minimum(d2Ld, 1./len(f))

        #     s_adj = (s.reshape(1,-1) - np.clip(dL/d2Ld, -rngf/(it+1),rngf/(it+1)) )  
        #     s_adj += 1e-6*np.random.randn(*s_adj.shape)   # dodge numerical roundoff issues

        #     self.dt = DecisionTreeRegressor(max_leaf_nodes=K if K>1 else 2, min_samples_leaf=1 if K>1 else len(self.dY),random_state=0)
        #     self.dt.fit(self.dX,s_adj[0,:], sample_weight=d2Ld.squeeze()/d2Ld.sum())
        #     c = self.dt.predict(self.dX)[None,:];
        #     sprev = s+0.
        #     s = s + (c-s)*.25 + (f[None,:]-s)*.25
        #     if np.abs(s-sprev).max()<.01: break
        # for it in range(num_iter):
        #     dL,d2Ld = 0.,0.
        #     np.random.seed(it)
        #     for _ in range(num_samples//num_batch):
        #         bs = sb_*np.random.randn(num_batch,f.size) + (al_-1.)*(f.reshape(1,-1)-f.mean())
        #         DP = (phi(al_*f.reshape(1,-1)+bs,axis=1)-phi(al_*s.reshape(1,-1)+bs,axis=1)+bias)
        #         psb = np.exp(al_*s.reshape(1,-1)+bs - phi(al_*s.reshape(1,-1)+bs,axis=1))
        #         dL += -2.*( DP * psb ).mean(0,keepdims=True)/(num_samples//num_batch)
        #         d2Ld += 2.*(psb**2 - DP*(psb-psb**2)).mean(0,keepdims=True)/(num_samples//num_batch)
            
        #     d2Ld = np.maximum(d2Ld, 1e-20) #1./len(f)**2);  ### Force d2L to be positive definite

        #     #s_adj = (s.reshape(1,-1) - dL/d2Ld)    # regression target w/ weights
        #     s_adj = (s.reshape(1,-1) - np.clip(dL/d2Ld, -rngf/(it+1),rngf/(it+1)) )    # I think this can't be larger than 0 or less than -1?
        #     s_adj += 1e-6*np.random.randn(*s_adj.shape)

        #     dt = DecisionTreeRegressor(max_leaf_nodes=K if K>1 else 2, min_samples_leaf=1 if K>1 else len(self.dY), random_state=0)
        #     dt.fit(self.dX,s_adj[0,:], sample_weight=d2Ld.squeeze()/d2Ld.sum())
        #     c = dt.predict(self.dX)[np.newaxis,:];
        #     # mhat = dt.predict(dX)
        #     if plot_incremental:
        #         from matplotlib import pyplot as plt
        #         pltrng = np.argsort(f.flatten())[-10:]
        #         plt.plot(f.flatten()[pltrng],'b-', label='f')
        #         plt.plot(s.flatten()[pltrng],'r-', label='s')
        #         plt.plot(s_adj.flatten()[pltrng],'m-',alpha=0.5, label='s_adj')
        #         plt.plot(c.flatten()[pltrng],'c:',alpha=0.9,lw=2, label='f_hat')
        #         plt.legend()
        #         plt.show()
        #         # plt.savefig('/home/cohenn1/NCE/notebooks/_September-2025/incremental_plots/it_'+str(it)+'.png')
        #     sprev = s+0.
        #     s = s + (c-s)*.25 + (f.reshape(1,-1)-s)/(4.) # +it?
        #     if np.abs(s-sprev).max()<.01: break
        #     #print(f.round(2)); print(s.round(2));
        # if it>=num_iter-1: print(f'Convergence issue? {it+1} iterations'); #warning.warn('Con')
        
        loss_function = self.loss_fn
        
        optimized_dt, loss_history = self.optimize_leaf_values()
        return optimized_dt
        mhat = optimized_dt.predict(self.dX)
        mhat = torch.tensor(mhat, dtype=self.true_values.dtype, device=self.true_values.device)
        mhat = self.convert_back_to_original(mhat)
        return (mhat)
    
    def fit_log_mse(self):
        K = self.num_leaves
        dY=self.true_values.cpu().numpy()
        msg = self.true_values.cpu().numpy()
        dX = self.assignments.cpu().numpy()
        
        dt = DecisionTreeRegressor(max_leaf_nodes=K if K>1 else 2, min_samples_leaf=1 if K>1 else len(msg), random_state=0)
        dt.fit(dX,msg)
        
        mhat = dt.predict(dX)
        mhat = torch.tensor(mhat, dtype=self.true_values.dtype, device=self.true_values.device)
        mhat = self.convert_back_to_original(mhat)
        return mhat
    
    def fit_lin_mse(self):
        K = self.num_leaves
        dY=self.true_values.cpu().numpy()
        msg = self.true_values.cpu().numpy()
        max_msg = np.max(msg)
        exp_msg = np.exp(msg - max_msg)  # for numerical stability
        dX = self.assignments.cpu().numpy()
        
        dt = DecisionTreeRegressor(max_leaf_nodes=K if K>1 else 2, min_samples_leaf=1 if K>1 else len(exp_msg), random_state=0)
        dt.fit(dX,exp_msg)
        
        mhat = dt.predict(dX)
        mhat = torch.tensor(mhat, dtype=self.true_values.dtype, device=self.true_values.device)
        # Convert back to original scale
        mhat = torch.log(mhat+ 1e-10) + max_msg
        mhat = self.convert_back_to_original(mhat)
        return mhat
    
    def fit_weighted_log_mse(self):
        K = self.num_leaves
        dY=self.true_values.cpu().numpy()
        msg = self.true_values.cpu().numpy()
        dX = self.assignments.cpu().numpy()
        
        dt = DecisionTreeRegressor(max_leaf_nodes=K if K>1 else 2, min_samples_leaf=1 if K>1 else len(msg), random_state=0)
        ln_max = np.max(self.dY)
        ln_min = np.min(self.dY)
        normalized_targets = (self.dY - ln_min) / (ln_max - ln_min)
        weights = normalized_targets / (np.sum(normalized_targets))
        assert np.isclose(np.sum(weights), 1.0), "Weights do not sum to 1"
        dt.fit(dX,msg, sample_weight=weights)
        
        mhat = dt.predict(dX)
        mhat = torch.tensor(mhat, dtype=self.true_values.dtype, device=self.true_values.device)
        mhat = self.convert_back_to_original(mhat)
        return mhat
    
    def fit_ukl(self):
        pass
        
    def fit_smg_altered(self, K=10, sf=1., sb=1., Sx=0., bias=0., plot_incremental=True):
        sf = self.sigma_f if self.sigma_f>0 else 0
        sb = self.sigma_g if self.sigma_g>0 else 0
        Sx = self.rho if self.rho>0 else 0
        
        K = self.num_leaves if K is None else K
        dY=self.true_values.cpu().numpy()
        msg = self.true_values.cpu().numpy()
        dX = self.assignments.cpu().numpy()

        f = 1.*dY; s=f+0.; c=s+0.;
        sf = sf+1e-12; sb = sb + 1e-12; rngf = np.std(f.flatten())+1e-12;
        forward_coeff = (1.+self.rho * sb/sf)
        backward_coeff = sb * np.sqrt(1. - self.rho**2)
        num_iter = self.num_iterations
        num_samples, num_batch = 1000, 10 # number of smg samples, num per batch
        print(f'     ... forward_coeff: {np.round(forward_coeff,3)}, backward_coeff: {np.round(backward_coeff,3)}, bias: {np.round(bias,3)}, K: {K} (sf:{np.round(sf,2)},sb:{np.round(sb,2)},Sx:{np.round(Sx,2)})')
        #bs = sb_*np.random.randn(num_samples,len(f))
        
        for it in range(num_iter):
            dL,d2Ld = 0.,0.
            np.random.seed(0*it)
            for _ in range(num_samples//num_batch):
                bs = backward_coeff*np.random.randn(num_batch,f.size)
                DP = (phi(forward_coeff*f.reshape(1,-1)+bs,axis=1)-phi(forward_coeff*s.reshape(1,-1)+bs,axis=1)+bias)
                psb = np.exp(forward_coeff*s.reshape(1,-1)+bs - phi(forward_coeff*s.reshape(1,-1)+bs,axis=1))
                dL += -2.*( DP * psb ).mean(0,keepdims=True)/(num_samples//num_batch)
                d2Ld += 2.*(psb**2 - DP*(psb-psb**2)).mean(0,keepdims=True)/(num_samples//num_batch)
            
            d2Ld = np.maximum(d2Ld, 1e-20) #1./len(f)**2);  ### Force d2L to be positive definite

            #s_adj = (s.reshape(1,-1) - dL/d2Ld)    # regression target w/ weights
            s_adj = (s.reshape(1,-1) - np.clip(dL/d2Ld, -rngf/(it+1),rngf/(it+1)) )    # I think this can't be larger than 0 or less than -1?
            s_adj += 1e-6*np.random.randn(*s_adj.shape)

            dt = DecisionTreeRegressor(max_leaf_nodes=K if K>1 else 2, min_samples_leaf=1 if K>1 else len(msg), random_state=0)
            dt.fit(dX,s_adj[0,:], sample_weight=d2Ld.squeeze()/d2Ld.sum())
            c = dt.predict(dX)[np.newaxis,:];
            # mhat = dt.predict(dX)
            if plot_incremental:
                from matplotlib import pyplot as plt
                pltrng = np.argsort(f.flatten())
                plt.plot(f.flatten()[pltrng],'b-', label='f')
                plt.plot(s.flatten()[pltrng],'r-', label='s')
                plt.plot(s_adj.flatten()[pltrng],'m-',alpha=0.5, label='s_adj')
                plt.plot(c.flatten()[pltrng],'c:',alpha=0.9,lw=2, label='f_hat')
                plt.legend()
                plt.show()
            sprev = s+0.
            s = s + (c-s)*.25 + (f.reshape(1,-1)-s)/(4.) # +it?
            if np.abs(s-sprev).max()<.01: break
        if it>=num_iter-1: print(f'Convergence issue? {it+1} iterations'); #warning.warn('Con')
        mhat = dt.predict(dX)
        sf = self.sigma_f if self.sigma_f>0 else 0
        sb = self.sigma_g if self.sigma_g>0 else 0
        Sx = self.rho * sf * sb
     
    def _compute_gradients(self, s, f, assignments):
        """
        Compute first and second derivatives of loss w.r.t. current approximation s.
        Uses finite differences for numerical derivatives.
        
        Parameters:
        -----------
        s : torch.Tensor
            Current approximation
        f : torch.Tensor  
            True values
        assignments : torch.Tensor
            Variable assignments
            
        Returns:
        --------
        dL : torch.Tensor
            First derivatives (gradient)
        d2L : torch.Tensor  
            Second derivatives (Hessian diagonal)
        """
        eps = 1e-6
        dL = torch.zeros_like(s)
        d2L = torch.zeros_like(s)
        
        # Compute baseline loss
        L0 = self.loss_fn(s, f)
        
        # Compute derivatives for each element
        for i in range(len(s)):
            # Forward difference for first derivative
            s_plus = s.clone()
            s_plus[i] += eps
            L_plus = self.loss_fn(s_plus, f)
            
            # Backward difference for first derivative  
            s_minus = s.clone()
            s_minus[i] -= eps
            L_minus = self.loss_fn(s_minus, f)
            
            # Central difference for first derivative
            dL[i] = (L_plus - L_minus) / (2 * eps)
            
            # Second derivative using three-point formula
            d2L[i] = (L_plus - 2*L0 + L_minus) / (eps**2)
            
        # Ensure second derivatives are positive (for stability)
        d2L = torch.clamp(d2L, min=1e-20)
        
        return dL, d2L

    def _load_bucket_data(self):
        """Load all assignments and true values from the bucket using same approach as neural networks."""
        from nce.sampling.sample_generator import SampleGenerator
        from nce.data.data_preprocessor import DataPreprocessor
        from nce.data.data_loader import DataLoader
        
        # Create sample generator exactly like neural networks do
        sg = SampleGenerator(gm=self.bucket.gm, bucket=self.bucket, random_seed=self.bucket.config.get('seed', 0))
        
        # Sample some values first to create data preprocessor (like neural networks do)
        sample_assignments = sg.sample_assignments(1000)
        sample_values = sg.compute_message_values(sample_assignments)
        sample_mg_values = sg.compute_gradient_values(sample_assignments)
        
        # Create data preprocessor exactly like in Trainer._make_dataloader
        fdb_setting = self.bucket.config.get('fdb', False)
        lower_dim = self.bucket.config.get('lower_dim', False)
        
        data_preprocessor = DataPreprocessor(
            y=sample_values, 
            mg=sample_mg_values, 
            is_logspace=True, 
            lower_dim=lower_dim, 
            device=self.bucket.config['device'],
            fdb=fdb_setting
        )
        
        # Create dataloader
        dataloader = DataLoader(
            bucket=self.bucket,
            sample_generator=sg,
            data_preprocessor=data_preprocessor
        )
        
        # Load all assignments and values
        data = dataloader.load_all()  # Returns list with one dict
        batch = data[0]
        
        self.assignments = batch['x']  # One-hot encoded assignments
        self.true_values = batch['y']  # True message values (normalized)
        self.data_preprocessor = data_preprocessor  # Store for later conversion back
        
        self.dY=self.true_values.cpu().numpy()
        self.dX = self.assignments.cpu().numpy()
        
        print(f"Loaded {len(self.assignments)} assignments from bucket {self.bucket.label}")
        print(f"Assignment shape: {self.assignments.shape}")
        print(f"Values shape: {self.true_values.shape}")
        print(f"Values range: [{self.true_values.min():.6f}, {self.true_values.max():.6f}]")
        print(f"FDB normalization: {fdb_setting}")
        
    def convert_back_to_original(self, normalized_values):
        """Convert normalized values back to original scale using data preprocessor."""
        if hasattr(self, 'data_preprocessor'):
            # Use the same data preprocessor to convert back
            return self.data_preprocessor.undo_normalization(normalized_values)
    
    def fit_old(self, assignments=None, true_values=None):
        """
        Fit decision tree approximation using iterative gradient-based optimization.
        
        Parameters:
        -----------
        assignments : torch.Tensor, optional
            One-hot encoded variable assignments. If None, uses bucket data.
        true_values : torch.Tensor, optional
            True function values. If None, uses bucket data.
            
        Returns:
        --------
        s : torch.Tensor
            Final approximation
        """
        # Use bucket data if no inputs provided
        if assignments is None or true_values is None:
            if self.bucket is None:
                raise ValueError("Must provide either bucket or assignments+true_values")
            assignments = self.assignments
            true_values = self.true_values
        
        # Ensure inputs are torch tensors
        if not isinstance(assignments, torch.Tensor):
            assignments = torch.tensor(assignments, dtype=torch.float32)
        if not isinstance(true_values, torch.Tensor):
            true_values = torch.tensor(true_values, dtype=torch.float32)
        
        # Ensure same device
        true_values = true_values.to(assignments.device)
        
        f = true_values.clone()
        s = f.clone()  # Initialize with true values
        c = s.clone()  # Placeholder for tree predictions
        
        rng_f = torch.std(f.flatten()) + 1e-12

        print(f"Starting optimization with {self.num_iterations} max iterations...")
        
        for it in range(self.num_iterations):
            s_prev = s.clone()
            
            # Compute gradients
            dL, d2L = self._compute_gradients(s, f, assignments)
            
            # Create regression targets (gradient step with clipping)
            rng_f = torch.std(f) + 1e-12
            step_size = rng_f / self.num_leaves / (it + 1)  # Adaptive step size
            gradient_step = torch.clamp(dL / d2L, -step_size, step_size)
            s_adj = s - gradient_step
            
            # Add small amount of noise for regularization
            s_adj += 1e-6 * torch.randn_like(s_adj)
            
            # Train decision tree on regression targets with importance weighting
            dt = DecisionTreeRegressor(
                max_leaf_nodes=self.num_leaves if self.num_leaves > 1 else 2,
                min_samples_leaf=1 if self.num_leaves > 1 else len(f),
                random_state=self.random_seed
            )
            
            # Normalize weights and convert to numpy for sklearn
            weights = d2L / d2L.sum()
            
            # Convert to numpy for sklearn (sklearn doesn't support torch tensors)
            assignments_np = assignments.detach().cpu().numpy()
            s_adj_np = s_adj.detach().cpu().numpy()
            weights_np = weights.detach().cpu().numpy()
            
            # Fit tree
            dt.fit(assignments_np, s_adj_np, sample_weight=weights_np)
            c_np = dt.predict(assignments_np)
            
            # Convert back to torch tensor on same device
            c = torch.tensor(c_np, dtype=s.dtype, device=s.device)
            
            # Update expansion point with momentum
            s_new = s + self.learning_rate * (c - s) + self.momentum * (f - s)
            
            # Check convergence
            max_change = torch.abs(s_new - s_prev).max()
            self.history['convergence'].append(float(max_change))
            
            # Compute current loss for tracking
            current_loss = float(self.loss_fn(s_new, f))
            self.history['losses'].append(current_loss)
            
            s = s_new
            
            if it % 10 == 0:
                print(f"Iteration {it}: loss={current_loss:.6f}, max_change={float(max_change):.6f}")
            
            if max_change < self.convergence_threshold:
                print(f"Converged after {it+1} iterations")
                break
        else:
            print(f"Reached maximum iterations ({self.num_iterations})")
        
        if hasattr(self, 'data_preprocessor'):
            c = self.convert_back_to_original(c)
            print("Converted result back to original scale")
        
        return c   
     
    def fit_and_convert_to_FastFactor(self, debug=False):
        """
        Convenience method that fits the decision tree and converts to FastFactor.
        
        Parameters:
        -----------
        debug : bool
            Whether to print debug information
            
        Returns:
        --------
        FastFactor
            A FastFactor with the decision tree approximation
        """
        if self.bucket is None:
            raise ValueError("Bucket is required for this method")
        
        print("Fitting decision tree...")
        trained_result = self.fit(K=self.num_leaves)
        
        print("Converting to FastFactor...")
        return self.decision_tree_to_FastFactor(trained_result, debug=debug)
        
    def decision_tree_to_FastFactor(self, trained_result, debug=False):
        """
        Convert trained decision tree result to a FastFactor.
        
        Parameters:
        -----------
        trained_result : torch.Tensor
            Result from fit() method (1D tensor in original scale)
        debug : bool
            Whether to print debug information
            
        Returns:
        --------
        FastFactor
            A FastFactor with tensor populated from decision tree predictions
        """
        if self.bucket is None:
            raise ValueError("Bucket is required to create FastFactor")
        
        from nce.inference.factor import FastFactor
        
        # Get the proper, ordered message scope
        scope = self.bucket.get_message_scope()
        
        # Get domain sizes for reshaping
        domain_sizes = [
            self.bucket.gm.vars[self.bucket.gm.matching_var(v)].states 
            for v in scope
        ]
        
        if debug:
            print(f"Scope: {scope}")
            print(f"Domain sizes: {domain_sizes}")
            print(f"Trained result shape: {trained_result.shape}")
        
        # Create tensor-less FastFactor with correct scope
        fast_factor = FastFactor(tensor=None, labels=scope)
        
        # Reshape the 1D fit() output to proper tensor shape and assign
        fast_factor.tensor = trained_result.reshape(domain_sizes)
        
        if debug:
            print(f"FastFactor tensor shape: {fast_factor.tensor.shape}")
            print(f"FastFactor labels: {fast_factor.labels}")
        
        return fast_factor

    def optimize_leaf_values(self, optimizer='adam', lr=0.01, max_iter=10000, tolerance=1e-6, device='cuda'):
        """
        Optimize leaf values of a fitted decision tree using PyTorch optimization.
        
        Parameters:
        -----------
        optimizer : str
            The optimizer to use ('adam' or 'lbfgs')
        lr : float
            Learning rate for the optimizer
        max_iter : int
            Maximum number of optimization iterations
        tolerance : float
            Tolerance for convergence
        device : str
            Device to run the optimization on ('cpu' or 'cuda')
        optimizer : str, 'adam' or 'lbfgs'
        lr : float, learning rate
        max_iter : int, maximum number of optimization steps
        tolerance : float, convergence tolerance
        device : str, 'cpu' or 'cuda'
        
        Returns:
        --------
        optimized_tree : modified tree with optimized leaf values
        loss_history : list of loss values during optimization
        """
        tree_model = self.dt
        X = self.dX
        y = self.dY
        loss_function = self.loss_fn
        # Convert inputs to tensors
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        else:
            X_tensor = X.to(device)
            
        if isinstance(y, np.ndarray):
            y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
        else:
            y_tensor = y.to(device)
        
        # Get leaf indices for each sample (this needs to stay on CPU for sklearn)
        X_cpu = X if isinstance(X, np.ndarray) else X.cpu().numpy()
        leaf_indices = tree_model.apply(X_cpu)
        unique_leaves = np.unique(leaf_indices)
        K = len(unique_leaves)
        
        # Create mapping from leaf_id to sample indices
        leaf_to_mask = {}
        for leaf_id in unique_leaves:
            mask = (leaf_indices == leaf_id)
            leaf_to_mask[leaf_id] = torch.tensor(mask, dtype=torch.bool, device=device)
        
        # Initialize leaf values as learnable parameters
        current_leaf_values = tree_model.tree_.value[unique_leaves].flatten()
        leaf_params = torch.tensor(current_leaf_values, dtype=torch.float32, 
                                device=device, requires_grad=True)
        
        # Setup optimizer
        if optimizer.lower() == 'adam':
            opt = Adam([leaf_params], lr=lr)
        elif optimizer.lower() == 'lbfgs':
            opt = LBFGS([leaf_params], lr=lr, max_iter=20, tolerance_grad=tolerance)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        loss_history = []
        best_loss = float('inf')
        best_params = leaf_params.clone()
        # patience_counter = 0
        
        def closure():
            opt.zero_grad()
            
            total_loss = 0
            for i, leaf_id in enumerate(unique_leaves):
                mask = leaf_to_mask[leaf_id]
                
                if mask.sum() > 0:  # Only if there are samples in this leaf
                    y_true_leaf = y_tensor[mask]
                    # Create y_pred_leaf while preserving gradients
                    y_pred_leaf = leaf_params[i].expand_as(y_true_leaf)
                    
                    leaf_loss = loss_function(y_pred_leaf, y_true_leaf)
                    total_loss += leaf_loss
            
            total_loss.backward()
            return total_loss
        
        print(f"Starting optimization with {K} leaves...")
        
        last_loss = 10e6
        for iteration in range(max_iter):
            if optimizer.lower() == 'lbfgs':
                loss = opt.step(closure)
            else:
                loss = closure()
                opt.step()
            
            loss_val = loss.item()
            loss_history.append(loss_val)
            
            # Early stopping
            # if loss_val < best_loss - tolerance:
            #     best_loss = loss_val
            #     best_params = leaf_params.clone()
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
                
            # if patience_counter > 200000 and optimizer.lower() == 'adam':  # Early stopping for Adam
            #     break
                
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss_val:.6f}")
                # early stopping
                if not loss_val < last_loss * 0.9:
                    print(f"Early stopping at iteration {iteration}")
                    break
                last_loss = loss_val
            # Convergence check
            if len(loss_history) > 1:
                if abs(loss_history[-1] - loss_history[-2]) < tolerance:
                    print(f"Converged at iteration {iteration}")
                    break
        
        # Update tree with optimized values
        optimized_params = best_params.detach().cpu().numpy()
        for i, leaf_id in enumerate(unique_leaves):
            tree_model.tree_.value[leaf_id] = optimized_params[i]
        
        print(f"Final loss: {best_loss:.6f}")
        return tree_model, loss_history


