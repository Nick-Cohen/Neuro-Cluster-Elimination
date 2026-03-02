"""
Interface to pyGMs WMB for backward message approximations.

This module provides functions to compute WMB backward messages using
Alex's pyGMs implementation, which can then be used in NCE's training.
"""

import sys
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any

# Add pyGMs to path if not already there
PYGMS_PATH = '/home/cohenn1/SDBE/PyGMs'
if PYGMS_PATH not in sys.path:
    sys.path.insert(0, PYGMS_PATH)

import pyGMs as gm
import pyGMs.wmb

from ..inference.factor import FastFactor


# Conversion constant
LN10 = np.log(10)


def nce_factors_to_pygms(factors: List[FastFactor], num_vars: int) -> gm.GraphModel:
    """
    Convert NCE FastFactors to a pyGMs GraphModel.

    Args:
        factors: List of NCE FastFactor objects (in log10 space)
        num_vars: Total number of variables in the model

    Returns:
        pyGMs GraphModel with factors in natural log space
    """
    # Create pyGMs variables
    # Assume all variables are binary (states=2) - adjust if needed
    X = [gm.Var(i, 2) for i in range(num_vars)]

    pygms_factors = []
    for ff in factors:
        if not ff.labels:
            # Scalar factor - pyGMs handles these differently
            # Convert log10 to ln and create a constant factor
            val_ln = ff.tensor.item() * LN10
            # Create a factor with no variables (constant)
            pf = gm.Factor([], [val_ln])
            pygms_factors.append(pf)
        else:
            # Get variable indices from labels
            var_indices = [v if isinstance(v, int) else v.label for v in ff.labels]
            vars_list = [X[i] for i in var_indices]

            # Convert tensor from log10 to natural log
            table_log10 = ff.tensor.cpu().numpy()
            table_ln = table_log10 * LN10

            # Create pyGMs factor
            # pyGMs expects table in a specific order based on variable ordering
            pf = gm.Factor(vars_list, table_ln.flatten())
            pygms_factors.append(pf)

    model = gm.GraphModel(pygms_factors)
    model.makePositive(1e-10)

    return model


def pygms_factor_to_nce(pf: gm.Factor, device: str = 'cpu') -> FastFactor:
    """
    Convert a pyGMs Factor to an NCE FastFactor.

    Args:
        pf: pyGMs Factor (in natural log space)
        device: Device for the tensor

    Returns:
        NCE FastFactor in log10 space
    """
    # Get variable labels
    labels = [v.label for v in pf.vars]

    # Convert from natural log to log10
    table_ln = pf.t
    table_log10 = table_ln / LN10

    # Create tensor with appropriate shape
    if labels:
        # Reshape based on variable dimensions
        shape = tuple(v.states for v in pf.vars)
        tensor = torch.tensor(table_log10.reshape(shape), dtype=torch.float32, device=device)
    else:
        # Scalar factor
        tensor = torch.tensor(float(table_log10.flatten()[0]), dtype=torch.float32, device=device)

    return FastFactor(tensor, labels)


class PyGMsWMBBackward:
    """
    Wrapper for pyGMs WMB backward message computation.

    This class sets up a pyGMs WMB inference engine and provides
    methods to extract backward messages for specific buckets.

    Supports weight optimization through:
    - setWeights(): Set per-variable weights
    - optimize_weights(): Iterative bound tightening via GDD
    """

    def __init__(
        self,
        model_file: Optional[str] = None,
        factors: Optional[List[FastFactor]] = None,
        elim_order: Optional[List[int]] = None,
        num_vars: Optional[int] = None,
        iB: int = 10,
        weights: float = 1.0,
        device: str = 'cpu',
        optimize: bool = False,
        num_gdd_iterations: int = 0
    ):
        """
        Initialize the pyGMs WMB backward interface.

        Args:
            model_file: Path to UAI model file (preferred - loads directly)
            factors: List of NCE FastFactors (alternative if no model_file)
            elim_order: Elimination order (list of variable indices)
            num_vars: Total number of variables (required if using factors)
            iB: Mini-bucket i-bound
            weights: WMB weight parameter. Can be:
                     - float: uniform weight for all variables (1.0 = upper bound)
                     - list: per-variable weights
                     - 'sum+': upper bound (equivalent to 1.0)
                     - 'sum-': lower bound (equivalent to -1.0)
            device: Device for output tensors
            optimize: If True, run weight optimization after initial forward/backward
            num_gdd_iterations: Number of GDD iterations for bound tightening (0=none)
        """
        self.iB = iB
        self.initial_weights = weights
        self.device = device
        self.model_file = model_file

        if model_file is not None:
            # Load directly from UAI file (preferred method)
            factors_raw = gm.readUai(model_file)
            self.pygms_model = gm.GraphModel(factors_raw)
            self.pygms_model.makePositive(1e-10)
            self.num_vars = len(self.pygms_model.X)
        elif factors is not None:
            # Convert NCE factors to pyGMs model
            if num_vars is None:
                raise ValueError("num_vars required when using factors")
            self.num_vars = num_vars
            self.pygms_model = nce_factors_to_pygms(factors, num_vars)
        else:
            raise ValueError("Either model_file or factors must be provided")

        if elim_order is None:
            # Use default order if not specified
            self.elim_order = list(range(self.num_vars))
        else:
            self.elim_order = list(elim_order)

        # Create WMB inference engine
        self.wmb = gm.wmb.WMB(
            self.pygms_model,
            self.elim_order,
            iBound=iB,
            weights=weights
        )

        # Run forward and backward passes
        self.lnZ = self.wmb.msgForward(0., 0.)  # No reparameterization
        self.wmb.msgBackward(0., 0.)

        # Initialize heuristic to create atElim structure (needed for proper backward message extraction)
        # atElim maps each variable to the mini-buckets that were created when it was eliminated
        pt = gm.PseudoTree(self.pygms_model, self.elim_order)
        self.wmb.initHeuristic(pt)

        # Optionally optimize weights
        if optimize or num_gdd_iterations > 0:
            self.optimize_weights(num_iterations=max(1, num_gdd_iterations))

        # Store bucket index mapping: var -> bucket_idx
        self.var_to_bucket_idx = {var: idx for idx, var in enumerate(self.elim_order)}

        # Track optimization history
        self.optimization_history = []

    def set_weights(self, weights) -> float:
        """
        Set new weights and recompute forward/backward messages.

        Args:
            weights: Can be:
                - float: uniform weight for all variables
                - list: per-variable weights (length = num_vars)
                - 'sum+': upper bound (1.0)
                - 'sum-': lower bound (-1.0)

        Returns:
            New lnZ estimate (natural log)
        """
        self.wmb.setWeights(weights)
        self.lnZ = self.wmb.msgForward(0., 0.)
        self.wmb.msgBackward(0., 0.)
        return self.lnZ

    def optimize_weights(
        self,
        num_iterations: int = 5,
        gdd_steps: int = 5,
        threshold: float = 0.01,
        verbose: bool = False
    ) -> List[float]:
        """
        Optimize weights using iterative GDD (Generalized Dual Decomposition).

        This performs bound tightening by reparameterizing factors.

        Args:
            num_iterations: Number of forward-GDD-backward iterations
            gdd_steps: Steps per GDD update
            threshold: Convergence threshold for GDD
            verbose: Print progress

        Returns:
            List of lnZ values at each iteration (natural log)
        """
        history = [self.lnZ]

        for i in range(num_iterations):
            # Reparameterize (incorporate messages into factors)
            self.wmb.reparameterize()

            # GDD update to tighten bound
            self.wmb.gdd_update(maxstep=1.0, threshold=threshold)

            # Recompute forward and backward
            self.lnZ = self.wmb.msgForward(0., 0.)
            self.wmb.msgBackward(0., 0.)

            history.append(self.lnZ)

            if verbose:
                improvement = history[-2] - history[-1]
                print(f"  GDD iter {i+1}: lnZ = {self.lnZ:.4f} (improvement: {improvement:.4f})")

            # Check convergence
            if len(history) > 1 and abs(history[-1] - history[-2]) < threshold:
                if verbose:
                    print(f"  Converged at iteration {i+1}")
                break

        self.optimization_history = history
        return history

    def learn_weights(
        self,
        num_iterations: int = 10,
        step_theta: float = 0.5,
        step_weights: float = 0.1,
        verbose: bool = False
    ) -> List[float]:
        """
        Learn optimal weights using entropy-based gradient descent.

        This uses pyGMs' built-in weight learning in msgForward/msgBackward,
        which updates weights based on the entropy of mini-bucket beliefs.
        Weights are pushed toward mini-buckets with lower entropy (more peaked).

        The update formula is:
            w_j *= exp(-stepWeights * w_j * (H_j - H_avg))
        where H_j is the entropy of mini-bucket j's belief.

        Args:
            num_iterations: Number of forward-backward iterations
            step_theta: Step size for reparameterization (moment matching)
            step_weights: Step size for weight updates (0 = no weight learning)
            verbose: Print progress

        Returns:
            List of lnZ values at each iteration (natural log)
        """
        history = [self.lnZ]

        if verbose:
            print(f"Learning weights: stepTheta={step_theta}, stepWeights={step_weights}")
            print(f"  Initial lnZ = {self.lnZ:.4f} ({self.lnZ/LN10:.4f} log10)")

        for i in range(num_iterations):
            # Forward pass with weight learning
            self.lnZ = self.wmb.msgForward(step_theta, step_weights)

            # Backward pass (typically without weight updates)
            self.wmb.msgBackward(0., 0.)

            history.append(self.lnZ)

            if verbose:
                improvement = history[-2] - history[-1]
                print(f"  Iter {i+1}: lnZ = {self.lnZ:.4f} ({self.lnZ/LN10:.4f} log10), "
                      f"improvement: {improvement:.4f}")

            # Check convergence
            if len(history) > 1 and abs(history[-1] - history[-2]) < 1e-6:
                if verbose:
                    print(f"  Converged at iteration {i+1}")
                break

        self.optimization_history = history
        return history

    def learn_weights_with_reparameterization(
        self,
        num_outer_iterations: int = 5,
        num_inner_iterations: int = 5,
        step_theta: float = 0.5,
        step_weights: float = 0.1,
        verbose: bool = False
    ) -> List[float]:
        """
        Learn weights with periodic reparameterization for better convergence.

        This alternates between:
        1. Weight learning via forward/backward passes
        2. Reparameterization to incorporate learned structure

        Args:
            num_outer_iterations: Number of reparameterization cycles
            num_inner_iterations: Forward-backward iterations per cycle
            step_theta: Step size for reparameterization
            step_weights: Step size for weight updates
            verbose: Print progress

        Returns:
            List of lnZ values at each iteration (natural log)
        """
        history = [self.lnZ]

        if verbose:
            print(f"Learning weights with reparameterization")
            print(f"  Initial lnZ = {self.lnZ:.4f} ({self.lnZ/LN10:.4f} log10)")

        for outer in range(num_outer_iterations):
            # Inner loop: weight learning
            for inner in range(num_inner_iterations):
                self.lnZ = self.wmb.msgForward(step_theta, step_weights)
                self.wmb.msgBackward(0., 0.)
                history.append(self.lnZ)

            # Reparameterize to lock in improvements
            self.wmb.reparameterize()

            if verbose:
                improvement = history[-num_inner_iterations-1] - history[-1]
                print(f"  Outer {outer+1}: lnZ = {self.lnZ:.4f} ({self.lnZ/LN10:.4f} log10), "
                      f"cycle improvement: {improvement:.4f}")

            # Check convergence
            if outer > 0 and abs(history[-1] - history[-num_inner_iterations-1]) < 1e-4:
                if verbose:
                    print(f"  Converged at outer iteration {outer+1}")
                break

        self.optimization_history = history
        return history

    def get_learned_weights_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current learned weights across all buckets.

        Returns:
            Dictionary with weight statistics
        """
        all_weights = []
        bucket_info = []

        for bucket_idx, bucket_var in enumerate(self.elim_order):
            mini_buckets = self.wmb.buckets[bucket_idx]
            if len(mini_buckets) > 1:  # Only interesting if multiple mini-buckets
                weights = [mb.weight for mb in mini_buckets]
                all_weights.extend(weights)
                bucket_info.append({
                    'bucket_idx': bucket_idx,
                    'var': bucket_var,
                    'num_mini_buckets': len(mini_buckets),
                    'weights': weights,
                    'weight_range': (min(weights), max(weights)),
                })

        return {
            'num_buckets_with_splits': len(bucket_info),
            'total_mini_buckets': len(all_weights),
            'weight_mean': float(np.mean(all_weights)) if all_weights else 1.0,
            'weight_std': float(np.std(all_weights)) if all_weights else 0.0,
            'weight_range': (float(min(all_weights)), float(max(all_weights))) if all_weights else (1.0, 1.0),
            'bucket_details': bucket_info,
        }

    def try_different_weights(
        self,
        weight_values: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5]
    ) -> Dict[float, float]:
        """
        Try different uniform weight values and return lnZ for each.

        This helps find the best weight setting for a given problem.

        Args:
            weight_values: List of weight values to try

        Returns:
            Dictionary mapping weight -> lnZ (natural log)
        """
        results = {}

        for w in weight_values:
            lnZ = self.set_weights(w)
            results[w] = lnZ

        # Restore to best weight (tightest bound for upper bound problem)
        best_weight = min(results, key=results.get)
        self.set_weights(best_weight)

        return results

    def get_mini_bucket_weights(self, bucket_var: int) -> List[float]:
        """
        Get the weights of mini-buckets at a specific bucket.

        Args:
            bucket_var: Variable index

        Returns:
            List of weights for each mini-bucket
        """
        if bucket_var not in self.var_to_bucket_idx:
            return []

        bucket_idx = self.var_to_bucket_idx[bucket_var]
        mini_buckets = self.wmb.buckets[bucket_idx]

        return [mb.weight for mb in mini_buckets]

    def find_optimal_weight(
        self,
        search_range: Tuple[float, float] = (0.1, 2.0),
        num_samples: int = 20,
        refine: bool = True
    ) -> Tuple[float, float]:
        """
        Find the optimal uniform weight that minimizes the bound gap.

        Uses grid search followed by optional local refinement.

        Args:
            search_range: (min_weight, max_weight) to search
            num_samples: Number of samples in initial grid search
            refine: If True, refine around best weight

        Returns:
            Tuple of (best_weight, best_lnZ)
        """
        # Grid search
        weights = np.linspace(search_range[0], search_range[1], num_samples)
        results = {}

        for w in weights:
            w_float = float(w)
            lnZ = self.set_weights(w_float)
            results[w_float] = lnZ

        # Find best (minimum lnZ for upper bound)
        best_weight = min(results, key=results.get)
        best_lnZ = results[best_weight]

        # Refine around best
        if refine:
            step = (search_range[1] - search_range[0]) / num_samples
            refined_weights = np.linspace(
                max(search_range[0], best_weight - step),
                min(search_range[1], best_weight + step),
                10
            )
            for w in refined_weights:
                w_float = float(w)
                lnZ = self.set_weights(w_float)
                if lnZ < best_lnZ:
                    best_lnZ = lnZ
                    best_weight = w_float

        # Set to best weight (ensure it's a Python float)
        self.set_weights(float(best_weight))

        return float(best_weight), best_lnZ

    def full_optimization(
        self,
        find_best_weight: bool = False,
        learn_weights: bool = True,
        num_learning_iterations: int = 20,
        num_gdd_iterations: int = 3,
        step_theta: float = 0.5,
        step_weights: float = 0.1,
        use_reparameterization: bool = False,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run full optimization: learn weights + GDD tightening.

        The optimization proceeds in stages:
        1. Grid search for best uniform initial weight (optional, often not needed)
        2. Entropy-based weight learning via forward/backward passes
        3. GDD reparameterization for final tightening (optional)

        Note: Grid search is disabled by default because weight learning from
        weights=1.0 typically gives better results. Enable it only if you want
        to compare against a fixed-weight baseline.

        Args:
            find_best_weight: If True, grid search for optimal uniform weight
                              (runs INSTEAD of learning, not before)
            learn_weights: If True, run entropy-based weight learning
            num_learning_iterations: Number of weight learning iterations
            num_gdd_iterations: Number of GDD iterations
            step_theta: Step size for theta reparameterization during learning
            step_weights: Step size for weight updates during learning
            use_reparameterization: If True, use learn_weights_with_reparameterization
                                    for more stable but slower optimization
            verbose: Print progress

        Returns:
            Dictionary with optimization results including:
            - initial_lnZ: Starting bound
            - final_lnZ: Final bound after all optimization
            - total_improvement: Reduction in bound (positive = tighter)
            - learned_weights_summary: Statistics on learned weights
        """
        results = {
            'initial_lnZ': self.lnZ,
            'initial_lnZ_log10': self.lnZ / LN10,
            'initial_weight': self.initial_weights,
        }

        if find_best_weight and not learn_weights:
            # Grid search only (no learning)
            if verbose:
                print("Stage 1: Finding optimal uniform weight (grid search)...")
            best_w, best_lnZ = self.find_optimal_weight()
            results['best_weight'] = best_w
            results['lnZ_after_weight_search'] = best_lnZ
            results['lnZ_after_weight_search_log10'] = best_lnZ / LN10
            if verbose:
                print(f"  Best weight: {best_w:.4f}, lnZ: {best_lnZ/LN10:.4f} (log10)")

        if learn_weights and num_learning_iterations > 0:
            if verbose:
                stage = "Stage 1" if not find_best_weight else "Stage 2"
                print(f"{stage}: Learning weights ({num_learning_iterations} iterations)...")

            if use_reparameterization:
                # More stable learning with periodic reparameterization
                num_outer = max(1, num_learning_iterations // 3)
                num_inner = 3
                history = self.learn_weights_with_reparameterization(
                    num_outer_iterations=num_outer,
                    num_inner_iterations=num_inner,
                    step_theta=step_theta,
                    step_weights=step_weights,
                    verbose=verbose
                )
            else:
                # Standard learning
                history = self.learn_weights(
                    num_iterations=num_learning_iterations,
                    step_theta=step_theta,
                    step_weights=step_weights,
                    verbose=verbose
                )
            results['learning_history'] = history
            results['lnZ_after_learning'] = self.lnZ
            results['lnZ_after_learning_log10'] = self.lnZ / LN10
            results['learning_improvement'] = history[0] - history[-1]
            if verbose:
                print(f"  Learning improvement: {(history[0] - history[-1])/LN10:.4f} log10")

        if num_gdd_iterations > 0:
            if verbose:
                print(f"Stage 3: GDD tightening ({num_gdd_iterations} iterations)...")
            history = self.optimize_weights(num_gdd_iterations, verbose=verbose)
            results['gdd_history'] = history
            results['lnZ_after_gdd'] = self.lnZ
            results['lnZ_after_gdd_log10'] = self.lnZ / LN10

        results['final_lnZ'] = self.lnZ
        results['final_lnZ_log10'] = self.lnZ / LN10
        results['total_improvement'] = results['initial_lnZ'] - self.lnZ
        results['total_improvement_log10'] = (results['initial_lnZ'] - self.lnZ) / LN10

        # Get summary of learned weights
        results['learned_weights_summary'] = self.get_learned_weights_summary()

        if verbose:
            print(f"\nOptimization complete:")
            print(f"  Initial: {results['initial_lnZ_log10']:.4f} log10")
            print(f"  Final:   {results['final_lnZ_log10']:.4f} log10")
            print(f"  Total improvement: {results['total_improvement_log10']:.4f} log10")

        return results

    def get_backward_message(
        self,
        bucket_var: int,
        return_factor_list: bool = False
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Get the WMB backward message for a specific bucket.

        Args:
            bucket_var: The variable whose bucket's backward message to retrieve
            return_factor_list: If True, returns list of factors instead of their sum.
                               This is useful for large scope messages where we want
                               to sample from the product without materializing it.

        Returns:
            Tuple of (backward_message, info_dict)
            - If return_factor_list=False: backward_message is a single FastFactor (sum of all msgBwd)
            - If return_factor_list=True: backward_message is a List[FastFactor] (one per mini-bucket)
            - info_dict: Dictionary with metadata (scope, num_mini_buckets, weights, etc.)
        """
        if bucket_var not in self.var_to_bucket_idx:
            raise ValueError(f"Variable {bucket_var} not in elimination order")

        bucket_idx = self.var_to_bucket_idx[bucket_var]

        # Get mini-buckets at this bucket AND their parents
        # This follows the pattern from check-nick-bwd.ipynb:
        #   all_msgs = set(mbBack.buckets[bucket_idx]) | set(mini.parent for mini in mbBack.atElim[var])
        bucket_mini_buckets = set(self.wmb.buckets[bucket_idx])

        # Get parent mini-buckets for all mini-buckets at the eliminated variable
        # atElim[var] contains all mini-buckets created when variable 'var' was eliminated
        parent_mini_buckets = set()
        if bucket_var < len(self.wmb.atElim) and self.wmb.atElim[bucket_var]:
            for mini in self.wmb.atElim[bucket_var]:
                if mini.parent is not None:
                    parent_mini_buckets.add(mini.parent)

        mini_buckets = list(bucket_mini_buckets | parent_mini_buckets)

        if not mini_buckets:
            # Empty bucket - return zero factor
            empty_factor = FastFactor(torch.tensor(0.0, device=self.device), [])
            info = {
                'num_mini_buckets': 0,
                'scope': [],
                'weights': [],
                'factor_scopes': [],
            }
            if return_factor_list:
                return [empty_factor], info
            else:
                return empty_factor, info

        # Collect info about mini-buckets
        info = {
            'num_mini_buckets': len(mini_buckets),
            'weights': [mb.weight for mb in mini_buckets],
            'clique_sizes': [len(mb.clique) for mb in mini_buckets],
            'lnZ_estimate': self.lnZ / LN10,  # Convert to log10
        }

        if return_factor_list:
            # Return list of factors (one per mini-bucket's msgBwd)
            # This allows sampling without materializing the full product
            factor_list = []
            factor_scopes = []

            for mb in mini_buckets:
                # Convert each mini-bucket's msgBwd to NCE FastFactor
                bw_factor = pygms_factor_to_nce(mb.msgBwd, self.device)
                factor_list.append(bw_factor)
                factor_scopes.append([v.label for v in mb.msgBwd.vars])

            info['factor_scopes'] = factor_scopes
            info['scope'] = list(set(v for scope in factor_scopes for v in scope))

            return factor_list, info
        else:
            # Return single factor (sum of all msgBwd in log space = product in prob space)
            back_sum = sum([mb.msgBwd for mb in mini_buckets])
            bw_factor = pygms_factor_to_nce(back_sum, self.device)
            info['scope'] = [v.label for v in back_sum.vars]

            return bw_factor, info

    def get_backward_factor_list(self, bucket_var: int) -> Tuple[List[FastFactor], Dict[str, Any]]:
        """
        Get the WMB backward message as a list of factors (convenience method).

        This is equivalent to get_backward_message(bucket_var, return_factor_list=True).

        Each factor in the list corresponds to one mini-bucket's backward message.
        The full backward message is the product of these factors (sum in log space).

        This is useful for:
        - Large scope messages where materializing the product is too expensive
        - Sampling from the backward distribution without explicit product
        - Understanding the structure of the WMB approximation

        Args:
            bucket_var: The variable whose bucket's backward message to retrieve

        Returns:
            Tuple of (factor_list, info_dict)
            - factor_list: List of FastFactors, one per mini-bucket
            - info_dict: Dictionary with metadata including per-factor scopes
        """
        return self.get_backward_message(bucket_var, return_factor_list=True)

    def get_all_backward_messages(self) -> Dict[int, Tuple[FastFactor, Dict[str, Any]]]:
        """
        Get backward messages for all buckets.

        Returns:
            Dictionary mapping bucket_var -> (backward_message, info)
        """
        results = {}
        for var in self.elim_order:
            results[var] = self.get_backward_message(var)
        return results


def get_pygms_backward_message(
    fastgm,
    bucket_var: int,
    iB: int = 10,
    weights: float = 1.0,
    model_file: Optional[str] = None
) -> Tuple[FastFactor, Dict[str, Any]]:
    """
    Convenience function to get pyGMs WMB backward message for an NCE FastGM.

    This function creates a pyGMs WMB engine and returns the backward message
    for the specified bucket.

    Args:
        fastgm: NCE FastGM object
        bucket_var: Variable whose backward message to compute
        iB: Mini-bucket i-bound
        weights: WMB weight parameter
        model_file: Path to UAI file (preferred). If None, tries to get from fastgm.

    Returns:
        Tuple of (backward_message_factor, info_dict)
    """
    # Try to get model file path
    if model_file is None:
        # Try to get from fastgm's model
        if hasattr(fastgm, 'model') and hasattr(fastgm.model, 'file'):
            model_file = fastgm.model.file
        elif hasattr(fastgm, 'model_file'):
            model_file = fastgm.model_file

    if model_file is None:
        raise ValueError("Could not determine model file path. Please provide model_file argument.")

    # Create interface using model file (preferred method)
    interface = PyGMsWMBBackward(
        model_file=model_file,
        elim_order=list(fastgm.elim_order),
        iB=iB,
        weights=weights,
        device=str(fastgm.device)
    )

    return interface.get_backward_message(bucket_var)


def create_pygms_wmb_interface(
    model_file: str,
    elim_order: List[int],
    iB: int = 10,
    weights: float = 1.0,
    device: str = 'cpu'
) -> PyGMsWMBBackward:
    """
    Create a pyGMs WMB interface from a model file.

    This is the recommended way to create the interface - it loads
    the model directly from the UAI file, ensuring correct factor values.

    Args:
        model_file: Path to UAI model file
        elim_order: Elimination order
        iB: Mini-bucket i-bound
        weights: WMB weight parameter
        device: Device for output tensors

    Returns:
        PyGMsWMBBackward interface object
    """
    return PyGMsWMBBackward(
        model_file=model_file,
        elim_order=elim_order,
        iB=iB,
        weights=weights,
        device=device
    )


def compare_backward_messages(
    nce_backward: FastFactor,
    pygms_backward: FastFactor,
    name: str = "backward"
) -> Dict[str, float]:
    """
    Compare NCE and pyGMs backward messages.

    Args:
        nce_backward: NCE's backward message (log10 space)
        pygms_backward: pyGMs backward message (log10 space, after conversion)
        name: Name for logging

    Returns:
        Dictionary with comparison statistics
    """
    nce_flat = nce_backward.tensor.cpu().numpy().flatten()
    pygms_flat = pygms_backward.tensor.cpu().numpy().flatten()

    if len(nce_flat) != len(pygms_flat):
        return {
            'size_match': False,
            'nce_size': len(nce_flat),
            'pygms_size': len(pygms_flat)
        }

    # Sort both for comparison (in case variable ordering differs)
    nce_sorted = np.sort(nce_flat)
    pygms_sorted = np.sort(pygms_flat)

    diff = pygms_sorted - nce_sorted

    return {
        'size_match': True,
        'max_diff': float(np.max(np.abs(diff))),
        'mean_diff': float(np.mean(diff)),
        'mean_abs_diff': float(np.mean(np.abs(diff))),
        'std_diff': float(np.std(diff)),
        'nce_mean': float(np.mean(nce_flat)),
        'pygms_mean': float(np.mean(pygms_flat)),
        'nce_range': (float(np.min(nce_flat)), float(np.max(nce_flat))),
        'pygms_range': (float(np.min(pygms_flat)), float(np.max(pygms_flat))),
    }


class PyGMsBackwardProvider:
    """
    Provider class that supplies pyGMs WMB backward messages for NCE training.

    This can be used to optionally replace NCE's backward computation with
    pyGMs backward messages during training.

    Supports weight optimization for tighter bounds.

    Usage:
        provider = PyGMsBackwardProvider(model_file, elim_order, iB=10)

        # Optionally optimize weights
        provider.optimize()

        # In training loop:
        backward_msg = provider.get_backward(bucket_var)
    """

    def __init__(
        self,
        model_file: str,
        elim_order: List[int],
        iB: int = 10,
        weights: float = 1.0,
        device: str = 'cpu',
        cache_messages: bool = True,
        optimize_weights: bool = False,
        learn_weights: bool = False,
        num_learning_iterations: int = 10,
        num_gdd_iterations: int = 0,
        verbose: bool = False
    ):
        """
        Initialize the backward provider.

        Args:
            model_file: Path to UAI model file
            elim_order: Elimination order
            iB: Mini-bucket i-bound
            weights: WMB weight parameter (or 'auto' to find optimal and learn)
            device: Device for output tensors
            cache_messages: If True, cache computed backward messages
            optimize_weights: If True, find optimal initial weight via grid search
            learn_weights: If True, learn weights via entropy-based gradient descent
            num_learning_iterations: Number of weight learning iterations
            num_gdd_iterations: Number of GDD iterations for bound tightening
            verbose: Print optimization progress
        """
        self.model_file = model_file
        self.elim_order = elim_order
        self._iB = iB
        self.device = device
        self.cache_messages = cache_messages

        # Initialize cache first (before optimize which may clear it)
        self._cache: Dict[int, FastFactor] = {}

        # Create interface
        self.interface = PyGMsWMBBackward(
            model_file=model_file,
            elim_order=elim_order,
            iB=iB,
            weights=1.0 if weights == 'auto' else weights,
            device=device
        )

        # Optimize if requested
        should_optimize = optimize_weights or weights == 'auto' or learn_weights or num_gdd_iterations > 0
        if should_optimize:
            self.optimize(
                find_best_weight=(optimize_weights or weights == 'auto'),
                learn_weights=(learn_weights or weights == 'auto'),
                num_learning_iterations=num_learning_iterations,
                num_gdd_iterations=num_gdd_iterations,
                verbose=verbose
            )

        # Pre-cache all backward messages if requested
        if cache_messages:
            self._precompute_all()

    def optimize(
        self,
        find_best_weight: bool = True,
        learn_weights: bool = True,
        num_learning_iterations: int = 10,
        num_gdd_iterations: int = 5,
        step_theta: float = 0.5,
        step_weights: float = 0.1,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run full weight optimization: grid search + learning + GDD.

        Args:
            find_best_weight: Search for optimal initial uniform weight
            learn_weights: Learn weights via entropy-based gradient descent
            num_learning_iterations: Number of weight learning iterations
            num_gdd_iterations: Number of GDD iterations
            step_theta: Step size for theta reparameterization
            step_weights: Step size for weight updates
            verbose: Print progress

        Returns:
            Optimization results dictionary
        """
        results = self.interface.full_optimization(
            find_best_weight=find_best_weight,
            learn_weights=learn_weights,
            num_learning_iterations=num_learning_iterations,
            num_gdd_iterations=num_gdd_iterations,
            step_theta=step_theta,
            step_weights=step_weights,
            verbose=verbose
        )

        # Clear cache since backward messages changed
        self._cache.clear()

        return results

    def _precompute_all(self):
        """Pre-compute and cache all backward messages (both single and list forms)."""
        for var in self.interface.elim_order:
            bw_msg, _ = self.interface.get_backward_message(var, return_factor_list=False)
            self._cache[var] = bw_msg

        # Also cache factor lists
        self._factor_list_cache: Dict[int, List[FastFactor]] = {}
        for var in self.interface.elim_order:
            factor_list, _ = self.interface.get_backward_message(var, return_factor_list=True)
            self._factor_list_cache[var] = factor_list

    def get_backward(
        self,
        bucket_var: int,
        return_factor_list: bool = False
    ) -> Any:
        """
        Get backward message for a bucket (from cache or compute).

        Args:
            bucket_var: Variable whose backward message to retrieve
            return_factor_list: If True, return list of factors instead of single product.
                               Use this for large scope messages to avoid materializing
                               the full product in memory.

        Returns:
            If return_factor_list=False: FastFactor containing the backward message
            If return_factor_list=True: List[FastFactor] that together comprise the backward
        """
        if return_factor_list:
            # Check factor list cache
            if self.cache_messages and hasattr(self, '_factor_list_cache') and bucket_var in self._factor_list_cache:
                return self._factor_list_cache[bucket_var]

            factor_list, _ = self.interface.get_backward_message(bucket_var, return_factor_list=True)

            if self.cache_messages:
                if not hasattr(self, '_factor_list_cache'):
                    self._factor_list_cache = {}
                self._factor_list_cache[bucket_var] = factor_list

            return factor_list
        else:
            # Check single-factor cache
            if self.cache_messages and bucket_var in self._cache:
                return self._cache[bucket_var]

            bw_msg, _ = self.interface.get_backward_message(bucket_var, return_factor_list=False)

            if self.cache_messages:
                self._cache[bucket_var] = bw_msg

            return bw_msg

    def get_backward_factor_list(self, bucket_var: int) -> List[FastFactor]:
        """
        Get backward message as a list of factors (convenience method).

        This returns the individual mini-bucket backward messages without
        multiplying them together. Useful for:
        - Large scope messages where full product is too expensive
        - Sampling from the backward distribution
        - Understanding WMB structure

        Args:
            bucket_var: Variable whose backward message to retrieve

        Returns:
            List of FastFactors that together comprise the backward message
            (their product in probability space = sum in log space)
        """
        return self.get_backward(bucket_var, return_factor_list=True)

    def get_backward_info(self, bucket_var: int) -> Dict[str, Any]:
        """
        Get metadata about the backward message structure.

        Args:
            bucket_var: Variable whose backward info to retrieve

        Returns:
            Dictionary with:
            - num_mini_buckets: Number of mini-buckets contributing
            - weights: Weight of each mini-bucket
            - factor_scopes: Scope of each factor in the list
            - scope: Combined scope of the full backward message
        """
        _, info = self.interface.get_backward_message(bucket_var, return_factor_list=True)
        return info

    def get_lnZ_estimate(self) -> float:
        """Get the WMB log partition function estimate (log10 space)."""
        return self.interface.lnZ / LN10

    @property
    def num_vars(self) -> int:
        """Number of variables in the model."""
        return self.interface.num_vars

    @property
    def iB(self) -> int:
        """Mini-bucket i-bound."""
        return self.interface.iB


def create_backward_provider(
    fastgm,
    iB: int = 10,
    weights: float = 1.0,
    cache_messages: bool = True
) -> PyGMsBackwardProvider:
    """
    Create a PyGMsBackwardProvider from an NCE FastGM.

    This is a convenience function to set up pyGMs backward messages
    that can be used during NCE training.

    Args:
        fastgm: NCE FastGM object (must have model.file attribute)
        iB: Mini-bucket i-bound
        weights: WMB weight parameter
        cache_messages: If True, pre-compute and cache all backward messages

    Returns:
        PyGMsBackwardProvider object

    Example:
        provider = create_backward_provider(fastgm, iB=10)

        # During training, use pyGMs backward instead of NCE:
        for bucket_var in training_buckets:
            pygms_backward = provider.get_backward(bucket_var)
            # Use pygms_backward for loss computation...
    """
    # Get model file
    model_file = None
    if hasattr(fastgm, 'model') and hasattr(fastgm.model, 'file'):
        model_file = fastgm.model.file
    elif hasattr(fastgm, 'model_file'):
        model_file = fastgm.model_file

    if model_file is None:
        raise ValueError("Could not determine model file path from FastGM")

    return PyGMsBackwardProvider(
        model_file=model_file,
        elim_order=list(fastgm.elim_order),
        iB=iB,
        weights=weights,
        device=str(fastgm.device),
        cache_messages=cache_messages
    )
