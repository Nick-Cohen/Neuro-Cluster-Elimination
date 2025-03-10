from .factor import FastFactor
from .factor_nn import FactorNN
from typing import List
import numpy as np

class FastBucket:
    
    # TODO: Multiply factors in a more sensible order, e.g. subsumed multiplications first
    
    def __init__(self, gm, label, factors, device, elim_vars, isRoot = False):
        self.gm = gm
        self.config = gm.config
        self.label = label
        self.factors = factors
        self.device = device
        self.elim_vars = elim_vars
        self.isRoot = isRoot
        self.approximate_downstream_factors: List[FastFactor] = None

        # Assert that all factors are on the specified device type
        for factor in self.factors:
            assert self.device in str(factor.device), f"Factor device {factor.device} does not match bucket device type {self.device}"
    
    def compute_message_nn(self):
        from NCE.neural_networks.net import Net
        from NCE.neural_networks.train import Trainer
        net = Net(self)
        t=Trainer(net=net, bucket=self)
        t.train()
        return FactorNN(net, t.data_preprocessor)
    
    def compute_dummy_nn(self):
        from NCE.neural_networks.net import Net, Memorizer
        from NCE.neural_networks.train import Trainer
        net = Net(self)
        t=Trainer(net=net, bucket=self)
        # use trainer to make dataloader
        x_all, y_all, _ = t.dataloader.load(all=True)
        mem = Memorizer(self, x_all, y_all)
        return FactorNN(mem, t.data_preprocessor)
    
    def compute_message_exact(self):
        # Multiply all factors
        if not self.factors:
            raise ValueError("No factors in the bucket to send message from")
        
        if self.factors[0].is_nn:
            message = self.factors[0].to_exact()
            assert message.tensor is not None
        else:
            message = self.factors[0]
        for factor in self.factors[1:]:
            if factor.is_nn:
                factor = factor.to_exact()
            message = message * factor

        # Eliminate variables
        message = message.eliminate(self.elim_vars)
        assert not (message.tensor is None and len(message.labels) > 0), f"{self.label}"
        return message
    
    def compute_wmb_message(self, iB: int, debug=False) -> List[FastFactor]:
        # todo: add weights functionality. Currently just doing mb
        """
        Compute the Weighted Mini-Bucket (WMB) message for the bucket with given i-bound.
        
        Args:
        iB (int): The i-bound parameter for mini-bucket elimination.
        
        Returns:
        List[FastFactor]: The list of factors representing the WMB message.
        """
        # Step 1: Split factors into mini-buckets
        mini_buckets = self._create_mini_buckets(iB)
        if debug:
            print(f"Number of mini-buckets: {len(mini_buckets)}")
        
        # Step 2: Compute weighted elimination for each mini-bucket
        wmb_factors = []
        first_bucket = True
        for mb in mini_buckets:
            if len(mb) == 1:
                wmb_factors.append(mb[0])
            else:
                combined_factor = mb[0]
                for factor in mb[1:]:
                    combined_factor *= factor
                eliminated_factor = combined_factor.eliminate(self.elim_vars) if first_bucket else combined_factor.eliminate(self.elim_vars, elimination_scheme='sum')
                first_bucket = False
                wmb_factors.append(eliminated_factor)
        
        return wmb_factors
    
    def _get_nn_input_size(self):
        dimensions = self.get_message_dimension()
        out = 0
        for nstates in dimensions:
            out += nstates - 1
        return out

    def _create_mini_buckets(self, iB: int) -> List[List[FastFactor]]:
        """
        Create mini-buckets from the factors in the bucket based on the i-bound.
        
        Args:
        iB (int): The i-bound parameter for mini-bucket creation.
        
        Returns:
        List[List[FastFactor]]: A list of mini-buckets, where each mini-bucket is a list of factors.
        """
        mini_buckets = []
        sorted_factors = sorted(self.factors, key=lambda f: len(f.vars), reverse=True)
        
        for factor in sorted_factors:
            placed = False
            for mb in mini_buckets:
                if len(set.union(*[set(f.vars) for f in mb], set(factor.vars))) <= iB:
                    mb.append(factor)
                    placed = True
                    break
            if not placed:
                mini_buckets.append([factor])
        
        return mini_buckets

    def send_message(self, bucket: 'FastBucket'):
        """
        Multiply all factors, eliminate variables, and send the resulting message to another bucket.
        """
        # Multiply all factors
        if not self.factors:
            raise ValueError("No factors in the bucket to send message from")
        
        message = self.factors[0]
        for factor in self.factors[1:]:
            message = message * factor

        # Eliminate variables
        message = message.eliminate(self.elim_vars)

        # Send the message to the receiving bucket
        bucket.receive_message(message)

    def receive_message(self, message: FastFactor):
        """
        Receive a message (factor) from another bucket and append it to this bucket's factors.
        """
        # Assert that the incoming message is on the correct device
        # assert str(self.device) in str(message.device), f"Message device {message.device} does not match bucket device {self.device}"

        # Append the message to the factors list
        self.factors.append(message)
        
    def get_message_scope(self):
        scope = set()
        for factor in self.factors:
            scope = scope.union(factor.labels)
        scope.discard(self.label)
        return sorted(list(scope))
    
    def get_width(self):
        return len(self.get_message_scope())
    
    def get_message_dimension(self):
        return [self.gm.matching_var(idx).states for idx in self.get_message_scope()]
    
    def get_message_size(self):
        scopes = self.get_message_dimension()
        return np.prod(scopes)
     