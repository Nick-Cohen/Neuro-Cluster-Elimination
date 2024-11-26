import torch
from inference.graphical_model import FastGM

class TestProblem:
    def __init__(self, config, device = 'cuda') -> None:
        self.name = config['name']
        self.width = config['width']
        self.uai_file = config['uai_file']
        self.Z = config['Z']
        self.interesting_buckets = config['interesting_buckets']
        self.device = device
        self.gm = None
    
    def load(self, device = None, doping=-5):
        if device == None:
            device = self.device
        self.gm = FastGM(uai_file=self.uai_file, device=device)
        self.gm.dope_factors(doping)

grid20x20_f2 = {
    "name": "grid20x20.f2",
    "width": 20,
    "uai_file": "/home/cohenn1/NCE/problems/width20-30/grid20x20.f2.uai",
    "Z": 291.7326354980469,
    "interesting_buckets": [114, 217, 93, 304, 30, 106, 285, 283, 308, 49, 175, 156, 179, 198, 177, 213, 194, 211, 232, 169, 251, 190, 207, 249, 289, 270, 331, 352, 142, 123, 144]
}
pedigree18 = {
    "name": "pedigree18",
    "width": 18,
    "uai_file": "/home/cohenn1/NCE/problems/width20-30/pedigree18.uai",
    "Z": -78.13997650146484, #pedigree18 doped -5
    "interesting_buckets": [64, 716, 37]
}
rbm_20 = {
    "name": "rbm_20",
    "width": 20,
    "uai_file": "/home/cohenn1/NCE/problems/width20-30/rbm_20.uai",
    "Z": 58.5306282043457, # doped -5
    "interesting_buckets": [28, 30, 35, 27, 33, 29, 23, 36, 31, 25, 24, 39, 37, 21, 26, 38, 34, 22, 32, 2]
}

test_problems = [TestProblem(grid20x20_f2), TestProblem(pedigree18), TestProblem(rbm_20)]


