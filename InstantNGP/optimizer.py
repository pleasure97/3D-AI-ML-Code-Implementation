import math
import torch
from torch.optim.optimizer import Optimizer, required

class RAdam(Optimizer):
    def __init__(self, params, lr = 1e-3, betas = [0.9, 0.999], eps = 1e-8, weight_decay = 0, sgd = False):

        self.sgd = sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['beta'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
