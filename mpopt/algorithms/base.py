import numpy as np

class BaseAlg(object):

    def __init__(self):
        self.set_params(self.default_params)
    
    def default_params(self, benchmark=None):
        raise NotImplementedError

    def set_params(self, params):
        for param in params:
            setattr(self, param, params[param])
    
    def optimize(self, evaluator):
        raise NotImplementedError

    def init(self):
        raise NotImplementedError
    
    