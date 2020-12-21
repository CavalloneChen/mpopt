import os
import time
import numpy as np

from ..population.cmaes import CMAESPop
from ..operator import operator as opt
from ..tools.distribution import MultiVariateNormalDistribution as MVND

EPS = 1e-8

class CMAES(object):

    def __init__(self):
        # params
        self.pop_size = None
        self.init_scale = None

        # problem related params
        self.dim = None
        self.lb = None
        self.ub = None
        
        # population
        self.pop = None

        # load default params
        self.set_params(self.default_params())

    def default_params(self, benchmark=None):
        params = {}
        params['pop_size'] = 300
        params['init_scale'] = 200
        return params

    def set_params(self, params):
        for param in params:
            setattr(self, param, params[param])
    
    def optimize(self, e):
        
        self.init(e)

        while not e.terminate():
            self.pop.evolve(e)
            #print(self.pop.dist.scale, min(self.pop.dist.eigvals), max(self.pop.dist.eigvals))

        return e.best_y

    def init(self, e):
        
        # record problem related params
        self.dim = opt.dim = e.obj.dim
        self.lb = opt.lb = e.obj.lb
        self.ub = opt.ub = e.obj.ub
        
        # init random seed
        self.seed = int(os.getpid()*time.clock())

        # init states
        shift = np.random.uniform(self.lb/2, self.ub/2, [self.dim])
        scale = self.init_scale
        cov = np.eye(self.dim)
        mvnd = MVND(shift, scale, cov)

        # init population
        self.pop = CMAESPop(mvnd, self.pop_size, lb=self.lb, ub=self.ub)
    
