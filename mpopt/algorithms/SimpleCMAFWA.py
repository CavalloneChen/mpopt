import os
import time
import numpy as np

from ..population.fireworks import SimpleCMAFirework
from ..operator import operator as opt
from ..tools.distribution import MultiVariateNormalDistribution as MVND

EPS = 1e-8

class SimpleCMAFWA(object):

    def __init__(self):
        # params
        self.fw_size = None
        self.sp_size = None
        self.init_scale = None

        # problem related params
        self.dim = None
        self.lb = None
        self.ub = None

        # population
        self.fireworks = None

        # load default params
        self.set_params(self.default_params())
    
    def default_params(self, benchmark=None):
        params = {}
        params['fw_size'] = 5
        params['sp_size'] = 300
        params['init_scale'] = 200
        return params

    def set_params(self, params):
        for param in params:
            setattr(self, param, params[param])
    
    def optimize(self, e):
        self.init(e)

        while not e.terminate():
            # independent cma evolving
            for fw in self.fireworks:
                fw.evolve(e)
        return e.best_y
        
    def init(self, e):

        # record problem related params
        self.dim = opt.dim = e.obj.dim
        self.lb = opt.lb = e.obj.lb
        self.ub = opt.ub = e.obj.ub

        # init random seed
        self.seed = int(os.getpid()*time.clock())

        # init states
        shifts = [np.random.uniform(self.lb, self.ub, [self.dim]) for _ in range(self.fw_size)]
        scales = [self.init_scale for _ in range(self.fw_size)]
        covs = [np.eye(self.dim) for _ in range(self.fw_size)]
        mvnds = [MVND(shifts[_], scales[_], covs[_]) for _ in range(self.fw_size)]

        # init population
        init_pop = np.random.uniform(self.lb, self.ub, [self.fw_size, self.dim])
        init_fit = e(init_pop)
        nspks = [int(self.sp_size / self.fw_size) for _ in range(self.fw_size)]
        nspks[0] += self.sp_size - sum(nspks)

        self.fireworks = [SimpleCMAFirework(init_pop[i,:], init_fit[i], mvnds[i], nspks[i]) for i in range(self.fw_size)]
