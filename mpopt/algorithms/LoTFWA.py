import os
import time
import numpy as np

from ..operator import operator as opt

EPS = 1e-8

class LoTFWA(object):

    def  __init__(self):
        # Definition of all parameters and states

        # params
        self.fw_size = None
        self.sp_size = None
        self.init_amp = None
        self.gm_ratio = None

        # states
        self.pop = None
        self.fit = None
        self.amps = None
        self.nspk = None

        # problem related params
        self.dim = None
        self.lb = None
        self.ub = None

        # load default params
        self.set_params(self.default_params())

    def default_params(self, benchmark=None):
        params = {}
        params['fw_size'] = 5
        params['sp_size'] = 300 if benchmark is None else 10*benchmark.dim
        params['init_amp'] = 200 if benchmark is None else benchmark.ub - benchmark.lb
        params['gm_ratio'] = 0.2
        return params
    
    def set_params(self, params):
        for param in params:
            setattr(self, param, params[param])

    def optimize(self, e):
        
        self.init(e)
  
        while not e.terminate():
            
            # explode
            spk_pops, spk_fits = self.explode(e)

            # mutate
            mut_pops, mut_fits = self.mutate(spk_pops, spk_fits, e)

            # select
            n_pop, n_fit = self.select(spk_pops, spk_fits, mut_pops, mut_fits)

            # restart
            improves = self.fit - n_fit
            rest_iter = (e.max_eval - e.num_eval) / (self.sp_size + self.fw_size)
            restart = (improves > EPS) * (improves * rest_iter < (n_fit-e.cur_y))
            restart_num = sum(restart.astype(np.int32))

            if restart_num > 0:
                rand_sample = np.random.uniform(self.lb, self.ub, (restart_num, self.dim))
                n_pop[restart, :] = rand_sample
                n_fit[restart] = e(n_pop[restart, :])
                self.amps[restart] = self.init_amp
            
            ## update states
            # dynamic amps
            for idx in range(self.fw_size):
                if restart[idx]:
                    continue
                if n_fit[idx] < self.fit[idx] - EPS:
                    self.amps[idx] *= 1.2
                else:
                    self.amps[idx] *= 0.9

            # change fireworks
            self.pop = n_pop
            self.fit = n_fit

        return e.best_y

    def init(self, e):

        # record problem related params
        self.dim = opt.dim = e.obj.dim
        self.lb = opt.lb = e.obj.lb
        self.ub = opt.ub = e.obj.ub

        # init states
        self.pop = np.random.uniform(self.lb, self.ub, [self.fw_size, self.dim])
        self.fit = e(self.pop)
        self.amps = np.array([self.init_amp] * self.fw_size)
        self.nspk = np.array([int(self.sp_size / self.fw_size)]*self.fw_size)
        
        # init random seed
        self.seed = int(os.getpid()*time.clock())

    def explode(self, e):
        spk_pops = []
        spk_fits = []
        for idx in range(self.fw_size):
            spk_pop = opt.box_explode(self.pop[idx,:], self.amps[idx], self.nspk[idx], remap=self.remap)
            spk_fit = e(spk_pop)
            spk_pops.append(spk_pop)
            spk_fits.append(spk_fit)
        return spk_pops, spk_fits

    def mutate(self, spk_pops, spk_fits, e):
        mut_pops = []
        mut_fits = []
        for idx in range(self.fw_size):
            mut_pop = opt.guided_mutate(self.pop[idx,:], spk_pops[idx], spk_fits[idx], self.gm_ratio, remap=self.remap)
            mut_fit = e(mut_pop)
            mut_pops.append(mut_pop)
            mut_fits.append(mut_fit)
        return mut_pops, mut_fits
    
    def select(self, spk_pops, spk_fits, mut_pops, mut_fits):
        n_pop = np.empty_like(self.pop)
        n_fit = np.empty_like(self.fit)

        for idx in range(self.fw_size):
            tot_pop = np.vstack([self.pop[idx,:], spk_pops[idx], mut_pops[idx]])
            tot_fit = np.concatenate([[self.fit[idx]], spk_fits[idx], mut_fits[idx]])
            n_pop[idx,:], n_fit[idx] = opt.elite_select(tot_pop, tot_fit)
        return n_pop, n_fit
    
    def remap(self, samples):
        return opt.random_map(samples, lb=self.lb, ub=self.ub)