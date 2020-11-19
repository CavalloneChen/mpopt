import numpy as np

from ..operator import operator as opt

class BasePop(object):
    """ Base class for population """
    def __init__(self, pop, fit, **kwargs):
        """ Innitializing a population

        Args:
            pop (np.ndarray): must be shaped [num_pop, dim,]
            fit (np.ndarray): must be shaped [dim,]
        """
        # init pop
        self.pop = pop
        self.fit = fit
        self.gen_pop = None
        self.gen_fit = None
        self.new_pop = None
        self.new_fit = None

        # load params and states
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])
        
        # alias
        if 'num_pop' not in self.__dict__:
            self.num_pop = self.pop.shape[0]
        if 'dim' not in self.__dict__:
            self.dim = self.pop.shape[1]

    def eval(self, e):
        self.gen_fit = e(self.gen_pop)

    def remap(self, samples):
        lb = self.lb if 'lb' in self.__dict__ else -float('inf')
        ub = self.ub if 'ub' in self.__dict__ else float('int')
        return opt.random_map(samples, lb=lb, ub=ub)

    def select(self):
        tot_pop = np.vstack(self.pop, self.gen_pop)
        tot_fit = np.concatenate([self.fit, self.gen_fit])
        self.new_pop, self.new_fit = opt.elite_select(tot_pop, tot_fit)

    def generate(self):
        raise NotImplementedError

    def update(self):
        self.pop = self.new_pop
        self.fit = self.new_fit
    
    def evolve(self, e):
        self.generate()
        self.eval(e)
        self.select()
        self.update()


class BaseFirework(object):
    """ Base class for firework """
    def __init__(self, idv, val, **kwargs):
        """ Initializing a firework

        Args:
            idv (np.ndarray):   idv.shape = [dim,]
            val (float):        fitness value of idv
        """
        # init pop
        self.idv = idv
        self.val = val
        self.spk = None
        self.spk_fit = None
        self.new_idv = None
        self.new_val = None

        # load params and states
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])
        
        # alias
        if 'dim' not in self.__dict__:
            self.dim = self.idv.shape[0]

    def eval(self, e):
        self.spk_fit = e(self.spk)
    
    def remap(self, samples):
        lb = self.lb if 'lb' in self.__dict__ else -float('inf') #pylint: disable-msg=no-member
        ub = self.ub if 'ub' in self.__dict__ else float('int') #pylint: disable-msg=no-member
        return opt.random_map(samples, lb=lb, ub=ub)
    
    def select(self):
        tot_pop = np.vstack([self.idv, self.spk])
        tot_fit = np.concatenate([np.array([self.val]), self.spk_fit])
        self.new_idv, self.new_val = opt.elite_select(tot_pop, tot_fit)

    def explode(self, amp=None, num_spk=None):
        if amp is None and 'amp' in self.__dict__:
            amp = self.amp #pylint: disable-msg=no-member
        if num_spk is None and 'num_spk' in self.__dict__:
            num_spk = self.num_spk #pylint: disable-msg=no-member

        self.spk = opt.box_explode(self.idv, amp, num_spk, remap=self.remap)

    def update(self):
        self.idv = self.new_idv
        self.val = self.new_val

    def evolve(self, e):
        self.explode()
        self.eval(e)
        self.select()
        self.update()

