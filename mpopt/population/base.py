import numpy as np

from ..operator import operator as opt


class BasePop(object):
    """ Base class for population """
    def __init__(self, pop, fit, lb=-float('inf'), ub=float('inf')):
        # init pop
        self.pop = pop
        self.fit = fit
        self.gen_pop = None
        self.gen_fit = None
        self.new_pop = None
        self.new_fit = None

        # params
        self.size = self.pop.shape[0]
        self.dim = self.pop.shape[1]
        self.lb = lb
        self.ub = ub

        # states
        # no states here

    def remap(self, samples):
        """ Always apply random_map on out-bounded samples """
        return opt.random_map(samples, self.lb, self.ub)
        
    def eval(self, e):
        """ Evaluate un-evaluated individuals here """
        raise NotImplementedError

    def select(self):
        """ Select 'new_pop' and 'new_fit' """
        raise NotImplementedError

    def generate(self):
        """ Generate offsprings """
        raise NotImplementedError

    def adapt(self):
        """ Adapt new states """
        raise NotImplementedError

    def update(self):
        """ Update pop and states """
        raise NotImplementedError

    def evolve(self):
        """ Define the evolve process in an iteration """
        raise NotImplementedError

class BaseEDAPop(object):
    """ Base class for EDA (Estimation of Distribution Algorithms) population """
    def __init__(self, dist, dim=None, lb=-float('inf'), ub=float('inf')):
        # init pop
        self.pop = None
        self.fit = None
        self.dist = dist
        self.new_dist = None

        # params
        self.dim = dim if dim is not None else dist.dim
        self.lb = lb
        self.ub = ub

        # states
        # no states here

    def remap(self, samples):
        """ Always apply random_map on out-bounded samples """
        return opt.random_map(samples, self.lb, self.ub)

    def eval(self, e):
        self.fit = e(self.pop)

    def sample(self, num_sample):
        """ Sample population from the distribution """
        raise NotImplementedError

    def adapt(self):
        """ Adapt the distribution """
        raise NotImplementedError
    
    def update(self):
        """ Update the distribution with adapted one """
        raise NotImplementedError

    def evolve(self):
        """ Define the evolve process in an iteration """
        raise NotImplementedError

class BaseFirework(object):
    """ Base Class for Fireworks """
    def __init__(self, idv, val, lb=-float('inf'), ub=float('inf')):
        # init pop
        self.idv = idv
        self.val = val
        self.spk_pop = None
        self.spk_fit = None
        self.new_idv = None
        self.new_val = None
        
        # params
        self.dim = self.idv.shape[0]
        self.lb = lb
        self.ub = ub

        # states
        # No states here
    
    def eval(self, e):
        """ Eval un-evaluated individuals here """
        raise NotImplementedError

    def remap(self, samples):
        """ Always apply random_map on out-bounded samples """
        return opt.random_map(samples, self.lb, self.ub)

    def select(self):
        """ Select 'new_pop' and 'new_fit' """
        raise NotImplementedError

    def explode(self):
        """ Generate explosion sparks """
        raise NotImplementedError

    def mutate(self):
        """ Generate mutation sparks """
        raise NotImplementedError

    def adapt(self):
        """ Adapt new states """
        raise NotImplementedError
    
    def update(self):
        """ Update pop and states """
        raise NotImplementedError

    def evolve(self):
        """ Define the evolve process in an iteration """
        raise NotImplementedError