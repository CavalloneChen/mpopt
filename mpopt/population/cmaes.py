import numpy as np

from ..operator import operator as opt
from ..tools.distribution import MultiVariateNormalDistribution as MVND
from .base import BaseEDAPop

EPS = 1e-8

class CMAESPop(BaseEDAPop):
    """ CMA-ES Population """
    def __init__(self, mvnd, size, lb=-float('inf'), ub=float('inf')):
        
        super().__init__(mvnd, lb=lb, ub=ub)

        # states
        self.size = size
        self.new_size = None

        self.ps = np.zeros(self.dim)
        self.new_ps = None

        self.pc = np.zeros(self.dim)
        self.new_pc = None

        self.num_iter = 0

    def sample(self):
        self.pop = self.dist.sample(self.size, remap=self.remap)

    def adapt(self):
        # alias
        lam, dim = self.pop.shape
        shift = self.dist.shift
        scale = self.dist.scale
        cov = self.dist.cov
        num_iter = self.num_iter

        # sort samples
        sort_idx = np.argsort(self.fit)
        self.pop = self.pop[sort_idx, :]
        self.fit = self.fit[sort_idx]

        # compute
        w = np.log(lam/2+0.5) - np.log(1+np.arange(lam))
        w[w<0] = 0
        w = w / np.sum(w)
        mueff = np.sum(w)**2/np.sum(w**2)

        cc = (4+mueff/dim) / (dim+4+2*mueff/dim)
        cs = (mueff+2)/(dim+mueff+5)
        ccn = (cc*(2-cc)*mueff)**0.5 / scale
        csn = (cs*(2-cs)*mueff)**0.5 / scale

        hsig = int((np.sum(self.ps**2) / dim / (1 - (1-cs)**(2*num_iter+1))) < 2 + 4./(dim+1))
        c1 = 2/((dim+1.3)**2+mueff)
        c1a = c1*(1-(1-hsig**2) * cc * (2-cc))
        cmu = min([1-c1, 2*(mueff-2+1/mueff) / ((dim+2)**2 + mueff)])

        # adapt shift
        new_shift = np.sum(w[:,np.newaxis]*self.pop, axis=0)

        # adapt evolution path
        y = new_shift - shift
        z = np.dot(y, self.dist.invsqrt_cov)

        new_pc = (1-cc) * self.pc + ccn * hsig * y
        new_ps = (1-cs) * self.ps + csn * z

        # adapt cov
        new_cov = EPS * np.eye(dim) + cov * (1 - EPS - c1a - cmu * sum(w))
        new_cov += c1 * np.dot(new_pc[:,np.newaxis], new_pc[np.newaxis,:])
        
        bias = self.pop - shift[np.newaxis,:]
        covs = np.matmul(bias[:,:,np.newaxis], bias[:,np.newaxis,:])
        for i in range(lam):
            if w[i] < 0:
                raise Exception("Negative weight found")
            new_cov += (w[i] * cmu / scale ** 2) * covs[i,:,:]
        
        # adapt scale
        damps = 2 * mueff / lam + 0.3 + cs
        cn, sum_square_ps = cs/damps, np.sum(new_ps ** 2)
        new_scale = scale * np.exp(min(1, cn * (sum_square_ps / dim - 1) / 2))

        self.new_dist = MVND(new_shift, new_scale, new_cov)
        self.new_ps = new_ps
        self.new_pc = new_pc

    def update(self):

        # update distribution
        try:
            self.new_dist.decompose()
        except:
            print(self.dist.scale)
            print(self.dist.eigvals)
            exit()
        self.dist = self.new_dist

        # update states
        self.ps = self.new_ps
        self.pc = self.new_pc
        self.num_iter += 1
    
    def evolve(self, e):
        self.sample()
        self.eval(e)
        self.adapt()
        self.update()
