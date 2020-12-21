import os
import time
import numpy as np

from ..population.fireworks import CMAFirework
from ..operator import operator as opt
from ..tools.distribution import MultiVariateNormalDistribution as MVND

EPS = 1e-8

class HCFWA(object):

    def __init__(self):
        # params
        self.fw_size = None
        self.sp_size = None
        self.init_scale = None
        self.fp_method = None

        # problem related params
        self.dim = None
        self.lb = None
        self.ub = None

        # population
        self.fireworks = None

        # states
        self.feature_points = None

        # load default params
        self.set_params(self.default_params())
    
    def default_params(self, benchmark=None):
        params = {}
        params['fw_size'] = 5
        params['sp_size'] = 300
        params['init_scale'] = 100
        params['fp_method'] = 'middle' # 'equal', 'rank'
        return params

    def set_params(self, params):
        for param in params:
            setattr(self, param, params[param])
    
    def optimize(self, e):
        self.init(e)

        while not e.terminate():
            # independent cma evolving
            for fw in self.fireworks:
                fw.explode()
                fw.eval(e)
                fw.select()
                fw.adapt()
 
            
            # cooperation
            mvnds = [fw.new_mvnd for fw in self.fireworks]
            for mvnd in mvnds:
                mvnd.decompose()

            fps = self.get_feature_point(mvnds)
            
            sort_idx = np.argsort([np.mean(fw.spk_fit) for fw in self.fireworks])
            alphas = [np.exp(i - self.fw_size) for i in sort_idx]

            new_mvnd = []
            for i in range(self.fw_size):
                if i == 0:
                    tmp = self.fit_global(mvnds[i], fps[i,:,:], alphas[i])
                else:
                    fp = self.filter_local_fp(mvnds[i], fps[i,:,:])
                    tmp = self.fit_local(mvnds[i], fp, alphas[i])

                new_mvnd.append(tmp)
 
            for i, fw in enumerate(self.fireworks):
                fw.new_mvnd = new_mvnd[i]
                fw.update()
            

        return e.best_y
        
    def init(self, e):

        # record problem related params
        self.dim = opt.dim = e.obj.dim
        self.lb = opt.lb = e.obj.lb
        self.ub = opt.ub = e.obj.ub

        # init random seed
        self.seed = int(os.getpid()*time.clock())

        # init population
        init_pop = np.random.uniform(self.lb, self.ub, [self.fw_size, self.dim])
        init_pop[0,:] = np.random.uniform(self.lb / 2, self.ub/2, [self.dim])
        init_fit = e(init_pop)

        shifts = [init_pop[_,:] for _ in range(self.fw_size)]
        scales = [self.init_scale / 2 for _ in range(self.fw_size)]
        scales[0] = self.init_scale
        covs = [np.eye(self.dim) for _ in range(self.fw_size)]
        mvnds = [MVND(shifts[_], scales[_], covs[_]) for _ in range(self.fw_size)]

        nspks = [int(self.sp_size / self.fw_size) for _ in range(self.fw_size)]
        nspks[0] += self.sp_size - sum(nspks)

        self.fireworks = [CMAFirework(init_pop[i,:], init_fit[i], mvnds[i], nspks[i], lb=self.lb, ub=self.ub) for i in range(self.fw_size)]

        # init states
        self.feature_points = [list() for _ in range(self.fw_size)]

    def get_paired_fp(self, mvnd1, mvnd2, relation='cp', method=None):
        
        if method is None:
            method = self.fp_method

        vec = mvnd2.shift - mvnd1.shift
        vec1 = mvnd1.scale * vec / np.sqrt(np.sum(np.dot(vec, mvnd1.rev.T) ** 2))
        vec2 = -mvnd2.scale * vec / np.sqrt(np.sum(np.dot(vec, mvnd2.rev.T) ** 2))

        l = np.linalg.norm(vec)
        l1 = np.linalg.norm(vec1)
        l2 = np.linalg.norm(vec2)

        if relation == 'cp':
            if method == 'middle':
                fp = (mvnd1.shift + mvnd2.shift) / 2
                fp1 = fp
                fp2 = fp
            
            elif method == 'equal':
                rescale = l / (l1 + l2)
                fp1 = mvnd1.shift + rescale * vec1
                fp2 = mvnd2.shift + rescale * vec2
                if np.sum(np.abs(fp1 - fp2)) > 1e-8:
                    raise Exception('Error in balancing.')
            
            elif method == 'rank':
                fp1 = mvnd1.shift + vec1
                if l1 >= l:
                    fp2 = mvnd2.shift + (EPS/l2) * vec2
                else:
                    fp2 = fp1
            
            else:
                pass
        
        elif relation == 'fa':
            if method == 'middle':
                rescale = max(0, (l1 - l) / l1)
                fp = mvnd2.shift + (rescale*vec1 - vec2) / 2
                fp1 = fp
                fp2 = fp

            elif method == 'equal':
                rescale = (l2 + np.sqrt(l2**2 + 4*l1*l)) / (2 * l1)
                fp = mvnd1.shift + rescale * vec1
                fp1 = fp
                fp2 = fp

            elif method == 'rank':
                fp1 = mvnd1.shift + vec1
                if l1 > l:
                    fp2 = fp1
                else:
                    fp2 = mvnd2.shift - (EPS/l2) * vec2

            else:
                pass
        
        elif relation == 'ch':
            if method == 'middle':
                rescale = max(0, (l2 - l) / l2)
                fp = mvnd1.shift + (rescale*vec2 - vec1) / 2
                fp1 = fp
                fp2 = fp
            
            elif method == 'equal':
                rescale = (l1 + np.sqrt(l1**2 + 4*l2*l)) / (2 * l2)
                fp = mvnd2.shift + rescale * vec2
                fp1 = fp
                fp2 = fp
            
            elif method == 'rank':
                fp2 = mvnd2.shift + vec2
                if l2 > l:
                    fp1 = fp2
                else:
                    fp1 = mvnd1.shift - (EPS/l1) * vec1

            else:
                pass
        
        else:
            raise Exception('Feature point method not implemented.')

        return fp1, fp2
    
    def get_feature_point(self, mvnds):
        lam = len(mvnds)
        dim = mvnds[0].dim

        fps = np.empty(shape=[lam, lam, dim])
        for i in range(lam):
            for j in range(i+1, lam):
                
                if self.fp_method in ['middle', 'equal']:
                    if i > 0:
                        fp1, fp2 = self.get_paired_fp(mvnds[i],
                                                      mvnds[j],
                                                      relation='cp',
                                                      method='middle')
                    
                    else:
                        fp1, fp2 = self.get_paired_fp(mvnds[i],
                                                      mvnds[j],
                                                      relation='fa',
                                                      method='equal')

                elif self.fp_method == 'rank':
                    
                    fw1 = self.fireworks[i]
                    fw2 = self.fireworks[j]

                    if np.min(fw1.spk_fit) > np.max(fw2.spk_fit):
                        # fw1 is in lower rank of fw2
                        if i == 0:
                            fp2, fp1 = self.get_paired_fp(mvnds[j],
                                                          mvnds[i],
                                                          relation='ch',
                                                          method='rank')
                        else:
                            fp2, fp1 = self.get_paired_fp(mvnds[j],
                                                          mvnds[i],
                                                          relation='cp',
                                                          method='rank')
                    
                    elif np.max(fw1.spk_fit) < np.min(fw2.spk_fit):
                        # fw2 is in lower rank of fw1
                        if i == 0:
                            fp1, fp2 = self.get_paired_fp(mvnds[i],
                                                          mvnds[j],
                                                          relation='fa',
                                                          method='rank')
                        else:
                            fp1, fp2 = self.get_paired_fp(mvnds[i],
                                                          mvnds[j],
                                                          relation='cp',
                                                          method='rank')

                    else:
                        if i == 0:
                            fp1, fp2 = self.get_paired_fp(mvnds[i],
                                                          mvnds[j],
                                                          relation='fa',
                                                          method='equal')
                        else:
                            fp1, fp2 = self.get_paired_fp(mvnds[i],
                                                          mvnds[j],
                                                          relation='cp',
                                                          method='equal')  
                
                fps[i,j,:] = fp1
                fps[j,i,:] = fp2
        return fps

    def filter_local_fp(self, mvnd, fp):        
        # prepare
        bias = fp - mvnd.shift
        bias_norm = np.linalg.norm(bias, axis=1)
        sort_idx = np.argsort(bias_norm)
        
        # check fp
        fp_idx = []
        for i in sort_idx:
            filtered = False
            for j in fp_idx:
                if np.sum(bias[i,:] * bias[j,:]) > np.sum(bias[j,:] ** 2):
                    filtered = True
                    break
            if not filtered:
                fp_idx.append(i)
        
        fp = np.vstack([fp[idx,:] for idx in fp_idx])

        return fp
            
    def fit_local(self, mvnd, fp, alpha):
        # alias
        lam, dim = fp.shape

        # fit shift
        new_shift = (1 - alpha) * mvnd.shift + alpha * np.mean(fp, axis=0)

        bias = fp - new_shift[np.newaxis,:]

        # fit sigma
        logmean_eigval = np.mean(np.log(mvnd.eigvals))
        logmean_length = np.mean(np.log(np.linalg.norm(bias, axis=1)))
        new_scale = np.exp((1-alpha) * logmean_eigval + alpha * logmean_length)

        # fit cov
        new_cov = (1 - alpha - EPS) * mvnd.cov + EPS * np.eye(dim)
        covs = np.matmul(bias[:,:,np.newaxis], bias[:,np.newaxis,:])
        for i in range(lam):
            new_cov += (alpha / lam / new_scale ** 2) * covs[i,:,:]
        
        return MVND(new_shift, new_scale, new_cov)

    def fit_global(self, mvnd, fp, alpha):
        # alias
        lam, dim = fp.shape

        # fit shift
        new_shift = (1 - alpha) * mvnd.shift + alpha * np.mean(fp, axis=0)

        bias = fp - new_shift[np.newaxis,:]

        # fit sigma
        logmean_eigval = np.mean(np.log(mvnd.eigvals))
        logmean_length = np.mean(np.log(np.linalg.norm(bias, axis=1)))
        new_scale = np.exp((1-alpha) * logmean_eigval + alpha * logmean_length)

        # fit cov
        new_cov = (1 - alpha - EPS) * mvnd.cov + EPS * np.eye(dim)
        covs = np.matmul(bias[:,:,np.newaxis], bias[:,np.newaxis,:])
        for i in range(lam):
            new_cov += (alpha / lam / new_scale ** 2) * covs[i,:,:]
        
        return MVND(new_shift, new_scale, new_cov)
