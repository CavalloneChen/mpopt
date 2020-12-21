#!/usr/bin/env python
import numpy as np

class MultiVariateNormalDistribution(object):

    def __init__(self, shift, scale, cov, dim=None):
        # main components
        self.shift = shift
        self.scale = scale
        self.cov = cov
        
        # params
        self.dim = dim if dim is not None else shift.shape[0]

        # states
        self.eigvecs = None
        self.eigvals = None
        self.inv_cov = None
        self.invsqrt_cov = None
        self.rev = None

        # decompose cov
        self.decomposed = False

    def decompose(self):

        # force symmetric 
        self.cov = (self.cov + self.cov.T) / 2.0

        # solve
        self.eigvals, self.eigvecs = np.linalg.eigh(self.cov)

        # rescale
        if 1 == 1:
            rescale = np.max(self.eigvals)
            self.cov /= rescale
            self.eigvals /= rescale
            self.scale *= np.sqrt(rescale)

        # inv cov
        self.inv_cov = np.dot(self.eigvecs, np.diag(self.eigvals ** -1)).dot(self.eigvecs.T)

        # inv sqrt cov
        self.invsqrt_cov = np.dot(self.eigvecs, np.diag(self.eigvals ** -0.5)).dot(self.eigvecs.T)

        # reverse projection matrix
        self.rev = np.dot(np.diag(self.eigvals ** -0.5), self.eigvecs.T)

    def sample(self, num, remap=None):
        if not self.decomposed:
            self.decompose()

        bias = np.random.normal(size=[num, self.dim])
        amp_bias = self.scale * (self.eigvals ** 0.5)[np.newaxis,:] * bias
        rot_bias = np.dot(amp_bias, self.eigvecs.T)
        samples = self.shift[np.newaxis,:] + rot_bias
        if remap is not None:
            samples = remap(samples)
        return samples
