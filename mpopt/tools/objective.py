#!/usr/bin/env python
""" Class defination for objective function and evaluator.

Class:
======
    class 'ObjFunction'
        Wrapper of a callable function to store additional information.

    class 'Evaluator'
        Wrapper of 'ObjFunction' to manage a succession of evaluations in optimization.
"""

import numpy as np


class ObjFunction(object):
    """Ojective funcitons.
    
    Manage an callable numeric objective function and important information for optimization.
    """
    def __init__(self, 
                 func,              # callable objective function 
                 dim=None,          # dim of solution. If None, accept any solution dim. 
                 lb=-float('inf'),  # lower bound (scalar or ndarray)
                 ub=float('inf'),   # upper bound (scalar or ndarray)
                 optimal_x=None,    # optimal solution
                 optimal_val=None,  # optimal value
                 **func_params,     # params passed to func
                 ):
        self.func = func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.optimal_x = optimal_x
        self.optimal_val = optimal_val
        self.func_params = func_params
    
    def __call__(self, X):
        X = self._preprocessing(X)
        y = self.func(X, **self.func_params)
        if y.shape[0] == 1:
            y = y[0]
        return y

    def _preprocessing(self, X):
        """Check and prepare input solution. """
        if X.ndim == 1:
            x = x.reshape((1, -1))
        out_bound = np.any(np.bitwise_or(X < self.lb, X > self.ub))
        if out_bound:
            raise Exception("Solution out of bound.")
        return x

class Evaluator(object):
    """Evaluator for optimization.

    Manage an callable numeric function, store problem parameters and update optimization information.
    ***Only support minimization currently.
    """
    def __init__(self, obj_func, max_eval=-1, max_batch=-1, traj_mod=0):
        self.obj = obj_func
        
        # parameters for problem
        self.max_eval = max_eval
        self.max_batch = max_batch

        # states
        self.cur_x = None # current best solution
        self.cur_y = None # current best value
        self.num_eval = 0
        self.num_batch = 0
        
        # trajectory
        self.traj_mod = traj_mod
        self.traj = None
        if self.traj_mod > 0:
            self.traj = []

    def __call__(self, X):
        """ Evaluation of solution.

        Conduct evaluation of X and record optimization information, including:
            cur_x:      current best solution
            cur_y:      current best value
            num_eval:   number of total evaluation
            num_batch:  number of total batch
            traj:       record the optimization process

        Args:
            X: np.ndarray, the matrix of solution(s)
        """
        y = self.obj(X)
        if X.ndim == 1:
            # single solution evaluated
            if y < self.cur_y:
                self.cur_x = X
                self.cur_y = y
            
            if self.traj_mod > 0 and ((self.num_eval+1) % self.num_eval == 0):
                self.traj.append(self.cur_y)

            self.num_eval += 1
            self.num_batch += 1
            
        else:
            # multiple solutions evaluated
            min_idx = np.argmin(y)
            min_x = X[min_idx,:]
            min_y = y[min_idx]

            if min_y < self.cur_y:
                self.cur_x = min_x
                self.cur_y = min_y
            
            if self.traj_mod > 0:
                mod = self.traj_mod
                num = X.shape[0]

                r = mod - (self.num_eval % mod)
                cnt = int((num - r) // mod) + 1

                self.traj += [self.cur_y] * cnt

            self.num_eval += X.shape[0]
            self.num_batch += 1
        
        return y

    def terminate(self):
        if self.max_eval > 0 and self.num_eval >= self.max_eval:
            return True
        if self.max_batch > 0 and self.num_batch >= self.max_batch:
            return True
        return False
