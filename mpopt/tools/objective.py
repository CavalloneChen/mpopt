#!/usr/bin/env python
""" Tools for objective function and evaluator.

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
    
    Attributes:
        func (callable function)
        batch (bool): whether 'func' accept batched input
        dim (integer)
        lb (float or np.ndarray[dim,] or list[dim])
        ub (float or np.ndarray[dim,] or list[dim])
        optimal_x (np.ndarray[dim,])
        optimal_val (float)
    """

    def __init__(
        self,
        func,  # callable objective function
        batch=False, # whether the func accept batched input
        dim=None,  # dim of solution. If None, accept any solution dim.
        lb=-float("inf"),  # lower bound (scalar or ndarray)
        ub=float("inf"),  # upper bound (scalar or ndarray)
        optimal_x=None,  # optimal solution
        optimal_val=None,  # optimal value
        **func_params,  # params passed to func
    ):
        self.func = func
        self.batch = batch
        self.dim = dim
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.optimal_x = np.array(optimal_x)
        self.optimal_val = optimal_val
        self.func_params = func_params

    def __call__(self, X):
        """ Evaluation of solution X 
        
        Args:
            X (np.ndarray): 2d array.

        Returns:
            y (np.ndarray): 1d array.
        """

        # handling input
        try:
            _X = np.array(X)
        except:
            raise Exception("ObjFunc Solution Error: Input cannot be converted to np.ndarray.")

        if _X.ndim > 2:
            raise Exception("ObjFunc Solution Error: Only support 2d np.ndarray input.")
        
        if _X.ndim == 1:
            if self.dim is not None:
                _X = _X.reshape(-1, self.dim)
            else:
                _X = _X.reshape(-1, 1)
        
        # call func
        if self.batch:
            y = self.func(X, **self.func_params)
        else:
            y = np.array([self.func(X[i,:], **self.func_params) for i in range(X.shape[0])])

        return y

class Evaluator(object):
    """Evaluator for optimization.

    Manage an callable numeric function, store problem parameters and update optimization states.
    """

    def __init__(self, obj_func, max_eval=-1, max_batch=-1, traj_mod=0):
        self.obj = obj_func

        # parameters for problem
        self.max_eval = max_eval
        self.max_batch = max_batch

        # states
        self.cur_x = None  # current best solution
        self.cur_y = None  # current best value
        self.best_x = None
        self.best_y = None
        self.num_eval = 0
        self.num_batch = 0

        # trajectory
        self.traj_mod = traj_mod
        self.traj = None
        if self.traj_mod > 0:
            self.traj = []

    def __call__(self, X):
        return self.eval(X)

    def eval(self, X):
        """ Evaluate solutions.

        Args:
            X (np.ndarray): 2d array.

        Returns:
            y (np.ndarray): 1d array.
        """

        num = X.shape[0]

        # evaluation
        y = self.obj(X)

        # update evaluator states
        if num == 1:
            # single solution evaluated
            if self.cur_y is None or y < self.cur_y:
                self.cur_x = X[0,:]
                self.cur_y = y[0]

            if self.traj_mod > 0 and ((self.num_eval + 1) % self.num_eval == 0):
                self.traj.append(self.cur_y)

        else:
            # multiple solutions evaluated
            min_idx = np.argmin(y)
            min_x = X[min_idx, :]
            min_y = y[min_idx]

            if self.cur_y is None or min_y < self.cur_y:
                self.cur_x = min_x
                self.cur_y = min_y

            if self.traj_mod > 0:
                mod = self.traj_mod

                r = mod - (self.num_eval % mod)
                cnt = int((num - r) // mod) + 1

                self.traj += [self.cur_y] * cnt
        
        self.num_eval += num
        self.num_batch += 1

        return y

    def terminate(self):
        stop = False
        
        # stop conditions
        if self.max_eval > 0 and self.num_eval >= self.max_eval:
            stop = True
        if self.max_batch > 0 and self.num_batch >= self.max_batch:
            stop = True
        
        # stop states
        if stop:
            self.best_x = self.cur_x
            self.best_y = self.cur_y
        
        return stop