#!/usr/bin/env python
""" Define the class for benchmark.

Unify multiple benchmark and is used to generate 'Evaluator'
"""

import os
import sys
import numpy as np
from mpopt.tools.objective import ObjFunction, Evaluator

current_path = os.path.dirname(__file__)
sys.path.append(os.path.join(current_path, "./cec2013"))
import cec13
sys.path.append(os.path.join(current_path, "./cec2017"))
import cec17
sys.path.append(os.path.join(current_path, "./cec2020"))
import cec20

class Benchmark(object):
    """ Class for benchmark.

    Use benchmark.generate to get a 'Evaluator' for optimization.

    Currently support 'CEC13', 'CEC17', 'CEC20'
    """
    def __init__(self, benchmark):
        """ Init objective functions from benchmark
        """
        if benchmark == 'CEC13':
            self.num_func = 28
            self.funcs = [self.wrap_func(cec13.eval, func_id+1) for func_id in range(self.num_func)] 
            self.bias = list(range(-1400, 0, 100)) + list(range(100, 1500, 100))
            self.lb = -100
            self.ub = 100

            self.dims = [2,5,10,20,30,40,50,60,70,80,90,100]
            self.dim2eval = {}
            for dim in self.dims:
                self.dim2eval[dim] = dim * 10000

        elif benchmark == 'CEC17':
            self.num_func = 30
            self.funcs = [self.wrap_func(cec17.eval, func_id+1) for func_id in range(self.num_func)] 
            self.bias = list(range(-1400, 0, 100)) + list(range(100, 1500, 100))
            self.lb = -100
            self.ub = 100

            self.dims = [2,10,20,30,50,100]
            self.dim2eval = {}
            for dim in self.dims:
                self.dim2eval[dim] = dim * 10000

        elif benchmark == 'CEC20':
            self.num_func = 10
            self.funcs = [self.wrap_func(cec20.eval, func_id+1) for func_id in range(self.num_func)] 
            self.bias = [100, 1100, 700, 1900, 1700, 1600, 2100, 2200, 2400, 2500]
            self.lb = -100
            self.ub = 100

            self.dims = [5, 10, 15, 20]
            self.dim2eval = {}
            self.dim2eval[5] = 50000    
            self.dim2eval[10] = 1000000
            self.dim2eval[15] = 3000000
            self.dim2eval[20] = 10000000

        else:
            raise Exception('Benchmark not Implemented.')

    def generate(self, func_id, dim, traj_mod=0):
        obj = ObjFunction(self.funcs[func_id], dim=dim, lb=self.lb, ub=self.ub, optimal_val=self.bias[func_id])
        evaluator = Evaluator(obj, max_eval=self.dim2eval[dim], traj_mod=traj_mod)
        return evaluator

    def wrap_func(self, func, func_id):
        def wrapped_obj(X):
            return func(X, func_id)
        return wrapped_obj