#!/usr/bin/env python
""" Define the class for benchmark.

Unify multiple benchmark and is used to generate 'Evaluator'
"""


import numpy as np
from ..tools.objective import ObjFunction, Evaluator
from .cec2013 import cec13
from .cec2017 import cec17
from .cec2020 import cec20


class Benchmark(object):
    """Class for benchmark.

    Use benchmark.generate to get a 'Evaluator' for optimization.

    Currently support 'CEC13', 'CEC17', 'CEC20'
    """

    def __init__(self, benchmark, dim):
        """Init objective functions from benchmark"""
        self.name = benchmark
        self.dim = dim

        if benchmark == "CEC13":
            if self.dim not in [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                raise Exception("Benchmark CEC13 do not support dimension {}.".format(self.dim))
            self.max_eval = 10000 * self.dim

            self.num_func = 28
            self.funcs = [
                self.wrap_func(cec13.eval, func_id + 1)
                for func_id in range(self.num_func)
            ]
            self.bias = np.array(list(range(-1400, 0, 100)) + list(range(100, 1500, 100)))
            self.lb = -100
            self.ub = 100

        elif benchmark == "CEC17":
            if self.dim not in [10, 30, 50, 100]:
                raise Exception("Benchmark CEC17 do not support dimension {}.".format(self.dim))
            self.max_eval = 10000 * self.dim
            
            self.num_func = 30
            self.funcs = [
                self.wrap_func(cec17.eval, func_id + 1)
                for func_id in range(self.num_func)
            ]
            self.bias = np.arange(100, 3100, 100)
            self.lb = -100
            self.ub = 100

        elif benchmark == "CEC20":
            if self.dim not in [10, 15, 20]:
                raise Exception("Benchmark CEC20 do not support dimension {}.".format(self.dim))
            if self.dim == 10:
                self.max_eval = 1000000
            elif self.dim == 15:
                self.max_eval = 3000000
            elif self.dim == 20:
                self.max_eval = 10000000

            self.num_func = 10
            self.funcs = [
                self.wrap_func(cec20.eval, func_id + 1)
                for func_id in range(self.num_func)
            ]
            self.bias = np.array([100, 1100, 700, 1900, 1700, 1600, 2100, 2200, 2400, 2500])
            self.lb = -100
            self.ub = 100

        else:
            raise Exception("Benchmark not Implemented.")

    def generate(self, func_id, traj_mod=0):

        if func_id < 0 or func_id >= self.num_func:
            raise Exception("Function id out of range.")

        obj = ObjFunction(
            self.funcs[func_id],
            dim=self.dim,
            lb=self.lb,
            ub=self.ub,
            optimal_val=self.bias[func_id],
        )
        evaluator = Evaluator(obj, max_eval=self.max_eval, traj_mod=traj_mod)
        return evaluator

    def wrap_func(self, func, func_id):
        def wrapped_obj(X):
            return func(X, func_id)

        return wrapped_obj
