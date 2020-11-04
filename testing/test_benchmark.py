# !/usr/bin/env python
""" Testing of implemented benchmark functions
"""

import time
import numpy as np
from ..benchmarks.benchmark import Benchmark

if __name__ == '__main__':
    bcn = ['CEC13', 'CEC17', 'CEC20']
    bcs = [Benchmark('CEC13'), Benchmark('CEC17'), Benchmark('CEC20')]

    # testing parameters
    repetition = 200
    batch_size = 300

    for idx in range(3):
        name = bcn[idx]
        bc = bcs[idx]

        for dim in bc.dims:
            for func_id in range(bc.num_func):
                evaluator = bc.generate(func_id, dim)

                ### test of single evaluation
                t0 = time.time()
                samples = []
                for rep in range(repetition):
                    samples.append(np.random.uniform(bc.lb, bc.ub, ()))

                t1 = time.time()
                for rep in range(repetition):
                    y = evaluator(samples[idx])

                t2 = time.time()

                print("{}. Dim: {}, Func_id: {}, SingleSolution, AveSampleTime: {}, AveEvalTime: {}".format(name, dim, func_id, (t1-t0)/repetition, (t2-t1)/repetition))

                #### test of batch evaluation
                t0 = time.time()
                samples = []
                for rep in range(repetition):
                    samples.append(np.random.uniform(bc.lb, bc.ub, (batch_size, dim)))

                t1 = time.time()
                for rep in range(repetition):
                    y = evaluator(samples[idx])

                t2 = time.time()

                print("{}. Dim: {}, Func_id: {}, BatchSolutions, AveSampleTime: {}, AveEvalTime: {}".format(name, dim, func_id, (t1-t0)/repetition, (t2-t1)/repetition))
