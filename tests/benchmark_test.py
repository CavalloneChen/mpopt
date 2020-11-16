# !/usr/bin/env python
""" Testing of implemented benchmark functions
"""

import time
import numpy as np
from mpopt.benchmarks.benchmark import Benchmark


def test_benchmark():

    # testing parameters
    repetition = 100
    batch_size = 300

    for name in ['CEC13', 'CEC17', 'CEC20']:
        if name == 'CEC20':
            dims = [10, 15, 20]
        else:
            dims = [30, 50, 100]

        for dim in dims:
            bc = Benchmark(name, dim)

            for func_id in range(bc.num_func):

                evaluator = bc.generate(func_id, dim)

                ### test of single evaluation
                t0 = time.time()
                samples = []
                for rep in range(repetition):
                    samples.append(np.random.uniform(bc.lb, bc.ub, (dim,)))

                t1 = time.time()
                for rep in range(repetition):
                    y = evaluator(samples[rep])

                t2 = time.time()

                print(
                    "{}. Dim: {}, Func_id: {}, SingleSolution, AveSampleTime: {:.3e}, AveEvalTime: {:.3e}".format(
                        name,
                        dim,
                        func_id+1,
                        (t1 - t0) / repetition,
                        (t2 - t1) / repetition,
                    )
                )

                #### test of batch evaluation
                t0 = time.time()
                samples = []
                for rep in range(repetition):
                    samples.append(np.random.uniform(bc.lb, bc.ub, (batch_size, dim)))

                t1 = time.time()
                for rep in range(repetition):
                    y = evaluator(samples[rep])

                t2 = time.time()

                print(
                    "{}. Dim: {}, Func_id: {}, BatchSolutions, AveSampleTime: {:.3e}, AveEvalTime: {:.3e}".format(
                        name,
                        dim,
                        func_id+1,
                        (t1 - t0) / repetition,
                        (t2 - t1) / repetition,
                    )
                )


if __name__ == "__main__":
    test_benchmark()
