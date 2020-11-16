# #!/usr/bin/env python
""" Script for testing algorithms on given benchmarks """

import os
import time
import json
import argparse
import importlib
import numpy as np
from functools import wraps
from multiprocessing import Pool

from mpopt.benchmarks.benchmark import Benchmark


def parsing():
    # experiment setting for benchmark testing
    parser = argparse.ArgumentParser(description="Benchmark testing for SIOA")

    # benchmark params
    parser.add_argument("--benchmark", "-b", help="Benchmark name")
    parser.add_argument("--dim", "-d", type=int, help="Benchmark dimension")

    # agortihm params
    parser.add_argument("--alg", "-a", help="Algorithm name")

    # testing params
    parser.add_argument("--name", "-n", default="", help="Name of experiment")
    parser.add_argument(
        "--rep", "-r", default=1, type=int, help="Repetition of each problem"
    )
    parser.add_argument(
        "--multiprocess",
        "-m",
        default=1,
        type=int,
        help="Number of threads for multiprocessing",
    )

    # results handling
    parser.add_argument(
        "--precision", default=1e-8, type=float, help="Precision in results comparing"
    )
    parser.add_argument("--alpha", default=0.05, type=float, help="Significant Level")

    return parser.parse_args()


def logging(opt):
    @wraps(opt)
    def opt_with_logging(func_id):
        start = time.time()
        val = opt(func_id)
        end = time.time()

        print(
            "Prob.{:<4}, res:{:.4e},\t time:{:.3f}".format(
                func_id + 1, val, end - start
            )
        )
        return val, end - start

    return opt_with_logging


if __name__ == "__main__":

    args = parsing()

    # get benchmark
    benchmark = Benchmark(args.benchmark, args.dim)

    # get opimizer
    alg_mod = importlib.import_module("mpopt.algorithms." + args.alg)
    model = getattr(alg_mod, args.alg)
    optimizer = model()

    # get algorithm params and update to provided params
    params = optimizer.default_params(benchmark=benchmark)
    for param in params:
        if param in args:
            params[param] = args[param]

    # opt function
    @logging
    def opt(idx):
        evaluator = benchmark.generate(idx)
        optimizer = model()
        optimizer.set_params(params)
        return optimizer.optimize(evaluator)

    # store results
    res = np.empty((benchmark.num_func, args.rep))
    cst = np.empty((benchmark.num_func, args.rep))

    for i in range(benchmark.num_func):
        if args.multiprocess > 1:
            # multiprocessing
            p = Pool(args.multiprocess)
            results = p.map(opt, [i] * args.rep)
            p.close()
            p.join()

            for j in range(args.rep):
                res[i, j], cst[i, j] = results[j][0], results[j][1]

        else:
            # sequential
            for j in range(args.rep):
                res[i, j], cst[i, j] = opt(i)

    # save
    info = {}
    info["args"] = vars(args)
    info["params"] = params
    info["optimals"] = res.tolist()
    info["times"] = cst.tolist()

    if args.name is not '':
        args.name = '_' + args.name
    dir_path = os.path.split(os.path.realpath(__file__))[0]
    with open(
        os.path.abspath(
            os.path.join(
                dir_path,
                "../logs/"
                + benchmark.name
                + "_"
                + str(benchmark.dim)
                + "D/"
                + args.alg
                + args.name
                + ".json",
            )
        ),
        "w",
    ) as f:
        json.dump(info, f)
