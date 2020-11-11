# #!/usr/bin/env python
""" Script for testing algorithms on given benchmarks """

import os
import time
import json
import argparse
import importlib
import numpy as np
from multiprocessing import Pool

from mpopt.benchmarks.benchmark import Benchmark


def parsing():
    # experiment setting for benchmark testing
    parser = argparse.ArgumentParser(description="Benchmark testing for SIOA")

    # benchmark params
    parser.add_argument("--bechmark", "-b", help="Benchmark name")
    parser.add_argument("--dim", "-d", help="Benchmark dimension")

    # agortihm params
    parser.add_argument("--alg", "-a", help="Algorithm name")

    # testing params
    parser.add_argument("--name", "-n", default="", help="Name of experiment")
    parser.add_argument("--rep", "-r", default=1, help="Repetition of each problem")
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

    # set params
    optimizer.set_params(params)

    # store results
    res = np.empty((benchmark.num_func, args.repetition))
    cst = np.empty((benchmark.num_func, args.repetition))

    for i in range(benchmark.num_func):
        # multiprocessing
        if args.multiprocess > 1:
            p = Pool(args.multiprocess)
            evaluators = [benchmark.generate(i) for _ in range(args.rep)]
            results = p.map(optimizer.opt, evaluators)
            p.close()
            p.join()

            for j in range(args.rep):
                res[i, j], cst[i, j] = results[j][0], results[j][1]
                print(
                    "Prob.{:<4}, res:{:.4e},\t time:{:.3f}".format(
                        i + 1, res[i, j], cst[i, j]
                    )
                )
        else:
            # sequential
            for j in range(args.rep):
                evaluator = benchmark.generate(i)
                res[i, j], cst[i, j] = optimizer.opt(evaluator)
                print(
                    "Prob.{:<4}, res:{:.4e},\t time:{:.3f}".format(
                        i + 1, res[i, j], cst[i, j]
                    )
                )

    # save
    info = {}
    info["benchmark"] = benchmark.__dict__
    info["optimizer"] = optimizer.__dict__
    info["optimals"] = res.tolist()
    info["times"] = cst.tolist()

    dir_path = os.path.split(os.path.realpath(__file__))[0]
    with open(
        os.path.join(
            dir_path,
            "../logs/"
            + benchmark.name
            + "D"
            + str(benchmark.dim)
            + "/"
            + args.alg
            + "_"
            + args.name,
        ),
        "w",
    ) as f:
        json.dump(info, f)
