# MPOPT: Multi-Population based Optimization

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

''mpopt'' is a flexible framework for complex optimization tasks by managing multiple populations.

**Note:** We only consider **Continuous**, **Black-Box**, **Minimization** optimization problems.

The repository contains:

    - 1. Basic operators used in [EA](#definations)s and [SIOA](#definations)s.

    - 2. Methods and examples for designing populations.

    - 3. Methods and examples for designing optimization algorithms with population.

    - 4. A new objective function interface and some pre-compiled benchmarks.

    - 5. Analysis tools for optimization results.

This repository is inspired from the framework of Fireworks Algorithm ([FWA](https://www.cil.pku.edu.cn/fwa/index.htm)). At present, it is mainly used in FWA-related research and applications.

## Table of Contents

- [Sections](#sections)
  - [Background](#background)
  - [Install](#install)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)
- [Definitions](#definitions)

## Background

''mpopt'' is motivated by the research in [FWA](https://link.springer.com/content/pdf/10.1007/978-3-642-13495-1_44.pdf), in which multiple population called fireworks are maintained for optimization. We believe that such kind of multi-population based optimization framework is of great significance in future research. Currently, we are also trying to prove its superiority theoritically.

> A lot of the latest information on FWA can be found at [here](https://www.cil.pku.edu.cn/fwa/index.htm).

## Install

''numpy'' is required for this repository.

To install, run the following commands:

```
git clone git@github.com:CavalloneChen/mpopt.git
cd mpopt
python3 setup.py install
```

If you do not have the super authority, run 'python3 setup.py install --user' instead.

## Usage

Here, we illustrate how to optimize using ''mpopt''. 'numpy' is needed in most parts of the package.

'''
import numpy as np
'''

### Objective Function

If you want to provide a handcraft objective, a callable function first is needed. For example:

'''
def my_func(X):
    return np.sum(X**2)
'''

Then, an ''ObjFunction'' instance should be created using the callable function with some additional infomation, which illustrate how the function should be called:

'''
from mpopt.tools.objective import ObjFunction
obj = ObjFunction(my_func, dim=2, lb=-1, ub=1)
'''

''ObjFunction'' instance can be called directly, but it is better to create a ''Evaluator'' instance for each run of optimization. 

'''
from mpopt.tools.objective import Evaluator
evaluator = Evaluator(obj, max_eval = 100)
'''

You can also get a evaluator from provided benchmark easily:

'''
from mpopt.benchmarks.benchmark import Benchmark
# get CEC20 benchmark in 10 dimension.
benchmark = Benchmark('CEC20', 10)
# get a evaluator for the first function
func_id = 0
evaluator = benchmark.generate(func_id)
'''

The evaluator holds the setting of optimization task and record states during the optimization. For example, a random search can simply be completed by following code:

'''
while not evaluator.terminate():
    lb, ub = evaluator.obj.lb, evaluator.obj.ub
    rand_samples = np.random.uniform(lb, ub, [10, 2]) # 10 random 2d sample each iteration
    fitness = evaluator(rand_samples)
print("Optimal: {}, Value: {}.".format(evaluator.best_x, evaluator.best_y))
'''

### Algorithm

Currently, ''mpopt'' provie some classic population based algorithms which can be used directly. We are planning on providing a lot more algorithms. For example, you can optimize a evaluator using following code:

'''
from mpopt.algorithms.LoTFWA import LoTFWA
alg = LoTFWA()
opt_val = alg.optimize(evaluator)
'''

You can also get and adjust the default params of the algorithm:

'''
params = alg.default_params()
alg.set_params(params)
'''

### Contributing

Please follow the README.md in each module for requirements to contirbutes.

If you have any question, contact Yifeng Li (liyifeng0039@gmail.com).

### License

## Definitions

- **EA**: Evolutionary Algorithm.

- **SIOA**: Swarm Intelligence Optimization Algorithm.