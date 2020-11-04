# !/usr/bin/env python
""" Testing of implemented benchmark functions
"""

import numpy as np
from benchmarks import objective as obj

if __name__ == '__main__':
    cec13 = obj.Benchmark('CEC13')
    cec17 = obj.Benchmark('CEC17')