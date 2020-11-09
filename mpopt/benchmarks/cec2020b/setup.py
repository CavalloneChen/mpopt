#! /usr/bin/env python

"""
setup.py file for SWIG CEC2020 benchmark
"""

from distutils.core import setup, Extension

cec20_module = Extension(
    "_cec20",
    sources=["cec20_wrap.c", "cec20.c"],
)

setup(
    name="cec20",
    version="1.0",
    author="Yifeng Li",
    description="CEC2020 benchmark functions for bound-constrained single-objective optimization.",
    ext_modules=[cec20_module],
    py_modules=["cec20"],
)
