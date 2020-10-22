#! /usr/bin/env python

"""
setup.py file for SWIG CEC2017 benchmark
"""

from distutils.core import setup, Extension

cec17_module = Extension('_cec17',
                         sources=['cec17_wrap.c', 'cec17.c'],
                        )

setup(name = 'cec17',
      version = '1.0',
      author = 'Yifeng Li',
      description = 'CEC2017 benchmark functions for bound-constrained single-objective optimization.',
      ext_modules = [cec17_module],
      py_modules = ['cec17'],
     )
