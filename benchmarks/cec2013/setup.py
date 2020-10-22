#! /usr/bin/env python

"""
setup.py file for SWIG CEC2013 benchmark
"""

from distutils.core import setup, Extension

cec13_module = Extension('_cec13',
                         sources=['cec13_wrap.c', 'cec13.c'],
                        )

setup(name = 'cec13',
      version = '1.0',
      author = 'Yifeng Li',
      description = 'CEC2013 benchmark functions for bound-constrained single-objective optimization.',
      ext_modules = [cec13_module],
      py_modules = ['cec13'],
     )
