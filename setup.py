"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
RMDL: Random Multimodel Deep Learning for Classification

* Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
* Last Update: Oct 26, 2018
* This file is part of  HDLTex project, University of Virginia.
* Free to use, change, share and distribute source code of RMDL
* Refrenced paper : HDLTex: Hierarchical Deep Learning for Text Classification
* Link: https://doi.org/10.1109/ICMLA.2017.0-134
* Comments and Error: email: kk7nc@virginia.edu
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import io
import os


import numpy
from setuptools import Extension
from setuptools import setup, find_packages

__author__ = 'Kamran Kowsari <kowsari.net>'

from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
def readfile(file):
    with open(path.join(here, file)) as f:
        long_description = f.read()
    return long_description


setup(
    name='HDLTex',
    version='1.0.5',
    description='HDLTex: Hierarchical Deep Learning for Text Classification',
    long_description=readfile('README.rst'),
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.5',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],
    url='https://github.com/kk7nc/HDLTex',
    author='Kamran Kowsari',
    author_email='kk7nc@virginia.edu',
    packages=['HDLTex'] + find_packages('HDLTex'),
    install_requires=[
        'matplotlib>=2.1.2',
        'numpy>=1.12.1',
        'pandas>=0.22.0',
        'scipy',
        'tensorflow',
        'keras>=2.0.9',
        'scikit-learn>=0.19.0',
        'nltk>=3.2.4'
    ],
    include_package_data=False
)
