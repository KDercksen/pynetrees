#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pip.req import parse_requirements
from setuptools import find_packages, setup
import pynetrees


install_reqs = parse_requirements('requirements.txt', session=False)

setup(
    name='pynetrees',
    version=pynetrees.__version__,
    description='Various tree-based classifiers',
    author='Koen Dercksen',
    author_email='mail@koendercksen.com',
    url='http://github.com/KDercksen/pynetrees',
    install_requires=[str(ir.req) for ir in install_reqs],
    packages=find_packages(exclude=['tests']),
)
