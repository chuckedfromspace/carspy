#!/usr/bin/env python

"""The setup script."""
from setuptools import setup, find_packages

requirements = ['numpy']

setup_requirements = []

test_requirements = ['numpy']

setup(
    author="Zhiyao Yin",
    author_email='zhiyao.yin@dlr.de',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description=("Synthesizing and fitting coherent anti-Stokes Raman spectra "
                 "(CARS) in Python"),
    install_requires=requirements,
    license="BSD license",
    # long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='carspy',
    name='carspy',
    packages=find_packages(include=['carspy', 'carspy.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/chuckedfromspace/carspy',
    version='0.1.0',
    zip_safe=False,
)
