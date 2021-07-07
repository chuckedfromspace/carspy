#!/usr/bin/env python

"""The setup script."""
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy']

setup_requirements = []

test_requirements = ['numpy']

setup(
    author="Zhiyao Yin",
    author_email='zhiyao.yin@dlr.de',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description=("Synthesizing and fitting coherent anti-Stokes Raman spectra "
                 "(CARS) in Python"),
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='carspy',
    name='carspy',
    packages=find_packages(include=['carspy', 'carspy.*']),
    package_data={'carspy': ['_constants/*.json']},
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/chuckedfromspace/carspy',
    version='0.6.1',
    zip_safe=False,
)
