#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst", encoding="utf-8") as readme_file:
    readme = readme_file.read()

# with open("HISTORY.rst") as history_file:
#     history = history_file.read()

# fmt: off
__version__ = '0.5.18'
# fmt: on

requirements = [
    "torch>=1.1.0",
    "pandas",
    "matplotlib",
    "numpy>=1.17",
]


test_requirements = ["pytest>=3"]

version = __version__

setup(
    author="Jason K. Eshraghian & Vincent Sun",
    author_email="jasonesh@umich.edu",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    description="Deep learning with spiking neural networks on IPUs.",
    # entry_points={
    #     "console_scripts": [
    #         "snntorch=snntorch.cli:main",
    #     ],
    # },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme,
    include_package_data=True,
    keywords="snntorch",
    name="snntorch-ipu",
    packages=find_packages(include=["snntorch-ipu", "snntorch", "snntorch.*"]),
    #package_data = { 'so_file' : ['custom_ops.so','fast_sigmoid_custom_ops.so', 
    #                'heaviside_custom_ops.so', 'straight_through_estimator_custom_ops.so'],
    #                 'custom_ops' : ['Makefile', 'fast_sigmoid.cpp', 'heaviside_custom_op.cpp',
    #                     'straight_through_estimator.cpp']},
    package_data = {'custom_ops' : ['Makefile', 'fast_sigmoid.cpp', 'heaviside_custom_op.cpp',
                                    'straight_through_estimator.cpp']},
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/vinniesun/snntorch-ipu",
    version=__version__,
    zip_safe=False,
)
