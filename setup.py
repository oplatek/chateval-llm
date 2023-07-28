#!/usr/bin/env python3
from setuptools import find_packages, setup


setup(
    name="chateval",
    version="0.0.1",
    python_requires=">=3.8",
    description="Shared utilities for experiments for Chateval (DSTC11 Track 4)",
    author="UFAL DSG/NLG group",
    packages=find_packages('./chateval/'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
