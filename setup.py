# setup.py
import sys

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup

ext_modules = [
    Pybind11Extension(
        "c.test_add",
        ["splat/c/test.cpp"],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="splat",
    version="0.1.0",
    description="Gaussian splatting",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=["splat"],
    package_dir={"": "splat"},
    zip_safe=False,
)