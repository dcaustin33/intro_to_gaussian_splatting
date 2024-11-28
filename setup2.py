# setup.py
import os
import sys

import numpy
import pybind11
from pybind11.setup_helpers import Pybind11Extension
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# Helper function to locate the CUDA toolkit
def find_cuda_home():
    cuda_home = os.environ.get('CUDAHOME') or os.environ.get('CUDA_HOME')
    if cuda_home:
        return cuda_home
    # Try to find nvcc in the PATH
    for path in os.environ['PATH'].split(os.pathsep):
        nvcc = os.path.join(path, 'nvcc')
        if os.path.exists(nvcc):
            return os.path.dirname(os.path.dirname(nvcc))
    raise EnvironmentError('Could not find CUDA toolkit')

CUDA_HOME = find_cuda_home()

class BuildExtensionWithCUDA(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')

        original_compile = self.compiler._compile

        def new_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith('.cu'):
                # Use nvcc for .cu files
                self.set_executable('compiler_so', os.path.join(CUDA_HOME, 'bin', 'nvcc'))
                postargs = extra_postargs.get('nvcc', [])
            else:
                postargs = extra_postargs.get('gcc', [])
            original_compile(obj, src, ext, cc_args, postargs, pp_opts)
            # Reset compiler_so to the default
            self.compiler.compiler_so = self.compiler.compiler_so
        self.compiler._compile = new_compile

        build_ext.build_extensions(self)


ext_modules = [
    Pybind11Extension(
        "c.test_add",
        ["splat/c/test.cpp"],
        extra_compile_args=["-O3"],
    ),
    Extension(
        'c.render_tile_cuda',
        sources=['splat/c/render_engine.cu'],
        include_dirs=[
            pybind11.get_include(),
            os.path.join(CUDA_HOME, 'include'),
        ],
        library_dirs=[os.path.join(CUDA_HOME, 'lib64')],
        libraries=['cudart'],
        language='c++',
        extra_compile_args={
            'gcc': ['-O3', '-std=c++14'],
            'nvcc': [
                '-O3',
                '-std=c++14',
                '-Xcompiler', '-fPIC',
                '-arch=sm_61',
            ]
        },
        extra_link_args=['-L' + os.path.join(CUDA_HOME, 'lib64'), '-lcudart'],
    )
]

setup(
    name="splat",
    version="0.1.0",
    description="Gaussian splatting",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtensionWithCUDA},
    packages=["splat"],
    package_dir={"": "splat"},
    zip_safe=False,
)