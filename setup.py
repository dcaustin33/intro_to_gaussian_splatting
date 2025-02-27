import pybind11
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="render_tile_cuda",
    ext_modules=[
        CUDAExtension(
            "splat.c.render_tile_cuda",
            ["splat/c/render_engine.cu"],
            include_dirs=[
                pybind11.get_include(),
            ],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-Xcompiler",
                    "-fPIC",
                    "-arch=sm_61",
                ],
                "cxx": ["-O3", "-std=c++17"],
            },
        ),
        CUDAExtension(
            "splat.c.render_engine_backwards",
            ["splat/c/render_engine_backwards.cu"],
            include_dirs=[
                pybind11.get_include(),
            ],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-Xcompiler",
                    "-fPIC",
                    "-arch=sm_61",
                ],
                "cxx": ["-O3", "-std=c++17"],
            },
        ),
        CUDAExtension(
            "splat.c.preprocessing",
            ["splat/c/preprocessing.cu"],
            include_dirs=[
                pybind11.get_include(),
            ],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-Xcompiler",
                    "-fPIC",
                    "-arch=sm_61",
                ],
                "cxx": ["-O3", "-std=c++17"],
            },
        ),
        CUDAExtension(
            "splat.custom_backwards_implementation.gaussian_weight_derivatives",
            ["splat/custom_backwards_implementation/gaussian_weight_derivatives.cu"],
            include_dirs=[
                pybind11.get_include(),
            ],
            extra_compile_args={
                "nvcc": ["-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-arch=sm_61"],
                "cxx": ["-O3", "-std=c++17"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
