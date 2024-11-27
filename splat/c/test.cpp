// src/my_module/main.cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(test_add, m) {
    m.doc() = "pybind11 example module"; // optional module docstring
    m.def("add", &add, "A function that adds two numbers");
}