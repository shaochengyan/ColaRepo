#include <pybind11/pybind11.h>
#include <iostream>
#include <string>
#include "../include/module.hpp"

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}


std::string name("Cola");

PYBIND11_MODULE(ColaModule, m) {
    m.doc() = "Cola Module."; // optional module docstring
    m.def("add", &add, "A function which adds two numbers", py::arg("i"), py::arg("j")=2); 

    // C++中的 name 变量定义为 python 中的 what
    py::object name = py::cast("name");
    m.attr("what") = name;
    m.attr("the_answer") = 42;
}