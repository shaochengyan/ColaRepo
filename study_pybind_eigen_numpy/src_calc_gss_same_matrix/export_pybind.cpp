#include "calc_gss_same_matrix.hpp"

// pybind11
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
namespace py = pybind11;

// export
// bind
PYBIND11_MODULE(gss_same_matrix, m) {
    // ransac with grades
    m.def("cola_calc_same_matrix", &calc_same_matrix, py::arg("gss_src_np"),
          py::arg("gss_dst_np"), py::arg("len_seg"));
    m.def("cola_calc_same_matrix_union", &calc_same_matrix_union, py::arg("gss_src_np"),
          py::arg("gss_dst_np"), py::arg("len_seg"));
}

