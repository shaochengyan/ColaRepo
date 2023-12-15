#include <torch/extension.h>
#include "func_kernel.cuh"

torch::Tensor tensorAdd(
    torch::Tensor a, 
    torch::Tensor b
) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    return tensorAdd_CU(a, b);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_add", &tensorAdd);
}