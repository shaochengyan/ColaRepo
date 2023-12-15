#include <torch/extension.h>

torch::Tensor tensorAdd(
    torch::Tensor a, 
    torch::Tensor b
) {
    return a + b;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_add", &tensorAdd);
}