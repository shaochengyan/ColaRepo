#include <torch/extension.h>

torch::Tensor tensorAdd(
    torch::Tensor a, 
    torch::Tensor b
) {
    a[0] = -888888;
    return a + b;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_add", &tensorAdd);
}