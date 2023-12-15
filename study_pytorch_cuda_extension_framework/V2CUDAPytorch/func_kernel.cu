#include <torch/extension.h>
#include "func_kernel.cuh"

torch::Tensor tensorAdd_CU(
    torch::Tensor a, 
    torch::Tensor b
) {
    // TODO: some cuda calculation.
    return a + b;
}

