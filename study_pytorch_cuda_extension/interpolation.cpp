#include <torch/extension.h>
#include "interpolation_kernel.cuh"


// step1: CPP interface for python 
/* FW
作用: 提供C++的接口, 让 Python 能对其进行访问! 但是实际计算都会在 cuda 端做!
*/
torch::Tensor trilnear_interpolation_fw(
    torch::Tensor feats, 
    torch::Tensor point
) {
    // check input is contiguous and is tensor
    CHECK_INPUT(feats);
    CHECK_INPUT(point);
    
    // return rslt from cuda
    return trilinear_fw_cu(feats, point);
}

// BW
torch::Tensor trilnear_interpolation_bw(
    torch::Tensor dL_dfeat_interp, 
    torch::Tensor feats, 
    torch::Tensor point
) {
    CHECK_INPUT(dL_dfeat_interp);
    CHECK_INPUT(feats);
    CHECK_INPUT(point);
    return trilinear_bw_cu(dL_dfeat_interp, feats, point);
}


// step2: python 的接口名称 + 对应的 CPP 的函数名.
// NOTE: TORCH_EXTENSION_NAME 不要写错了!
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("triliear_interpolation_fw", &trilnear_interpolation_fw);
    m.def("triliear_interpolation_bw", &trilnear_interpolation_bw);
}