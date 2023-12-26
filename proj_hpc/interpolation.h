#include <torch/torch.h>


torch::Tensor trilinear_fw_cu(
    torch::Tensor feats, 
    torch::Tensor point
);


torch::Tensor trilinear_fw_gpu_torch(
    torch::Tensor feats, 
    torch::Tensor point
);

torch::Tensor trilinear_fw_cpu(
    torch::Tensor feats, 
    torch::Tensor point
);

torch::Tensor trilinear_fw_cpu_mp(
    torch::Tensor feats, 
    torch::Tensor point
);




