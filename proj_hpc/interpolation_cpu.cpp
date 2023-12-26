#include <torch/torch.h>
#include "interpolation.h"
#include "cola_utils.h"
#include <iostream>
using namespace std;



// torch::Tensor trilinear_fw_cpu(
//     torch::Tensor feats, 
//     torch::Tensor point
// ) {
//     // init
//     int N = feats.size(0);
//     int F = feats.size(2);
//     torch::Tensor feats_interp = torch::zeros({N, F}, feats.options());

//     // fill
    


//     return feats_interp;
// }


torch::Tensor trilinear_fw_gpu_torch(torch::Tensor feats, torch::Tensor points) {
    int N = feats.size(0);
    int F = feats.size(2);
    
    torch::Tensor u = (points.slice(1, 0, 1) + 1) / 2;  // Nx1
    torch::Tensor v = (points.slice(1, 1, 2) + 1) / 2;  // Nx1
    torch::Tensor w = (points.slice(1, 2, 3) + 1) / 2;  // Nx1

    torch::Tensor a = (1 - v) * (1 - w);
    torch::Tensor b = (1 - v) * w;
    torch::Tensor c = v * (1 - w);
    torch::Tensor d = 1 - a - b - c;

    torch::Tensor feats_interp = (1 - u) * (a * feats.slice(1, 0, 1).squeeze(1) +
                                             b * feats.slice(1, 1, 2).squeeze(1) +
                                             c * feats.slice(1, 2, 3).squeeze(1) +
                                             d * feats.slice(1, 3, 4).squeeze(1)) +
                                  u * (a * feats.slice(1, 4, 5).squeeze(1) +
                                       b * feats.slice(1, 5, 6).squeeze(1) +
                                       c * feats.slice(1, 6, 7).squeeze(1) +
                                       d * feats.slice(1, 7, 8).squeeze(1));


    return feats_interp;
}

torch::Tensor trilinear_fw_cpu(torch::Tensor feats, torch::Tensor points) {
    int N = feats.size(0);
    int F = feats.size(2);
    
    torch::Tensor u = (points.slice(1, 0, 1) + 1) / 2;  // Nx1
    torch::Tensor v = (points.slice(1, 1, 2) + 1) / 2;  // Nx1
    torch::Tensor w = (points.slice(1, 2, 3) + 1) / 2;  // Nx1

    torch::Tensor a = (1 - v) * (1 - w);
    torch::Tensor b = (1 - v) * w;
    torch::Tensor c = v * (1 - w);
    torch::Tensor d = 1 - a - b - c;

    torch::Tensor feats_interp = (1 - u) * (a * feats.slice(1, 0, 1).squeeze(1) +
                                             b * feats.slice(1, 1, 2).squeeze(1) +
                                             c * feats.slice(1, 2, 3).squeeze(1) +
                                             d * feats.slice(1, 3, 4).squeeze(1)) +
                                  u * (a * feats.slice(1, 4, 5).squeeze(1) +
                                       b * feats.slice(1, 5, 6).squeeze(1) +
                                       c * feats.slice(1, 6, 7).squeeze(1) +
                                       d * feats.slice(1, 7, 8).squeeze(1));

    return feats_interp;
}

