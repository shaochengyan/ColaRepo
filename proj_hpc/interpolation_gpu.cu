#include <torch/torch.h>
#include "interpolation.h"


// template <typename scalar_t>
// __global__ void trilinear_fw_kernel(
//     const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
//     const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
//     torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp
// ){
//     const int n = blockIdx.x * blockDim.x + threadIdx.x;
//     const int f = blockIdx.y * blockDim.y + threadIdx.y;

//     if (n>=feats.size(0) || f>=feats.size(2)) return;

//     // point -1~1
//     const scalar_t u = (points[n][0]+1)/2;
//     const scalar_t v = (points[n][1]+1)/2;
//     const scalar_t w = (points[n][2]+1)/2;
    
//     const scalar_t a = (1-v)*(1-w);
//     const scalar_t b = (1-v)*w;
//     const scalar_t c = v*(1-w);
//     const scalar_t d = 1-a-b-c;
//     feat_interp[n][f] = (1-u)*(a*feats[n][0][f] +
//                                b*feats[n][1][f] +
//                                c*feats[n][2][f] +
//                                d*feats[n][3][f]) + 
//                             u*(a*feats[n][4][f] +
//                                b*feats[n][5][f] +
//                                c*feats[n][6][f] +
//                                d*feats[n][7][f]);
// }

template <typename scalar_t>
__global__ void trilinear_fw_kernel(
    // 写出对应的参数即可
    // input
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    // output
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp
) {
    // 每个线程对应 第 i 个各自 的第 j 个 维度

    // step1: 固定的下标计算
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    const int f = blockDim.y * blockIdx.y + threadIdx.y;

    // step2: 排除不必要的 thread
    // TODO: 绘制图
    if ( n >= feats.size(0) || f >= feats.size(2) ) return;

    // 计算第 n 个点 的 第 f 个维度的特征 ( 8 个点的线性组合)
    // TODO: 四点现象组合 和 8点线性组合的公式
    // point \in (-1, 1) -> ratio: (x - (-1)) / (1 - (-1)
    const scalar_t u = (points[n][0] + 1) / 2;
    const scalar_t v = (points[n][1] + 1) / 2;
    const scalar_t w = (points[n][2] + 1) / 2;

    const scalar_t a = (1 - v) * (1 - w);
    const scalar_t b = (1 - v) * w;
    const scalar_t c = v * (1 - w);
    const scalar_t d = 1 - a - b - c;
    feat_interp[n][f] = (1 - u) * (
        a * feats[n][0][f] + 
        b * feats[n][1][f] + 
        c * feats[n][2][f] + 
        d * feats[n][3][f] 
    ) + 
    u * (
        a * feats[n][4][f] + 
        b * feats[n][5][f] + 
        c * feats[n][6][f] + 
        d * feats[n][7][f] 
    );

}



// template <typename scalar_t>
// __global__ void trilinear_fw_kernel(
//     // 写出对应的参数即可
//     // input
//     const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
//     const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
//     // output
//     torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp
// ) {
//     // 每个线程对应 第 i 个各自 的第 j 个 维度

//     // step1: 固定的下标计算
//     const int n = blockDim.x * blockIdx.x + threadIdx.x;
//     const int f = blockDim.y * blockIdx.y + threadIdx.y;

//     // step2: 排除不必要的 thread
//     // TODO: 绘制图
//     if ( n >= feats.size(0) || f >= feats.size(2) ) return;

//     // 计算第 n 个点 的 第 f 个维度的特征 ( 8 个点的线性组合)
//     // TODO: 四点现象组合 和 8点线性组合的公式
//     // point \in (-1, 1) -> ratio: (x - (-1)) / (1 - (-1)
//     const scalar_t u = (points[n][0] + 1) / 2;
//     const scalar_t v = (points[n][1] + 1) / 2;
//     const scalar_t w = (points[n][2] + 1) / 2;

//     const scalar_t a = (1 - v) * (1 - w);
//     const scalar_t b = (1 - v) * w;
//     const scalar_t c = v * (1 - w);
//     const scalar_t d = 1 - a - b - c;
//     feat_interp[n][f] = (1 - u) * (
//         a * feats[n][0][f] + 
//         b * feats[n][1][f] + 
//         c * feats[n][2][f] + 
//         d * feats[n][3][f] 
//     ) + 
//     u * (
//         a * feats[n][4][f] + 
//         b * feats[n][5][f] + 
//         c * feats[n][6][f] + 
//         d * feats[n][7][f] 
//     );

// }


template <typename scalar_t>
__global__ void trilinear_bw_kernel(
    // 写出对应的参数即可
    // input
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dfeats_interp,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    // output
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dL_dfeats
) {
    // 目的就是: 一个一个将 dL_dfeats 的值算出来! 即可!
    // 每个线程对应 第 i 个各自 的第 j 个 维度

    // step1: 固定的下标计算
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    const int f = blockDim.y * blockIdx.y + threadIdx.y;

    // step2: 排除不必要的 thread
    // TODO: 绘制图
    if ( n >= feats.size(0) || f >= feats.size(2) ) return;

    // 计算第 n 个点 的 第 f 个维度的特征 ( 8 个点的线性组合)
    // TODO: 四点现象组合 和 8点线性组合的公式
    // point \in (-1, 1) -> ratio: (x - (-1)) / (1 - (-1)
    const scalar_t u = (points[n][0] + 1) / 2;
    const scalar_t v = (points[n][1] + 1) / 2;
    const scalar_t w = (points[n][2] + 1) / 2;

    const scalar_t a = (1 - v) * (1 - w);
    const scalar_t b = (1 - v) * w;
    const scalar_t c = v * (1 - w);
    const scalar_t d = 1 - a - b - c;

    // 根据偏微分公式计算出偏微分的结果
    dL_dfeats[n][0][f] = (1 - u) * a * dL_dfeats_interp[n][f];
    dL_dfeats[n][1][f] = (1 - u) * b * dL_dfeats_interp[n][f];
    dL_dfeats[n][2][f] = (1 - u) * c * dL_dfeats_interp[n][f];
    dL_dfeats[n][3][f] = (1 - u) * d * dL_dfeats_interp[n][f];
    dL_dfeats[n][4][f] = (u) * a * dL_dfeats_interp[n][f];
    dL_dfeats[n][5][f] = (u) * b * dL_dfeats_interp[n][f];
    dL_dfeats[n][6][f] = (u) * c * dL_dfeats_interp[n][f];
    dL_dfeats[n][7][f] = (u) * d * dL_dfeats_interp[n][f];

}

torch::Tensor trilinear_fw_cu(
    torch::Tensor feats, 
    torch::Tensor points 
) {
    // 需要有一个 output, first 定义一个空的, 然后不断填进去,初始先明确其形状.
    const int N = feats.size(0), F = feats.size(2);  // shape[0]

    // torch.zeros(N, F, dtype=torch.float32, device="cuda:0")
    // dtype, device all store in tensor.options() 
    // way1: same to feats
    torch::Tensor feat_inpterp = torch::zeros({N, F}, feats.options());  

    // if specify other dtype and device
    // torch::zeros({N, F}, torch::dtype(torch::kInt32).device(feats.device()));

    // step1: grid and block size
    // 先明确需要多少个线程: N 和 F 的维度都可以并行

    // 两个维度需要平行运算! 第三个维度默认为1
    // 每个 block 的大小
    const dim3 block_size(16, 16);  // 128, 256, 512 需要多方尝试!

    // 计算出 grid 的大小: 一定要包含全部的 N 和 F
    const dim3 grid_size(
        (N + block_size.x - 1) / block_size.x, 
        (F + block_size.y - 1) / block_size.y
    );   // X 方向对应 N, 即不同的方块, Y 方向对应该方块中特征不同维度

    // TODO: 图解

    // 指定运算类型 && 启动 kernel 函数.
    AT_DISPATCH_FLOATING_TYPES(
        feats.type(),  // 对应的计算类型
        "trilinear_bw_cu", // 函数名, 报错名
        ([&] {
            trilinear_fw_kernel<scalar_t><<<grid_size, block_size>>>(  // kernel 的名字 <各种运算的数据类型&占位>
                // 对 torch::Tensor 需要将其转换为 CUDA 可见的
                // 将数据类型转型为 kernel (cuda) 可以识别的数据类型才可以被 kernel 函数处理 
                // scalar_t: 代替 float32, double, int 等, 或者一定是 float32 则可以直接用对应的数据类型即可
                // 3/2: 对应 tensor 的维度.
                // torch::RestrictPtrTraits: 所有元素不和其他元素有交集
                // size_t: idx 的形式 (int64)
                feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),  // input1
                points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),  // input2
                feat_inpterp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()  // output
                // 如果是普通的类型, lirujcjiw例如 bool, 则直接用即可
                // a,
            );
        }
        )
    );

    return feat_inpterp;
}


torch::Tensor trilinear_bw_cu(
    const torch::Tensor dL_dfeat_interp, 
    torch::Tensor feats, 
    torch::Tensor points 
) {

    // 需要有一个 output, first 定义一个空的, 然后不断填进去,初始先明确其形状.
    const int N = feats.size(0), F = feats.size(2);  // shape[0]

    // torch.zeros(N, F, dtype=torch.float32, device="cuda:0")
    // dtype, device all store in tensor.options() 
    // way1: same to feats
    torch::Tensor dl_dfeats = torch::zeros({N, 8, F}, feats.options());  

    // if specify other dtype and device
    // torch::zeros({N, F}, torch::dtype(torch::kInt32).device(feats.device()));

    // step1: grid and block size
    // 先明确需要多少个线程: N 和 F 的维度都可以并行

    // 两个维度需要平行运算! 第三个维度默认为1
    // 每个 block 的大小
    const dim3 block_size(16, 16);  // 128, 256, 512 需要多方尝试!

    // 计算出 grid 的大小: 一定要包含全部的 N 和 F
    const dim3 grid_size(
        (N + block_size.x - 1) / block_size.x, 
        (F + block_size.y - 1) / block_size.y
    );   // X 方向对应 N, 即不同的方块, Y 方向对应该方块中特征不同维度
    // TODO: 图解

    // 指定运算类型 && 启动 kernel 函数.
    AT_DISPATCH_FLOATING_TYPES(
        feats.type(),  // 对应的计算类型
        "trilinear_bw_cu", // 函数名, 报错名
        ([&] {
            trilinear_bw_kernel<scalar_t><<<grid_size, block_size>>>(  // kernel 的名字 <各种运算的数据类型&占位>
                // 对 torch::Tensor 需要将其转换为 CUDA 可见的
                // 将数据类型转型为 kernel (cuda) 可以识别的数据类型才可以被 kernel 函数处理 
                // scalar_t: 代替 float32, double, int 等, 或者一定是 float32 则可以直接用对应的数据类型即可
                // 3/2: 对应 tensor 的维度.
                // torch::RestrictPtrTraits: 所有元素不和其他元素有交集
                // size_t: idx 的形式 (int64)
                dL_dfeat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),  // input1
                feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),  // input1
                points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),  // input2
                dl_dfeats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()  // output
                // 如果是普通的类型, lirujcjiw例如 bool, 则直接用即可
                // a,
            );
        }
        )
    );

    return dl_dfeats;
}

