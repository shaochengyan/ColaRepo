#include "evaluation.h"

std::tuple<int, int, int, bool> runEvaluation(const int N,
                                              const int F,
                                              const int T,
                                              bool is_print) {
    torch::Device device(torch::kCUDA);
    torch::Tensor points_gpu = torch::rand({N, 3}, device) * 2 - 1;
    torch::Tensor feats_gpu = torch::rand({N, 8, F}, device);

    torch::Tensor points_cpu = points_gpu.cpu();
    torch::Tensor feats_cpu = feats_gpu.cpu();

    cola::Timer timer;

    torch::Tensor feats_interp_gpu_ours;
    torch::Tensor feats_interp_gpu_torch;
    torch::Tensor feats_interp_cpu_torch;

    // CPU
    timer.start();
    FOR_N_TIMES(T) {
        feats_interp_cpu_torch = trilinear_fw_cpu(feats_cpu, points_cpu);
    }
    int duration_cpu_torch = timer.end_ms("CPU_Pytorch: ", is_print);

    // gpu_torch
    timer.start();
    FOR_N_TIMES(T) {
        feats_interp_gpu_torch = trilinear_fw_gpu_torch(feats_gpu, points_gpu);
    }
    int duration_gpu_pytorch = timer.end_ms("GPU_Pytorch: ", is_print);

    // GPU
    timer.start();
    FOR_N_TIMES(T) {
        feats_interp_gpu_ours = trilinear_fw_cu(feats_gpu, points_gpu);
    }
    int duration_gpu_ours = timer.end_ms("GPU_Ours: ", is_print);

    //  compare
    double threshold = 1e-6;  // 设置阈值
    torch::Tensor diff =
        torch::abs(feats_interp_gpu_ours.cpu() - feats_interp_gpu_torch.cpu());
    bool are_equal = torch::all(diff < threshold).item<bool>();

    if (is_print) {
        if (are_equal) {
            std::cout << "feats_interp equal." << std::endl;
        } else {
            std::cout << "feats_interp not equal." << std::endl;
        }
    }
    return std::make_tuple(duration_gpu_ours, duration_gpu_pytorch,
                           duration_cpu_torch, are_equal);
}
