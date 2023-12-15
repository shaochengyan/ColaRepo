#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace at;


int64_t integer_round(int64_t num, int64_t denom){
  return (num + denom - 1) / denom;
}


template<class T>
__global__ void add_one_kernel(const T *const input, T *const output, const int64_t N){
  // Grid-strided loop
  for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x){
    output[i] = input[i] + 1;
  }
}


///Adds one to each element of a tensor
Tensor add_one(const Tensor &input){
  auto output = torch::zeros_like(input);

  // Common values:
  // AT_DISPATCH_INDEX_TYPES
  // AT_DISPATCH_FLOATING_TYPES
  // AT_DISPATCH_INTEGRAL_TYPES
  AT_DISPATCH_ALL_TYPES(
    input.scalar_type(), "add_one_cuda", [&](){
      const auto block_size = 128;
      const auto num_blocks = std::min(65535L, integer_round(input.numel(), block_size));
      add_one_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        input.numel()
      );
      // Always test your kernel launches
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  );

  return output;
}


///Note that we can have multiple implementations spread across multiple files, though there should only be one `def`
TORCH_LIBRARY(pytorch_cmake_example, m) {
  m.def("add_one(Tensor input) -> Tensor");
  m.impl("add_one", c10::DispatchKey::CUDA, TORCH_FN(add_one));
  //c10::DispatchKey::CPU is also an option
}