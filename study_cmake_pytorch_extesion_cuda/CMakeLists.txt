# 这句话指定了最低所需的CMake版本。在此示例中，至少需要CMake 3.28 版本才能运行
cmake_minimum_required (VERSION 3.28)

# 这句话定义了项目的名称（pytorch_cmake_example），以及所使用的编程语言（C++和CUDA）。这表明这是一个混合编程项目，涉及到C++和CUDA。
# 即需要使用 nvcc
project(pytorch_cmake_example LANGUAGES CXX CUDA)

# 查找Python的安装。它会查找Python，并检查是否安装了开发组件（Development components）。通常，这用于在CMake中与Python进行交互，例如，将Python代码嵌入到C++代码中。
find_package(Python REQUIRED COMPONENTS Development)
# 这句话用于查找PyTorch库的安装。它确保了你的项目可以使用PyTorch。
find_package(Torch REQUIRED)

# 设置 CUDA 计算架构:这个条件语句检查是否已经定义了CMAKE_CUDA_ARCHITECTURES变量。如果没有定义，它会将该变量设置为75。这个变量用于指定要生成的CUDA代码的架构。在此示例中，它被设置为CUDA架构75。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
  # set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# 这句话用于创建一个名为pytorch_cmake_example的共享库，并指定要编译的源文件为main.cu。这个库将包含你的CUDA代码。
add_library(pytorch_cmake_example SHARED
  main.cu
)

# 这句话指定了编译pytorch_cmake_example库时要使用的C++标准。在此示例中，使用C++11标准。
target_compile_features(pytorch_cmake_example PRIVATE cxx_std_11)

# 这句话用于指定pytorch_cmake_example库的链接依赖项。它链接了PyTorch库和Python库，以便在代码中使用它们的功能。
target_link_libraries(pytorch_cmake_example PRIVATE ${TORCH_LIBRARIES} Python::Python)

# 这句话用于为CUDA编译选项设置g++编译器的路径。这通常用于指定CUDA代码的编译器。
target_compile_options(pytorch_cmake_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ccbin g++-9>)