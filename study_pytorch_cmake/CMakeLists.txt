cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(example-app)

#Add the torch library directory
# list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/envs/lib/python3.8/site-packages/torch")
list(APPEND CMAKE_PREFIX_PATH "/home/cola/anaconda3/envs/cppcuda/lib/python3.8/site-packages/torch/share/cmake/Torch")
# list()

#This is needed to be able to find the mkl and other dependent libraries
link_directories("/home/cola/anaconda3/envs/cppcuda/lib")
set(ENV{MKLROOT} "/home/cola/anaconda3/envs/cppcuda/lib")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp)

#We need to add pthread and omp manually here
target_link_libraries(example-app "${TORCH_LIBRARIES}" pthread omp)

set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
