cmake_minimum_required(VERSION 3.12)
project(YourProjectName)

set(CMAKE_CXX_STANDARD 11)

# include dir
set(EIGEN_INCLUDE_DIR "/usr/include/eigen3")
include_directories("cnpy")

# sub project
add_subdirectory(cnpy)
add_subdirectory(pybind11)
add_subdirectory(src_calc_gss_same_matrix)
