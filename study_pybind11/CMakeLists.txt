cmake_minimum_required(VERSION 3.28)
project(example LANGUAGES CXX)

add_subdirectory(pybind11)  # 使用 pybind header 库，提供pybind11_add_module 
add_subdirectory(src_exp_function)