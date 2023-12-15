# 导出计算 gss same matrix的函数
pybind11_add_module(gss_same_matrix calc_gss_same_matrix.cpp calc_gss_same_matrix.hpp export_pybind.cpp)
target_include_directories(gss_same_matrix PUBLIC ${EIGEN_INCLUDE_DIR})


add_executable(test_calc_same_matrix main.cpp calc_gss_same_matrix.cpp)
target_include_directories(test_calc_same_matrix PRIVATE ${EIGEN_INCLUDE_DIR})
target_link_libraries(test_calc_same_matrix cnpy)