#include <Eigen/Core>
#include <bitset>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

// 定义接口函数
Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> calc_same_matrix(
    const Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>& gss_src_np,
    const Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>& gss_dst_np,
    const int len_seg);

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> calc_same_matrix_union(
    const Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>& gss_src_np,
    const Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>& gss_dst_np,
    const int len_seg);
