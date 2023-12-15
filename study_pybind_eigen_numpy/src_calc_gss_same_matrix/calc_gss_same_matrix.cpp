#include <Eigen/Core>
#include <bitset>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "calc_gss_same_matrix.hpp"



Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> calc_same_matrix(
    const Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>& gss_src_np,
    const Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>& gss_dst_np,
    const int len_seg) {
    // return gss_src_np;
    // check size
    if (gss_src_np.cols() != gss_dst_np.cols()) {
        throw std::runtime_error(
            "Input matrices must have the same columns size.");
    }

    // result
    Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> result;
    result.resize(gss_src_np.rows(), gss_dst_np.rows());
    int num_src = gss_src_np.rows();
    int num_dst = gss_dst_np.rows();
    const int num_seg = 20;

    // 用这种方法, 类似于空间换时间(但是空间也不多), 可以快很多!
    // bit data
    std::vector<std::bitset<num_seg * 8>> gss_src(num_src);  //
    std::vector<std::bitset<num_seg * 8>> gss_dst(num_src);

    // std::cout << 1 << std::endl;

    // trans to bit data
    int label_to_bit_idx[100];
    label_to_bit_idx[71] = 0;
    label_to_bit_idx[80] = 1;
    label_to_bit_idx[10] = 2;
    label_to_bit_idx[13] = 3;
    label_to_bit_idx[18] = 4;
    label_to_bit_idx[81] = 5;
    int tmp_label;

    for (int i = 0; i < num_src; ++i) {
        for (int j = 0; j < num_seg; j++) {
            for (int k = 0; k < len_seg; k++) {
                tmp_label = gss_src_np(i, j * len_seg + k);

                if (tmp_label != -1)
                    gss_src[i].set(j * 8 + label_to_bit_idx[tmp_label], true);
                else
                    break;
            }
        }
    }

    for (int i = 0; i < num_dst; ++i) {
        for (int j = 0; j < num_seg; j++) {
            for (int k = 0; k < len_seg; k++) {
                tmp_label = gss_dst_np(i, j * len_seg + k);
                if (tmp_label != -1)
                    gss_dst[i].set(j * 8 + label_to_bit_idx[tmp_label], true);
                else
                    break;
            }
        }
    }
    // calc same matrix
    for (int i = 0; i < num_src; ++i) {
        for (int j = 0; j < num_dst; ++j) {
            result(i, j) = (gss_src[i] & gss_dst[j]).count();
        }
    }

    // for (int i = 0; i < result.rows(); ++i) {
    //     for (int j = 0; j < result.cols(); ++j) {
    //         std::cout << result(i, j) << " ";
    //     }
    //     putchar('\n');
    // }

    return result;
}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> calc_same_matrix_union(
    const Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>& gss_src_np,
    const Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>& gss_dst_np,
    const int len_seg) {
    // return gss_src_np;
    // check size
    if (gss_src_np.cols() != gss_dst_np.cols()) {
        throw std::runtime_error(
            "Input matrices must have the same columns size.");
    }

    // result
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> result;
    result.resize(gss_src_np.rows(), gss_dst_np.rows());
    int num_src = gss_src_np.rows();
    int num_dst = gss_dst_np.rows();
    const int num_seg = 20;

    // 用这种方法, 类似于空间换时间(但是空间也不多), 可以快很多!
    // bit data
    std::vector<std::bitset<num_seg * 8>> gss_src(num_src);  //
    std::vector<std::bitset<num_seg * 8>> gss_dst(num_src);

    // std::cout << 1 << std::endl;

    // trans to bit data
    int label_to_bit_idx[100];
    label_to_bit_idx[71] = 0;
    label_to_bit_idx[80] = 1;
    label_to_bit_idx[10] = 2;
    label_to_bit_idx[13] = 3;
    label_to_bit_idx[18] = 4;
    label_to_bit_idx[81] = 5;
    int tmp_label;

    for (int i = 0; i < num_src; ++i) {
        for (int j = 0; j < num_seg; j++) {
            for (int k = 0; k < len_seg; k++) {
                tmp_label = gss_src_np(i, j * len_seg + k);

                if (tmp_label != -1)
                    gss_src[i].set(j * 8 + label_to_bit_idx[tmp_label], true);
                else
                    break;
            }
        }
    }

    for (int i = 0; i < num_dst; ++i) {
        for (int j = 0; j < num_seg; j++) {
            for (int k = 0; k < len_seg; k++) {
                tmp_label = gss_dst_np(i, j * len_seg + k);
                if (tmp_label != -1)
                    gss_dst[i].set(j * 8 + label_to_bit_idx[tmp_label], true);
                else
                    break;
            }
        }
    }
    // calc same matrix
    Eigen::VectorXf len_src(num_src);
    Eigen::VectorXf len_dst(result.cols());

    for (int i = 0; i < num_src; ++i) {
        len_src(i) = gss_src[i].count();
    }

    for (int i = 0; i < num_dst; ++i) {
        len_dst(i) = gss_dst[i].count();
    }

    float num_inter;

    for (int i = 0; i < num_src; ++i) {
        for (int j = 0; j < num_dst; ++j) {
            // result(i, j) = (gss_src[i] & gss_dst[j]).count();
            num_inter = (gss_src[i] & gss_dst[j]).count();
            result(i, j) = (num_inter) / (len_src(i) + len_dst(j) - num_inter);

        }
    }

    // for (int i = 0; i < result.rows(); ++i) {
    //     for (int j = 0; j < result.cols(); ++j) {
    //         std::cout << result(i, j) << " ";
    //     }
    //     putchar('\n');
    // }

    return result;
}




/*

make && cp ~/coding/pcr/setoreg/ColaExtension/NumpyEigen/build_/src_calc_gss_same_matrix/gss_same_matrix.cpython-39-x86_64-linux-gnu.so /home/cola/anaconda3/envs/setoreg/lib/python3.9/site-packages/gss_same_matrix.cpython-39-x86_64-linux-gnu.so

*/