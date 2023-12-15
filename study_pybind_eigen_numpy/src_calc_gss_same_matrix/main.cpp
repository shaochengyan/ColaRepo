#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>
#include "calc_gss_same_matrix.hpp"
#include "cnpy.h"

using namespace Eigen;
using namespace std;

MatrixXi loadGSS(const string& path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    int* data_ptr = arr.data<int>();

    // data to eigen
    MatrixXi gss(arr.shape[0], arr.shape[1]);
    for (int i = 0; i < arr.shape[0]; ++i) {
        for (int j = 0; j < arr.shape[1]; ++j) {
            gss(i, j) = data_ptr[i * arr.shape[1] + j];
            // cout << data_ptr[i * arr.shape[1] + j];
        }
    }
    return gss;
}

#define MEASURE_EXECUTION_TIME(code, repetitions) \
    do { \
        auto start_time = std::chrono::high_resolution_clock::now(); \
        for (int _rep = 0; _rep < repetitions; ++_rep) { \
            code; \
        } \
        auto end_time = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count(); \
        double average_time = static_cast<double>(duration) / repetitions / 1000000.0; \
        std::cout << "Average execution time: " << average_time << " seconds." << std::endl; \
    } while (false)

int main(int argc, const char** argv) {
    if (argc < 3) {
        printf("Please input npy file.\n");
        return 0;
    }
    // const string filepath_src = "/home/cola/coding/pcr/setoreg/ColaExtension/NumpyEigen/Assets/gss_src.npy";
    const string filepath_src(argv[1]);
    auto gss_src = loadGSS(filepath_src);
    // const string filepath_dst = "/home/cola/coding/pcr/setoreg/ColaExtension/NumpyEigen/Assets/gss_dst.npy";
    const string filepath_dst(argv[2]);
    auto gss_dst = loadGSS(filepath_dst);

    // cout << gss_src.row(0) << endl;
    // cout << gss_dst.row(gss_dst.rows() - 1) << endl;

    // auto same_matrix = calc_same_matrix(gss_src, gss_dst, 3, true);
    // auto same_matrix = calc_same_matrix(gss_dst, gss_src, 3);

    printf("GSS_SRC: (%ld, %ld)\n", gss_src.rows(), gss_src.cols());
    printf("GSS_DST: (%ld, %ld)\n", gss_dst.rows(), gss_dst.cols());

    MEASURE_EXECUTION_TIME(
        auto same_matrix = calc_same_matrix(gss_dst, gss_src, 3);, 
        10
    );


    MEASURE_EXECUTION_TIME(
        auto same_matrix2 = calc_same_matrix_union(gss_dst, gss_src, 3);, 
        10
    );

    return 0;
}