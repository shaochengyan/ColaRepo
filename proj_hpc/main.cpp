#include <iostream>
#include "interpolation.h"

// cola
#include "cola_utils.h"
#include "evaluation.h"

using namespace std;


int main(int argc, char **argv) {

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <N> <F>" << std::endl;
        return 1; // 返回非零值表示参数错误
    }

    // 解析命令行参数 N 和 F
    int N = std::atoi(argv[1]);
    int F = std::atoi(argv[2]);
    int T = std::atoi(argv[3]);

    runEvaluation(N, F, T, 1);
}


/*
./main 1024 256 1000
*/