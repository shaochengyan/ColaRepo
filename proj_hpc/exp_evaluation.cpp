#include <torch/extension.h>
#include "evaluation.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_hpc_evaluation", &runEvaluation);
}