import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR)]
sources = glob.glob("*.cpp") + glob.glob("*.cu")

setup(
    name="hpc_evaluation", 
    version="1.0", 
    author="cola", 
    ext_modules=[
        CUDAExtension(
            name="hpc_evaluation", 
            sources=sources, 
            include_dirs=include_dirs, 
            extra_compile_args={
                "cxx": ['-O2'], 
                "nvcc": ["-O2"]
            }
        )
    ], 
    cmdclass={
        "build_ext": BuildExtension
    }

)

