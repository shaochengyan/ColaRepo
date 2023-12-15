import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]
sources = glob.glob("*.cpp") + glob.glob("*.cu")  # all cpp and cu


setup(
    name="cppcudapytorch", 
    version="1.0", 
    author="cola", 
    ext_modules=[
        CUDAExtension(
            name="cppcudapytorch_lib", 
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

"""
pip install .
"""
