from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

# step3. 让Python能build CPP
setup(
    name="cppcuda", 
    version="1.0", 
    auto="cola", 
    author_email="shaochengyan@whu.edu.cn", 
    description="study", 
    ext_modules=[
        CppExtension(
            name="cppcuda_study", 
            sources=["interpolation.cpp"]  # 制定程序, 逗号分割
        )
    ], 
    cmdclass={
        "build_ext": BuildExtension
    }
)


"""
pip install .
"""

