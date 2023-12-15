from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

# step3. 让Python能build CPP
setup(
    name="cpptorch", 
    version="1.0", 
    auto="cola", 
    author_email="shaochengyan@whu.edu.cn", 
    description="study", 
    ext_modules=[
        CppExtension(
            name="cpptorch_lib", 
            sources=["func.cpp"]  # 制定程序, 逗号分隔
        )
    ], 
    cmdclass={
        "build_ext": BuildExtension
    }
)


"""
pip install .
"""
