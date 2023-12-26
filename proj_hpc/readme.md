# 项目说明
- 利用 cuda 对三线性插值进行加速
- 编译可以用 CMake 来编译 libtorch, 也可以用 setup 安装为 python 的库
- 原理参考 [[VN_CPP_CUD_FOR_Pytorch拓展_20231213141245]]

完成了
- 在 C++ 中用 libtorch 进行测试
- 将实验接口导出到了 Python, 方便 Python 端进行调用



# 环境配准
```shell
conda create -n cppcuda python=3.8
conda activate cppcuda
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```


install python 接口
```shell
conda activate cppcuda
pip install .


cmake 也可直接编译
```