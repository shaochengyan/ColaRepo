> - [B站教程](https://www.bilibili.com/video/BV1pG411F7Yx/?p=2&spm_id_from=pageDriver&vd_source=884f4bc9c9cd33b8196db78368e7fce9) [Youtub教程](https://www.youtube.com/watch?v=oG0WUq3bRz0)
> - [github:code](https://github.com/kwea123/pytorch-cppcuda-tutorial/blob/master/interpolation_kernel.cu)
> - [Pytorch的CPP文档](https://pytorch.org/cppdocs/#pytorch-c-api)


obsidian note: [[VN_CPP_CUD_FOR_Pytorch拓展_20231213141245]]


```shell
conda create -n cppcuda python=3.8
conda activate cppcuda
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```


run test code
```shell
conda activate cppcuda
pip install .

# test
python -m test_cu_bw
python -m test_cu_fw
```

