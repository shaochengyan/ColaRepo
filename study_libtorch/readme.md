```shell
conda activate cppcuda
mkdir build_
cd build_


# 使用下载好的 libtorch
cmake -DCMAKE_PREFIX_PATH=./libtorch ..

# 使用 python 虚拟环境
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake -DCMAKE_PREFIX_PATH=/home/cola/anaconda3/envs/cppcuda/lib/python3.8/site-packages/torch/share/cmake ..

make


# run
./example-app
```