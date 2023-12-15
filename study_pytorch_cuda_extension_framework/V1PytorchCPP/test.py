import torch
import cpptorch_lib


print(dir(cpptorch_lib))

a = torch.rand(3, 4)
b = torch.rand(3, 4)
print(cpptorch_lib.tensor_add(a, b))
print(a+b)
