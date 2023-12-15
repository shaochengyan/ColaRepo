import torch
import cppcudapytorch_lib


print(dir(cppcudapytorch_lib))

a = torch.rand(3, 4).cuda()  # NOTE: must cuda
b = torch.rand(3, 4).cuda()
print(cppcudapytorch_lib.tensor_add(a, b))
print(a+b)
