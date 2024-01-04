import torch
import cpptorch_lib


print(dir(cpptorch_lib))

a = torch.rand(3, 4).cuda()
b = torch.rand(3, 4).cuda()
print(a)
# print(cpptorch_lib.tensor_add(a, b))
cpptorch_lib.tensor_add(a, b)
print(a)

# print(a+b)



