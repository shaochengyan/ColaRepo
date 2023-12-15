import torch
import cppcuda_study

from typing import Any
import time


class TriliearInterpolationCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, points):
        # ctx: 存储反向传播需要的模块输入数据
        feats_interp = cppcuda_study.triliear_interpolation_fw(feats, points)

        # 存储前向传播的输入输出, 这样在反向传播时: 输入 X 偏微分 -> 才能得到最终的 delta
        ctx.save_for_backward(feats, points)

        # 返回该模块的输出数据
        return feats_interp

    @staticmethod
    def backward(ctx, dL_dfeat_interp) -> Any:
        # INPUT: ctx + 模型后端传回的梯度

        # take out the input of this module (save in forward function)
        feats, points = ctx.saved_tensors

        # calculate gradient
        dL_dfeats = cppcuda_study.triliear_interpolation_bw(dL_dfeat_interp.contiguous(), feats, points)

        # 返回L对所有输入的梯度
        return dL_dfeats, None
        # 第一个返回值对应 对 输入1: feats 的偏微分
        # 但是不想对 points 做偏微分, 因此第二个返回 None
        
        
if __name__=="__main__":
    # input
    # N, F = 60240, 256
    N, F = 1024, 256

    points = torch.rand(size=(N, 3), device="cuda:0") * 2 - 1
    feats = torch.rand(size=(N, 8, F), device="cuda:0").requires_grad_()
    feats2 = feats.clone().requires_grad_()
    feats2.retain_grad()

    # test
    t = time.time()
    out_cuda = TriliearInterpolationCuda.apply(feats, points)
    torch.cuda.synchronize()
    print(time.time() - t)

    # check output
    from test_cu_fw import trilinear_interpolation_py
    t = time.time()
    out_py = trilinear_interpolation_py(feats2, points)
    torch.cuda.synchronize()
    print(time.time() - t)

    print(torch.allclose(out_cuda, out_py))

    # backward
    loss = out_cuda.sum()
    loss.backward()

    loss2 = out_py.sum()
    loss2.backward()

    print(feats2.grad)
    print("bw all close: ", torch.allclose(feats.grad, feats2.grad))


"""
python -m test_cu_bw
"""