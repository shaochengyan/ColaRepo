import torch
import cppcuda_study
import time

def trilinear_interpolation_py(feats, points):
    """
    Inputs:
        feats: (N, 8, F)
        points: (N, 3) local coordinates in [-1, 1]
    
    Outputs:
        feats_interp: (N, F)
    """
    u = (points[:, 0:1]+1)/2
    v = (points[:, 1:2]+1)/2
    w = (points[:, 2:3]+1)/2
    a = (1-v)*(1-w)
    b = (1-v)*w
    c = v*(1-w)
    d = 1-a-b-c

    feats_interp = (1-u)*(a*feats[:, 0] +
                          b*feats[:, 1] +
                          c*feats[:, 2] +
                          d*feats[:, 3]) + \
                       u*(a*feats[:, 4] +
                          b*feats[:, 5] +
                          c*feats[:, 6] +
                          d*feats[:, 7])
    
    return feats_interp


def test_cuda():
    print(dir(cppcuda_study))


    # input
    # N, F = 60240, 256
    N, F = 1024, 256

    feats = torch.rand(size=(N, 8, F), device="cuda:0")
    points = torch.rand(size=(N, 3), device="cuda:0") * 2 - 1


    # test
    t = time.time()
    for i in range(10):
        out = cppcuda_study.triliear_interpolation_fw(feats, points)
    print(time.time() - t)
    print(out.requires_grad) 

    # check output
    t = time.time()
    for i in range(10):
        out_py = trilinear_interpolation_py(feats, points)
    print(time.time() - t)

    print(torch.allclose(out, out_py))

if __name__=="__main__":
    test_cuda()

"""
python -m test_cu
"""
