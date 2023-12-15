import numpy as np
import gss_same_matrix

gss_src = np.load("Assets/gss_src.npy")
gss_dst = np.load("Assets/gss_dst.npy")


print(gss_src.shape)
print(gss_dst.shape)
out = gss_same_matrix.cola_calc_same_matrix(gss_dst, gss_dst, 3)
print(out.shape)