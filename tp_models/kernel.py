import torch
import numpy as np
import os
from torch.fft import fft2

def load_kernel(kernel_path, M, N, device):
    kt = torch.tensor(np.loadtxt(os.path.join("kernels", kernel_path)))
    (m,n) = kt.shape

    # Embed the kernel in a MxNx3 image, and put center at pixel (0,0)
    k = torch.zeros((M,N),device=device)
    k[0:m,0:n] = kt/torch.sum(kt)
    k = torch.roll(k,(-int(m/2),-int(n/2)),(0,1))
    k = k[None,None,:,:]
    return fft2(k)