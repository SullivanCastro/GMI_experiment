import torch
from torch.fft import fft2, ifft2

def A(x, fk):
    return ifft2(fk * fft2(x)).real

def blurring(x0, fk, nu, device):
    return A(x0, fk) + nu * torch.randn_like(x0, device=device)