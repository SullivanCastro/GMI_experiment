
import torch
from torch.fft import fft2, ifft2

def f(x, fk, y, A, nu):
    return 0.5 * torch.sum((A(x, fk)-y)**2) / nu**2

def proxf(x, fk, y, nu, tau):
    ### TODO ###
    a = torch.conj(fk)*fft2(y)/nu**2 + fft2(x)/tau
    b = torch.abs(fk)**2/nu**2 + 1/tau
    return ifft2(a/b).real