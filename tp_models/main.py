from time import time 

from math import gamma

import torch

from .utils import psnr, load_image
from .kernel import load_kernel
from .prox_operator import f, proxf
from .denoiser import load_denoiser
from .blurring import blurring, A

def pnp_pgd_denoising(image_path, nu= 2/255, denoiser_type="DRUNet", kernel_path="kernel8.txt", niter=100):
    # Define the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load image
    x0, M, N = load_image(image_path, device)

    # load kernel
    fk = load_kernel(kernel_path, M, N, device)

    # blurring operator
    y = blurring(x0, fk, nu, device)

    # load denoiser
    D = load_denoiser(denoiser_type, device)

    tau = 1.9*nu**2
    s = 2*nu  # strength of the denoiser (corresponding to notation sigma)

    # initialize
    x = y.clone()
    x.requires_grad_(True)
    normxinit = torch.linalg.vector_norm(x)

    psnrtab = []
    rtab = []
    t0 = time()

    print('[%4d/%4d] [%.5f s] PSNR = %.2f'%(0,niter,0,psnr(x0,y)))

    for it in range(niter):
        ### TODO ###
        grad = torch.autograd.grad(f(x, fk, y, A, nu), x, create_graph=False, retain_graph=False)[0]

        with torch.no_grad():
            x_prev = x.clone()
            x -= tau * grad
            x = D(x, sigma=s)
        x.requires_grad_(True)

        rtab.append((torch.linalg.vector_norm(x - x_prev) / normxinit).detach().cpu())

        psnrt = psnr(x0, x.detach())
        psnrtab.append(psnrt)

        if (it+1)%10==0:
            print('[%4d/%4d] [%.5f s] PSNR = %.2f'%(it+1,niter,time()-t0,psnrt))

    return x.squeeze().permute(1, 2, 0).detach().cpu().numpy(), psnrtab, rtab


def red_denoising(image_path, nu=2/255, tau=2e-5, lam=1600, gamma=0.4, eta=0.9, kernel_path="kernel8.txt", niter=100):
    # Define the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load image
    x0, M, N = load_image(image_path, device)

    # load kernel
    fk = load_kernel(kernel_path, M, N, device)

    # blurring operator
    y = blurring(x0, fk, nu, device)

    # load denoiser
    D = load_denoiser("GSDRUNet", device)

    # Parameters
    s = 1.8*nu  # strength of the denoiser

    # Define the objective function
    def F(x, lam):
        return lam*D.potential(x,sigma=s) + f(x, fk, y, A, nu)

    # initialize
    x = torch.clone(y).requires_grad_(True)
    normxinit = torch.linalg.vector_norm(x.detach())
    
    psnrtab = []
    rtab = []
    t0 = time()

    t0 = time()
    print('[%4d/%4d] [%.5f s] PSNR = %.2f'%(0,niter,0,psnr(x0,y)))

    for it in range(niter):

        tau_it = tau

        T_x = proxf(x - tau_it * lam * torch.autograd.grad(D.potential(x,sigma=s), x, create_graph=False, retain_graph=False)[0], fk, y, nu, tau_it)
        left_term = F(x, lam) - F(T_x, lam)
        right_term = gamma * torch.sum((T_x - x)**2)
        while left_term < right_term / tau_it:
            tau_it *= eta

        x_prev = x.clone()
        x = proxf(x, fk, y, nu, tau_it)

        rtab.append((torch.linalg.vector_norm(x - x_prev) / normxinit).detach().cpu())

        psnrt = psnr(x0, x.detach())
        psnrtab.append(psnrt)

        if (it+1)%10==0:
            print('[%4d/%4d] [%.5f s] PSNR = %.2f'%(it+1,niter,time()-t0,psnr(x0,x.detach())))

    return x.squeeze().permute(1, 2, 0).detach().cpu().numpy(), psnrtab, rtab