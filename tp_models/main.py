from time import time 

import torch

import os
import sys
sys.path.append("convex_ridge_regularizers")
sys.path.append("convex_ridge_regularizers/inverse_problems")
from models import utils
from inverse_problems import AdaGD_Recon


sys.path.append("tp_models")
from utils_tp import psnr, load_image
from kernel import load_kernel
from prox_operator import f, proxf
from denoiser import load_denoiser
from blurring import blurring, A


def init_denoising(image_path, kernel_path="kernel8.txt", nu=1/255, is_crr_nn=False):
    # Define the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load image
    x0, M, N = load_image(image_path, device, is_crr_nn)

    # load kernel
    fk = load_kernel(kernel_path, M, N, device)

    # blurring operator
    y = blurring(x0, fk, nu, device)

    return x0, fk, y, device


def pnp_pgd_denoising(image_path, nu= 1/255, denoiser_type="DRUNet", kernel_path="kernel8.txt", niter=100):
    # Initialize
    x0, fk, y, device = init_denoising(image_path, kernel_path, nu)

    # load denoiser
    D = load_denoiser(denoiser_type, device)

    tau = 1.9*nu**2
    s = 2*nu  # strength of the denoiser (corresponding to notation sigma)

    # initialize
    x = y.clone()
    x.requires_grad_(True)
    normxinit = torch.linalg.vector_norm(x)

    rtab = []
    psnrtab = []
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

    return x.squeeze().permute(1, 2, 0).detach().cpu(), psnr(x0.detach().cpu(), x.detach().cpu()), psnrtab


def red_denoising(image_path, nu=1/255, tau=2e-5, lam=1600, gamma=0.4, eta=0.9, kernel_path="kernel8.txt", niter=100):
    # Initialize
    x0, fk, y, device = init_denoising(image_path, kernel_path, nu)

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

    return x.squeeze().permute(1, 2, 0).detach().cpu(), psnr(x0.detach().cpu(), x.detach().cpu()), psnrtab


def crr_nn_denoising(image_path, nu=1/255, lmbd=25, mu=4, sigma_training=5, t=10, kernel_path="kernel8.txt", niter=100):
    # Initialize
    x0, fk, y, device = init_denoising(image_path, kernel_path, nu, is_crr_nn=True)

    # def H, Ht
    H = lambda x: A(x, fk)
    Ht = lambda y: A(y, torch.conj(fk))

    # Model
    exp_name = f'Sigma_{sigma_training}_t_{t}'
    model = utils.load_model(exp_name, str(device))
    model.prune(change_splines_to_clip=False, prune_filters=True, collapse_filters=True)
    model.initializeEigen(size=100)
    model.precise_lipschitz_bound(n_iter=100)

    # Denoise
    x_out, psnr_, _, _, _, _, _= AdaGD_Recon(y=y, H=H, Ht=Ht, model=model, lmbd=lmbd, mu=mu, x_gt=x0, tol=1e-6, max_iter=niter, track_cost=True)

    return x_out.squeeze().detach().cpu(), psnr(x0.detach().cpu(), x_out.detach().cpu()), psnr_[1:]
