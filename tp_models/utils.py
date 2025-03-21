import torch

import numpy as np

import matplotlib.pyplot as plt

def rgb2gray(u):
    return 0.2989 * u[:,:,0] + 0.5870 * u[:,:,1] + 0.1140 * u[:,:,2]

def str2(chars):
    return "{:.2f}".format(chars)

def psnr(uref,ut,M=1):
    rmse = np.sqrt(np.mean((np.array(uref.cpu())-np.array(ut.cpu()))**2))
    return 20*np.log10(M/rmse)

def tensor2im(x):
    return x.detach().cpu().permute(2,3,1,0).squeeze().clip(0,1)

def load_image(img_path, device):
    x0 = torch.tensor(plt.imread(img_path),device=device)
    M, N, _ = x0.shape
    return x0.permute(2,0,1).unsqueeze(0), M, N


if __name__ == "__main__":
    img_path = 'images/simpson512crop.png'
    device = torch.device('cpu')
    img = load_image(img_path, device)
    plt.figure()
    plt.imshow(tensor2im(img))
    plt.show()