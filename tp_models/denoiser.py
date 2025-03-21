import deepinv as dinv

def load_denoiser(denoiser_type, device):
    if denoiser_type == "DRUNet":
        return dinv.models.DRUNet(pretrained='ckpts/drunet_color.pth').to(device)
    elif denoiser_type == "GSDRUNet":
        return dinv.models.GSDRUNet(pretrained='ckpts/GSDRUNet.ckpt').to(device)
    elif denoiser_type == "BM3D":
        return dinv.models.BM3D().to(device)
    elif denoiser_type == "DnCNN":
        dinv.models.DnCNN(pretrained='ckpts/dncnn_sigma2_color.pth').to(device)
    elif denoiser_type == "DnCNN-lip":
        return dinv.models.DnCNN(pretrained='ckpts/dncnn_sigma2_color.pth').to(device)
    else:
        raise ValueError(f"Unknown denoiser type: {denoiser_type}")
    