import argparse

import sys
sys.path += ['../']

from utils_inverse_problems.batch_wrapper import validate

modality = 'mri'
method = 'wcrr'
n_hyperparameters = 2

# constrain the grid search
p2_max = 50
p2_init = 2
p1_init = 0.5

tol = 5e-6
max_iter = 3000
gamma_stop = 1.1

if __name__ == "__main__":
    # argpars
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")

    device = parser.parse_args().device


    EXP_NAMES = ['WCRR-CNN']
    DATA_TYPES = ['pd', 'pdfs']
    opt = '2'
    noise_sd = 0.002

    if (opt == '1'):
        coil_type = 'single'
        acc = 2
        cf = 0.16

    elif (opt == '2'):
        coil_type = 'single'
        acc = 4
        cf = 0.08

    elif (opt == '3'):
        coil_type = 'multi'
        acc = 4
        cf = 0.08

    elif (opt == '4'):
        coil_type = 'multi'
        acc = 8
        cf = 0.04

    
    for data_type in DATA_TYPES:
        for exp_name in EXP_NAMES:
            job_name = f"{modality}_coiltype_{coil_type}_acc_{acc}_cf_{cf}_noisesd_{noise_sd}_datatype_{data_type}_wcrr_{exp_name}_v2"
            validate(method = method, modality = modality, job_name=job_name, coil_type=coil_type, acc=acc, cf=cf, noise_sd=noise_sd, data_type=data_type, device=device, model_name = exp_name,n_hyperparameters=n_hyperparameters,\
            p1_init=p1_init, p2_init=p2_init, p2_max=p2_max, tol=tol, crop=True, max_iter=max_iter, gamma_stop=gamma_stop)
