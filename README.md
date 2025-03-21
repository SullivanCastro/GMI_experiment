# CRR-NN: Convex Ridge Regularization for Inverse Problems

## Overview

This project focuses on solving inverse problems in imaging using Plug-and-Play (PnP) methods and Convex Ridge Regularization Neural Networks (CRR-NN). It provides different denoising strategies, including:

- PnP-PGD: Plug-and-Play proximal gradient descent denoising.
- RED: Regularization by Denoising.
- CRR-NN: Convex Ridge Regularization-based neural network for iterative reconstruction.

All implementations are found in `tp_models/main.py`, with experiments detailed in `experiments.ipynb`.

---

## Installation

To install the required dependencies, use pip:

```bash
pip install -r requirements.txt
```

or Pixi:

```bash
pixi install
pixi run jupyter notebook
```

If Pixi is not installed, get it with:

```bash
curl -sSL https://install.prefix.dev | bash
```

---

## Usage

Run denoising with one of the following methods:

### PnP-PGD Denoising

```python
from main import pnp_pgd_denoising
denoised_img, psnr_value = pnp_pgd_denoising("images/noisy_image.png", nu=1/255)
```

### RED Denoising

```python
from main import red_denoising
denoised_img, psnr_value = red_denoising("images/noisy_image.png", nu=1/255)
```

### CRR-NN Denoising

```python
from main import crr_nn_denoising
denoised_img, psnr_value = crr_nn_denoising("images/noisy_image.png", nu=1/255)
```

All methods return the denoised image and the corresponding PSNR value.

---

## Debugging Tips

### Import Errors (`ModuleNotFoundError`)
If you encounter module import errors, ensure that your `sys.path` is set correctly:

```python
import sys
import os
sys.path.append(os.path.abspath("."))
```

### PyTorch `weights_only=False` Fix
If you encounter this error:

```
UnpicklingError: Weights only load failed...
```

It's due to PyTorch 2.6+ using `weights_only=True` by default.

#### Fix Option 
Modify the file:

```
/usr/local/lib/python3.11/dist-packages/torch/serialization.py
```

At line 1470, set:

```python
weights_only=False
```

---

## Project Structure

```bash
├── tp_models/                      # Scripts for training/testing models
│   ├── main.py                     # Main entry point for inference
│   ├── blurring.py                  # Implements blurring operators
│   ├── kernel.py                    # Loads kernels for degradation
│   ├── denoiser.py                   # Loads pre-trained denoisers
│   ├── prox_operator.py              # Implements proximal operators
│   ├── utils_tp.py                   # Utility functions (PSNR, image loading)
│   ├── visualization.py              # Visualization utilities
│   ├── __init__.py                    # Module initialization
│
├── convex_ridge_regularizers/       # Core implementation of CRR-NN
│   ├── models/                       # Neural networks & architectures
│   │   ├── convex_ridge_regularizer.py
│   │   ├── multi_conv.py
│   │   ├── linear_spline.py
│   │   ├── quadratic_spline.py
│   │   ├── utils.py
│   │   ├── __init__.py
│   ├── inverse_problems/             # Tools for handling inverse problems
│   │   ├── reconstruction_map_crr.py
│   │   ├── utils_inverse_problems.py
│   │   ├── __init__.py
│
├── images/                          # Input/output images
├── experiments.ipynb                 # Notebook with all results
├── results.pdf                        # PSNR & comparison charts
├── README.md                         # This documentation
├── requirements.txt                   # Python dependencies
├── pixi.lock / pixi.toml              # Pixi environment config
├── .gitignore                         # Git exclusions
```

---

## Acknowledgments

This project was developed as part of the GMI course at École des Ponts ParisTech (2024-2025). It is inspired by recent research in Plug-and-Play optimization and learned convex regularization.


