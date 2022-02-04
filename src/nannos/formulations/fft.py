#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


from .. import HAS_CUDA, get_backend
from .. import numpy as np

_device = "cuda" if HAS_CUDA else "cpu"
_BACKEND = get_backend()
if _BACKEND == "torch":
    from .. import torch
_fft2 = torch.fft.fft2 if _BACKEND == "torch" else np.fft.fft2
_ifft2 = torch.fft.ifft2 if _BACKEND == "torch" else np.fft.ifft2

BACKEND = get_backend()


def fourier_transform(u):
    if _BACKEND == "torch":
        u = torch.Tensor(u).to(_device)
    uft = _fft2(u)
    nx, ny = np.shape(uft)[:2]
    return uft / (nx * ny)


def inverse_fourier_transform(uft, shape=None, axes=(-2, -1)):
    if _BACKEND == "torch":
        uft = torch.Tensor(uft).to(_device)
    u = _ifft2(uft, s=shape, axes=axes)
    nx, ny = np.shape(u)[:2]
    return u * (nx * ny)
