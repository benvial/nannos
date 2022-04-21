#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


from scipy.fftpack import fft2 as _fft2_scipy
from scipy.fftpack import ifft2 as _ifft2_scipy

from .. import BACKEND, HAS_TORCH
from .. import backend as bk
from .. import jit

if HAS_TORCH:
    from torch.fft import fft2 as _fft2_torch
    from torch.fft import ifft2 as _ifft2_torch


def fourier_transform(u, s=None, axes=(-2, -1)):
    if BACKEND == "scipy":
        uft = _fft2_scipy(u, shape=s, axes=axes)
    elif BACKEND == "torch":
        uft = _fft2_torch(u, s=s, dim=axes)
    else:
        uft = bk.fft.fft2(u, s=s, axes=axes)
    nx, ny = uft.shape[axes[0]], uft.shape[axes[1]]
    return uft / (nx * ny)


def inverse_fourier_transform(uft, s=None, axes=(-2, -1)):
    if BACKEND == "scipy":
        u = _ifft2_scipy(uft, shape=s, axes=axes)
    elif BACKEND == "torch":
        u = _ifft2_torch(uft, s=s, dim=axes)
    else:
        u = bk.fft.ifft2(uft, s=s, axes=axes)
    nx, ny = uft.shape[axes[0]], uft.shape[axes[1]]
    return u * (nx * ny)


fourier_transform = jit(fourier_transform, static_argnums=(1, 2))
inverse_fourier_transform = jit(inverse_fourier_transform, static_argnums=(1, 2))
