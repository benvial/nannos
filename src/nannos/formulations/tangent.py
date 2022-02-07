#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


import numpy as npo

from .. import backend as bk
from ..utils import filter, norm
from .fft import fourier_transform, inverse_fourier_transform


def _normalize(x, n):
    with npo.errstate(invalid="ignore"):
        f = x / (n)
    return bk.where(n == 0.0, 0.0 * x, f)


def _ft_filt(x, expo):
    with npo.errstate(divide="ignore"):
        f = 1 / (x**expo)
    return bk.where(x == 0.0, 0.0 * x, f)


def get_tangent_field(grid, normalize=False, alt=False, rfilt=4, expo=0.5):
    Nx, Ny = grid.shape
    xf = filter(grid, rfilt)
    v = bk.gradient(grid)
    vf = bk.gradient(xf)
    norma = norm(v)
    N = [bk.array(v[i]) * bk.abs(vf[i]) for i in range(2)]
    N = [_normalize(N[i], norma) for i in range(2)]
    fx = bk.fft.fftfreq(Nx)
    fy = bk.fft.fftfreq(Ny)
    Fx, Fy = bk.meshgrid(fx, fy, indexing="ij")
    ghat = _ft_filt(Fx**2 + Fy**2, expo=expo) * Nx * Ny
    Nhat = [fourier_transform(N[i]) for i in range(2)]
    Nstar = [(inverse_fourier_transform(Nhat[i] * ghat)) for i in range(2)]
    if normalize:
        norm_Nstar = norm(Nstar)
        Nstar = [_normalize(Nstar[i], norm_Nstar) for i in range(2)]

    if alt:
        return [-Nstar[0], -Nstar[1]]
    else:
        return [Nstar[0], Nstar[1]]
