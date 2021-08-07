#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import warnings

from .. import numpy as np
from ..helpers import filter, norm


def _normalize(x, n):
    with np.errstate(invalid="ignore"):
        f = x / (n)
    return np.where(n == 0.0, 0, f)


def _ft_filt(x, expo):
    with np.errstate(divide="ignore"):
        f = 1 / (x ** expo)
    return np.where(x == 0.0, 0, f)


def get_tangent_field(grid, normalize=True, alt=False, rfilt=4, expo=0.5):

    Nx, Ny = grid.shape
    xf = filter(grid, rfilt)
    v = np.gradient(grid)
    vf = np.gradient(xf)
    norma = norm(v)
    N = np.array(v) * np.abs(np.array(vf))
    N = _normalize(N, norma)

    fx = np.fft.fftfreq(Nx)
    fy = np.fft.fftfreq(Ny)
    Fx, Fy = np.meshgrid(fx, fy)

    ghat = _ft_filt(Fx ** 2 + Fy ** 2, expo=expo)
    Nhat = np.fft.fft2(N)
    Nstar = np.real(np.fft.ifft2(Nhat * ghat))
    if normalize:
        norm_Nstar = norm(Nstar)
        Nstar = _normalize(Nstar, norm_Nstar)

    if alt:
        return np.array([Nstar[1], Nstar[0]])
    else:
        return np.array([-Nstar[1], Nstar[0]])
