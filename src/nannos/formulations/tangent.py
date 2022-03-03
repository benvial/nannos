#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


import numpy as npo

from .. import backend as bk
from ..utils import apply_filter, norm
from .fft import fourier_transform, inverse_fourier_transform


def _normalize(x, n):
    with npo.errstate(invalid="ignore"):
        f = x / (n)
    return bk.array(bk.where(n == 0.0, 0.0 * x, f))


def _ft_filt(x, expo):
    with npo.errstate(divide="ignore"):
        f = 1 / (x**expo)
    return bk.array(bk.where(x == 0.0, 0.0 * x, f))


def _get_tangent_field_fft(grid, normalize=False, rfilt=4, expo=0.5):
    Nx, Ny = grid.shape
    xf = apply_filter(grid, rfilt)
    v = bk.gradient(grid)
    vf = bk.gradient(xf)
    norma = norm(v)
    N = [bk.array(v[i]) * bk.abs(vf[i]) for i in range(2)]
    N = [_normalize(N[i], norma) for i in range(2)]
    fx = bk.fft.fftfreq(Nx)
    fy = bk.fft.fftfreq(Ny)
    Fx, Fy = bk.meshgrid(fx, fy, indexing="ij")
    ghat = _ft_filt(Fx**2 + Fy**2, expo=expo) * Nx * Ny
    # ghat = apply_filter(Fx**2 + Fy**2, rfilt)
    Nhat = [fourier_transform(N[i]) for i in range(2)]
    Nstar = [(inverse_fourier_transform(Nhat[i] * ghat)) for i in range(2)]
    if normalize:
        norm_Nstar = norm(Nstar)
        Nstar = [_normalize(Nstar[i], norm_Nstar) for i in range(2)]

    return [Nstar[1], -Nstar[0]]
    # return [Nstar[0], Nstar[1]]


def _get_tangent_field_min(grid, harmonics, normalize=False, rfilt=4):
    from autograd import grad
    from autograd import numpy as npg
    from scipy.optimize import minimize

    def norm(v):
        ## avoid division by 0
        eps = npg.finfo(float).eps
        return npg.sqrt(eps + v[0] * npg.conj(v[0]) + v[1] * npg.conj(v[1]))

    def _normalize(x, n):
        with npo.errstate(invalid="ignore"):
            f = x / (n)
        return npg.array(npg.where(n == 0.0, 0.0 * x, f))

    Nx, Ny = grid.shape
    Nx_ds = min(2**6, Nx)
    Ny_ds = min(2**6, Ny)

    downsample_x = int(Nx / Nx_ds)
    downsample_y = int(Ny / Ny_ds)
    shape_small = (Nx_ds, Ny_ds)

    nh = len(harmonics[0])

    nh = min(nh, 150)
    harmonics = harmonics[:, :nh]

    xf = apply_filter(grid, rfilt=rfilt)
    dgrid_f = npg.array(npg.gradient(xf))

    normdgrid_f = norm(dgrid_f)
    maxi = npg.max(normdgrid_f)
    # dgrid_f = npg.array([dgrid_f[i] /maxi for i in range(2)])
    dgrid_f = npg.array([_normalize(dgrid_f[i], normdgrid_f) for i in range(2)])

    def get_amps(amplitudes, shape=(Nx, Ny)):
        amplitudes = npg.array([amplitudes])
        if len(amplitudes.shape) == 1:
            amplitudes = npg.reshape(amplitudes, amplitudes.shape + (1,))
        f = npg.zeros(shape + (amplitudes.shape[0],) + (nh,), dtype=npg.complex128)
        f[harmonics[0], harmonics[1], 0, :] = npg.eye(nh)
        s = npg.sum(amplitudes * f, axis=-1)
        ft = npg.fft.ifft2(s, axes=(0, 1)) * Nx * Ny
        return ft[:, :, 0]

    def minifun(x):
        coef = x[:nh] + 1j * x[nh:]
        a = get_amps(coef, shape_small)
        da = npg.array(npg.gradient(a))
        I = npg.mean(npg.abs(da - dgrid_f[:, ::downsample_x, ::downsample_y]) ** 2)
        return I

    minifun_der = grad(minifun)

    x0 = npg.zeros(2 * nh)
    res = minimize(
        minifun,
        x0,
        method="BFGS",
        jac=minifun_der,
        options={"gtol": 1e-3, "disp": False, "maxiter": 20},
    )

    xopt = res.x
    coef = xopt[:nh] + 1j * xopt[nh:]

    a = get_amps(coef)
    t = npg.array(npg.gradient(a))
    t = [t[1], -t[0]]

    # maxi = npg.max(norm(t))

    # t = [_normalize(t[i], norm_t) for i in range(2)]

    # t = [t[i] / maxi for i in range(2)]

    if normalize:
        norm_t = norm(t)
        t = [_normalize(t[i], norm_t) for i in range(2)]

    return bk.array(t)


def get_tangent_field(grid, harmonics, normalize=False, rfilt=4, type="fft"):
    if type == "fft":
        return _get_tangent_field_fft(grid, normalize, rfilt)
    elif type == "opt":
        return _get_tangent_field_min(grid, harmonics, normalize, rfilt)
    else:
        raise ValueError(
            f"Wrong type of tangent field {type}. Please choose 'fft' or 'opt'"
        )
