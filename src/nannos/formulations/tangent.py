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


def _normalize(x, n, invalid=0, threshold=0.0):
    with npo.errstate(invalid="ignore"):
        f = x / (n)
    return bk.array(bk.where(n <= threshold, invalid * bk.ones_like(x), f))


def _normalize_vec(V, n, threshold=1e-6):
    Vx = _normalize(V[0], n, invalid=1, threshold=threshold)
    Vy = _normalize(V[1], n, invalid=0, threshold=threshold)
    return [Vx, Vy]


def _ft_filt(x, expo):
    with npo.errstate(divide="ignore"):
        f = 1 / (x**expo)
    return bk.array(bk.where(x == 0.0, 0.0 * x, f))


def _get_tangent_field_fft(grid, normalize=False, rfilt=4, expo=0.5):
    Nx, Ny = grid.shape

    grid = bk.hstack([grid, grid, grid])
    grid = bk.vstack([grid, grid, grid])

    xf = apply_filter(grid, rfilt)
    v = bk.gradient(grid)
    vf = bk.gradient(xf)
    norma = norm(v)
    N = [bk.array(v[i]) * bk.abs(vf[i]) for i in range(2)]
    N = [_normalize(N[i], norma) for i in range(2)]
    fx = bk.fft.fftfreq(3 * Nx)
    fy = bk.fft.fftfreq(3 * Ny)
    Fx, Fy = bk.meshgrid(fx, fy, indexing="ij")
    ghat = _ft_filt(Fx**2 + Fy**2, expo=expo) * Nx * Ny
    # ghat = apply_filter(Fx**2 + Fy**2, rfilt)
    Nhat = [fourier_transform(N[i]) for i in range(2)]
    Nstar = [(inverse_fourier_transform(Nhat[i] * ghat)) for i in range(2)]

    Nstar = [Nstar[i][Nx : 2 * Nx, Ny : 2 * Ny] for i in range(2)]
    t = [Nstar[1].real, -Nstar[0].real]
    # t = bk.array(t).real
    if normalize:
        norm_t = norm(t)
        t = _normalize_vec(t, norm_t)
    else:
        t = [t[i] / bk.max(norm(t)) for i in range(2)]
    return t


def _get_tangent_field_min(
    grid, harmonics, normalize=False, rfilt=4, opt_backend="autograd"
):

    if opt_backend == "jax":
        from jax import grad
        from jax import numpy as npg
        from jax.config import config

        # config.update("jax_platform_name", "cpu")
        config.update("jax_enable_x64", True)
        from jax import jit
    else:
        from autograd import grad
        from autograd import numpy as npg

        def jit(x):
            return x

    from scipy.optimize import minimize

    def norm(v):
        # avoid division by 0
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
    nh = min(nh, 51)
    harmonics = harmonics[:, :nh]

    xf = apply_filter(grid, rfilt=rfilt)
    dgrid_f = npg.array(npg.gradient(xf))

    normdgrid_f = norm(dgrid_f)
    # maxi = npg.max(normdgrid_f)
    # dgrid_f = npg.array([dgrid_f[i] /maxi for i in range(2)])
    dgrid_f = npg.array([_normalize(dgrid_f[i], normdgrid_f) for i in range(2)])

    def _set_idx(mat, idx, val):
        if opt_backend == "jax":
            mat = mat.at[tuple(idx)].set(val)
        else:
            # idx += [None]
            mat[tuple(idx)] = val
        return mat

    def _get_amps(amplitudes, shape):
        amplitudes = npg.array([amplitudes])
        if len(amplitudes.shape) == 1:
            amplitudes = npg.reshape(amplitudes, amplitudes.shape + (1,))
        f = npg.zeros(shape + (amplitudes.shape[0],) + (nh,), dtype=npg.complex128)
        # f[harmonics[0], harmonics[1],0] = npg.eye(nh)
        idx = [harmonics[0], harmonics[1], 0]
        # f[tuple(idx)] =  npg.eye(nh)
        f = _set_idx(f, idx, npg.eye(nh))
        s = npg.sum(amplitudes * f, axis=-1)
        ft = npg.fft.ifft2(s, axes=(0, 1)) * Nx * Ny
        return ft[:, :, 0]

    @jit
    def get_amps(amplitudes):
        return _get_amps(amplitudes, shape_small)

    def minifun(x):
        coef = x[:nh] + 1j * x[nh:]
        a = get_amps(coef)
        da = npg.array(npg.gradient(a))
        obj = npg.mean(npg.abs(da - dgrid_f[:, ::downsample_x, ::downsample_y]) ** 2)
        return obj

    @jit
    def minifun_der(x):
        d = grad(minifun)(x)
        return d

    x0 = npg.zeros(2 * nh)
    # x0 = npo.random.rand(2 * nh)
    res = minimize(
        minifun,
        x0,
        method="BFGS",
        jac=minifun_der,
        tol=5e-4,
        options={"disp": False, "maxiter": 50},
    )

    xopt = res.x
    coef = xopt[:nh] + 1j * xopt[nh:]

    a = _get_amps(coef, (Nx, Ny))
    n = npg.array(npg.gradient(a))
    t = [n[1], -n[0]]

    t = bk.array(t).real

    if normalize:
        norm_t = norm(t)
        t = _normalize_vec(t, norm_t)
    else:
        t = [t[i] / bk.max(norm(t)) for i in range(2)]

    return t


def get_tangent_field(grid, harmonics, normalize=False, rfilt=4, type="fft"):
    if type == "fft":
        return _get_tangent_field_fft(grid, normalize, rfilt=rfilt)
    elif type == "opt":
        return _get_tangent_field_min(grid, harmonics, normalize, rfilt=rfilt)
    else:
        raise ValueError(
            f"Wrong type of tangent field {type}. Please choose 'fft' or 'opt'"
        )
