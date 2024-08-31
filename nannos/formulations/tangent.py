#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import numpy as npo
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

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
    grid,
    harmonics,
    normalize=False,
    rfilt=0,
    opt_backend="autograd",
    maxiter=1,
):
    if opt_backend == "jax":
        from jax import grad
        from jax import numpy as npg
        from jax.config import config

        # config.update("jax_platform_name", "cpu")
        config.update("jax_enable_x64", True)
        from jax import jit
    else:
        from autograd import grad, hessian_vector_product
        from autograd import numpy as npg

        def jit(x):
            return x

    def norm(v):
        # avoid division by 0
        eps = npg.finfo(float).eps
        return npg.sqrt(eps + v[0] * npg.conj(v[0]) + v[1] * npg.conj(v[1]))

    def _normalize(x, n):
        with npo.errstate(invalid="ignore"):
            f = x / (n)
        return npg.array(npg.where(n == 0.0, 0.0 * x, f))

    # try:
    #     grid = grid.detach().cpu().numpy()
    # except Exception:
    #     pass

    grid = bk.abs(grid)
    grid = grid / bk.max(grid)
    Nx, Ny = grid.shape
    Nx_ds = min(2**6, Nx)
    Ny_ds = min(2**6, Ny)
    # Nx_ds = min(int(Nx/2**3), Nx)
    # Ny_ds = min(int(Nx/2**3), Ny)

    downsample_x = int(Nx / Nx_ds)
    downsample_y = int(Ny / Ny_ds)
    shape_small = (Nx_ds, Ny_ds)

    xi = npg.linspace(0, 1, Nx)
    yi = npg.linspace(0, 1, Ny)
    points_dsx = npg.linspace(0, 1, Nx_ds)
    points_dsy = npg.linspace(0, 1, Ny_ds)
    points = npg.meshgrid(xi, yi, indexing="ij")

    # grid += 0.0001 * npg.sin(2 * npg.pi * points[0]) * npg.sin(2 * npg.pi * points[1])

    nh = len(harmonics[0])
    # nh = min(nh, 51)

    harmonics = harmonics[:, :nh]

    # xf = apply_filter(grid, rfilt=rfilt)
    xf = grid

    dgrid = bk.stack(bk.gradient(xf))
    dgrid_fx = apply_filter(dgrid[0], rfilt=rfilt)
    dgrid_fy = apply_filter(dgrid[1], rfilt=rfilt)
    dgrid_f = bk.stack([dgrid_fx, dgrid_fy])

    norm_dgrid_f = bk.array(norm(dgrid_f))
    # if normalize:
    #     dgrid_f = bk.array(_normalize_vec(dgrid_f, norm_dgrid_f))
    # else:
    dgrid_f = bk.stack([dgrid_f[i] / bk.max(norm_dgrid_f) for i in range(2)])
    # dgrid_f = bk.array(_normalize_vec(dgrid_f, norm_dgrid_f))

    def _set_idx(mat, idx, val):
        if opt_backend == "jax":
            mat = mat.at[tuple(idx)].set(val)
        else:
            mat[tuple(idx)] = val
        return mat

    def _get_amps(amplitudes, shape):
        amplitudes = npg.array([amplitudes])
        if len(amplitudes.shape) == 1:
            amplitudes = npg.reshape(amplitudes, amplitudes.shape + (1,))
        f = npg.zeros(shape + (amplitudes.shape[0],) + (nh,), dtype=npg.complex128)
        idx = [harmonics[0], harmonics[1], 0]
        f = _set_idx(f, idx, npg.eye(nh))
        s = npg.sum(amplitudes * f, axis=-1)
        ft = npg.fft.ifft2(s, axes=(0, 1)) * Nx * Ny
        return ft[:, :, 0]

    @jit
    def get_amps(amplitudes):
        return _get_amps(amplitudes, shape_small)

    dgrid_f_downspl = dgrid_f[:, ::downsample_x, ::downsample_y]
    try:
        dgrid_f_downspl = dgrid_f_downspl.detach().cpu().numpy()
    except Exception:
        pass

    def minifun(x):
        coefx = x[:nh] + 1j * x[nh : 2 * nh]
        coefy = x[2 * nh : 3 * nh] + 1j * x[3 * nh :]
        ax = _get_amps(coefx, (Nx_ds, Nx_ds))
        ay = _get_amps(coefy, (Ny_ds, Ny_ds))
        da = npg.real(npg.array([ax, ay]))
        obj = npg.mean(npg.abs(da - dgrid_f_downspl) ** 2)
        ### smoothness
        # obj_smooth = npg.mean(da[0] ** 2 + da[1] ** 2) * 1
        # obj += obj_smooth
        # obj += obj_hf
        return obj

    x0 = npg.zeros(4 * nh)
    # x0 = npg.ones(4 * nh)
    # x0 = (npg.random.rand(4 * nh)*0.5 - 1) * 1
    res = minimize(
        minifun,
        x0,
        method="Newton-CG",
        jac=grad(minifun),
        hessp=hessian_vector_product(minifun),
        tol=1e-6,
        options={"disp": False, "maxiter": maxiter},
    )
    xopt = res.x
    coefx = xopt[:nh] + 1j * xopt[nh : 2 * nh]
    coefy = xopt[2 * nh : 3 * nh] + 1j * xopt[3 * nh :]

    ax = _get_amps(coefx, (Nx_ds, Nx_ds))
    ay = _get_amps(coefy, (Ny_ds, Ny_ds))

    # points_dsx = xi[::downsample_x]
    # points_dsy = yi[::downsample_y]
    interpx = RegularGridInterpolator(
        (points_dsx, points_dsy), ax, bounds_error=False, fill_value=0
    )
    interpy = RegularGridInterpolator(
        (points_dsx, points_dsy), ay, bounds_error=False, fill_value=0
    )

    points = npg.stack(points).T
    ax = interpx(points).T
    ay = interpy(points).T

    t = bk.real(bk.stack([bk.array(ay), -bk.array(ax)]))

    if normalize:
        try:
            t1 = t.detach().cpu().numpy()
        except Exception:
            t1 = t
        norm_t = bk.array(norm(t1))
        t = _normalize_vec(t, norm_t)
    else:
        # t *= norm_dgrid_f
        t = [t[i] / bk.max(norm(t)) for i in range(2)]

    return t


def get_tangent_field(grid, harmonics, normalize=False, rfilt=4, type="opt", **kwargs):
    if type == "fft":
        return _get_tangent_field_fft(grid, normalize, rfilt=rfilt, **kwargs)
    elif type == "opt":
        return _get_tangent_field_min(grid, harmonics, normalize, rfilt=rfilt, **kwargs)
    else:
        raise ValueError(
            f"Wrong type of tangent field {type}. Please choose 'fft' or 'opt'"
        )
