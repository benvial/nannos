#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

all = [
    "unique",
    "is_scalar",
    "set_index",
    "next_power_of_2",
    "next_power_of_2",
    "norm",
    "block",
    "get_block",
    "blockmatmul",
    "block2list",
    "inv2by2block",
    "apply_filter",
]

import numpy as npo

from .. import backend as bk
from .. import get_backend, get_types, jit
from ..formulations.fft import fourier_transform, inverse_fourier_transform

FLOAT, COMPLEX = get_types()


def allclose(a, b, **kwargs):
    try:
        a = a.detach().cpu().numpy()
    except Exception:
        pass

    try:
        b = b.detach().cpu().numpy()
    except Exception:
        pass

    return npo.allclose(a, b, **kwargs)


def unique(x):
    seen = []
    return not any(i in seen or seen.append(i) for i in x)


def is_scalar(x):
    return not hasattr(x, "__len__")


def is_anisotropic(x):
    return False if is_scalar(x) else x.shape[:2] == (3, 3)


def is_uniform(x):
    if is_scalar(x):
        return True
    if is_anisotropic(x):
        return x.shape == (3, 3)
    else:
        return x.shape[0] == 1 if len(x.shape) == 1 else x.shape == ()


def set_index(mat, idx, val):
    if get_backend() == "jax":
        mat = mat.at[tuple(idx)].set(val)
    else:
        idx += [None]
        mat[tuple(idx)] = val
    return mat


def next_power_of_2(x):
    return 1 if x == 0 else int(2 ** bk.ceil(bk.log2(x)))


def norm(v):
    # avoid division by 0
    eps = bk.finfo(float).eps
    out = bk.array(
        0j + eps + bk.sqrt(v[0] * bk.conj(v[0]) + v[1] * bk.conj(v[1])),
        dtype=COMPLEX,
    )
    return bk.real(out)


def block(a):
    l1 = bk.array(bk.hstack([bk.array(a[0][0]), bk.array(a[0][1])]))
    l2 = bk.array(bk.hstack([bk.array(a[1][0]), bk.array(a[1][1])]))
    return bk.vstack([l1, l2])


def get_block(M, i, j, n):
    return M[i * n : (i + 1) * n, j * n : (j + 1) * n]


def blockmatmul(A, B, N):
    a = [[get_block(A, i, j, N) for j in range(2)] for i in range(2)]
    b = [[get_block(B, i, j, N) for j in range(2)] for i in range(2)]
    out = [
        [sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2)] for i in range(2)
    ]
    return block(out)


def block2list(M, N):
    return [[get_block(M, i, j, N) for j in range(2)] for i in range(2)]


def inv2by2block(T, N):
    M = block2list(T, N)
    detT = M[0][0] * M[1][1] - M[1][0] * M[0][1]
    return block(
        [
            [M[1][1] / detT, -M[0][1] / detT],
            [-M[1][0] / detT, M[0][0] / detT],
        ]
    )


def _reseter(prop, attr=None):
    try:
        if attr is None:
            del prop
        else:
            delattr(prop, attr)
    except Exception:
        pass


def _apply_filter(x, rfilt, vectors=None):
    if is_scalar(rfilt):
        rfilt = (rfilt, rfilt)

    if rfilt == (0, 0):
        return x
    Nx, Ny = x.shape

    if vectors is None:
        tx = bk.array(bk.linspace(-Nx / 2, Nx / 2, Nx))
        ty = bk.array(bk.linspace(-Ny / 2, Ny / 2, Ny))

    else:
        tx = bk.array(bk.linspace(-0.5 * vectors[0][0], 0.5 * vectors[0][0], Nx))
        ty = bk.array(bk.linspace(-0.5 * vectors[1][1], 0.5 * vectors[1][1], Ny))
    bumpx = bk.exp(-(tx**2) / rfilt[0] ** 2)
    bumpx /= bk.trapz(bumpx)  # normalize the integral to 1
    bumpy = bk.exp(-(ty**2) / rfilt[1] ** 2)
    bumpy /= bk.trapz(bumpy)  # normalize the integral to 1
    # make a 2-D kernel out of it
    kernel = bumpx[:, None] * bumpy[None, :]
    kernel_ft = fourier_transform(kernel, s=x.shape[:2], axes=(0, 1))
    img_ft = fourier_transform(x, axes=(0, 1))
    # convolve
    img2_ft = kernel_ft * img_ft
    out = bk.real(inverse_fourier_transform(img2_ft, axes=(0, 1)))
    return bk.fft.fftshift(out) * (Nx * Ny)


apply_filter = jit(_apply_filter, static_argnums=(1))
