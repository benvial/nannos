#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


from .. import backend as bk
from .. import get_backend, jit
from ..formulations.fft import fourier_transform, inverse_fourier_transform


def unique(x):
    seen = list()
    return not any(i in seen or seen.append(i) for i in x)


def set_index(mat, idx, val):
    if get_backend() == "jax":
        mat = mat.at[tuple(idx)].set(val)
    else:
        idx += [None]
        mat[tuple(idx)] = val


def next_power_of_2(x):
    return 1 if x == 0 else int(2 ** bk.ceil(bk.log2(x)))


def norm(v):
    ## avoid division by 0
    eps = bk.finfo(float).eps
    out = bk.array(
        0j + eps + bk.sqrt(v[0] * bk.conj(v[0]) + v[1] * bk.conj(v[1])),
        dtype=bk.complex128,
    )
    return bk.real(out)


def block(a):
    l1 = bk.array(bk.hstack([a[0][0], a[0][1]]))
    l2 = bk.array(bk.hstack([a[1][0], a[1][1]]))
    return bk.vstack([l1, l2])


def get_block(M, i, j, n):
    return M[i * n : (i + 1) * n, j * n : (j + 1) * n]


def blockmatmul(A, B, N):
    a = [[get_block(A, i, j, N) for j in range(2)] for i in range(2)]
    b = [[get_block(B, i, j, N) for j in range(2)] for i in range(2)]
    out = [
        [sum([a[i][k] * b[k][j] for k in range(2)]) for j in range(2)] for i in range(2)
    ]
    return block(out)


def _apply_filter(x, rfilt):
    if rfilt == 0:
        return x
    else:
        Nx = x.shape[0]
        # First a 1-D  Gaussian
        # t = bk.linspace(0, Nx-1, Nx)
        t = bk.array(bk.linspace(-Nx / 2, Nx / 2, Nx))
        bump = bk.exp(-(t**2) / rfilt**2)
        bump /= bk.trapz(bump)  # normalize the integral to 1

        # make a 2-D kernel out of it
        kernel = bump[:, None] * bump[None, :]

        kernel_ft = fourier_transform(kernel, s=x.shape[:2], axes=(0, 1))
        img_ft = fourier_transform(x, axes=(0, 1))

        # convolve
        img2_ft = kernel_ft * img_ft

        out = bk.real(inverse_fourier_transform(img2_ft, axes=(0, 1)))

        return bk.fft.fftshift(out) * (Nx**2)


apply_filter = jit(_apply_filter, static_argnums=(1))
