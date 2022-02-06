#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


from .. import backend as bk
from .. import get_backend


def next_power_of_2(x):
    return 1 if x == 0 else int(2 ** bk.ceil(bk.log2(x)))


def norm(v):
    ## avoid division by 0
    eps = bk.finfo(float).eps
    # return bk.sqrt(eps + bk.power(v[0], 2) + bk.power(v[1], 2))
    return bk.sqrt(eps + v[0] * bk.conj(v[0]) + v[1] * bk.conj(v[1]))


def block(a):
    l1 = bk.hstack([a[0][0], a[0][1]])
    l2 = bk.hstack([a[1][0], a[1][1]])
    return bk.vstack([l1, l2])


def get_block(M, i, j, n):
    return M[i * n : (i + 1) * n, j * n : (j + 1) * n]


def filter(x, rfilt):
    if rfilt == 0:
        return x
    else:
        Nx = x.shape[0]
        # First a 1-D  Gaussian
        # t = bk.linspace(0, Nx-1, Nx)
        t = bk.linspace(-Nx / 2, Nx / 2, Nx)
        bump = bk.exp(-(t**2) / rfilt**2)
        bump /= bk.trapz(bump)  # normalize the integral to 1

        # make a 2-D kernel out of it
        kernel = bump[:, None] * bump[None, :]

        BACKEND = get_backend()

        if BACKEND == "torch":

            kernel_ft = bk.fft.fft2(kernel, s=x.shape[:2], dim=(0, 1))
            img_ft = bk.fft.fft2(x, dim=(0, 1))
        else:

            kernel_ft = bk.fft.fft2(kernel, s=x.shape[:2], axes=(0, 1))
            img_ft = bk.fft.fft2(x, axes=(0, 1))

        # convolve
        img2_ft = kernel_ft * img_ft

        if BACKEND == "torch":
            out = bk.real(bk.fft.ifft2(img2_ft, dim=(0, 1)))
        else:
            out = bk.real(bk.fft.ifft2(img2_ft, axes=(0, 1)))

        return bk.fft.fftshift(out)
