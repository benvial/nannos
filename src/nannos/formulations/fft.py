#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


from .. import BACKEND, HAS_CUDA
from .. import backend as bk
from .. import get_backend

_device = "cuda" if HAS_CUDA else "cpu"


def fourier_transform(u):
    if BACKEND == "torch":
        u = bk.array(u).to(_device)
    uft = bk.fft.fft2(u)
    nx, ny = uft.shape[:2]
    return uft / (nx * ny)


def inverse_fourier_transform(uft, shape=None, axes=(-2, -1)):
    if BACKEND == "torch":
        uft = bk.array(uft).to(_device)
    u = bk.fft.ifft2(uft, s=shape, axes=axes)
    nx, ny = u.shape[:2]
    return u * (nx * ny)
