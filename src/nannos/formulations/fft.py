#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


from .. import backend as bk


def fourier_transform(u):
    uft = bk.fft.fft2(u)
    nx, ny = uft.shape[:2]
    return uft / (nx * ny)


def inverse_fourier_transform(uft, shape=None, axes=(-2, -1)):
    u = bk.fft.ifft2(uft, s=shape, axes=axes)
    nx, ny = u.shape[:2]
    return u * (nx * ny)
