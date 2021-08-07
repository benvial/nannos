#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from .. import numpy as np


def fourier_transform(u):
    uft = np.fft.fft2(u)
    nx, ny = np.shape(uft)[:2]
    return uft / (nx * ny)


def inverse_fourier_transform(uft, shape=None, axes=(-2, -1)):
    u = np.fft.ifft2(uft, s=shape, axes=axes)
    nx, ny = np.shape(u)[:2]
    return u * (nx * ny)
