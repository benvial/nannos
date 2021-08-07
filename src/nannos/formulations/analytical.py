#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from scipy import special as sp

from nannos import numpy as np
from nannos import pi


def jinc(x):
    with np.errstate(invalid="ignore"):
        f = sp.j1(x) / x
    return np.where(x == 0.0, 0.5, f)


def fourier_transform_circle(radius, Nx, Ny):
    dx = 1 / (Nx - 1)
    dy = 1 / (Ny - 1)
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    kx, ky = np.meshgrid(fx, fy, indexing="ij")
    r = (kx ** 2 + ky ** 2) ** 0.5
    ft = 2 * pi * radius ** 2 * jinc(2 * pi * r * radius) + 0j
    ft *= np.exp(-2 * np.pi * 1j * (0.5 * kx + 0.5 * ky))
    return ft


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    plt.ion()

    plt.close("all")

    Nx = 2 ** 6
    Ny = Nx

    radius = 0.4
    x0 = np.linspace(0, 1, Nx)
    y0 = np.linspace(0, 1, Ny)
    x, y = np.meshgrid(x0, y0, indexing="ij")

    r = ((x - 0.5) ** 2 + (y - 0.5) ** 2) ** 0.5
    hole = r < radius
    from nannos.formulations.fft import fourier_transform

    eb = 4
    eh = 1

    # ids = np.zeros((Nx, Ny), dtype=float)
    ids = np.ones((Nx, Ny), dtype=float) * eb
    ids[hole] = eh

    dx = x0[-1] - x0[0]
    dy = y0[-1] - y0[0]
    d = x0[1] - x0[0]

    # ft = np.fft.fft2(ids)
    ft = fourier_transform(ids)

    # ft = np.fft.fftshift(ft) * (d ** 2)
    # nx, ny = np.shape(ft)
    # ft /= (nx * ny)

    #
    # plt.figure()
    # plt.imshow(ids)
    # plt.colorbar()

    plt.figure()
    plt.imshow(np.real(ft))
    plt.colorbar()
    plt.title("fft")

    f0 = np.fft.fftfreq(Nx, d=d)
    f1 = np.fft.fftshift(f0)

    ana = fourier_transform_circle(radius, Nx, Nx)
    bg = np.zeros_like(ana)
    bg[0, 0] = eb
    ana = bg + (eh - eb) * ana
    # ana = np.fft.fftshift(ana)
    plt.figure()
    plt.imshow(np.real(ana))
    plt.colorbar()
    plt.title("analytical")

    n = 0  # int(Nx / 2)

    plt.figure()
    plt.plot(np.real(ana[:, n]), "o--", label="analytical Re")
    plt.plot(np.real(ft[:, n]), "s-", label="fft Re")
    plt.plot(np.imag(ana[:, n]), "o--", label="analytical Im")
    plt.plot(np.imag(ft[:, n]), "s-", label="fft Im")
    plt.legend()

    plt.figure()
    plt.imshow(np.log10(np.abs(ft - ana)))
    plt.colorbar()
    plt.title("error")

    xs

    n1 = 3
    q = (np.abs(ana - ft) ** 2)[n - n1 : n + n1, n - n1 : n + n1]

    assert np.mean(np.abs(ana - ft) ** 2) < 1e-3
    t = 1e-2
    cond = np.abs(ana) > t
    q = np.abs(ana[cond] - ft[cond]) ** 2

    assert np.mean(q) < 1e-2

    #
    # if ana:
    #     Nx, Ny = u.shape
    #     eb = 4
    #     eh = 1
    #     uft = fourier_transform_circle(0.25, Nx, Ny)
    #     bg = np.zeros_like(uft)
    #     bg[0, 0] = eb
    #     uft = bg + (eh - eb) * uft
