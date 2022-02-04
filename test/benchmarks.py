#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import numpy as np

import nannos as nn


def build_simu(nh, npts):
    L1 = [1.0, 0]
    L2 = [0, 1.0]
    freq = 1.4
    theta = 0.0 * np.pi / 180
    phi = 0.0 * np.pi / 180
    psi = 0.0 * np.pi / 180

    Nx = npts
    Ny = npts

    eps_sup = 1.0
    eps_pattern = 12.0
    eps_hole = 1.0
    eps_sub = 1.0
    h = 0.5
    radius = 0.2
    epsgrid = np.ones((Nx, Ny), dtype=float) * eps_pattern
    x0 = np.linspace(0, 1.0, Nx)
    y0 = np.linspace(0, 1.0, Ny)
    x, y = np.meshgrid(x0, y0, indexing="ij")
    hole = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius ** 2
    epsgrid[hole] = eps_hole
    lattice = nn.Lattice((L1, L2))
    sup = nn.Layer("Superstrate", epsilon=eps_sup)
    ms = nn.Layer("Metasurface", thickness=h)
    sub = nn.Layer("Substrate", epsilon=eps_sub)
    pattern = nn.Pattern(epsgrid, name="hole")
    ms.add_pattern(pattern)
    pw = nn.PlaneWave(frequency=freq, angles=(theta, phi, psi))
    stack = [sup, ms, sub]
    sim = nn.Simulation(lattice, stack, pw, nh)

    return sim


NH = [50, 100, 200, 400, 800]

NH = [800]

npts = 2 ** 9
backend = "numpy"
# backend = "jax"
nn.set_backend(backend)


def build_matrix(layer):
    return sim._build_matrix(layer)


for nh in NH:
    print("----")
    sim = build_simu(nh, npts)

    layer = sim.layers[1]
    t = nn.tic()
    build_matrix(layer)
    t = nn.toc(t)

    mat = sim.layers[1].matrix
    t = nn.tic()
    nn.formulations.fft.fourier_transform(mat)
    t = nn.toc(t)

    # R, T = sim.diffraction_efficiencies()

#
# from nannos.utils import block
#

#
#
# def _build_Kmatrix_old(u, Kx, Ky):
#     return block(
#         [
#             [Kx @ u @ Kx, Kx @ u @ Ky],
#             [Ky @ u @ Kx, Ky @ u @ Ky],
#         ]
#     )
# #


from nannos.utils import block

Kx, Ky = sim.Kx, sim.Ky


epsilon_zz = layer.patterns[0].epsilon
eps_hat = sim._get_toeplitz_matrix(epsilon_zz)
eps_hat_inv = np.linalg.inv(eps_hat)


u = eps_hat_inv
# Keps = _build_Kmatrix(u, Ky, -Kx)
# # Keps_old = _build_Kmatrix_old(u, Ky, -Kx)
# assert np.allclose(Keps,Keps_old,atol=1e-6)


#
# backend = "jax"
# nn.set_backend(backend)
# from jax import jit
#
# def test(mat):
#     return nn.numpy.fft.fft2(mat)
#
# def test(mat):
#     return nn.numpy.linalg.inv(mat)
#
#
# jit is probably not very helpful...

#
# mat = nn.numpy.array(mat)
#
# test_jit = jit(test)
# %timeit test(mat).block_until_ready()
# %timeit test_jit(mat).block_until_ready()
