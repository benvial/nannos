#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


import pytest

devices = ["cpu", "gpu"]
formulations = ["original", "tangent", "jones"]
backends = ["numpy", "scipy", "autograd", "jax", "torch"]

# formulations = ["original"]
# backends = ["torch"]
# devices = ["gpu"]
# backends = ["torch"]
# formulations = [ "jones"]
#


@pytest.mark.parametrize("formulation", formulations)
@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("device", devices)
def test_simulations(formulation, backend, device):
    import nannos as nn

    if backend in ["numpy", "scipy", "autograd", "jax"] and device == "gpu":
        return
    if backend == "torch" and (not nn.HAS_TORCH and not nn.HAS_CUDA):
        return

    nn.set_backend(backend)
    if device == "gpu":
        nn.use_gpu()
    print(nn._nannos_device)
    if nn.BACKEND == "torch":
        device = nn.backend.device(nn._nannos_device)
        print(device.type)
    elif nn.BACKEND == "jax":
        import jax

        print(jax.default_backend())

    nh = 51
    L1 = [1.0, 0]
    L2 = [0, 1.0]
    Nx = 2**9
    Ny = 2**9

    eps_pattern = 4.0 + 0j
    eps_hole = 1.0
    mu_pattern = 1.0
    mu_hole = 1.0

    h = 2

    radius = 0.25
    x0 = nn.backend.linspace(0, 1.0, Nx)
    y0 = nn.backend.linspace(0, 1.0, Ny)
    x, y = nn.backend.meshgrid(x0, y0, indexing="ij")
    hole = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius**2
    hole = nn.backend.array(hole)
    # hole.to(nn.backend.device(nn._nannos_device))

    lattice = nn.Lattice((L1, L2))
    sup = nn.Layer("Superstrate", epsilon=1, mu=1)
    sub = nn.Layer("Substrate", epsilon=1, mu=1)

    ids = nn.backend.ones((Nx, Ny), dtype=float)
    zs = nn.backend.zeros_like(ids)

    # epsgrid = ids * eps_pattern
    eps1 = nn.backend.array(eps_pattern, dtype=nn.backend.complex128)
    eps2 = nn.backend.array(1, dtype=nn.backend.complex128)
    epsgrid = nn.backend.where(hole, eps1, eps2)
    # epsgrid[hole] = eps_pattern
    mugrid = 1 + 0j

    pattern = nn.Pattern(epsgrid, mugrid)
    st = nn.Layer("Structured", h)
    st.add_pattern(pattern)

    pw = nn.PlaneWave(
        frequency=1.1,
    )

    for i in range(2):
        t0 = nn.tic()
        sim = nn.Simulation(lattice, [sup, st, sub], pw, nh, formulation=formulation)
        R, T = sim.diffraction_efficiencies()
        nn.toc(t0)
    B = R + T

    print(">>> formulation = ", formulation)
    print("T = ", T)
    print("R = ", R)
    print("R + T = ", B)
    assert nn.backend.allclose(
        B, nn.backend.array(1.0, dtype=nn.backend.float64), atol=5e-3
    )

    a, b = sim._get_amplitudes(1, z=0.1)
    field_fourier = sim.get_field_fourier(1, z=0.1)

    return R, T, sim


#
# import nannos
# nannos.set_backend("torch")
# nannos.use_gpu()
# from nannos import backend as bk
# pi = bk.pi
# nh=10
# Lk = bk.array(((1,0),(0,1)),dtype=float)
#
# u = bk.array([bk.linalg.norm(l) for l in Lk])
# udot = bk.dot(Lk[0], Lk[1])
# ucross = bk.array(Lk[0][0] * Lk[1][1] - Lk[0][1] * Lk[1][0])
#
# circ_area = nh * bk.abs(ucross)
# circ_radius = bk.sqrt(circ_area / pi) + u[0] + u[1]
#
# u_extent = bk.array([
#     1 + int(circ_radius / (q * bk.sqrt(1.0 - udot**2 / (u[0] * u[1]) ** 2)))
#     for q in u
# ])
#
# xG, yG = [bk.array(bk.arange(-q, q + 1)) for q in u_extent]
# G = bk.meshgrid(xG, yG, indexing="ij")
# G = [g.flatten() for g in G]
#
# # u[0].get_device()
#
# Gl2 = bk.array(G[0] ** 2 * u[0] ** 2 + G[1] ** 2 * u[0] ** 2 + 2 * G[0] * G[1] * udot)
# jsort = bk.argsort(Gl2)
# Gsorted = [g[jsort] for g in G]
# Gl2 = Gl2[jsort]
#
# nGtmp = (2 * u_extent[0] + 1) * (2 * u_extent[1] + 1)
# if nh < nGtmp:
#     nGtmp = nh
#
# tol = 1e-10 * max(u[0] ** 2, u[1] ** 2)
# for i in bk.arange(nGtmp - 1, -1, -1):
#     if bk.abs(Gl2[i] - Gl2[i - 1]) > tol:
#         break
# nh = i
#
# G = bk.vstack(Gsorted)[:, :nh]
