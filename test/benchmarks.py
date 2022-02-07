#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


import sys

import numpy as npo
import pytest

try:
    threads = sys.argv[1]
except:
    threads = 1

devices = ["cpu", "gpu"]
formulations = ["original", "tangent", "jones"]
backends = ["numpy", "scipy", "autograd", "jax", "torch"]

formulations = ["original"]
# backends = ["torch"]
# devices = ["gpu"]
# backends = ["torch"]
# formulations = [ "jones"]


@pytest.mark.parametrize("formulation", formulations)
@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("device", devices)
def test_simulations(formulation, backend, device):
    import nannos as nn

    if backend in ["numpy", "scipy", "autograd", "jax"] and device == "gpu":
        return
    if backend == "torch" and (not nn.HAS_TORCH or not nn.HAS_CUDA):
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

    truncation = "circular"

    nfreq = 11
    frequencies = np.ones(nfreq) * 1.1

    NH = [100, 200, 400, 600, 800, 1000]
    NH_real = []
    TIMES = []

    for nh in NH:
        print(f"number of harmonics = {nh}")
        for ifreq, freq in enumerate(frequencies):
            pw = nn.PlaneWave(
                frequency=freq,
            )
            t0 = nn.tic()
            sim = nn.Simulation(
                lattice,
                [sup, st, sub],
                pw,
                nh,
                formulation=formulation,
                truncation=truncation,
            )
            R, T = sim.diffraction_efficiencies()
            t1 = nn.toc(t0)
            if ifreq > 0:
                TIMES.append(t1)
        npo.mean(TIMES)

        NH_real.append(sim.nh)
    B = R + T

    import numpy as npo

    npo.savez(
        f"benchmark_{backend}_{device}.npz",
        times=npo.mean(TIMES),
        real_nh=NH_real,
        nh=NH,
    )

    # print("T = ", T)
    # print("R = ", R)
    # print("R + T = ", B)
    # assert nn.backend.allclose(
    #     B, nn.backend.array(1.0, dtype=nn.backend.float64), atol=5e-3
    # )
    #
    # a, b = sim._get_amplitudes(1, z=0.1)
    # field_fourier = sim.get_field_fourier(1, z=0.1)

    # return R, T, sim
