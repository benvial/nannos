#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


import os
import sys

import numpy as npo

try:
    threads = int(sys.argv[1])
except:
    threads = 1

try:
    nfreq = int(sys.argv[2])
except:
    nfreq = 2


devices = ["cpu", "gpu"]
backends = ["numpy", "scipy", "autograd", "jax", "torch"]


def test_simulations(backend, device):
    print("--------------------------")
    print(f"{backend} {device}")
    print("--------------------------")
    formulation = "original"
    import nannos as nn

    if backend in ["numpy", "scipy", "autograd", "jax"] and device == "gpu":
        return
    if backend == "torch":
        if not nn.HAS_TORCH:
            return
        if not nn.HAS_CUDA and device == "gpu":
            return

    nn.set_backend(backend)
    if device == "gpu":
        nn.use_gpu()
    if nn.BACKEND == "torch":
        device = nn.backend.device(nn._nannos_device)

    elif nn.BACKEND == "jax":
        import jax

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

    # nfreq = 11
    frequencies = nn.backend.ones(nfreq) * 1.1

    NH = [100, 200, 400, 600, 800, 1000]
    NH_real = []
    TIMES = []
    TIMES_ALL = []

    for nh in NH:
        print(f"number of harmonics = {nh}")

        TIMES_NH = []
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
                TIMES_NH.append(t1)

        TIMES.append(sum(TIMES_NH) / len(TIMES_NH))
        TIMES_ALL.append(TIMES_NH)

        NH_real.append(sim.nh)
    B = R + T

    npo.savez(
        f"benchmark_{backend}_{device}.npz",
        times=TIMES,
        times_all=TIMES_ALL,
        real_nh=NH_real,
        nh=NH,
    )


if __name__ == "__main__":
    for device in devices:
        for backend in backends:
            test_simulations(backend, device)

    # test_simulations("torch", "gpu")
    # test_simulations("numpy", "cpu")
    # test_simulations("torch", "cpu")
    # test_simulations("jax", "cpu")
