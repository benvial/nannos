#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import nannos as nn


def test_simulation():
    gpu = [False]
    if nn.HAS_CUDA:
        gpu.append(True)
    for backend in nn.available_backends:
        nn.set_backend(backend)
        for use_gpu in gpu:
            nn.use_gpu(use_gpu)
            print(f"backend: {backend}, GPU: {use_gpu}")
            backend = nn.get_backend()
            bk = nn.backend
            lattice = nn.Lattice(
                basis_vectors=[[1.0, 0], [0, 1.0]], discretization=2**9
            )
            sup = lattice.Layer("Superstrate", epsilon=1)
            ms = lattice.Layer("Metasurface", thickness=0.5)
            sub = lattice.Layer("Substrate", epsilon=2)
            ms.epsilon = lattice.ones() * 12.0
            circ = lattice.circle(center=(0.5, 0.5), radius=0.2)
            if backend == "jax":
                ms.epsilon = ms.epsilon.at[circ].set(1)
            else:
                ms.epsilon[circ] = 1
            stack = [sup, ms, sub]
            pw = nn.PlaneWave(wavelength=0.9, angles=(0, 0, 0))
            sim = nn.Simulation(stack, pw, nh=200)
            R, T = sim.diffraction_efficiencies()
            assert bk.allclose(R + T, bk.array([1], dtype=bk.float64))
