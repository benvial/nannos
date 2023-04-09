#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import time

import numpy as npo

import nannos as nn
from nannos import numpy as np
from nannos.utils import allclose

print("##########")
print(nn.BACKEND)
print("##########")

bk = nn.backend


def test_para():
    npo.random.seed(1984)
    Nx = Ny = 2**5

    x = npo.random.rand(Nx * Ny)
    x = nn.backend.array(x)

    res = []
    timing = []
    WL = [0.6, 0.7, 0.8, 0.9]

    for n_jobs in [1, 2]:

        @nn.parloop(n_jobs=n_jobs)
        def sim(wl):
            lattice = nn.Lattice(([1, 0], [0, 1]))
            xa = nn.backend.reshape(x, (Nx, Ny))
            eps_pattern = 2 + 1 * xa
            sup = lattice.Layer("Superstrate")
            sub = lattice.Layer("Substrate")
            ms = lattice.Layer("ms", 1)
            ms.epsilon = eps_pattern
            sim = nn.Simulation([sup, ms, sub], nn.PlaneWave(wl), 50)
            R, T = sim.diffraction_efficiencies()
            return R

        t0 = nn.tic()
        res.append(sim(WL))
        t1 = nn.toc(t0)
        timing.append(t1)

    speedup = timing[0] / np.array(timing)

    print(f"speedup = {speedup}")

    assert allclose(bk.stack(res[0]), bk.stack(res[1]))
