#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


import time

import pytest

import nannos as nn
from nannos import numpy as np

np.random.seed(1984)
Nx = Ny = 2 ** 5

x = np.random.rand(Nx * Ny)


def test_para():
    res = []
    timing = []
    F = [1.1, 1.2, 1.3, 1.4]

    for n_jobs in [1, 2]:

        @nn.parloop(n_jobs=n_jobs)
        def sim(f):
            xa = np.reshape(x, (Nx, Ny))
            eps_pattern = 2 + 1 * xa
            sup = nn.Layer("Superstrate")
            sub = nn.Layer("Substrate")
            ms = nn.Layer("ms", 1)
            pattern = nn.Pattern(eps_pattern)
            ms.add_pattern(pattern)
            sim = nn.Simulation(
                nn.Lattice(([1, 0], [0, 1])), [sup, ms, sub], nn.PlaneWave(f), 50
            )
            R, T = sim.diffraction_efficiencies()
            return R

        t0 = nn.tic()
        res.append(sim(F))
        t1 = nn.toc(t0)
        timing.append(t1)

    speedup = timing[0] / np.array(timing)

    print(f"speedup = {speedup}")

    assert np.allclose(res[0], res[1])
