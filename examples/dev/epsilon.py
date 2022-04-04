#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

plt.ion()

bk = nn.backend
# formulation = "tangent"
formulation = "original"

lattice = nn.Lattice([[1.0, 0], [0, 1.0]], discretization=2**9)
sup = lattice.Layer("Superstrate", epsilon=1)
sub = lattice.Layer("Substrate", epsilon=1)
o = lattice.ones()
hole = lattice.circle(center=(0.5, 0.5), radius=0.2)
epsilon = o * 4
epsilon[hole] = 1
ms = lattice.Layer("Metasurface", thickness=0.5, epsilon=epsilon)
pw = nn.PlaneWave(frequency=1.4, angles=(0, 0, 0 * nn.pi / 2))


plt.close("all")
for n in [1, 3, 5, 7, 11, 21, 41]:
    nh = n**2
    print("------------------------")
    print("Number of harmonics: ", nh)
    sim = nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)
    # eps = sim.get_epsilon("Substrate")
    t = nn.tic()
    print("Building matrix...")
    ms = sim._build_matrix(ms)
    print("Done build matrix")
    nn.toc(t)
    t = nn.tic()
    print("Retrieving epsilon...")
    eps = sim.get_epsilon(ms)
    print("Done retrieving epsilon...")
    nn.toc(t)
    plt.figure()
    plt.imshow(eps.real)
    plt.colorbar()
    plt.title(nh)
    plt.pause(0.1)
