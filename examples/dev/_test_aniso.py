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

nh = 300
bk = nn.backend
# formulation = "tangent"
formulation = "original"

lattice = nn.Lattice([[1.0, 0], [0, 1.0]], discretization=2**9)
sup = lattice.Layer("Superstrate", epsilon=1)
sub = lattice.Layer("Substrate", epsilon=1)
o = lattice.ones()
z = bk.zeros_like(o)
epsilon_xx = 3 * o
epsilon_yy = 3 * o
epsilon_zz = 4 * o
epsilon = bk.array([[epsilon_xx, z, z], [z, epsilon_yy, z], [z, z, epsilon_zz]])

ms = lattice.Layer("Metasurface", thickness=0.5, epsilon=epsilon)
pw = nn.PlaneWave(wavelength=1 / 1.4, angles=(0, 0, 0 * nn.pi / 2))
sim = nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)
t = nn.tic()
sim.solve()
nn.toc(t)
w0 = bk.sort(ms.eigenvalues.real)
print(ms.is_uniform)
# print(w0)

epsilon = 4

z = 0
epsilon_xx = 3
epsilon_yy = 3
epsilon_zz = 4
epsilon = bk.array([[epsilon_xx, z, z], [z, epsilon_yy, z], [z, z, epsilon_zz]])
ms = lattice.Layer("Metasurface", thickness=0.5, epsilon=epsilon)
pw = nn.PlaneWave(wavelength=1 / 1.4, angles=(0, 0, 0 * nn.pi / 2))
sim = nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)
t = nn.tic()
sim.solve()
nn.toc(t)
print(ms.is_uniform)
w1 = bk.sort(ms.eigenvalues.real)


# print(w1)

assert np.allclose(w0, w1)
