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
formulation = "tangent"
formulation = "original"

lattice = nn.Lattice([[1.0, 0], [0, 1.0]], discretization=2**9)
sup = lattice.Layer("Superstrate", epsilon=1)
sub = lattice.Layer("Substrate", epsilon=1)
o = lattice.ones()
hole = lattice.circle(center=(0.5, 0.5), radius=0.2)

z = bk.zeros_like(o)
epsilon_xx = 3 * o
epsilon_yy = 4 * o
epsilon_zz = 1 * o
epsilon_xx[hole] = epsilon_yy[hole] = epsilon_zz[hole] = 1
epsilon = bk.array([[epsilon_xx, z, z], [z, epsilon_yy, z], [z, z, epsilon_zz]])
ms_aniso = lattice.Layer("Metasurface", thickness=0.5, epsilon=epsilon)
ms = lattice.Layer("Metasurface", thickness=0.5, epsilon=epsilon_xx)
pw = nn.PlaneWave(frequency=1.4, angles=(0, 0, 0 * nn.pi / 2))
nh = 200
sim = nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)
R, T = sim.diffraction_efficiencies()
print(R, T, R + T)
sim_aniso = nn.Simulation([sup, ms_aniso, sub], pw, nh=nh, formulation=formulation)
R, T = sim_aniso.diffraction_efficiencies()
print(R, T, R + T)

epsilon_xx = 4 * o
epsilon_yy = 3 * o
epsilon_zz = 1 * o
epsilon_xx[hole] = epsilon_yy[hole] = epsilon_zz[hole] = 1
epsilon = bk.array([[epsilon_xx, z, z], [z, epsilon_yy, z], [z, z, epsilon_zz]])
ms_aniso = lattice.Layer("Metasurface", thickness=0.5, epsilon=epsilon)
pw = nn.PlaneWave(frequency=1.4, angles=(0, 0, 1 * nn.pi / 2))
sim_aniso = nn.Simulation([sup, ms_aniso, sub], pw, nh=nh, formulation=formulation)
R, T = sim_aniso.diffraction_efficiencies()
print(R, T, R + T)
#
#
# plt.close("all")
# for nh in [50,100,200,300,400]:
#     sim = nn.Simulation(stack, pw, nh=nh)
#     R, T = sim.diffraction_efficiencies()
#     print(R,T,R+T)
#     # eps = sim.get_epsilon("Substrate")
#     # eps = sim.get_epsilon(ms)
#     # plt.figure()
#     # plt.imshow(eps.real)
#     # plt.colorbar()
#     # plt.title(nh)
#     # plt.pause(0.1)
