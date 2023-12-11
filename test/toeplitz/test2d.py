#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3
#################   2D

import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy

import nannos as nn

# plt.close("all")
plt.ion()

form = "original"
# form = "tangent"
nh = int(sys.argv[1])
theta = 30
psi = 0
pw = nn.PlaneWave(wavelength=1, angles=(theta, 0, psi))

lattice = nn.Lattice(
    [[1.0, 0], [0, 1.0]], discretization=(2**9, 2**9), truncation="parallelogrammic"
)


sup = lattice.Layer("Superstrate", epsilon=1)
ms = lattice.Layer("Metasurface", thickness=0.5)
sub = lattice.Layer("Substrate", epsilon=1)

ms.epsilon = lattice.ones() * 12.0
circ = lattice.circle(center=(0.5, 0.5), radius=0.2)
ms.epsilon[circ] = 1

stack = [sup, ms, sub]
sim = nn.Simulation(stack, pw, nh=nh)

print(sim.harmonics)

Ri, Ti = sim.diffraction_efficiencies(orders=True)
print(Ri)
print(Ti)
R, T = sim.diffraction_efficiencies()
print(R, T, R + T)
