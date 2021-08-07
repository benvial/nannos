#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

"""
Photonic crystal slab
=====================

Metasurface with holes.
"""

import nannos

nannos.set_backend("autograd")

# import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

from nannos import Lattice, Layer, Pattern, PlaneWave, Simulation

##############################################################################
# We will study a benchmark of hole in a dielectric surface

nG = 100
L1 = [1.0, 0]
L2 = [0, 1.0]
freq = 1.1
theta = 0.0 * np.pi / 180
phi = 0.0 * np.pi / 180
psi = 0.0 * np.pi / 180

Nx = 400
Ny = 400

eps_sup = 1.0
eps_pattern = 12.0
eps_hole = 1.0
eps_sub = 1.0


h = 1.0

radius = 0.2
epsgrid = np.ones((Nx, Ny), dtype=float) * eps_pattern
x0 = np.linspace(0, 1.0, Nx)
y0 = np.linspace(0, 1.0, Ny)
x, y = np.meshgrid(x0, y0, indexing="ij")
hole = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius ** 2
epsgrid[hole] = eps_hole
mugrid = np.ones((Nx, Ny), dtype=float) * 1
# mugrid[hole] = eps_pattern

pattern = Pattern(epsgrid, mugrid)

lattice = Lattice((L1, L2))

sup = Layer("Superstrate", epsilon=eps_sup)
ms = Layer("Metasurface", epsilon=2, thickness=3)
ms.add_pattern(pattern)
sub = Layer("Substrate", epsilon=eps_sub)

pw = PlaneWave(frequency=freq, angles=(theta, phi, psi))

simu = Simulation(lattice, [sup, ms, sub], pw, nG)

R, T = simu.diffraction_efficiencies()


plt.figure()
plt.bar(range(2), [R, T], color=["#4a77ba", "#e69049"])
plt.xticks
plt.xticks((0, 1), labels=("R", "T"))
plt.title("Diffraction efficiencies")

print("R = ", R)
print("T = ", T)
print("R+T = ", R + T)


def fun_reflection(epsgrid):

    mugrid = np.ones_like(epsgrid)
    pattern = Pattern(epsgrid, mugrid)
    lattice = Lattice((L1, L2))

    sup = Layer("Superstrate", epsilon=eps_sup)
    ms = Layer("Metasurface", h)
    ms.add_pattern(pattern)
    sub = Layer("Substrate", epsilon=eps_sub)

    pw = PlaneWave(frequency=freq, angles=(theta, phi, psi))

    simu = Simulation(lattice, [sup, ms, sub], pw, nG)

    R, T = simu.diffraction_efficiencies()

    return R


R = fun_reflection(epsgrid)

grad_fun_reflection = grad(fun_reflection)
dR_deps = grad_fun_reflection(epsgrid)

plt.figure()
plt.imshow(dR_deps)
