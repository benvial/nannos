#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

"""
Photonic crystal slab
=====================

Metasurface with holes.
"""

import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

##############################################################################
# We will study a benchmark of hole in a dielectric surface

nG = 100
L1 = [1.0, 0]
L2 = [0, 1.0]
freq = 1.1
theta = 0.0 * np.pi / 180
phi = 0.0 * np.pi / 180
psi = 0.0 * np.pi / 180

Nx = 2 ** 7
Ny = 2 ** 7

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


##############################################################################
# Visualize the permittivity
import matplotlib as mpl

cmap = mpl.colors.ListedColormap(["#ffe7c2", "#232a4e"])

bounds = [1, 12]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
plt.imshow(epsgrid, cmap=cmap, extent=(0, 1, 0, 1))
plt.colorbar(ticks=bounds)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(r"permittitivity $\varepsilon(x,y)$")
plt.tight_layout()
plt.show()


##############################################################################
# Define the lattice

lattice = nn.Lattice((L1, L2))

##############################################################################
# Define the layers

sup = nn.Layer("Superstrate", epsilon=eps_sup)
ms = nn.Layer("Metasurface", epsilon=2, thickness=3)
sub = nn.Layer("Substrate", epsilon=eps_sub)


##############################################################################
# Define the pattern and add it to the metasurface layer

pattern = nn.Pattern(epsgrid, name="hole")
ms.add_pattern(pattern)

##############################################################################
# Define the incident plane wave

pw = nn.PlaneWave(frequency=freq, angles=(theta, phi, psi))

##############################################################################
# Define the simulation

simu = nn.Simulation(lattice, [sup, ms, sub], pw, nG)


##############################################################################
# Compute diffraction efficiencies

R, T = simu.diffraction_efficiencies()

##############################################################################
# Compute diffraction efficiencies per order

Ri, Ti = simu.diffraction_efficiencies(orders=True)


plt.figure()
plt.bar(range(2), [R, T], color=["#4a77ba", "#e69049"])
plt.xticks((0, 1), labels=("R", "T"))
plt.title("Diffraction efficiencies")
nmax = 5
print("Ri = ", Ri[:nmax])
print("Ti = ", Ti[:nmax])
print("R = ", R)
print("T = ", T)
print("R+T = ", R + T)
