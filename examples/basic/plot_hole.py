#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

"""
Photonic crystal slab
=====================

Metasurface with holes.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

##############################################################################
# We will study a benchmark of hole in a dielectric surface

nh = 200
L1 = [1.0, 0]
L2 = [0, 1.0]
freq = 1.4
theta = 0.0 * np.pi / 180
phi = 0.0 * np.pi / 180
psi = 0.0 * np.pi / 180

Nx = 2 ** 9
Ny = 2 ** 9

eps_sup = 1.0
eps_pattern = 6.0
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


##############################################################################
# Visualize the permittivity

cmap = mpl.colors.ListedColormap(["#ffe7c2", "#232a4e"])

bounds = [eps_hole, eps_pattern]
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

simu = nn.Simulation(lattice, [sup, ms, sub], pw, nh)


##############################################################################
# Compute diffraction efficiencies

R, T = simu.diffraction_efficiencies()

##############################################################################
# Compute diffraction efficiencies per order

Ri, Ti = simu.diffraction_efficiencies(orders=True)
nmax = 5
print("Ri = ", Ri[:nmax])
print("Ti = ", Ti[:nmax])
print("R = ", R)
print("T = ", T)
print("R+T = ", R + T)


##############################################################################
# Plot

simu.G[:, :nmax][:, 1]

fig, (axR, axT) = plt.subplots(1, 2, figsize=(4, 2))

labels = [f"({g[0]},{g[1]})" for g in (simu.G[:, :nmax]).T]


axR.bar(range(nmax), Ri[:nmax], color=["#e69049"])
axR.set_xticks(range(nmax))
axR.set_xticklabels(labels=labels)
axR.set_xlabel("order")
axR.set_ylabel("reflection $R_{i,j}$")
axR.annotate(
    r"$R = \sum_i\,\sum_j\, R_{i,j}=$" + f"{sum(Ri[:nmax]):0.4f}",
    (0.5, 0.9),
    xycoords="axes fraction",
)

axT.bar(range(nmax), Ti[:nmax], color=["#4a77ba"])
axT.set_xticks(range(nmax))
axT.set_xticklabels(labels=labels)
axT.set_xlabel("order")
axT.set_ylabel("transmission $T_{i,j}$")
axT.annotate(
    r"$T =\sum_i\,\sum_j\, T_{i,j}=$" + f"{sum(Ti[:nmax]):0.4f}",
    (0.5, 0.9),
    xycoords="axes fraction",
)


plt.suptitle("Diffraction efficiencies: $R+T=$" + f"{sum(Ri[:nmax]+Ti[:nmax]):0.4f}")
plt.tight_layout()
plt.show()
