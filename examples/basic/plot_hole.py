#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
Photonic crystal slab
=====================

Metasurface with holes.
"""

# sphinx_gallery_thumbnail_number = 1

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

##############################################################################
# We will study a benchmark of hole in a dielectric surface similar to
# those studied in :cite:p:`Fan2002`.

##############################################################################
# Define the lattice

lattice = nn.Lattice([[1.0, 0], [0, 1.0]], discretization=(2**9, 2**9))


##############################################################################
# Define the layers

sup = lattice.Layer("Superstrate", epsilon=1)
ms = lattice.Layer("Metasurface", thickness=0.5)
sub = lattice.Layer("Substrate", epsilon=1)


##############################################################################
# Define the pattern and add it to the metasurface layer

ms.epsilon = lattice.ones() * 12.0
circ = lattice.circle(center=(0.5, 0.5), radius=0.2)
ms.epsilon[circ] = 1

##############################################################################
# Visualize the permittivity

cmap = mpl.colors.ListedColormap(["#ffe7c2", "#232a4e"])
bounds = [1, 12]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
ims = ms.plot(cmap=cmap)
plt.axis("scaled")
plt.colorbar(ims[0], ticks=bounds)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(r"permittitivity $\varepsilon(x,y)$")
plt.tight_layout()
plt.show()

##############################################################################
# Define the incident plane wave

pw = nn.PlaneWave(frequency=1.4, angles=(0, 0, 0))

##############################################################################
# Define the simulation

stack = [sup, ms, sub]
sim = nn.Simulation(stack, pw, nh=100)


##############################################################################
# Compute diffraction efficiencies

R, T = sim.diffraction_efficiencies()

##############################################################################
# Compute diffraction efficiencies per order

Ri, Ti = sim.diffraction_efficiencies(orders=True)
nmax = 5
print("Ri = ", Ri[:nmax])
print("Ti = ", Ti[:nmax])
print("R = ", R)
print("T = ", T)
print("R+T = ", R + T)


##############################################################################
# Plot


fig, (axR, axT) = plt.subplots(1, 2, figsize=(4, 2))

labels = [f"({g[0]},{g[1]})" for g in (sim.harmonics[:, :nmax]).T]


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


##############################################################################
# Fig 12 (c) from :cite:p:`Fan2002`.


def compute_transmission(fn):
    pw = nn.PlaneWave(frequency=fn, angles=(0, 0, 0))
    sim = nn.Simulation(stack, pw, 100)
    R, T = sim.diffraction_efficiencies()
    print(f"f = {fn} (normalized)")
    print("T = ", T)
    return T


freqs_norma = np.linspace(0.25, 0.6, 100)
freqs_adapted, transmission = nn.adaptive_sampler(
    compute_transmission,
    freqs_norma,
)


plt.figure()
plt.plot(freqs_adapted, transmission, c="#be4c83")
plt.xlim(freqs_norma[0], freqs_norma[-1])
plt.ylim(0, 1)
plt.xlabel(r"frequency ($2\pi c / a$)")
plt.ylabel("Transmission")
plt.tight_layout()
