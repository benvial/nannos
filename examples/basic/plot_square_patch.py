#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
Dielectric patch array
======================

Transmission spectrum.
"""


# sphinx_gallery_thumbnail_number = -1


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.constants import c, e, h

import nannos as nn

#########################################################################
# Results are compared to the reference :cite:p:`Tikhodeev2002`.


eps_quartz = 2.132
eps_active = 3.97

N = 2**7
period = 0.68
l_patch = 0.8 * period


#########################################################################
# Define the lattice

lattice = nn.Lattice(([period, 0], [0, period]), discretization=(N, N))


#########################################################################
# Define the slab layer with a square patch

epsilon = lattice.ones() * eps_quartz
square = lattice.square(center=(0.5 * period, 0.5 * period), width=l_patch)
epsilon[square] = eps_active

slab = lattice.Layer("Slab", thickness=0.12)
slab.epsilon = epsilon

cmap = ListedColormap(["#dddddd", "#73a0e8"])


plt.figure(figsize=(3, 2.5))
im = slab.plot(cmap=cmap)
cbar = plt.colorbar(im[0], ticks=[eps_quartz, eps_active])
plt.xlabel(r"$x$ ($\mu$m)")
plt.ylabel(r"$y$ ($\mu$m)")
plt.title(r"$\varepsilon$")
plt.axis("scaled")
plt.tight_layout()
plt.show()

#########################################################################
# Define the simulation

sup = lattice.Layer("Superstrate", epsilon=1)
sub = lattice.Layer("Substrate", epsilon=eps_quartz)
stack = [sup, slab, sub]


def compute_transmission(fev):
    w = h * c / e / (fev * 1e-6)
    f = 1 / w
    pw = nn.PlaneWave(frequency=f, angles=(0, 0, np.pi / 2))
    sim = nn.Simulation(stack, pw, 100, formulation="tangent")
    R, T = sim.diffraction_efficiencies()
    print(f"f = {fev}eV")
    print("T = ", T)
    return T


freqsev = np.linspace(1, 2.6, 101)
fev_adapted, transmission = nn.adaptive_sampler(
    compute_transmission, freqsev, max_bend=10, max_x_rel=0.001, max_df=0.005
)


#########################################################################
# Figure 4 from :cite:p:`Tikhodeev2002`.

plt.figure()
plt.plot(fev_adapted * 1000, transmission, c="#be4c83")
plt.ylim(0.4, 1)
plt.xlabel("frequency (meV)")
plt.ylabel("Transmissivity")
plt.tight_layout()


#########################################################################
# Plot the fields at the resonant frequency of 2456meV

fev = 2.456
w = h * c / e / (fev * 1e-6)  # /1000
f = 1 / w
pw = nn.PlaneWave(frequency=f, angles=(0, 0, np.pi / 2))
sim = nn.Simulation(stack, pw, 151, formulation="tangent")
E, H = sim.get_field_grid("Superstrate", shape=(N, N))

Ex, Ey, Ez = E[:, :, :, 0]
Hx, Hy, Hz = H[:, :, :, 0]
nE2 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2  # + np.abs(Ez)**2
nH2 = np.abs(Hx) ** 2 + np.abs(Hy) ** 2  # + np.abs(Hz)**2

#########################################################################
# Electric field

extent = [0, period, 0, period]
x, y = np.linspace(0, period, N), np.linspace(0, period, N)

plt.figure()
plt.imshow(epsilon.real, cmap="Greys", origin="lower", extent=extent)
plt.imshow(nE2, alpha=0.9, origin="lower", extent=extent)
plt.colorbar()
s = 3
plt.quiver(x[::s], y[::s], Ex[::s, ::s].real, Ey[::s, ::s].real, color="w")
plt.xlabel(r"$x$ ($\mu$m)")
plt.ylabel(r"$y$ ($\mu$m)")
plt.title("$E$")
plt.tight_layout()
plt.show()

#########################################################################
# Magnetic field

plt.figure()
plt.imshow(epsilon.real, cmap="Greys", origin="lower", extent=extent)
plt.imshow(nH2, alpha=0.9, origin="lower", extent=extent)
plt.colorbar()
plt.quiver(x[::s], y[::s], Hx[::s, ::s].real, Hy[::s, ::s].real, color="w")
plt.xlabel(r"$x$ ($\mu$m)")
plt.ylabel(r"$y$ ($\mu$m)")
plt.title("$H$")
plt.tight_layout()
plt.show()
