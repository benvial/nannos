#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
Two photonic crystal slabs
==========================

Mechanically tunable photonic crystal structure consisting of coupled photonic crystal slabs.
"""

# sphinx_gallery_thumbnail_number = 1

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

plt.ion()
plt.close("all")

##############################################################################
# We will code the structures studied in :cite:p:`Suh2003`.

nh = 40
L1 = [1.0, 0]
L2 = [0, 1.0]
theta = 0.0 * np.pi / 180
phi = 0.0 * np.pi / 180
psi = 0.0 * np.pi / 180

Nx = 2**8
Ny = 2**8

eps_sup = 1.0
eps_pattern = 12.0
eps_hole = 1.0
eps_sub = 1.0
h = 0.55

radius = 0.4
epsgrid = np.ones((Nx, Ny), dtype=float) * eps_pattern
x0 = np.linspace(0, 1.0, Nx)
y0 = np.linspace(0, 1.0, Ny)
x, y = np.meshgrid(x0, y0, indexing="ij")
hole = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius**2
epsgrid[hole] = eps_hole

##############################################################################
# Define the problem

lattice = nn.Lattice((L1, L2))
sup = nn.Layer("Superstrate", epsilon=eps_sup)
phc_slab = nn.Layer("PC slab", thickness=h)
sub = nn.Layer("Substrate", epsilon=eps_sub)
pattern = nn.Pattern(epsgrid, name="hole")
phc_slab.add_pattern(pattern)
stack = [sup, phc_slab, sub]


##############################################################################
# Fig 2 (a) from :cite:p:`Suh2003`.


def compute_transmission(fn):
    pw = nn.PlaneWave(frequency=fn, angles=(0, 0, 0))
    sim = nn.Simulation(lattice, stack, pw, nh)
    R, T = sim.diffraction_efficiencies()
    print(f"f = {fn} (normalized)")
    print("T = ", T)
    return T


#
freqs_norma = np.linspace(0.49, 0.6, 30)
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


##############################################################################
# Figs 2 (b-c) from :cite:p:`Suh2003`.

phc_slab_top = nn.Layer("PC slab top", thickness=h)
phc_slab_top.add_pattern(pattern)
phc_slab_bot = phc_slab_top.copy("PC slab bottom")

plt.figure()

seps = [1.35, 1.1, 0.95, 0.85, 0.75, 0.65, 0.55]
colors = plt.cm.Spectral(np.linspace(0, 1, len(seps)))

for i, sep in enumerate(seps):
    spacer = nn.Layer("Spacer", epsilon=1, thickness=sep)
    stack = [sup, phc_slab_top, spacer, phc_slab_bot, sub]

    def compute_transmission(fn):
        pw = nn.PlaneWave(frequency=fn, angles=(0, 0, 0))
        sim = nn.Simulation(lattice, stack, pw, nh)
        R, T = sim.diffraction_efficiencies()
        print(f"f = {fn} (normalized)")
        print("T = ", T)
        return T

    freqs_norma = np.linspace(0.49, 0.6, 30)
    freqs_adapted, transmission = nn.adaptive_sampler(
        compute_transmission,
        freqs_norma,
    )

    plt.plot(freqs_adapted, transmission, c=colors[i], label=rf"$d = {sep}a$")
    plt.xlim(freqs_norma[0], freqs_norma[-1])
    plt.ylim(0, 1)
    plt.xlabel(r"frequency ($2\pi c / a$)")
    plt.ylabel("Transmission")
    plt.tight_layout()
    plt.pause(0.1)


plt.legend(loc=(1.05, 0.3))
plt.tight_layout()
