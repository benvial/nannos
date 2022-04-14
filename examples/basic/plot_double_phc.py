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

##############################################################################
# We will code the structures studied in :cite:p:`Suh2003`.

nh = 51
L1 = [1.0, 0]
L2 = [0, 1.0]
theta = 0.0 * np.pi / 180
phi = 0.0 * np.pi / 180
psi = 0.0 * np.pi / 180

Nx = 2**8
Ny = 2**8

lattice = nn.Lattice((L1, L2), (Nx, Ny))

epsgrid = lattice.ones() * 12.0
hole = lattice.circle((0.5, 0.5), 0.4)
epsgrid[hole] = 1.0

##############################################################################
# Define the problem

sup = lattice.Layer("Superstrate", epsilon=1.0)
phc_slab = lattice.Layer("PC slab", thickness=0.55)
sub = lattice.Layer("Substrate", epsilon=1.0)
phc_slab.epsilon = epsgrid
stack = [sup, phc_slab, sub]


##############################################################################
# Fig 2 (a) from :cite:p:`Suh2003`.


def compute_transmission(fn):
    pw = nn.PlaneWave(wavelength=1 / fn, angles=(0, 0, 0))
    sim = nn.Simulation(stack, pw, nh)
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

phc_slab_top = lattice.Layer("PC slab top", thickness=0.55)
phc_slab_top.epsilon = epsgrid
phc_slab_bot = phc_slab_top.copy("PC slab bottom")

plt.figure()

seps = [1.35, 1.1, 0.95, 0.85, 0.75, 0.65, 0.55]
colors = plt.cm.turbo(np.linspace(0, 1, len(seps)))

for i, sep in enumerate(seps):
    spacer = lattice.Layer("Spacer", epsilon=1, thickness=sep)
    stack = [sup, phc_slab_top, spacer, phc_slab_bot, sub]

    def compute_transmission(fn):
        pw = nn.PlaneWave(wavelength=1 / fn, angles=(0, 0, 0))
        sim = nn.Simulation(stack, pw, nh)
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
plt.show()
