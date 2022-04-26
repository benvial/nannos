#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
Polarization conversion
=========================

Simulation of conversion efficiency of a geometric metasurface.
"""

import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

##############################################################################
# The unit cell is a rectangular Si structure on a SiO2 substrate
# as in :cite:p:`Yoon2021`.
# Circularly-polarized light of wavelength 635 nm is normally incident from the
# substrate to the structure

wl = 635  # wavelength
P = 350  # period
W = 100  # pillar width along x
L = 190  # pillar length along y
eps_Si = (3.87 + 0.02j) ** 2
eps_SiO2 = 1.4573**2
nh = 100


##############################################################################
# Define a function to initialize simulation


def simu(H, psi):
    lattice = nn.Lattice([[P, 0], [0, P]], discretization=2**9)
    sup = lattice.Layer("Superstrate", epsilon=eps_SiO2)
    sub = lattice.Layer("Substrate", epsilon=1)
    epsilon = lattice.ones()
    metaatom = lattice.rectangle((0.5 * P, 0.5 * P), (W, L))
    epsilon[metaatom] = eps_Si
    ms = lattice.Layer("Metasurface", thickness=H, epsilon=epsilon)
    pw = nn.PlaneWave(wavelength=wl, angles=(0, 0, psi))
    return nn.Simulation([sup, ms, sub], pw, nh=nh)


##############################################################################
# Since the layer eigenmodes do not change with thickness we compute them
# only once for the first iteration.


nb_thick = 100
thicknesses = np.linspace(200, 500, nb_thick)
conv_effs = np.zeros(nb_thick)

for ih, H in enumerate(thicknesses):

    # x-polarization
    if ih == 0:
        simx = simu(H, 0)
    else:
        simx.layers[1].thickness = H
        simx.reset("S")
    rxi, txi = simx.diffraction_efficiencies(orders=True, complex=True)
    txx = simx.get_order(txi[0], (0, 0))

    # y-polarization
    if ih == 0:
        simy = simu(H, 90)
    else:
        simy.layers[1].thickness = H
        # print(self.is_solved)
        simy.reset("S")
    ryi, tyi = simy.diffraction_efficiencies(orders=True, complex=True)
    tyy = simy.get_order(tyi[1], (0, 0))

    conv_effs[ih] = np.abs((tyy - txx) / 2) ** 2

##############################################################################
# Plot the efficiency

plt.clf()
plt.plot(thicknesses, conv_effs)
plt.xlabel("H (nm)")
plt.ylabel("conversion efficiency")
plt.tight_layout()
plt.show()


##############################################################################
# Plot the unit cell

p = simx.plot_structure()
p.show_axes()
p.show()
