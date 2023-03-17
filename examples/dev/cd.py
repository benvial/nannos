#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io

# Simulation of a chiral metasurface. (a) Unit cell configuration consisting
# of a Z-shaped Si structure. Circularly-polarized light is normally incident from
# the substrate to structures. P: 400 nm; H: 300 nm; L0 : 300 nm; L1 : 100 nm; L2 :
# 100 nm; W0 : 300 nm; W1 : 200 nm; W2 : 200 nm. (b) Comparison of calculated
# circular dichroism (CD) in the visible regime. Truncation orders are set to 20 for
# both directions.

import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

nh = 51
bk = nn.backend
# formulation = "tangent"
formulation = "original"

P = 400
H = 300
L0 = 300
L1 = L2 = 100
W0 = 300
W1 = W2 = 200

# eps_Si = (3.87870 + 0.019221j) ** 2
eps_Si = (3.86 + 1j * 0.015) ** 2

eps_SiO2 = 1.4570**2


wavelengths = np.linspace(400, 800, 400)


def get_order_comp(self, A, order, comp):
    c = 0 if comp == "x" else self.nh
    return A[c + self.get_order_index(order)]


plt.close("all")
plt.ion()
plt.clf()
Wc = W0 - 0.5 * W1 - 0.5 * W2

for wl in wavelengths:
    lattice = nn.Lattice([[P, 0], [0, P]], discretization=2**9)
    sup = lattice.Layer("Superstrate", epsilon=eps_SiO2)
    sub = lattice.Layer("Substrate", epsilon=1)
    epsilon = lattice.ones()
    metaatom = lattice.rectangle((0.5 * P, 0.5 * P), (Wc, L0))
    metaatom += lattice.rectangle(
        (0.5 * P + Wc / 2, 0.5 * P + L0 / 2 - L2 / 2), (W2, L2)
    )
    metaatom += lattice.rectangle(
        (0.5 * P - Wc / 2, 0.5 * P - L0 / 2 + L2 / 2), (W2, L2)
    )
    epsilon[metaatom] = eps_Si
    ms = lattice.Layer("Metasurface", thickness=H, epsilon=epsilon)

    # ms.plot()

    pwx = nn.PlaneWave(wavelength=1 / wl, angles=(0, 0, 1 * nn.pi / 2))
    simx = nn.Simulation([sup, ms, sub], pwx, nh=nh, formulation=formulation)
    pwy = nn.PlaneWave(wavelength=1 / wl, angles=(0, 0, 0 * nn.pi / 2))
    simy = nn.Simulation([sup, ms, sub], pwy, nh=nh, formulation=formulation)

    norma = 1  # eps_SiO2

    axN = simx.get_field_fourier("Substrate")[0, 0, 0:2]
    txx = simx.get_order(axN[0], (0, 0)) * norma
    tyx = simx.get_order(axN[1], (0, 0)) * norma

    ayN = simy.get_field_fourier("Substrate")[0, 0, 0:2]
    txy = simy.get_order(ayN[0], (0, 0)) * norma
    tyy = simy.get_order(ayN[1], (0, 0)) * norma
    J = 0.5 * np.array(
        [
            [txx + tyy + 1j * (tyx - txy), txx - tyy - 1j * (txy + tyx)],
            [txx - tyy + 1j * (tyx + txy), txx + tyy - 1j * (txy - tyx)],
        ]
    )

    Tl = np.abs(J[0, 0]) ** 2 + np.abs(J[1, 0]) ** 2
    Tr = np.abs(J[0, 1]) ** 2 + np.abs(J[1, 1]) ** 2
    CD = Tl - Tr

    # CE = np.abs((J[0, 0] - J[0, 1]) / 2) ** 2

    plt.plot(wl, CD, "or")
    plt.pause(0.1)
