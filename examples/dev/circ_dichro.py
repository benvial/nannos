#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
Geometry tools
==============

Defining patterns using shapely.
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg

import nannos as nn
from nannos.geometry import shape_mask

plt.close("all")
plt.ion()

N = 2**8

####################################################################
# Double-sided scythe (DSS)
P = 850
R = 280
L1 = L2 = 220
W1 = W2 = 191
W1 = 191
W2 = 231
H = 350
center = P / 2, P / 2
n_Si = 3.2

####################################################################
# Various patterns

x = np.linspace(0, P, N)
y = np.linspace(0, P, N)

epsilon = np.ones((N, N))
circle = sg.Point(*center).buffer(R)
mask = shape_mask(circle, x, y)

epsilon[mask] = n_Si**2  # alpha-Si

arm_1 = sg.Polygon(
    [
        (center[0] - R + L1, center[1] + R),
        (center[0] - R + L1, center[1] + R - W1),
        (center[0] - R, center[1] + R - W1),
        (center[0] - R, center[1] + R),
    ]
)
mask1 = shape_mask(arm_1, x, y)
epsilon[mask1] = 1

arm_2 = sg.Polygon(
    [
        (center[0] + R - L2, center[1] - R),
        (center[0] + R - L2, center[1] - R + W2),
        (center[0] + R, center[1] - R + W2),
        (center[0] + R, center[1] - R),
    ]
)
mask2 = shape_mask(arm_2, x, y)
epsilon[mask2] = 1

cmap = mpl.colors.ListedColormap(["#ffe7c2", "#232a4e"])

plt.imshow(epsilon, cmap=cmap, origin="lower", extent=[0, P, 0, P])
plt.colorbar()
plt.show()

##############################################################################
# Define the lattice

lattice = nn.Lattice(([P, 0], [0, P]))

##############################################################################
# Define the layers
eps_sub = 1
eps_sup = 1.45**2  # SiO2

sup = nn.Layer("Superstrate", epsilon=eps_sup)
ms = nn.Layer("Metasurface", thickness=H)
sub = nn.Layer("Substrate", epsilon=eps_sub)

##############################################################################
# Define the pattern and add it to the metasurface layer

pattern = nn.Pattern(epsilon, name="DSS")
ms.add_pattern(pattern)


##############################################################################
# Define the incident plane wave

nh = 151
spectra = []

wls = np.linspace(1330, 1480, 51)

# wls = np.linspace(1400, 1550, 1)


theta, phi, psi = 3.7 * nn.pi / 180, 0, 0
freq = 1 / wls[0]


Tpola = dict()


plt.figure()
CD_spetra = []
i = 0
for wl in wls:
    freq = 1 / wl

    for orientation in ["right", "left"]:
        # pw = nn.PlaneWave(frequency=freq, angles=(theta, phi, psi))
        pw = nn.excitation.CircPolPlaneWave(
            frequency=freq, angles=(theta, phi, psi), orientation=orientation
        )
        stack = [sup, ms, sub]
        sim = nn.Simulation(lattice, stack, pw, nh)
        R, T = sim.diffraction_efficiencies()
        print("R = ", R)
        print("T = ", T)
        print("R+T = ", R + T)
        Tpola[orientation] = T

    CD = (Tpola["right"] - Tpola["left"]) / (Tpola["right"] + Tpola["left"])

    print("CD = ", CD)
    CD_spetra.append(CD)

    plt.plot(wl, CD, "sr")
    i += 1
    plt.plot(wls[:i], CD_spetra, "r")
    plt.pause(0.1)


xsxs

plt.figure()

for wl in wls:
    freq = 1 / wl
    Tmatrix = []
    for psi in [0, np.pi / 2]:
        # freq = 1 / 1400
        theta, phi = 8 * np.pi / 180, 0
        pw = nn.PlaneWave(frequency=freq, angles=(theta, phi, psi))
        # pw = nn.excitation.CircPolPlaneWave(frequency=freq, angles=(theta, phi, psi))

        ##############################################################################
        # Define the simulation
        stack = [sup, ms, sub]
        sim = nn.Simulation(lattice, stack, pw, nh)

        ##############################################################################
        # Compute diffraction efficiencies per order

        # R, T = sim.diffraction_efficiencies()
        # print("R = ", R)
        # print("T = ", T)
        # print("R+T = ", R + T)
        #
        # ai, bi = sim._get_amplitudes(0)
        # at, bt = sim._get_amplitudes(2)
        #
        #
        # np.abs(at[0] / ai[0]) ** 2
        # np.abs(at[sim.nh] / ai[0]) ** 2

        htot = sum([s.thickness for s in sim.layers])

        e, h = sim.get_field_fourier(2, z=htot)[0]
        # e, h = sim.fields_fourier[0]
        ex, ey, ez = e
        hx, hy, hz = h

        # Ax = np.sum(ex, axis=-1)
        # Ay = np.sum(ey, axis=-1)

        Ax = ex[0]
        Ay = ey[0]

        Tmatrix.append([Ax, Ay])

    Tmatrix = np.array(Tmatrix).T

    #### circular basis

    Lambda = np.array([[1, 1], [1j, -1j]]) / 2**0.5

    Tmatrix_circ = np.linalg.inv(Lambda) @ Tmatrix @ Lambda

    spectra.append(Tmatrix_circ)

    absTmatrix_circ = np.abs(Tmatrix_circ) ** 2

    plt.plot(wl, absTmatrix_circ[0, 0], "sr")
    plt.plot(wl, absTmatrix_circ[1, 0], "<g")
    plt.plot(wl, absTmatrix_circ[0, 1], "ob")
    plt.plot(wl, absTmatrix_circ[1, 1], "^k")
    plt.tight_layout()
    plt.pause(0.1)
