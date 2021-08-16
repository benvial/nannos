#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


"""
Convergence
===========

Convergence of the various FMM formulations.
"""

import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

#########################################################################
# We will study the convergence on a benchmark case from
# :cite:p:`Li1997`.


def checkerboard(nh, formulation):
    la = 1
    lattice = nn.Lattice(([2 * 1.25 * la, 0], [0, 2 * 1.25 * la]))

    freq = 1 / la
    theta = 0.0 * nn.pi / 180
    phi = 0.0 * nn.pi / 180
    psi = 0.0 * nn.pi / 180
    pw = nn.PlaneWave(frequency=freq, angles=(theta, phi, psi))

    Nx = 2 ** 9
    Ny = 2 ** 9

    x0 = np.linspace(0, 1.0, Nx)
    y0 = np.linspace(0, 1.0, Ny)
    x, y = np.meshgrid(x0, y0, indexing="ij")
    epsgrid = 2.25 * np.ones((Nx, Ny), dtype=float)
    epsgrid[np.logical_and(x > 0.5, y > 0.5)] = 1
    epsgrid[np.logical_and(x < 0.5, y < 0.5)] = 1

    sup = nn.Layer("Superstrate", epsilon=2.25)
    sub = nn.Layer("Substrate", epsilon=1)
    st = nn.Layer("Structured", la)
    pattern = nn.Pattern(epsgrid)
    st.add_pattern(pattern)

    simu = nn.Simulation(lattice, [sup, st, sub], pw, nh, formulation=formulation)
    order = (1, 1)
    R, T = simu.diffraction_efficiencies(orders=True)
    t = simu.get_order(T, order)
    return R, T, t, simu


NH = [100, 200, 300, 400, 600]
formulations = ["original", "tangent", "jones"]

nhs = {f: [] for f in formulations}
ts = {f: [] for f in formulations}
markers = {"original": "^", "tangent": "o", "jones": "s"}
colors = {
    "original": "#d4b533",
    "tangent": "#d46333",
    "jones": "#3395d4",
}

for nh in NH:
    for formulation in formulations:
        Ri, Ti, t, simu = checkerboard(nh, formulation=formulation)
        R = np.sum(Ri)
        T = np.sum(Ti)
        print("formulation = ", formulation)
        print("nh0 = ", nh)
        print("nh = ", simu.nh)
        print("t = ", t)
        print("R = ", R)
        print("T = ", T)
        print("R+T = ", R + T)
        print("-----------------")
        nhs[formulation].append(simu.nh)
        ts[formulation].append(t)

for formulation in formulations:
    plt.plot(
        nhs[formulation],
        ts[formulation],
        "-",
        color=colors[formulation],
        marker=markers[formulation],
        label=formulation,
    )
    plt.pause(0.1)
plt.legend()
plt.xlabel("number of Fourier harmonics $n_h$")
plt.ylabel("$T_{0,-1}$")
plt.tight_layout()
