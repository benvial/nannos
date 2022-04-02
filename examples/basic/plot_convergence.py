#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


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
# First we define the main function that performs the simulation.


def checkerboard(nh, formulation):
    la = 1
    d = 2 * 1.25 * la
    Nx = 2**9
    Ny = 2**9
    lattice = nn.Lattice(([d, 0], [0, d]), discretization=(Nx, Ny))

    freq = 1 / la
    pw = nn.PlaneWave(frequency=freq, angles=(0, 0, 0))
    epsgrid = lattice.ones() * 2.25
    sq1 = lattice.square((0.25 * d, 0.25 * d), 0.5 * d)
    sq2 = lattice.square((0.75 * d, 0.75 * d), 0.5 * d)
    epsgrid[sq1] = 1
    epsgrid[sq2] = 1

    sup = lattice.Layer("Superstrate", epsilon=2.25)
    sub = lattice.Layer("Substrate", epsilon=1)
    st = lattice.Layer("Structured", la)
    st.epsilon = epsgrid

    sim = nn.Simulation([sup, st, sub], pw, nh, formulation=formulation)
    order = (
        -1,
        -1,
    )  # this actually corresponds to order (0,-1) for the other unit cell in [Li1997]
    R, T = sim.diffraction_efficiencies(orders=True)
    t = sim.get_order(T, order)
    return R, T, t, sim


#########################################################################
# Perform the simulation for different formulations and number
# of retained harmonics:

NH = [100, 200, 300, 400, 600]
formulations = ["original", "tangent", "pol", "jones"]
nhs = {f: [] for f in formulations}
ts = {f: [] for f in formulations}


for nh in NH:
    print("============================")
    print("number of harmonics = ", nh)
    print("============================")

    for formulation in formulations:
        Ri, Ti, t, sim = checkerboard(nh, formulation=formulation)
        R = np.sum(Ri)
        T = np.sum(Ti)
        print("formulation = ", formulation)
        print("nh0 = ", nh)
        print("nh = ", sim.nh)
        print("t = ", t)
        print("R = ", R)
        print("T = ", T)
        print("R+T = ", R + T)
        print("-----------------")
        nhs[formulation].append(sim.nh)
        ts[formulation].append(t)

#########################################################################
# Plot the results:


markers = {"original": "^", "tangent": "o", "jones": "s", "pol": "^"}
colors = {
    "original": "#d4b533",
    "tangent": "#d46333",
    "jones": "#3395d4",
    "pol": "#54aa71",
}

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
plt.ylim(0.1255, 0.129)
plt.tight_layout()
