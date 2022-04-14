#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of nannos
# License: GPLv3
# See the documentation at nannos.gitlab.io


"""
Elliptical holes
=================

Convergence checks.
"""

import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

#########################################################################
# We will study the convergence on a benchmark case from
# :cite:p:`Schuster2007`.
# First we define the main function that performs the simulation.


def array_ellipse(nh, formulation, psi):
    wl = 500 + 1e-6  # avoid Wood-Rayleigh anomaly
    d = 1000
    Nx = 2**9
    Ny = 2**9
    lattice = nn.Lattice(([d, 0], [0, d]), discretization=(Nx, Ny))
    pw = nn.PlaneWave(wavelength=wl, angles=(0, 0, psi))
    epsgrid = lattice.ones() * (1.75 + 1.5j) ** 2
    ell = lattice.ellipse((0.5 * d, 0.5 * d), (1000 / 2, 500 / 2), rotate=45)
    epsgrid[ell] = 1

    sup = lattice.Layer("Superstrate", epsilon=1)
    sub = lattice.Layer("Substrate", epsilon=1.5**2)
    st = lattice.Layer("Structured", thickness=50)
    st.epsilon = epsgrid

    sim = nn.Simulation([sup, st, sub], pw, nh, formulation=formulation)
    order = (0, 0)
    R, T = sim.diffraction_efficiencies(orders=True)
    r = sim.get_order(R, order)
    return R, T, r, sim


#
# sim = array_ellipse(100, "original")
# lay = sim.get_layer("Structured")
# lay.plot()
# plt.show()

#########################################################################
# Perform the simulation for different formulations and number
# of retained harmonics:

NH = [100, 200, 300, 400, 500, 600]
formulations = ["original", "tangent"]


def run_convergence(psi):
    nhs = {f: [] for f in formulations}
    rs = {f: [] for f in formulations}

    for nh in NH:
        print("============================")
        print("number of harmonics = ", nh)
        print("============================")
        for formulation in formulations:
            Ri, Ti, r, sim = array_ellipse(nh, formulation=formulation, psi=psi)
            R = np.sum(Ri)
            T = np.sum(Ti)
            print("formulation = ", formulation)
            print("nh0 = ", nh)
            print("nh = ", sim.nh)
            print("r = ", r)
            print("R = ", R)
            print("T = ", T)
            print("R+T = ", R + T)
            print("-----------------")
            nhs[formulation].append(sim.nh)
            rs[formulation].append(r)

    return nhs, rs


#########################################################################
# Plot the results:


markers = {"original": "^", "tangent": "o"}
colors = {
    "original": "#d4b533",
    "tangent": "#4cb7c6",
}

plt.ion()

for psi in [45, -45]:
    nhs, rs = run_convergence(psi)
    plt.figure(figsize=(2, 2))

    for formulation in formulations:
        plt.plot(
            nhs[formulation],
            rs[formulation],
            "-",
            color=colors[formulation],
            marker=markers[formulation],
            label=formulation,
        )
        plt.pause(0.1)
    plt.legend()
    plt.xlabel("number of Fourier harmonics $n_h$")
    plt.ylabel("$R_{0,0}$")
    t = "" if psi == 45 else "-"
    plt.title(rf"$\psi = {t}45\degree$")
    plt.ylim(0.16, 0.2)
    plt.tight_layout()
    plt.show()
