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

import time

import matplotlib.pyplot as plt

import nannos as nn

bk = nn.backend

#########################################################################
# We will study the convergence on a benchmark case from
# :cite:p:`Li1997`.
# First we define the main function that performs the simulation.

wavelength = 1
sq_size = 1.25 * wavelength
eps_diel = 2.25


def checkerboard_cellA(nh, formulation):
    d = 2 * sq_size
    Nx = 2**9
    Ny = 2**9
    lattice = nn.Lattice(([d, 0], [0, d]), discretization=(Nx, Ny))
    pw = nn.PlaneWave(wavelength=wavelength, angles=(0, 0, 0))
    epsgrid = lattice.ones() * eps_diel
    sq1 = lattice.square((0.25 * d, 0.25 * d), sq_size)
    sq2 = lattice.square((0.75 * d, 0.75 * d), sq_size)
    epsgrid[sq1] = 1
    epsgrid[sq2] = 1

    sup = lattice.Layer("Superstrate", epsilon=eps_diel)
    sub = lattice.Layer("Substrate", epsilon=1)
    st = lattice.Layer("Structured", wavelength)
    st.epsilon = epsgrid

    sim = nn.Simulation([sup, st, sub], pw, nh, formulation=formulation)
    # this actually corresponds to order (0,-1) for the other unit cell in [Li1997]
    order = (-1, -1)
    R, T = sim.diffraction_efficiencies(orders=True)
    t = sim.get_order(T, order)
    return t, sim


def checkerboard_cellB(nh, formulation):
    d = sq_size * 2**0.5
    Nx = 2**9
    Ny = 2**9
    lattice = nn.Lattice(([d, 0], [0, d]), discretization=(Nx, Ny))
    pw = nn.PlaneWave(wavelength=wavelength, angles=(0, 45, 0))
    epsgrid = lattice.ones() * eps_diel
    sq = lattice.square((0.5 * d, 0.5 * d), sq_size, rotate=45)
    epsgrid[sq] = 1

    sup = lattice.Layer("Superstrate", epsilon=eps_diel)
    sub = lattice.Layer("Substrate", epsilon=1)
    st = lattice.Layer("Structured", wavelength)
    st.epsilon = epsgrid

    # st.plot()
    sim = nn.Simulation([sup, st, sub], pw, nh, formulation=formulation)
    order = (0, -1)
    R, T = sim.diffraction_efficiencies(orders=True)
    t = sim.get_order(T, order)
    return t, sim


#########################################################################
# Perform the simulation for different formulations and number
# of retained harmonics:


def plot_cell(sim):
    axin = plt.gca().inset_axes([0.77, 0.0, 0.25, 0.25])
    lay = sim.get_layer_by_name("Structured")
    lay.plot(ax=axin)
    axin.set_axis_off()


NH = [100, 200, 300, 400, 600, 800, 1000]
formulations = ["original", "tangent", "pol", "jones"]

for icell, cell_fun in enumerate([checkerboard_cellA, checkerboard_cellB]):
    celltype = "A" if icell == 0 else "B"

    print("============================")
    print(f"cell type {celltype}")
    print("============================")

    nhs = {f: [] for f in formulations}
    ts = {f: [] for f in formulations}
    times = {f: [] for f in formulations}

    for nh in NH:

        print("number of harmonics = ", nh)

        for formulation in formulations:
            t0 = -time.time()
            t, sim = cell_fun(nh, formulation=formulation)
            t0 += time.time()
            print("formulation = ", formulation)
            print(f"number of harmonics: asked={nh}, actual={sim.nh}")
            print(f"elapsed time = {t0}s")
            print("T(0,-1) = ", t)
            print("-----------------")
            nhs[formulation].append(sim.nh)
            ts[formulation].append(t)
            times[formulation].append(t0)

    #########################################################################
    # Plot the results:

    markers = {"original": "^", "tangent": "o", "jones": "s", "pol": "^"}
    colors = {
        "original": "#d4b533",
        "tangent": "#d46333",
        "jones": "#3395d4",
        "pol": "#54aa71",
    }

    plt.figure()
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
    plt.legend(loc=5, ncols=2)
    plt.xlabel("number of Fourier harmonics $n_h$")
    plt.ylabel("$T_{0,-1}$")
    # plt.ylim(0.1255, 0.129)
    plt.title(f"cell {celltype}")
    plot_cell(sim)
    plt.tight_layout()

    plt.figure()

    for formulation in formulations:
        plt.plot(
            nhs[formulation],
            times[formulation],
            "-",
            color=colors[formulation],
            marker=markers[formulation],
            label=formulation,
        )
        plt.pause(0.1)
    plt.yscale("log")
    plt.legend(ncols=2)
    plt.xlabel("number of Fourier harmonics $n_h$")
    plt.ylabel("CPU time (s)")
    plt.title(f"cell {celltype}")
    plot_cell(sim)
    plt.tight_layout()
