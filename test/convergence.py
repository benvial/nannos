#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import matplotlib.pyplot as plt
import numpy as np

from nannos import Lattice, Layer, Pattern, PlaneWave, Simulation, pi

plt.ion()
plt.clf()

la = 1

lattice = Lattice(([2 * 1.25 * la, 0], [0, 2 * 1.25 * la]))

freq = 1 / la
theta = 0.0 * pi / 180
phi = 0.0 * pi / 180
psi = 0.0 * pi / 180


pw = PlaneWave(
    frequency=freq, angles=(theta * pi / 180, phi * pi / 180, psi * pi / 180)
)

Nx = 2 ** 8
Ny = 2 ** 8

eps_pattern = 2.0
eps_hole = 7.0
mu_pattern = 5.0
mu_hole = 3.0

radius = 0.2
x0 = np.linspace(0, 1.0, Nx)
y0 = np.linspace(0, 1.0, Ny)
x, y = np.meshgrid(x0, y0, indexing="ij")
hole = (x - 0.5) ** 2 + (y - 0.5) ** 2 < radius ** 2

sup = Layer("Superstrate", epsilon=1, mu=1)
sub = Layer("Substrate", epsilon=1, mu=1)

ids = np.ones((Nx, Ny), dtype=float)
zs = np.zeros_like(ids)

epsgrid = ids * 2.25
epsgrid[np.logical_and(x > 0.5, y > 0.5)] = 1
epsgrid[np.logical_and(x < 0.5, y < 0.5)] = 1
mugrid = ids

sup = Layer("Superstrate", epsilon=2.25, mu=1)
sub = Layer("Substrate", epsilon=1, mu=1)
pattern = Pattern(epsgrid, mugrid)
st = Layer("Structured", la)
st.add_pattern(pattern)
#
#
# z = 0 * ids
# from nannos.formulations.tangent import get_tangent_field
#
# epsgrid = np.array([[epsgrid, z, z], [z, epsgrid, z], [z, z, epsgrid]])
#
# t = get_tangent_field(epsgrid[0, 0, :, :])
#
# plt.pcolor(x, y, epsgrid[0, 0, :, :], cmap="Pastel1", shading="auto")
# plt.colorbar()
# plt.quiver(x, y, *t, scale=10)
# plt.axis("scaled")


def checkerboard(nG, formulation="fft"):
    simu = Simulation(lattice, [sup, st, sub], pw, nG, formulation=formulation)
    order = (1, 1)
    R, T = simu.diffraction_efficiencies(True)
    t = simu.get_order(T, order)
    return R, T, t, simu


plt.clf()
NG = np.arange(100, 600, 100, dtype=int)
tt = []
ngs = []
tt1 = []
ngs1 = []

formulations = ["original", "normal", "jones", "pol"]

ngs = {f: [] for f in formulations}
tt = {f: [] for f in formulations}
markers = {"original": "^", "normal": "x", "jones": "s", "pol": "o"}
colors = {
    "original": "#d4b533",
    "normal": "#d46333",
    "jones": "#3395d4",
    "pol": "#33d48b",
}

for nG in NG:
    for formulation in formulations:
        R, T, t, simu = checkerboard(nG, formulation=formulation)
        R = np.sum(R)
        T = np.sum(T)
        print("formulation = ", formulation)
        print("nG0 = ", nG)
        print("nG = ", simu.nG)
        print("t = ", t)
        print("R = ", R)
        print("T = ", T)
        print("R+T = ", R + T)
        print("-----------------")
        ngs[formulation].append(simu.nG)
        tt[formulation].append(t)
        # plt.clf()
        plt.plot(
            ngs[formulation],
            tt[formulation],
            "-",
            color=colors[formulation],
            marker=markers[formulation],
        )
        plt.pause(0.1)
