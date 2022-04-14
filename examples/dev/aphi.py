#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT
#
# @article{ou_high_2018,
# 	title = {High efficiency focusing vortex generation and detection with polarization-insensitive dielectric metasurfaces},
# 	volume = {10},
# 	issn = {2040-3372},
# 	doi = {10.1039/C8NR07480A},
# 	language = {en},
# 	number = {40},
# 	urldate = {2022-04-10},
# 	journal = {Nanoscale},
# 	author = {Ou, Kai and Li, Guanhai and Li, Tianxin and Yang, Hui and Yu, Feilong and Chen, Jin and Zhao, Zengyue and Cao, Guangtao and Chen, Xiaoshuang and Lu, Wei},
# 	month = oct,
# 	year = {2018},
# 	note = {Publisher: The Royal Society of Chemistry},
# 	pages = {19154--19161},
# }

import matplotlib.pyplot as plt
import numpy as np

import nannos as nn

plt.ion()

nh = 51
formulation = "original"

wl = 1.55
P = 0.615
H = 0.7
eps_Si = 3.57**2
eps_SiO2 = 1.4648**2
patch_size = np.linspace(0.05, 0.25, 51)


fig, ax = plt.subplots(1, 2)
A, Phi = [], []
for Rad in patch_size:
    lattice = nn.Lattice([[P, 0], [0, P]], discretization=2**9)
    sup = lattice.Layer("Superstrate", epsilon=eps_SiO2)
    sub = lattice.Layer("Substrate", epsilon=1)
    epsilon = lattice.ones()
    metaatom = lattice.circle((0.5 * P, 0.5 * P), Rad)
    epsilon[metaatom] = eps_Si
    ms = lattice.Layer("Metasurface", thickness=H, epsilon=epsilon)
    pw = nn.PlaneWave(wavelength=1 / 1 / wl, angles=(0.0 * nn.pi / 2, 0, 0))
    sim = nn.Simulation([sup, ms, sub], pw, nh=nh, formulation=formulation)
    R, T = sim.diffraction_efficiencies()
    aN = sim.S @ sim.a0
    aN = nn.block(sim.S) @ np.hstack([sim.a0, sim.bN])
    t = sim.get_order(aN, (0, 0))
    t = np.abs(t) ** 2
    # aN = sim.get_field_fourier("Substrate")[0, 0, 0:3]
    # tx = sim.get_order(aN[0], (0, 0))
    # ty = sim.get_order(aN[1], (0, 0))
    # tz = sim.get_order(aN[2], (0, 0))
    # t = np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2

    b0 = sim.get_field_fourier("Superstrate")[0, 0, 0:3]
    rx = sim.get_order(b0[0], (0, 0))
    ry = sim.get_order(b0[1], (0, 0))
    rz = sim.get_order(b0[2], (0, 0))
    r = np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2
    qx = sim.get_order(sim.kx, (0, 0))
    qy = sim.get_order(sim.ky, (0, 0))

    p = (sim.omega**2 - qx**2 - qy**2) ** 0.5
    p1 = (sim.omega**2 * eps_SiO2 - qx**2 - qy**2) ** 0.5

    T1 = t * p1 / p

    phi = np.angle(t) / (nn.pi * 2)
    ax[0].plot(Rad, T, "sb")
    ax[0].plot(Rad, T1, "+r")
    ax[1].plot(Rad, phi, "sb")
    plt.pause(0.1)
    A.append(T)
    Phi.append(phi)


Phi = np.array(Phi)
Phi = np.unwrap(Phi * (nn.pi * 2)) / (nn.pi * 2)
Phi -= Phi[0]


fig, ax1 = plt.subplots()
color = "#e47d32"
ax1.set_xlabel(r"radius ($\mu$m)")
ax1.set_ylabel("Amplitude", color=color)
ax1.set_ylim(0, 1.0)
ax1.plot(patch_size, A, color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax2 = ax1.twinx()
color = "#3c75cb"
ax2.set_ylabel(r"Phase ($2\pi$)", color=color)
ax2.plot(patch_size, Phi, color=color)
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylim(0, 1.0)
ax1.set_xlim(0.05, 0.25)
fig.tight_layout()
plt.show()
